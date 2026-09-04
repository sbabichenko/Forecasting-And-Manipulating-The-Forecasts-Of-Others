#!/usr/bin/env python3
"""Experimental neural/Galerkin solver for the stationary two-sided equations.

This script is intentionally separate from the production C++ stationary solver.
It uses fixed random tanh features and trains only the linear readout weights.
The default backend uses JAX when available, giving a JIT-compiled residual and
an exact autodiff Jacobian for SciPy's least-squares driver. The NumPy residual
is retained as a fallback and for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


D_W = 3
E0 = np.array([1.0, 0.0, 0.0])
E1 = np.array([0.0, 1.0, 0.0])
E2 = np.array([0.0, 0.0, 1.0])

PI1 = np.diag([0.0, 1.0, 0.0])
PI2 = np.diag([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Config:
    p1: float
    p2: float
    r1: float
    r2: float
    sigma: float
    A: float
    n: int
    lag_max: float
    quadrature: str
    inner_quad: int
    n_features: int
    seed: int
    max_nfev: int
    tol: float
    accept_rms: float
    coef_penalty: float
    w_state: float
    w_filter: float
    w_hx: float
    w_H: float
    w_wedge: float
    w_policy: float
    w_boundary: float
    w_tail_left: float


class FeatureMap:
    """One-hidden-layer tanh feature map with analytic first derivatives."""

    def __init__(
        self,
        dim: int,
        n_hidden: int,
        bounds: Iterable[Tuple[float, float]],
        rng: np.random.Generator,
    ) -> None:
        self.dim = dim
        self.n_hidden = n_hidden
        self.bounds = np.array(list(bounds), dtype=float)
        center = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        half_width = 0.5 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.center = center
        self.half_width = np.maximum(half_width, 1e-12)
        self.W = rng.normal(loc=0.0, scale=1.35, size=(n_hidden, dim))
        self.b = rng.uniform(low=-1.0, high=1.0, size=n_hidden)
        self.n_out = 1 + dim + n_hidden

    def scaled(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, self.dim)
        return (x - self.center) / self.half_width

    def eval(self, x: np.ndarray) -> np.ndarray:
        u = self.scaled(x)
        z = u @ self.W.T + self.b
        hidden = np.tanh(z)
        return np.concatenate([np.ones((u.shape[0], 1)), u, hidden], axis=1)

    def deriv(self, x: np.ndarray, axis: int) -> np.ndarray:
        u = self.scaled(x)
        z = u @ self.W.T + self.b
        hidden_deriv = (1.0 - np.tanh(z) ** 2) * self.W[None, :, axis]
        out = np.zeros((u.shape[0], self.n_out))
        out[:, 1 + axis] = 1.0
        out[:, 1 + self.dim :] = hidden_deriv
        return out / self.half_width[axis]


@dataclass
class Basis:
    a: np.ndarray
    b: np.ndarray
    h: float
    offset: int
    weights_a: np.ndarray
    inner_unit_nodes: np.ndarray
    inner_unit_weights: np.ndarray
    fmap_a: FeatureMap
    fmap_b: FeatureMap
    fmap_ab: FeatureMap
    phi_a: np.ndarray
    dphi_a: np.ndarray
    phi_b: np.ndarray
    dphi_b: np.ndarray
    phi_ab: np.ndarray
    dphi_ab_a: np.ndarray
    dphi_ab_b: np.ndarray


class Packer:
    def __init__(self, basis: Basis) -> None:
        self.specs: List[Tuple[str, str, int]] = [
            ("x", "a", D_W),
            ("d1", "a", D_W),
            ("d2", "a", D_W),
            ("c1", "a", D_W),
            ("c2", "a", D_W),
            ("xt1", "a", D_W),
            ("xt2", "a", D_W),
            ("hx1", "b", D_W),
            ("hx2", "b", D_W),
            ("w1", "b", D_W),
            ("w2", "b", D_W),
            ("H1", "ab", D_W * D_W),
            ("H2", "ab", D_W * D_W),
        ]
        self.feature_sizes = {
            "a": basis.fmap_a.n_out,
            "b": basis.fmap_b.n_out,
            "ab": basis.fmap_ab.n_out,
        }
        self.offsets: Dict[str, Tuple[int, int, str, int]] = {}
        pos = 0
        for name, domain, out_dim in self.specs:
            size = self.feature_sizes[domain] * out_dim
            self.offsets[name] = (pos, pos + size, domain, out_dim)
            pos += size
        self.size = pos

    def coeff(self, theta: np.ndarray, name: str) -> np.ndarray:
        lo, hi, domain, out_dim = self.offsets[name]
        return theta[lo:hi].reshape(self.feature_sizes[domain], out_dim)

    def set_coeff(self, theta: np.ndarray, name: str, coeff: np.ndarray) -> None:
        lo, hi, _, _ = self.offsets[name]
        theta[lo:hi] = coeff.reshape(-1)


def make_basis(cfg: Config) -> Basis:
    n = max(5, int(cfg.n))
    lag_max = max(1e-8, float(cfg.lag_max))
    h = lag_max / (n - 1)
    if cfg.quadrature == "gauss":
        a_unit, a_w_unit = np.polynomial.legendre.leggauss(n)
        a = 0.5 * lag_max * (a_unit + 1.0)
        weights = 0.5 * lag_max * a_w_unit
        b_unit, _ = np.polynomial.legendre.leggauss(2 * n - 1)
        b = lag_max * b_unit
        offset = int(np.searchsorted(b, 0.0, side="left"))
    else:
        a = np.linspace(0.0, lag_max, n)
        b = np.linspace(-lag_max, lag_max, 2 * n - 1)
        weights = np.full(n, h)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        offset = n - 1
    inner_n = max(2, int(cfg.inner_quad or n))
    inner_unit_nodes, inner_unit_weights = np.polynomial.legendre.leggauss(inner_n)

    rng = np.random.default_rng(cfg.seed)
    fmap_a = FeatureMap(1, cfg.n_features, [(0.0, lag_max)], rng)
    fmap_b = FeatureMap(1, cfg.n_features, [(-lag_max, lag_max)], rng)
    fmap_ab = FeatureMap(2, cfg.n_features, [(0.0, lag_max), (-lag_max, lag_max)], rng)

    aa, bb = np.meshgrid(a, b, indexing="ij")
    ab = np.column_stack([aa.reshape(-1), bb.reshape(-1)])

    return Basis(
        a=a,
        b=b,
        h=h,
        offset=offset,
        weights_a=weights,
        inner_unit_nodes=inner_unit_nodes,
        inner_unit_weights=inner_unit_weights,
        fmap_a=fmap_a,
        fmap_b=fmap_b,
        fmap_ab=fmap_ab,
        phi_a=fmap_a.eval(a[:, None]),
        dphi_a=fmap_a.deriv(a[:, None], 0),
        phi_b=fmap_b.eval(b[:, None]),
        dphi_b=fmap_b.deriv(b[:, None], 0),
        phi_ab=fmap_ab.eval(ab),
        dphi_ab_a=fmap_ab.deriv(ab, 0),
        dphi_ab_b=fmap_ab.deriv(ab, 1),
    )


def fit_coeff(phi: np.ndarray, target: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    gram = phi.T @ phi
    rhs = phi.T @ target
    return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)


def initial_theta(cfg: Config, basis: Basis, packer: Packer) -> np.ndarray:
    theta = np.zeros(packer.size)
    inv_sum = 1.0 / max(cfg.r1, 1e-10) + 1.0 / max(cfg.r2, 1e-10)
    s = math.sqrt(1.0 / max(inv_sum, 1e-12))
    decay = inv_sum * s

    x0 = (cfg.sigma * np.exp(-decay * basis.a))[:, None] * E0[None, :]
    d1 = -(s / max(cfg.r1, 1e-10)) * x0
    d2 = -(s / max(cfg.r2, 1e-10)) * x0
    c1 = d1 @ PI1.T
    c2 = d2 @ PI2.T
    xt1 = x0 - x0 @ PI1.T
    xt2 = x0 - x0 @ PI2.T

    for name, target in [
        ("x", x0),
        ("d1", d1),
        ("d2", d2),
        ("c1", c1),
        ("c2", c2),
        ("xt1", xt1),
        ("xt2", xt2),
    ]:
        packer.set_coeff(theta, name, fit_coeff(basis.phi_a, target))

    hx1 = np.zeros((basis.b.size, D_W))
    hx2 = np.zeros((basis.b.size, D_W))
    positive = basis.b >= 0.0
    if np.any(positive):
        x_pos = (cfg.sigma * np.exp(-decay * basis.b[positive]))[:, None] * E0[None, :]
        hx1[positive] = cfg.r1 * (s / max(cfg.r1, 1e-10)) * x_pos
        hx2[positive] = cfg.r2 * (s / max(cfg.r2, 1e-10)) * x_pos
    packer.set_coeff(theta, "hx1", fit_coeff(basis.phi_b, hx1))
    packer.set_coeff(theta, "hx2", fit_coeff(basis.phi_b, hx2))
    return theta


def eval_state(theta: np.ndarray, basis: Basis, packer: Packer) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    dout: Dict[str, np.ndarray] = {}
    for name, domain, out_dim in packer.specs:
        coeff = packer.coeff(theta, name)
        if domain == "a":
            out[name] = basis.phi_a @ coeff
            dout[name + "_da"] = basis.dphi_a @ coeff
        elif domain == "b":
            out[name] = basis.phi_b @ coeff
            dout[name + "_db"] = basis.dphi_b @ coeff
        else:
            n = basis.a.size
            nb = basis.b.size
            val = basis.phi_ab @ coeff
            da = basis.dphi_ab_a @ coeff
            db = basis.dphi_ab_b @ coeff
            out[name] = val.reshape(n, nb, D_W, D_W)
            dout[name + "_da"] = da.reshape(n, nb, D_W, D_W)
            dout[name + "_db"] = db.reshape(n, nb, D_W, D_W)
    out.update(dout)
    return out


def eval_a_coeff(basis: Basis, coeff: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1)
    return basis.fmap_a.eval(pts[:, None]) @ coeff


def eval_b_coeff(basis: Basis, coeff: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1)
    return basis.fmap_b.eval(pts[:, None]) @ coeff


def eval_ab_coeff(
    basis: Basis,
    coeff: np.ndarray,
    a_points: np.ndarray,
    b_points: np.ndarray,
) -> np.ndarray:
    a_pts = np.asarray(a_points, dtype=float).reshape(-1)
    b_pts = np.asarray(b_points, dtype=float).reshape(-1)
    if a_pts.shape != b_pts.shape:
        raise ValueError("a_points and b_points must have the same shape")
    pts = np.column_stack([a_pts, b_pts])
    return (basis.fmap_ab.eval(pts) @ coeff).reshape(-1, D_W, D_W)


def eval_a_name(theta: np.ndarray, basis: Basis, packer: Packer, name: str, points: np.ndarray) -> np.ndarray:
    return eval_a_coeff(basis, packer.coeff(theta, name), points)


def eval_b_name(theta: np.ndarray, basis: Basis, packer: Packer, name: str, points: np.ndarray) -> np.ndarray:
    return eval_b_coeff(basis, packer.coeff(theta, name), points)


def eval_ab_name(
    theta: np.ndarray,
    basis: Basis,
    packer: Packer,
    name: str,
    a_points: np.ndarray,
    b_points: np.ndarray,
) -> np.ndarray:
    return eval_ab_coeff(basis, packer.coeff(theta, name), a_points, b_points)


def causal_on_b(theta: np.ndarray, basis: Basis, packer: Packer) -> np.ndarray:
    out = np.zeros((basis.b.size, D_W))
    positive = basis.b >= 0.0
    if np.any(positive):
        out[positive] = eval_a_name(theta, basis, packer, "x", basis.b[positive])
    return out


def stationary_filter_kernel(
    a_val: float,
    b_val: float,
    xtilde_coeff: np.ndarray,
    obs_gain: float,
    obs_idx: int,
    precision: float,
    basis: Basis,
) -> np.ndarray:
    e = np.zeros(D_W)
    e[obs_idx] = 1.0
    f = np.zeros((D_W, D_W))
    if a_val > b_val + 1e-12:
        f += obs_gain * np.outer(eval_a_coeff(basis, xtilde_coeff, np.array([a_val - b_val]))[0], e)
    elif b_val > a_val + 1e-12:
        f += obs_gain * np.outer(e, eval_a_coeff(basis, xtilde_coeff, np.array([b_val - a_val]))[0])

    upper = min(a_val, b_val)
    if upper > 1e-14:
        c = 0.5 * upper * (basis.inner_unit_nodes + 1.0)
        w = 0.5 * upper * basis.inner_unit_weights
        xa = eval_a_coeff(basis, xtilde_coeff, a_val - c)
        xb = eval_a_coeff(basis, xtilde_coeff, b_val - c)
        for wi, vai, vbi in zip(w, xa, xb):
            f += wi * precision * np.outer(vai, vbi)
    return f


def stationary_filter_matrix(
    a_points: np.ndarray,
    b_points: np.ndarray,
    xtilde_coeff: np.ndarray,
    obs_gain: float,
    obs_idx: int,
    precision: float,
    basis: Basis,
) -> np.ndarray:
    a_pts = np.asarray(a_points, dtype=float).reshape(-1)
    b_pts = np.asarray(b_points, dtype=float).reshape(-1)
    na = a_pts.size
    nb = b_pts.size
    f = np.zeros((na, nb, D_W, D_W))
    e = np.zeros(D_W)
    e[obs_idx] = 1.0

    diff = a_pts[:, None] - b_pts[None, :]
    gt = diff > 1e-12
    lt = diff < -1e-12
    if np.any(gt):
        vals = eval_a_coeff(basis, xtilde_coeff, diff[gt])
        rows, cols = np.nonzero(gt)
        f[rows, cols] += obs_gain * vals[:, :, None] * e[None, None, :]
    if np.any(lt):
        vals = eval_a_coeff(basis, xtilde_coeff, (-diff)[lt])
        rows, cols = np.nonzero(lt)
        f[rows, cols] += obs_gain * e[None, :, None] * vals[:, None, :]

    upper = np.minimum(a_pts[:, None], b_pts[None, :])
    active = upper > 1e-14
    if np.any(active):
        rows, cols = np.nonzero(active)
        u = upper[rows, cols]
        c_nodes = 0.5 * u[:, None] * (basis.inner_unit_nodes[None, :] + 1.0)
        c_weights = 0.5 * u[:, None] * basis.inner_unit_weights[None, :]
        left = eval_a_coeff(
            basis,
            xtilde_coeff,
            (a_pts[rows, None] - c_nodes).reshape(-1),
        ).reshape(rows.size, basis.inner_unit_nodes.size, D_W)
        right = eval_a_coeff(
            basis,
            xtilde_coeff,
            (b_pts[cols, None] - c_nodes).reshape(-1),
        ).reshape(rows.size, basis.inner_unit_nodes.size, D_W)
        f[rows, cols] += precision * np.einsum(
            "pq,pqr,pqs->prs", c_weights, left, right
        )
    return f


def project_filter(
    x: np.ndarray,
    d: np.ndarray,
    xtilde_coeff: np.ndarray,
    obs_gain: float,
    obs_idx: int,
    precision: float,
    pi: np.ndarray,
    basis: Basis,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = stationary_filter_matrix(
        basis.a, basis.a, xtilde_coeff, obs_gain, obs_idx, precision, basis
    )
    xhat = x @ pi.T + np.einsum("a,abrj,ar->bj", basis.weights_a, f, x)
    chat = d @ pi.T + np.einsum("a,abrj,ar->bj", basis.weights_a, f, d)
    zero_rows = np.where(np.isclose(basis.a, 0.0, atol=1e-14))[0]
    for row in zero_rows:
        chat[row] = pi @ chat[row]
    return xhat, x - xhat, chat


def project_filter_at(
    x: np.ndarray,
    d: np.ndarray,
    x_b: np.ndarray,
    d_b: np.ndarray,
    xtilde_coeff: np.ndarray,
    obs_gain: float,
    obs_idx: int,
    precision: float,
    pi: np.ndarray,
    basis: Basis,
    b_val: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = stationary_filter_matrix(
        basis.a, np.array([b_val]), xtilde_coeff, obs_gain, obs_idx, precision, basis
    )[:, 0]
    xb = pi @ x_b + np.einsum("a,arj,ar->j", basis.weights_a, f, x)
    cb = pi @ d_b + np.einsum("a,arj,ar->j", basis.weights_a, f, d)
    # At zero lag, a player can react to the contemporaneously observed
    # innovation coordinate. Only unobserved primitive shock coordinates are
    # forced to have no instantaneous action load.
    if abs(b_val) <= 1e-14:
        cb = pi @ cb
    return xb, x_b - xb, cb


def wedge_from_H(
    H: np.ndarray,
    xtilde_k: np.ndarray,
    H_coeff: np.ndarray,
    obs_gain_k: float,
    obs_idx_k: int,
    precision_k: float,
    basis: Basis,
) -> np.ndarray:
    e_obs = np.zeros(D_W)
    e_obs[obs_idx_k] = 1.0
    acc = np.zeros((basis.b.size, D_W))
    integral = np.einsum("a,abrs,ar->bs", basis.weights_a, H, xtilde_k)
    # Diagonal innovation-birth term:
    # Gamma^T E H^k(0,b), from delta w^k_t(t).
    H0b = eval_ab_coeff(basis, H_coeff, np.zeros_like(basis.b), basis.b)
    diagonal = obs_gain_k * np.einsum("brs,r->bs", H0b, e_obs)
    acc = diagonal + precision_k * integral
    return acc


def residual_blocks(
    theta: np.ndarray,
    cfg: Config,
    basis: Basis,
    packer: Packer,
) -> List[Tuple[str, np.ndarray]]:
    y = eval_state(theta, basis, packer)
    p1 = max(cfg.p1, 0.0)
    p2 = max(cfg.p2, 0.0)
    g1 = math.sqrt(p1)
    g2 = math.sqrt(p2)

    x = y["x"]
    d1 = y["d1"]
    d2 = y["d2"]
    c1 = y["c1"]
    c2 = y["c2"]
    xt1 = y["xt1"]
    xt2 = y["xt2"]
    hx1 = y["hx1"]
    hx2 = y["hx2"]
    w1 = y["w1"]
    w2 = y["w2"]
    H1 = y["H1"]
    H2 = y["H2"]

    x_ext = causal_on_b(theta, basis, packer)
    hx1_pos = eval_b_name(theta, basis, packer, "hx1", basis.a)
    hx2_pos = eval_b_name(theta, basis, packer, "hx2", basis.a)
    x0 = eval_a_name(theta, basis, packer, "x", np.array([0.0]))[0]
    x_tail = eval_a_name(theta, basis, packer, "x", np.array([cfg.lag_max]))[0]
    d1_0 = eval_a_name(theta, basis, packer, "d1", np.array([0.0]))[0]
    d2_0 = eval_a_name(theta, basis, packer, "d2", np.array([0.0]))[0]
    c1_0 = eval_a_name(theta, basis, packer, "c1", np.array([0.0]))[0]
    c2_0 = eval_a_name(theta, basis, packer, "c2", np.array([0.0]))[0]
    xt1_0 = eval_a_name(theta, basis, packer, "xt1", np.array([0.0]))[0]
    xt2_0 = eval_a_name(theta, basis, packer, "xt2", np.array([0.0]))[0]
    hx1_left = eval_b_name(theta, basis, packer, "hx1", np.array([-cfg.lag_max]))[0]
    hx2_left = eval_b_name(theta, basis, packer, "hx2", np.array([-cfg.lag_max]))[0]
    hx1_right = eval_b_name(theta, basis, packer, "hx1", np.array([cfg.lag_max]))[0]
    hx2_right = eval_b_name(theta, basis, packer, "hx2", np.array([cfg.lag_max]))[0]
    H1_a_tail = eval_ab_name(theta, basis, packer, "H1", np.full_like(basis.b, cfg.lag_max), basis.b)
    H2_a_tail = eval_ab_name(theta, basis, packer, "H2", np.full_like(basis.b, cfg.lag_max), basis.b)
    H1_b_right = eval_ab_name(theta, basis, packer, "H1", basis.a, np.full_like(basis.a, cfg.lag_max))
    H2_b_right = eval_ab_name(theta, basis, packer, "H2", basis.a, np.full_like(basis.a, cfg.lag_max))

    _, xt1_projected, c1_projected = project_filter(
        x, d1, packer.coeff(theta, "xt1"), g1, 1, p1, PI1, basis
    )
    _, xt2_projected, c2_projected = project_filter(
        x, d2, packer.coeff(theta, "xt2"), g2, 2, p2, PI2, basis
    )
    _, xt1_0_projected, c1_0_projected = project_filter_at(
        x, d1, x0, d1_0, packer.coeff(theta, "xt1"), g1, 1, p1, PI1, basis, 0.0
    )
    _, xt2_0_projected, c2_0_projected = project_filter_at(
        x, d2, x0, d2_0, packer.coeff(theta, "xt2"), g2, 2, p2, PI2, basis, 0.0
    )

    w1_projected = wedge_from_H(H1, xt2, packer.coeff(theta, "H1"), g2, 2, p2, basis)
    w2_projected = wedge_from_H(H2, xt1, packer.coeff(theta, "H2"), g1, 1, p1, basis)

    H1_res = (
        y["H1_da"]
        + y["H1_db"]
        + d2[:, None, :, None] * hx1[None, :, None, :]
        - x[:, None, :, None] * w1[None, :, None, :]
    )
    H2_res = (
        y["H2_da"]
        + y["H2_db"]
        + d1[:, None, :, None] * hx2[None, :, None, :]
        - x[:, None, :, None] * w2[None, :, None, :]
    )

    blocks: List[Tuple[str, np.ndarray]] = [
        ("state", cfg.w_state * (y["x_da"] - cfg.A * x - c1 - c2)),
        ("filter1_xtilde", cfg.w_filter * (xt1 - xt1_projected)),
        ("filter1_control", cfg.w_filter * (c1 - c1_projected)),
        ("filter1_zero_xtilde", cfg.w_filter * (xt1_0 - xt1_0_projected)),
        ("filter1_zero_control", cfg.w_filter * (c1_0 - c1_0_projected)),
        ("filter2_xtilde", cfg.w_filter * (xt2 - xt2_projected)),
        ("filter2_control", cfg.w_filter * (c2 - c2_projected)),
        ("filter2_zero_xtilde", cfg.w_filter * (xt2_0 - xt2_0_projected)),
        ("filter2_zero_control", cfg.w_filter * (c2_0 - c2_0_projected)),
        ("hx1", cfg.w_hx * (-y["hx1_db"] - x_ext - cfg.A * hx1 - w1)),
        ("hx2", cfg.w_hx * (-y["hx2_db"] - x_ext - cfg.A * hx2 - w2)),
        ("H1", cfg.w_H * H1_res),
        ("H2", cfg.w_H * H2_res),
        ("wedge1", cfg.w_wedge * (w1 - w1_projected)),
        ("wedge2", cfg.w_wedge * (w2 - w2_projected)),
        ("policy1", cfg.w_policy * (d1 + hx1_pos / max(cfg.r1, 1e-10))),
        ("policy2", cfg.w_policy * (d2 + hx2_pos / max(cfg.r2, 1e-10))),
        ("x0", cfg.w_boundary * (x0 - cfg.sigma * E0)),
        ("control1_zero_unobserved", cfg.w_boundary * ((np.eye(D_W) - PI1) @ c1_0)),
        ("control2_zero_unobserved", cfg.w_boundary * ((np.eye(D_W) - PI2) @ c2_0)),
        ("x_tail", cfg.w_boundary * x_tail),
        ("hx1_right", cfg.w_boundary * hx1_right),
        ("hx2_right", cfg.w_boundary * hx2_right),
        ("H1_a_tail", cfg.w_boundary * H1_a_tail),
        ("H2_a_tail", cfg.w_boundary * H2_a_tail),
        ("H1_b_right", cfg.w_boundary * H1_b_right),
        ("H2_b_right", cfg.w_boundary * H2_b_right),
        ("left_tail", cfg.w_tail_left * np.stack([hx1_left, hx2_left], axis=0)),
    ]
    if cfg.coef_penalty > 0.0:
        blocks.append(("coef_penalty", cfg.coef_penalty * theta))
    return blocks


def residual_vector(theta: np.ndarray, cfg: Config, basis: Basis, packer: Packer) -> np.ndarray:
    blocks = residual_blocks(theta, cfg, basis, packer)
    return np.concatenate([block.reshape(-1) for _, block in blocks])


class JaxResidual:
    """JIT-compiled residual and exact dense Jacobian for the stationary fit."""

    def __init__(self, cfg: Config, basis: Basis, packer: Packer) -> None:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX backend requested, but JAX is not importable")
        self.cfg = cfg
        self.packer = packer
        self.n = int(basis.a.size)
        self.nb = int(basis.b.size)
        self.coef_penalty = float(cfg.coef_penalty)

        self.a = jnp.asarray(basis.a, dtype=jnp.float64)
        self.b = jnp.asarray(basis.b, dtype=jnp.float64)
        self.weights_a = jnp.asarray(basis.weights_a, dtype=jnp.float64)
        self.phi_a = jnp.asarray(basis.phi_a, dtype=jnp.float64)
        self.dphi_a = jnp.asarray(basis.dphi_a, dtype=jnp.float64)
        self.phi_b = jnp.asarray(basis.phi_b, dtype=jnp.float64)
        self.dphi_b = jnp.asarray(basis.dphi_b, dtype=jnp.float64)
        self.phi_ab = jnp.asarray(basis.phi_ab, dtype=jnp.float64)
        self.dphi_ab_a = jnp.asarray(basis.dphi_ab_a, dtype=jnp.float64)
        self.dphi_ab_b = jnp.asarray(basis.dphi_ab_b, dtype=jnp.float64)

        self.e0 = jnp.asarray(E0, dtype=jnp.float64)
        self.e1 = jnp.asarray(E1, dtype=jnp.float64)
        self.e2 = jnp.asarray(E2, dtype=jnp.float64)
        self.pi1 = jnp.asarray(PI1, dtype=jnp.float64)
        self.pi2 = jnp.asarray(PI2, dtype=jnp.float64)
        self.eye = jnp.eye(D_W, dtype=jnp.float64)
        self.zero_a_mask = jnp.asarray(np.isclose(basis.a, 0.0, atol=1e-14))
        self.b_nonnegative = jnp.asarray((basis.b >= 0.0)[:, None], dtype=jnp.float64)

        self.phi_a_zero = jnp.asarray(basis.fmap_a.eval(np.array([[0.0]])), dtype=jnp.float64)
        self.phi_a_tail = jnp.asarray(basis.fmap_a.eval(np.array([[cfg.lag_max]])), dtype=jnp.float64)
        self.phi_b_left = jnp.asarray(basis.fmap_b.eval(np.array([[-cfg.lag_max]])), dtype=jnp.float64)
        self.phi_b_right = jnp.asarray(basis.fmap_b.eval(np.array([[cfg.lag_max]])), dtype=jnp.float64)
        self.phi_b_at_a = jnp.asarray(basis.fmap_b.eval(basis.a[:, None]), dtype=jnp.float64)
        self.phi_a_at_b = jnp.asarray(basis.fmap_a.eval(np.maximum(basis.b, 0.0)[:, None]), dtype=jnp.float64)
        self.phi_ab_0b = jnp.asarray(
            basis.fmap_ab.eval(np.column_stack([np.zeros_like(basis.b), basis.b])),
            dtype=jnp.float64,
        )
        self.phi_ab_a_tail = jnp.asarray(
            basis.fmap_ab.eval(np.column_stack([np.full_like(basis.b, cfg.lag_max), basis.b])),
            dtype=jnp.float64,
        )
        self.phi_ab_b_right = jnp.asarray(
            basis.fmap_ab.eval(np.column_stack([basis.a, np.full_like(basis.a, cfg.lag_max)])),
            dtype=jnp.float64,
        )

        self.filter_aa = self._make_filter_cache(basis.a, basis.a, basis)
        self.filter_a0 = self._make_filter_cache(basis.a, np.array([0.0]), basis)

        self.offsets = packer.offsets
        self.feature_sizes = packer.feature_sizes
        self._value_jac = jax.jit(self._build_value_jac())
        self._cache_x: np.ndarray | None = None
        self._cache_residual: np.ndarray | None = None
        self._cache_jacobian: np.ndarray | None = None

    @staticmethod
    def _make_filter_cache(a_points: np.ndarray, b_points: np.ndarray, basis: Basis):
        a_pts = np.asarray(a_points, dtype=float).reshape(-1)
        b_pts = np.asarray(b_points, dtype=float).reshape(-1)
        diff = a_pts[:, None] - b_pts[None, :]
        abs_diff = np.abs(diff)
        upper = np.maximum(np.minimum(a_pts[:, None], b_pts[None, :]), 0.0)
        c_nodes = 0.5 * upper[:, :, None] * (basis.inner_unit_nodes[None, None, :] + 1.0)
        c_weights = 0.5 * upper[:, :, None] * basis.inner_unit_weights[None, None, :]
        left_points = a_pts[:, None, None] - c_nodes
        right_points = b_pts[None, :, None] - c_nodes
        return {
            "gt": jnp.asarray(diff > 1e-12),
            "lt": jnp.asarray(diff < -1e-12),
            "phi_abs": jnp.asarray(
                basis.fmap_a.eval(abs_diff.reshape(-1, 1)).reshape(*diff.shape, basis.fmap_a.n_out),
                dtype=jnp.float64,
            ),
            "weights": jnp.asarray(c_weights, dtype=jnp.float64),
            "phi_left": jnp.asarray(
                basis.fmap_a.eval(left_points.reshape(-1, 1)).reshape(*diff.shape, -1, basis.fmap_a.n_out),
                dtype=jnp.float64,
            ),
            "phi_right": jnp.asarray(
                basis.fmap_a.eval(right_points.reshape(-1, 1)).reshape(*diff.shape, -1, basis.fmap_a.n_out),
                dtype=jnp.float64,
            ),
        }

    def _coeff(self, theta, name: str):
        lo, hi, domain, out_dim = self.offsets[name]
        return theta[lo:hi].reshape((self.feature_sizes[domain], out_dim))

    def _eval_state(self, theta):
        out = {}
        for name, domain, _ in self.packer.specs:
            coeff = self._coeff(theta, name)
            if domain == "a":
                out[name] = self.phi_a @ coeff
                out[name + "_da"] = self.dphi_a @ coeff
            elif domain == "b":
                out[name] = self.phi_b @ coeff
                out[name + "_db"] = self.dphi_b @ coeff
            else:
                out[name] = (self.phi_ab @ coeff).reshape((self.n, self.nb, D_W, D_W))
                out[name + "_da"] = (self.dphi_ab_a @ coeff).reshape((self.n, self.nb, D_W, D_W))
                out[name + "_db"] = (self.dphi_ab_b @ coeff).reshape((self.n, self.nb, D_W, D_W))
        return out

    def _filter_matrix(self, xtilde_coeff, obs_gain: float, e_obs, precision: float, cache):
        vals = jnp.einsum("...f,fd->...d", cache["phi_abs"], xtilde_coeff)
        direct_gt = vals[..., :, None] * e_obs[None, None, None, :]
        direct_lt = e_obs[None, None, :, None] * vals[..., None, :]
        direct = obs_gain * (
            jnp.where(cache["gt"][..., None, None], direct_gt, 0.0)
            + jnp.where(cache["lt"][..., None, None], direct_lt, 0.0)
        )
        left = jnp.einsum("...qf,fd->...qd", cache["phi_left"], xtilde_coeff)
        right = jnp.einsum("...qf,fd->...qd", cache["phi_right"], xtilde_coeff)
        integral = precision * jnp.einsum("...q,...qr,...qs->...rs", cache["weights"], left, right)
        return direct + integral

    def _project_filter(self, x, d, xtilde_coeff, obs_gain: float, e_obs, precision: float, pi):
        f = self._filter_matrix(xtilde_coeff, obs_gain, e_obs, precision, self.filter_aa)
        xhat = x @ pi.T + jnp.einsum("a,abrj,ar->bj", self.weights_a, f, x)
        chat = d @ pi.T + jnp.einsum("a,abrj,ar->bj", self.weights_a, f, d)
        chat = jnp.where(self.zero_a_mask[:, None], chat @ pi.T, chat)
        return xhat, x - xhat, chat

    def _project_filter_at_zero(self, x, d, x_b, d_b, xtilde_coeff, obs_gain: float, e_obs, precision: float, pi):
        f = self._filter_matrix(xtilde_coeff, obs_gain, e_obs, precision, self.filter_a0)[:, 0]
        xb = pi @ x_b + jnp.einsum("a,arj,ar->j", self.weights_a, f, x)
        cb = pi @ d_b + jnp.einsum("a,arj,ar->j", self.weights_a, f, d)
        # At zero lag, only the contemporaneously observed innovation coordinate
        # may enter the primitive control.
        cb = pi @ cb
        return xb, x_b - xb, cb

    def _wedge_from_H(self, H, xtilde_k, H_coeff, obs_gain_k: float, e_obs, precision_k: float):
        integral = jnp.einsum("a,abrs,ar->bs", self.weights_a, H, xtilde_k)
        H0b = (self.phi_ab_0b @ H_coeff).reshape((self.nb, D_W, D_W))
        # Diagonal innovation-birth term:
        # Gamma^T E H^k(0,b), from delta w^k_t(t).
        diagonal = obs_gain_k * jnp.einsum("brs,r->bs", H0b, e_obs)
        return diagonal + precision_k * integral

    def _residual(self, theta):
        cfg = self.cfg
        p1 = max(cfg.p1, 0.0)
        p2 = max(cfg.p2, 0.0)
        g1 = math.sqrt(p1)
        g2 = math.sqrt(p2)
        y = self._eval_state(theta)

        x = y["x"]
        d1 = y["d1"]
        d2 = y["d2"]
        c1 = y["c1"]
        c2 = y["c2"]
        xt1 = y["xt1"]
        xt2 = y["xt2"]
        hx1 = y["hx1"]
        hx2 = y["hx2"]
        w1 = y["w1"]
        w2 = y["w2"]
        H1 = y["H1"]
        H2 = y["H2"]

        x_c = self._coeff(theta, "x")
        d1_c = self._coeff(theta, "d1")
        d2_c = self._coeff(theta, "d2")
        c1_c = self._coeff(theta, "c1")
        c2_c = self._coeff(theta, "c2")
        xt1_c = self._coeff(theta, "xt1")
        xt2_c = self._coeff(theta, "xt2")
        hx1_c = self._coeff(theta, "hx1")
        hx2_c = self._coeff(theta, "hx2")
        H1_c = self._coeff(theta, "H1")
        H2_c = self._coeff(theta, "H2")

        x_ext = (self.phi_a_at_b @ x_c) * self.b_nonnegative
        hx1_pos = self.phi_b_at_a @ hx1_c
        hx2_pos = self.phi_b_at_a @ hx2_c
        x0 = (self.phi_a_zero @ x_c)[0]
        x_tail = (self.phi_a_tail @ x_c)[0]
        d1_0 = (self.phi_a_zero @ d1_c)[0]
        d2_0 = (self.phi_a_zero @ d2_c)[0]
        c1_0 = (self.phi_a_zero @ c1_c)[0]
        c2_0 = (self.phi_a_zero @ c2_c)[0]
        xt1_0 = (self.phi_a_zero @ xt1_c)[0]
        xt2_0 = (self.phi_a_zero @ xt2_c)[0]
        hx1_left = (self.phi_b_left @ hx1_c)[0]
        hx2_left = (self.phi_b_left @ hx2_c)[0]
        hx1_right = (self.phi_b_right @ hx1_c)[0]
        hx2_right = (self.phi_b_right @ hx2_c)[0]
        H1_a_tail = (self.phi_ab_a_tail @ H1_c).reshape((self.nb, D_W, D_W))
        H2_a_tail = (self.phi_ab_a_tail @ H2_c).reshape((self.nb, D_W, D_W))
        H1_b_right = (self.phi_ab_b_right @ H1_c).reshape((self.n, D_W, D_W))
        H2_b_right = (self.phi_ab_b_right @ H2_c).reshape((self.n, D_W, D_W))

        _, xt1_projected, c1_projected = self._project_filter(x, d1, xt1_c, g1, self.e1, p1, self.pi1)
        _, xt2_projected, c2_projected = self._project_filter(x, d2, xt2_c, g2, self.e2, p2, self.pi2)
        _, xt1_0_projected, c1_0_projected = self._project_filter_at_zero(
            x, d1, x0, d1_0, xt1_c, g1, self.e1, p1, self.pi1
        )
        _, xt2_0_projected, c2_0_projected = self._project_filter_at_zero(
            x, d2, x0, d2_0, xt2_c, g2, self.e2, p2, self.pi2
        )

        w1_projected = self._wedge_from_H(H1, xt2, H1_c, g2, self.e2, p2)
        w2_projected = self._wedge_from_H(H2, xt1, H2_c, g1, self.e1, p1)

        H1_res = (
            y["H1_da"]
            + y["H1_db"]
            + d2[:, None, :, None] * hx1[None, :, None, :]
            - x[:, None, :, None] * w1[None, :, None, :]
        )
        H2_res = (
            y["H2_da"]
            + y["H2_db"]
            + d1[:, None, :, None] * hx2[None, :, None, :]
            - x[:, None, :, None] * w2[None, :, None, :]
        )

        blocks = [
            cfg.w_state * (y["x_da"] - cfg.A * x - c1 - c2),
            cfg.w_filter * (xt1 - xt1_projected),
            cfg.w_filter * (c1 - c1_projected),
            cfg.w_filter * (xt1_0 - xt1_0_projected),
            cfg.w_filter * (c1_0 - c1_0_projected),
            cfg.w_filter * (xt2 - xt2_projected),
            cfg.w_filter * (c2 - c2_projected),
            cfg.w_filter * (xt2_0 - xt2_0_projected),
            cfg.w_filter * (c2_0 - c2_0_projected),
            cfg.w_hx * (-y["hx1_db"] - x_ext - cfg.A * hx1 - w1),
            cfg.w_hx * (-y["hx2_db"] - x_ext - cfg.A * hx2 - w2),
            cfg.w_H * H1_res,
            cfg.w_H * H2_res,
            cfg.w_wedge * (w1 - w1_projected),
            cfg.w_wedge * (w2 - w2_projected),
            cfg.w_policy * (d1 + hx1_pos / max(cfg.r1, 1e-10)),
            cfg.w_policy * (d2 + hx2_pos / max(cfg.r2, 1e-10)),
            cfg.w_boundary * (x0 - cfg.sigma * self.e0),
            cfg.w_boundary * ((self.eye - self.pi1) @ c1_0),
            cfg.w_boundary * ((self.eye - self.pi2) @ c2_0),
            cfg.w_boundary * x_tail,
            cfg.w_boundary * hx1_right,
            cfg.w_boundary * hx2_right,
            cfg.w_boundary * H1_a_tail,
            cfg.w_boundary * H2_a_tail,
            cfg.w_boundary * H1_b_right,
            cfg.w_boundary * H2_b_right,
            cfg.w_tail_left * jnp.stack([hx1_left, hx2_left], axis=0),
        ]
        if self.coef_penalty > 0.0:
            blocks.append(self.coef_penalty * theta)
        return jnp.concatenate([jnp.ravel(block) for block in blocks])

    def _build_value_jac(self):
        jac_fn = jax.jacfwd(self._residual)

        def value_jac(theta):
            return self._residual(theta), jac_fn(theta)

        return value_jac

    def _evaluate(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta_arr = np.asarray(theta, dtype=float)
        if (
            self._cache_x is not None
            and theta_arr.shape == self._cache_x.shape
            and np.array_equal(theta_arr, self._cache_x)
        ):
            return self._cache_residual, self._cache_jacobian
        residual, jacobian = self._value_jac(jnp.asarray(theta_arr, dtype=jnp.float64))
        self._cache_x = theta_arr.copy()
        self._cache_residual = np.asarray(residual)
        self._cache_jacobian = np.asarray(jacobian)
        return self._cache_residual, self._cache_jacobian

    def fun(self, theta: np.ndarray) -> np.ndarray:
        residual, _ = self._evaluate(theta)
        return residual

    def jac(self, theta: np.ndarray) -> np.ndarray:
        _, jacobian = self._evaluate(theta)
        return jacobian


def block_norms(
    theta: np.ndarray,
    cfg: Config,
    basis: Basis,
    packer: Packer,
) -> Dict[str, float]:
    out = {}
    for name, block in residual_blocks(theta, cfg, basis, packer):
        flat = block.reshape(-1)
        out[name] = float(np.sqrt(np.mean(flat * flat))) if flat.size else 0.0
    return out


def solution_arrays(theta: np.ndarray, basis: Basis, packer: Packer) -> Dict[str, np.ndarray]:
    y = eval_state(theta, basis, packer)
    return {name: y[name] for name in ["x", "d1", "d2", "c1", "c2", "xt1", "xt2", "hx1", "hx2", "w1", "w2"]}


def run_solver(cfg: Config, verbose: bool = False, init_path: str = "", backend: str = "auto"):
    basis = make_basis(cfg)
    packer = Packer(basis)
    theta0 = initial_theta(cfg, basis, packer)
    if init_path:
        with np.load(init_path) as saved:
            saved_theta = np.asarray(saved["theta"], dtype=float)
        if saved_theta.shape != theta0.shape:
            raise ValueError(
                f"init theta has shape {saved_theta.shape}, expected {theta0.shape}; "
                "use matching N/features/seed"
            )
        theta0 = saved_theta.copy()
    if backend == "auto":
        actual_backend = "jax" if JAX_AVAILABLE else "numpy"
    else:
        actual_backend = backend
    if actual_backend == "jax" and not JAX_AVAILABLE:
        raise RuntimeError("JAX backend requested, but JAX is not importable")

    if actual_backend == "jax":
        jax_residual = JaxResidual(cfg, basis, packer)
        initial_residual = jax_residual.fun(theta0)
        result = least_squares(
            jax_residual.fun,
            theta0,
            jac=jax_residual.jac,
            method="trf",
            max_nfev=cfg.max_nfev,
            ftol=cfg.tol,
            xtol=cfg.tol,
            gtol=cfg.tol,
            x_scale="jac",
            verbose=2 if verbose else 0,
        )
        final_residual = jax_residual.fun(result.x)
    else:
        initial_residual = residual_vector(theta0, cfg, basis, packer)
        result = least_squares(
            lambda z: residual_vector(z, cfg, basis, packer),
            theta0,
            method="trf",
            max_nfev=cfg.max_nfev,
            ftol=cfg.tol,
            xtol=cfg.tol,
            gtol=cfg.tol,
            x_scale="jac",
            verbose=2 if verbose else 0,
        )
        final_residual = residual_vector(result.x, cfg, basis, packer)
    rms = float(np.sqrt(np.mean(final_residual * final_residual)))
    initial_rms = float(np.sqrt(np.mean(initial_residual * initial_residual)))
    return basis, packer, result, initial_rms, rms, actual_backend


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Experimental random-feature neural solver for the stationary two-sided equations."
    )
    ap.add_argument("p1", type=float, nargs="?", default=0.1, help="player 1 precision p1")
    ap.add_argument("p2", type=float, nargs="?", default=0.1, help="player 2 precision p2")
    ap.add_argument("r1", type=float, nargs="?", default=0.1, help="player 1 control cost r1")
    ap.add_argument("r2", type=float, nargs="?", default=0.1, help="player 2 control cost r2")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--A", type=float, default=0.0)
    ap.add_argument("--N", type=int, default=9, help="one-sided lag collocation points")
    ap.add_argument("--L", type=float, default=3.0, help="lag truncation window")
    ap.add_argument(
        "--quadrature",
        choices=["uniform", "gauss"],
        default="uniform",
        help="collocation/integration rule for the NN residual",
    )
    ap.add_argument(
        "--inner-quad",
        type=int,
        default=0,
        help="Gauss nodes for nested prefix integrals; defaults to N",
    )
    ap.add_argument("--features", type=int, default=8, help="random tanh features per map")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-nfev", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-6, help="SciPy optimizer termination tolerance")
    ap.add_argument(
        "--backend",
        choices=["auto", "jax", "numpy"],
        default="auto",
        help="residual/Jacobian backend; auto uses JAX when installed",
    )
    ap.add_argument(
        "--accept-rms",
        type=float,
        default=3e-2,
        help="weighted residual RMS threshold used for the converged flag and exit code",
    )
    ap.add_argument("--coef-penalty", type=float, default=1e-5)
    ap.add_argument("--w-state", type=float, default=1.0)
    ap.add_argument("--w-filter", type=float, default=1.0)
    ap.add_argument("--w-hx", type=float, default=1.0)
    ap.add_argument("--w-H", type=float, default=0.25)
    ap.add_argument("--w-wedge", type=float, default=1.0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-boundary", type=float, default=3.0)
    ap.add_argument("--w-tail-left", type=float, default=0.1)
    ap.add_argument("--init", type=str, default="", help="optional .npz warm-start file with theta")
    ap.add_argument("--out", type=str, default="", help="optional .npz file for fitted arrays")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        p1=args.p1,
        p2=args.p2,
        r1=max(args.r1, 1e-10),
        r2=max(args.r2, 1e-10),
        sigma=args.sigma,
        A=args.A,
        n=max(args.N, 5),
        lag_max=max(args.L, 1e-8),
        quadrature=args.quadrature,
        inner_quad=max(args.inner_quad, 0),
        n_features=max(args.features, 1),
        seed=args.seed,
        max_nfev=max(args.max_nfev, 1),
        tol=args.tol,
        accept_rms=max(args.accept_rms, 0.0),
        coef_penalty=max(args.coef_penalty, 0.0),
        w_state=args.w_state,
        w_filter=args.w_filter,
        w_hx=args.w_hx,
        w_H=args.w_H,
        w_wedge=args.w_wedge,
        w_policy=args.w_policy,
        w_boundary=args.w_boundary,
        w_tail_left=args.w_tail_left,
    )


def main() -> int:
    args = parse_args()
    cfg = config_from_args(args)
    t0 = time.perf_counter()
    basis, packer, result, initial_rms, final_rms, backend = run_solver(
        cfg, args.verbose, args.init, args.backend
    )
    elapsed_s = time.perf_counter() - t0
    norms = block_norms(result.x, cfg, basis, packer)
    arrays = solution_arrays(result.x, basis, packer)
    zero_c1 = eval_a_name(result.x, basis, packer, "c1", np.array([0.0]))[0]
    zero_c2 = eval_a_name(result.x, basis, packer, "c2", np.array([0.0]))[0]
    tail_x = eval_a_name(result.x, basis, packer, "x", np.array([cfg.lag_max]))[0]
    tail_c1 = eval_a_name(result.x, basis, packer, "c1", np.array([cfg.lag_max]))[0]
    tail_c2 = eval_a_name(result.x, basis, packer, "c2", np.array([cfg.lag_max]))[0]
    hx1_left = eval_b_name(result.x, basis, packer, "hx1", np.array([-cfg.lag_max]))[0]
    hx2_left = eval_b_name(result.x, basis, packer, "hx2", np.array([-cfg.lag_max]))[0]
    hx1_right = eval_b_name(result.x, basis, packer, "hx1", np.array([cfg.lag_max]))[0]
    hx2_right = eval_b_name(result.x, basis, packer, "hx2", np.array([cfg.lag_max]))[0]

    if args.out:
        np.savez(
            args.out,
            theta=result.x,
            lag=basis.a,
            b_lag=basis.b,
            zero_c1=zero_c1,
            zero_c2=zero_c2,
            p1=cfg.p1,
            p2=cfg.p2,
            r1=cfg.r1,
            r2=cfg.r2,
            sigma=cfg.sigma,
            A=cfg.A,
            N=cfg.n,
            L=cfg.lag_max,
            quadrature=cfg.quadrature,
            inner_quad=int(basis.inner_unit_nodes.size),
            features=cfg.n_features,
            seed=cfg.seed,
            backend=backend,
            initial_rms=initial_rms,
            residual_rms=final_rms,
            accept_rms=cfg.accept_rms,
            converged=bool(final_rms < cfg.accept_rms),
            cost=float(result.cost),
            optimality=float(result.optimality),
            nfev=int(result.nfev),
            elapsed_s=elapsed_s,
            block_rms_json=json.dumps(norms, sort_keys=True),
            **arrays,
        )

    payload = {
        "success": bool(result.success),
        "converged": bool(final_rms < cfg.accept_rms),
        "status": int(result.status),
        "message": str(result.message),
        "initial_rms": initial_rms,
        "residual_rms": final_rms,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "elapsed_s": elapsed_s,
        "seconds_per_nfev": elapsed_s / max(int(result.nfev), 1),
        "backend": backend,
        "jax_available": bool(JAX_AVAILABLE),
        "n_params": int(packer.size),
        "accept_rms": cfg.accept_rms,
        "p1": cfg.p1,
        "p2": cfg.p2,
        "obs_gain1": math.sqrt(max(cfg.p1, 0.0)),
        "obs_gain2": math.sqrt(max(cfg.p2, 0.0)),
        "r1": cfg.r1,
        "r2": cfg.r2,
        "N": cfg.n,
        "L": cfg.lag_max,
        "quadrature": cfg.quadrature,
        "inner_quad": int(basis.inner_unit_nodes.size),
        "features": cfg.n_features,
        "seed": cfg.seed,
        "block_rms": norms,
        "tail": {
            "x": float(np.linalg.norm(tail_x)),
            "calD": float(max(np.linalg.norm(tail_c1), np.linalg.norm(tail_c2))),
            "hx_left": float(max(np.linalg.norm(hx1_left), np.linalg.norm(hx2_left))),
            "hx_right": float(max(np.linalg.norm(hx1_right), np.linalg.norm(hx2_right))),
        },
        "zero_lag": {
            "calD1_own_obs": float(zero_c1[1]),
            "calD2_own_obs": float(zero_c2[2]),
            "calD1_unobserved_norm": float(np.linalg.norm(zero_c1[[0, 2]])),
            "calD2_unobserved_norm": float(np.linalg.norm(zero_c2[[0, 1]])),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if final_rms < cfg.accept_rms else 2


if __name__ == "__main__":
    raise SystemExit(main())
