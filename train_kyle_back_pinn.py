#!/usr/bin/env python3
"""Experimental PINN for the stationary Kyle-Back adjoint block.

This script implements a scalar, two-trader specialization of the new
Kyle-Back stationary chapter. It solves the discounted adjoint equations in
Theorem 1.7 and evaluates the weak seed-motion readout in Section 1.9 for a
fixed stationary candidate profile.

It is intentionally separate from the Noise-State-Games solver. The chapter
does not specify a single numeric benchmark closure for the filter and policy
kernels, so the defaults below provide a transparent scalar candidate profile:

    q = 1, two strategic traders, primitive Brownian coordinates
    (fundamental, public order-flow noise, private signal 1, private signal 2).

The learned objects are the observer adjoint matrices

    H^{0,1}, H^{2,1}, H^{0,2}, H^{1,2}

as functions of stationary lag. The loss enforces the adjoint ODEs with the
alpha/beta contractions computed from the learned adjoints. The exported JSON
also includes the weak readout H^i from equations (1.79)-(1.81), with old-history
and diagonal-birth terms kept separate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

# NVIDIA 580.142 reports a two-part kernel driver version. Current XLA logs
# that harmless parse failure as an error even though CUDA execution works.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import optax


D = 4
EV = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
EZ = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
EY1 = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
EY2 = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
EYS = [EY1, EY2]

BLOCK_NAMES = ["H01", "H21", "H02", "H12"]
METRIC_NAMES = [
    "adjoint_H01",
    "adjoint_H21",
    "adjoint_H02",
    "adjoint_H12",
    "tail",
    "regularization",
]


def observer_rows(observer: int, cfg):
    """Direct Brownian coordinates revealed by observer b in normalized units."""
    if observer == 0:
        return jnp.stack([EZ], axis=0)
    if observer == 1:
        return jnp.stack([EZ, cfg["std_y1"] * EY1], axis=0)
    if observer == 2:
        return jnp.stack([EZ, cfg["std_y2"] * EY2], axis=0)
    raise ValueError(f"unknown observer {observer}")


def simpson_weights(n: int, h: float) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    intervals = n - 1
    if intervals % 2 == 0:
        w = np.empty(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or i == intervals:
                w[i] = h / 3.0
            else:
                w[i] = (4.0 if i % 2 else 2.0) * h / 3.0
        return w.astype(np.float32)

    # Simpson 1/3 on the prefix and Simpson 3/8 on the final three intervals.
    w = np.zeros(n, dtype=np.float64)
    tail0 = intervals - 3
    if tail0 > 0:
        for i in range(tail0 + 1):
            if i == 0 or i == tail0:
                w[i] += h / 3.0
            else:
                w[i] += (4.0 if i % 2 else 2.0) * h / 3.0
    for i in range(tail0, intervals + 1):
        w[i] += (3.0 * h / 8.0) if (i == tail0 or i == intervals) else (9.0 * h / 8.0)
    return w.astype(np.float32)


def init_mlp(key, dims: List[int], last_scale: float):
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for layer_idx, (k, din, dout) in enumerate(zip(keys, dims[:-1], dims[1:])):
        scale = math.sqrt(2.0 / max(din + dout, 1))
        if layer_idx == len(dims) - 2:
            scale = last_scale
        params.append(
            {
                "W": scale * jax.random.normal(k, (din, dout), dtype=jnp.float32),
                "b": jnp.zeros((dout,), dtype=jnp.float32),
            }
        )
    return params


def mlp(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def tree_to_jsonable(params):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32).tolist(), params)


def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        name: [
            {
                "W": jnp.asarray(layer["W"], dtype=jnp.float32),
                "b": jnp.asarray(layer["b"], dtype=jnp.float32),
            }
            for layer in layers
        ]
        for name, layers in payload["nets"].items()
    }


def vec_outer(a, b):
    return a[..., :, None] * b[..., None, :]


def mse(x):
    return jnp.mean(jnp.square(x))


def eval_H(params, ell, cfg):
    ell = jnp.asarray(ell, dtype=jnp.float32)
    x = (2.0 * ell / cfg["L"] - 1.0)[..., None]
    raw = mlp(params["H"], x).reshape(ell.shape + (len(BLOCK_NAMES), D, D))
    tail = jnp.exp(-cfg["tail_decay"] * ell)[..., None, None, None]
    return tail * raw


def value_and_deriv_H(params, ell, cfg):
    return jax.jvp(
        lambda aa: eval_H(params, aa, cfg),
        (ell,),
        (jnp.ones_like(ell),),
    )


def split_H(y):
    return {name: y[..., idx, :, :] for idx, name in enumerate(BLOCK_NAMES)}


def candidate_kernels(ell, cfg):
    """Stationary scalar candidate profile in primitive coordinates.

    Demand kernels D_i(ell) are row vectors from primitive shocks to raw demand.
    Observation-drift kernels follow equations (1.17)-(1.18). The unresolved
    kernels ce^b are fixed candidate Kalman-gain kernels for this experiment.
    """
    ell = jnp.asarray(ell, dtype=jnp.float32)
    # Kyle-Back standing convention: predictable policy kernels have zero
    # diagonal trace, so raw demand kernels vanish at lag zero.
    demand_trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell))[..., None]
    fd = demand_trace * jnp.exp(-cfg["d_decay"] * ell)[..., None]
    fyd = demand_trace * jnp.exp(-cfg["private_decay"] * ell)[..., None]
    ff = jnp.exp(-cfg["filter_decay"] * ell)[..., None]

    d1 = fd * (cfg["d1_v"] * EV + cfg["d1_z"] * EZ) + fyd * cfg["d1_y"] * EY1
    d2 = fd * (cfg["d2_v"] * EV + cfg["d2_z"] * EZ) + fyd * cfg["d2_y"] * EY2
    dtot = d1 + d2

    c0 = dtot / cfg["std_z"]
    cy1 = jnp.broadcast_to((cfg["gamma1"] * cfg["std_v"] / cfg["std_y1"]) * EV, ell.shape + (D,))
    cy2 = jnp.broadcast_to((cfg["gamma2"] * cfg["std_v"] / cfg["std_y2"]) * EV, ell.shape + (D,))

    ce0 = ff * (
        cfg["ce0_v"] * EV
        + cfg["ce0_z"] * EZ
        + cfg["ce0_y1"] * EY1
        + cfg["ce0_y2"] * EY2
    )
    ce1_z = ce0 + ff * cfg["ce_trader_z_private"] * EY1
    ce2_z = ce0 + ff * cfg["ce_trader_z_private"] * EY2
    ce1_y = ff * (cfg["ce_y_v"] * EV + cfg["ce_y_private"] * EY1)
    ce2_y = ff * (cfg["ce_y_v"] * EV + cfg["ce_y_private"] * EY2)

    return {
        "D1": d1,
        "D2": d2,
        "Dtot": dtot,
        "c0": c0,
        "c1_z": c0,
        "c2_z": c0,
        "cY1": cy1,
        "cY2": cy2,
        "ce0": ce0,
        "ce1_z": ce1_z,
        "ce1_y": ce1_y,
        "ce2_z": ce2_z,
        "ce2_y": ce2_y,
    }


def observer_ctilde(observer: int, ell, cfg):
    k = candidate_kernels(ell, cfg)
    if observer == 0:
        return k["ce0"][..., None, :]
    if observer == 1:
        return jnp.stack([k["ce1_z"], k["ce1_y"]], axis=-2)
    if observer == 2:
        return jnp.stack([k["ce2_z"], k["ce2_y"]], axis=-2)
    raise ValueError(f"unknown observer {observer}")


def filter_kernel_pairs(observer: int, first_lag, second_lag, cfg):
    """Stationary filter kernel f^b(first_lag, second_lag).

    This is the characteristic solution of the note's equations (15)-(17).
    It is used only for diagnostics/export, to convert a noise-state policy
    kernel D_i into its primitive-coordinate kernel d_i via equations (8)-(9).
    """
    first_lag = jnp.asarray(first_lag, dtype=jnp.float32)
    second_lag = jnp.asarray(second_lag, dtype=jnp.float32)
    first_ge_second = first_lag >= second_lag
    first_gt_second = first_lag > second_lag + 1e-7
    second_gt_first = second_lag > first_lag + 1e-7
    base = jnp.abs(first_lag - second_lag)
    width = jnp.minimum(first_lag, second_lag)
    rows = observer_rows(observer, cfg)
    nodes = cfg.get("filter_nodes")
    weights = cfg.get("filter_weights")

    c_base = observer_ctilde(observer, base, cfg)
    boundary_right = jnp.einsum("...rd,re->...de", c_base, rows)
    boundary_left = jnp.einsum("rd,...re->...de", rows, c_base)
    # Birth terms are side-boundary traces, not diagonal density mass.
    boundary = jnp.where(
        first_gt_second[..., None, None],
        boundary_right,
        jnp.where(second_gt_first[..., None, None], boundary_left, jnp.zeros_like(boundary_right)),
    )

    tau = 0.5 * width[..., None] * (nodes + 1.0)
    quad_w = 0.5 * width[..., None] * weights
    c_first = observer_ctilde(
        observer,
        jnp.where(first_ge_second[..., None], base[..., None] + tau, tau),
        cfg,
    )
    c_second = observer_ctilde(
        observer,
        jnp.where(first_ge_second[..., None], tau, base[..., None] + tau),
        cfg,
    )
    integral = jnp.einsum("...q,...qrd,...qre->...de", quad_w, c_first, c_second)
    return boundary + integral


def project_kernel(observer: int, kernel, cfg):
    """Apply P^b from equation (8) to a primitive-coordinate kernel."""
    grid = cfg["grid"]
    weights = cfg["weights"]
    kernel = jnp.asarray(kernel, dtype=jnp.float32)
    rows = observer_rows(observer, cfg)
    direct = jnp.einsum("nd,rd,re->ne", kernel, rows, rows)
    first = jnp.broadcast_to(grid[:, None], (grid.shape[0], grid.shape[0]))
    second = jnp.broadcast_to(grid[None, :], (grid.shape[0], grid.shape[0]))
    f = filter_kernel_pairs(observer, first, second, cfg)
    indirect = jnp.einsum("a,ad,alde->le", weights, kernel, f)
    return direct + indirect


def integrate_vec(weights, x):
    return jnp.einsum("n,n...->...", weights, x)


def alpha_terms(params, H_grid, cfg):
    k = candidate_kernels(cfg["grid"], cfg)
    w = cfg["weights"]

    H = split_H(H_grid)
    Hz = {
        name: split_H(eval_H(params, jnp.array([0.0], dtype=jnp.float32), cfg))[name][0]
        for name in BLOCK_NAMES
    }

    a01_z = integrate_vec(w, jnp.einsum("nij,nj->ni", H["H01"], k["ce0"])) + Hz["H01"] @ EZ
    a02_z = integrate_vec(w, jnp.einsum("nij,nj->ni", H["H02"], k["ce0"])) + Hz["H02"] @ EZ

    a21_z = integrate_vec(w, jnp.einsum("nij,nj->ni", H["H21"], k["ce2_z"])) + Hz["H21"] @ EZ
    a21_y = integrate_vec(w, jnp.einsum("nij,nj->ni", H["H21"], k["ce2_y"])) + Hz["H21"] @ (cfg["std_y2"] * EY2)

    a12_z = integrate_vec(w, jnp.einsum("nij,nj->ni", H["H12"], k["ce1_z"])) + Hz["H12"] @ EZ
    a12_y = integrate_vec(w, jnp.einsum("nij,nj->ni", H["H12"], k["ce1_y"])) + Hz["H12"] @ (cfg["std_y1"] * EY1)

    return {
        "a01_z": a01_z,
        "a21_z": a21_z,
        "a21_y": a21_y,
        "beta1": a01_z + a21_z,
        "a02_z": a02_z,
        "a12_z": a12_z,
        "a12_y": a12_y,
        "beta2": a02_z + a12_z,
    }


def source_blocks(ell, alphas, cfg):
    k = candidate_kernels(ell, cfg)
    zinv_dtot = k["Dtot"] / cfg["std_z"]

    src01 = vec_outer(k["D1"], cfg["std_v"] * EV) - vec_outer(alphas["a01_z"], zinv_dtot)
    src21 = (
        vec_outer(alphas["beta1"], k["D2"] / cfg["std_z"])
        - vec_outer(alphas["a21_z"], zinv_dtot)
        - vec_outer(alphas["a21_y"], k["cY2"])
    )
    src02 = vec_outer(k["D2"], cfg["std_v"] * EV) - vec_outer(alphas["a02_z"], zinv_dtot)
    src12 = (
        vec_outer(alphas["beta2"], k["D1"] / cfg["std_z"])
        - vec_outer(alphas["a12_z"], zinv_dtot)
        - vec_outer(alphas["a12_y"], k["cY1"])
    )
    return {
        "H01": src01,
        "H21": src21,
        "H02": src02,
        "H12": src12,
    }


def residual_metrics(params, cfg):
    H_grid = eval_H(params, cfg["grid"], cfg)
    alphas = alpha_terms(params, H_grid, cfg)

    H_ode, dH_ode = value_and_deriv_H(params, cfg["ode_grid"], cfg)
    H = split_H(H_ode)
    dH = split_H(dH_ode)
    src = source_blocks(cfg["ode_grid"], alphas, cfg)

    blocks = [
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H01"] - dH["H01"] - src["H01"]),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H21"] - dH["H21"] - src["H21"]),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H02"] - dH["H02"] - src["H02"]),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H12"] - dH["H12"] - src["H12"]),
        cfg["w_tail"] * mse(H_ode[-1]),
        cfg["w_reg"] * mse(H_ode),
    ]
    return jnp.stack(blocks)


def readout_for_deviator(H_name0, H_name_opp, opp: int, alphas, params, cfg):
    grid = cfg["grid"]
    weights = cfg["weights"]
    H_grid, dH_grid = value_and_deriv_H(params, grid, cfg)
    H = split_H(H_grid)
    src = source_blocks(grid, alphas, cfg)
    k = candidate_kernels(grid, cfg)

    if opp == 1:
        ce_opp_z = k["ce1_z"]
        ce_opp_y = k["ce1_y"]
        c_opp_z = k["c1_z"]
        c_opp_y = k["cY1"]
        D_opp = k["D1"]
        std_y = cfg["std_y1"]
        ey = EY1
    else:
        ce_opp_z = k["ce2_z"]
        ce_opp_y = k["ce2_y"]
        c_opp_z = k["c2_z"]
        c_opp_y = k["cY2"]
        D_opp = k["D2"]
        std_y = cfg["std_y2"]
        ey = EY2

    phi0 = k["ce0"] / cfg["std_z"]
    phi_opp = ce_opp_z / cfg["std_z"]
    psi = EZ / cfg["std_z"]

    R = integrate_vec(weights, jnp.sum(D_opp * phi_opp, axis=-1))

    bracket0 = integrate_vec(weights, jnp.sum(k["c0"] * phi0, axis=-1)) - R / cfg["std_z"]
    bracket_opp_z = integrate_vec(weights, jnp.sum(c_opp_z * phi_opp, axis=-1)) - R / cfg["std_z"]
    bracket_opp_y = integrate_vec(weights, jnp.sum(c_opp_y * phi_opp, axis=-1))

    q0 = k["ce0"] * bracket0
    q_opp = ce_opp_z * bracket_opp_z + ce_opp_y * bracket_opp_y
    q0_diag = EZ * bracket0
    q_opp_diag = EZ * bracket_opp_z + std_y * ey * bracket_opp_y

    H0 = H[H_name0]
    Hopp = H[H_name_opp]
    S0 = src[H_name0]
    Sopp = src[H_name_opp]
    H_zero = split_H(eval_H(params, jnp.array([0.0], dtype=jnp.float32), cfg))
    H0_zero = H_zero[H_name0][0]
    Hopp_zero = H_zero[H_name_opp][0]
    k_zero = candidate_kernels(jnp.array([0.0], dtype=jnp.float32), cfg)
    phi0_zero = k_zero["ce0"][0] / cfg["std_z"]
    phi_opp_zero = (k_zero["ce1_z"][0] if opp == 1 else k_zero["ce2_z"][0]) / cfg["std_z"]

    density0 = jnp.einsum("nij,nj->ni", S0, phi0) + jnp.einsum("nij,nj->ni", H0, q0)
    density_opp = jnp.einsum("nij,nj->ni", Sopp, phi_opp) + jnp.einsum("nij,nj->ni", Hopp, q_opp)
    old0 = integrate_vec(weights, density0)
    old_opp = integrate_vec(weights, density_opp)

    diag0 = H0_zero @ (cfg["rho"] * psi - phi0_zero + q0_diag)
    diag_opp = Hopp_zero @ (cfg["rho"] * psi - phi_opp_zero + q_opp_diag)

    return {
        "density_market_maker": density0,
        "density_opponent": density_opp,
        "density_total": density0 + density_opp,
        "old_history_market_maker": old0,
        "old_history_opponent": old_opp,
        "diagonal_market_maker": diag0,
        "diagonal_opponent": diag_opp,
        "diagonal_total": diag0 + diag_opp,
        "total": old0 + old_opp + diag0 + diag_opp,
        "R": R,
        "bracket0": bracket0,
        "bracket_opp_z": bracket_opp_z,
        "bracket_opp_y": bracket_opp_y,
    }


def diagnostics(params, cfg, step, elapsed, loss, metrics, grad_norm):
    H_grid = eval_H(params, cfg["grid"], cfg)
    alphas = alpha_terms(params, H_grid, cfg)
    ro1 = readout_for_deviator("H01", "H21", 2, alphas, params, cfg)
    ro2 = readout_for_deviator("H02", "H12", 1, alphas, params, cfg)

    k = candidate_kernels(cfg["grid"], cfg)
    lam = cfg["std_v"] / cfg["std_z"] * integrate_vec(cfg["weights"], k["ce0"][:, 0])
    inv_lam = 1.0 / jnp.where(jnp.abs(lam) > 1e-8, lam, jnp.inf)
    local_info = cfg["std_z"] * k["ce0"]
    policy_shadow1 = ro1["total"] * inv_lam
    policy_shadow2 = ro2["total"] * inv_lam
    raw_shadow_density1 = ro1["density_total"] * inv_lam
    raw_shadow_density2 = ro2["density_total"] * inv_lam

    # Equation-sheet demand readout:
    # first project the local weak shadow H_i onto trader i's information,
    # then convert the resulting noise-state policy kernel D_i to primitive
    # coordinates d_i with the same P_i operator from equations (8)-(9).
    policy_noise_state1 = project_kernel(1, local_info + raw_shadow_density1, cfg)
    policy_noise_state2 = project_kernel(2, local_info + raw_shadow_density2, cfg)
    policy_best_response1 = project_kernel(1, policy_noise_state1, cfg)
    policy_best_response2 = project_kernel(2, policy_noise_state2, cfg)
    fixed_point_residual1 = policy_best_response1 - k["D1"]
    fixed_point_residual2 = policy_best_response2 - k["D2"]

    block_rms = {
        name: float(math.sqrt(max(float(v), 0.0)))
        for name, v in zip(METRIC_NAMES, np.asarray(metrics))
    }
    return {
        "step": int(step),
        "elapsed_s": float(elapsed),
        "loss": float(loss),
        "rms": float(math.sqrt(max(float(loss) / len(METRIC_NAMES), 0.0))),
        "grad_norm": float(grad_norm),
        "block_rms": block_rms,
        "lambda": float(lam),
        "alpha": {k: np.asarray(v, dtype=float).tolist() for k, v in alphas.items()},
        "weak_readout": {
            "player1": {k: np.asarray(v, dtype=float).tolist() if hasattr(v, "shape") else float(v) for k, v in ro1.items()},
            "player2": {k: np.asarray(v, dtype=float).tolist() if hasattr(v, "shape") else float(v) for k, v in ro2.items()},
        },
        "policy_shadow": {
            "player1": np.asarray(policy_shadow1, dtype=float).tolist(),
            "player2": np.asarray(policy_shadow2, dtype=float).tolist(),
        },
        "theorem_1_12_policy": {
            "formula": "local range equation plus equations (8)-(9): D_i=Pi_i(Sigma_Z^{1/2} ctilde0 + Lambda^dagger H_i), d_i=Pi_i D_i",
            "lambda": float(lam),
            "local_information_gap": np.asarray(local_info, dtype=float).tolist(),
            "shadow_density": {
                "player1": np.asarray(raw_shadow_density1, dtype=float).tolist(),
                "player2": np.asarray(raw_shadow_density2, dtype=float).tolist(),
            },
            "noise_state_policy": {
                "player1": np.asarray(policy_noise_state1, dtype=float).tolist(),
                "player2": np.asarray(policy_noise_state2, dtype=float).tolist(),
            },
            "best_response": {
                "player1": np.asarray(policy_best_response1, dtype=float).tolist(),
                "player2": np.asarray(policy_best_response2, dtype=float).tolist(),
            },
            "candidate_fixed_point_residual": {
                "player1": np.asarray(fixed_point_residual1, dtype=float).tolist(),
                "player2": np.asarray(fixed_point_residual2, dtype=float).tolist(),
            },
            "candidate_fixed_point_rms": {
                "player1": float(jnp.sqrt(jnp.mean(jnp.square(fixed_point_residual1)))),
                "player2": float(jnp.sqrt(jnp.mean(jnp.square(fixed_point_residual2)))),
            },
        },
    }


def make_cfg(args):
    if args.quadrature == "gauss":
        x, w = np.polynomial.legendre.leggauss(args.N)
        grid = 0.5 * args.L * (x + 1.0)
        weights = 0.5 * args.L * w
    else:
        grid = np.linspace(0.0, args.L, args.N, dtype=np.float64)
        h = args.L / max(args.N - 1, 1)
        weights = simpson_weights(args.N, h) if args.quadrature == "simpson" else np.full(args.N, h)
        if args.quadrature == "trapezoid":
            weights[0] *= 0.5
            weights[-1] *= 0.5

    if args.ode_N > 0:
        theta = np.linspace(0.0, math.pi, args.ode_N, dtype=np.float64)
        ode_grid = 0.5 * args.L * (1.0 - np.cos(theta))
    else:
        ode_grid = grid

    cfg = vars(args).copy()
    cfg["grid"] = jnp.asarray(grid.astype(np.float32))
    cfg["weights"] = jnp.asarray(weights.astype(np.float32))
    cfg["ode_grid"] = jnp.asarray(ode_grid.astype(np.float32))
    filter_nodes, filter_weights = np.polynomial.legendre.leggauss(args.filter_quad)
    cfg["filter_nodes"] = jnp.asarray(filter_nodes.astype(np.float32))
    cfg["filter_weights"] = jnp.asarray(filter_weights.astype(np.float32))
    return cfg


def init_params(args):
    key = jax.random.PRNGKey(args.seed)
    return {
        "H": init_mlp(
            key,
            [1] + [args.hidden] * args.depth + [len(BLOCK_NAMES) * D * D],
            args.last_scale,
        )
    }


def save_json(path, params, cfg, args, diag):
    H_grid = eval_H(params, cfg["grid"], cfg)
    k = candidate_kernels(cfg["grid"], cfg)
    payload = {
        "version": 1,
        "kind": "kyle_back_stationary_adjoint_pinn",
        "description": "Scalar two-trader Kyle-Back stationary adjoint PINN for Theorem 1.7 and weak readout equations (1.79)-(1.81).",
        "assumptions": {
            "scope": "fixed candidate profile; not a full equilibrium/filter-closure solve",
            "primitive_coordinates": ["fundamental", "order_flow_noise", "private_signal_1", "private_signal_2"],
            "raw_order_flow_dimension": 1,
            "traders": 2,
            "diagonal_birth_terms": True,
            "weak_seed_motion_readout": True,
        },
        "nets": tree_to_jsonable(params),
        "architecture": {
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": "tanh",
            "n_params": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))),
        },
        "grid": np.asarray(cfg["grid"], dtype=float).tolist(),
        "quadrature": args.quadrature,
        "params": {
            k: float(v)
            for k, v in vars(args).items()
            if isinstance(v, (int, float)) and k not in {"steps", "log_every", "save_every", "seed"}
        },
        "fields": {
            "H_blocks": BLOCK_NAMES,
            "candidate_kernels": ["D1", "D2", "Dtot", "c0", "cY1", "cY2", "ce0", "ce1_z", "ce1_y", "ce2_z", "ce2_y"],
        },
        "candidate": {name: np.asarray(val, dtype=float).tolist() for name, val in k.items()},
        "H": {name: np.asarray(split_H(H_grid)[name], dtype=float).tolist() for name in BLOCK_NAMES},
        "diagnostics": diag,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def maybe_plot(path, params, cfg):
    if not path:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping plot {path}: matplotlib unavailable ({exc})", flush=True)
        return False

    grid = np.asarray(cfg["grid"], dtype=float)
    H = split_H(eval_H(params, cfg["grid"], cfg))

    fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for axis, name in zip(ax.flat, BLOCK_NAMES):
        mat = np.asarray(H[name], dtype=float)
        axis.plot(grid, mat[:, 0, 0], label="fundamental->fundamental")
        axis.plot(grid, mat[:, 0, 1], label="order-flow noise")
        axis.plot(grid, mat[:, 0, 2], label="private 1")
        axis.plot(grid, mat[:, 0, 3], label="private 2")
        axis.set_title(name)
        axis.grid(alpha=0.25)
    ax[1, 0].set_xlabel("lag")
    ax[1, 1].set_xlabel("lag")
    ax[0, 0].set_ylabel("adjoint")
    ax[1, 0].set_ylabel("adjoint")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="Experimental scalar Kyle-Back stationary adjoint PINN.")
    ap.add_argument("--out", default="data/kyle_back_pinn.json")
    ap.add_argument("--plot", default="figs/kyle_back_pinn.pdf")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-decay", type=float, default=0.6)
    ap.add_argument("--weight-decay", type=float, default=1e-7)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260525)
    ap.add_argument("--init-from", default="", help="Resume from a saved stationary-adjoint JSON export.")
    ap.add_argument("--N", type=int, default=129)
    ap.add_argument("--ode-N", type=int, default=0)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--quadrature", choices=["trapezoid", "simpson", "gauss"], default="simpson")
    ap.add_argument("--filter-quad", type=int, default=21,
                    help="Gauss nodes for stationary filter projection diagnostics.")
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--std-v", type=float, default=1.0)
    ap.add_argument("--std-z", type=float, default=1.0)
    ap.add_argument("--std-y1", type=float, default=1.0)
    ap.add_argument("--std-y2", type=float, default=1.0)
    ap.add_argument("--gamma1", type=float, default=3.0)
    ap.add_argument("--gamma2", type=float, default=10.0)
    ap.add_argument("--d1-v", type=float, default=0.65)
    ap.add_argument("--d2-v", type=float, default=0.55)
    ap.add_argument("--d1-z", type=float, default=-0.10)
    ap.add_argument("--d2-z", type=float, default=-0.08)
    ap.add_argument("--d1-y", type=float, default=0.30)
    ap.add_argument("--d2-y", type=float, default=0.28)
    ap.add_argument("--d-decay", type=float, default=0.95)
    ap.add_argument("--diag-ramp", type=float, default=4.0)
    ap.add_argument("--private-decay", type=float, default=1.20)
    ap.add_argument("--filter-decay", type=float, default=1.05)
    ap.add_argument("--cy-v", type=float, default=0.35)
    ap.add_argument("--ce0-v", type=float, default=0.42)
    ap.add_argument("--ce0-z", type=float, default=0.22)
    ap.add_argument("--ce0-y1", type=float, default=0.04)
    ap.add_argument("--ce0-y2", type=float, default=0.03)
    ap.add_argument("--ce-trader-z-private", type=float, default=0.08)
    ap.add_argument("--ce-y-v", type=float, default=0.30)
    ap.add_argument("--ce-y-private", type=float, default=0.55)
    ap.add_argument("--tail-decay", type=float, default=0.08)
    ap.add_argument("--last-scale", type=float, default=1e-3)
    ap.add_argument("--w-adjoint", type=float, default=1.0)
    ap.add_argument("--w-tail", type=float, default=0.5)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = make_cfg(args)
    params = load_params(args.init_from) if args.init_from else init_params(args)
    n_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))
    print(
        f"kyle-back adjoint PINN params={n_params} N={args.N} L={args.L} "
        f"quadrature={args.quadrature}",
        flush=True,
    )

    schedule = optax.exponential_decay(
        args.lr,
        transition_steps=max(args.steps // 4, 1),
        decay_rate=args.lr_decay,
        staircase=False,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    opt_state = opt.init(params)

    def loss_fn(params):
        metrics = residual_metrics(params, cfg)
        return jnp.sum(metrics), metrics

    @jax.jit
    def train_step(params, opt_state):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics, optax.global_norm(grads)

    t0 = time.perf_counter()
    best = None
    best_loss = float("inf")
    last_diag = None

    for step in range(1, args.steps + 1):
        params, opt_state, loss, metrics, grad_norm = train_step(params, opt_state)
        if step == 1:
            jax.block_until_ready(loss)

        if step == 1 or step % args.log_every == 0:
            jax.block_until_ready(loss)
            elapsed = time.perf_counter() - t0
            rms = math.sqrt(max(float(loss) / len(METRIC_NAMES), 0.0))
            block_rms = {
                name: math.sqrt(max(float(v), 0.0))
                for name, v in zip(METRIC_NAMES, np.asarray(metrics))
            }
            worst = sorted(block_rms.items(), key=lambda kv: kv[1], reverse=True)[:3]
            worst_s = " ".join(f"{k}={v:.2e}" for k, v in worst)
            print(
                f"step={step} loss={float(loss):.4e} rms={rms:.4e} "
                f"grad={float(grad_norm):.3e} elapsed={elapsed:.1f}s {worst_s}",
                flush=True,
            )
            last_diag = diagnostics(params, cfg, step, elapsed, loss, metrics, grad_norm)
            if float(loss) < best_loss:
                best_loss = float(loss)
                best = last_diag

        if step % args.save_every == 0 and last_diag is not None:
            diag = dict(last_diag)
            diag["best"] = best
            save_json(args.out, params, cfg, args, diag)

    loss, metrics = loss_fn(params)
    jax.block_until_ready(loss)
    final_diag = diagnostics(
        params,
        cfg,
        args.steps,
        time.perf_counter() - t0,
        loss,
        metrics,
        0.0,
    )
    final_diag["best"] = best
    save_json(args.out, params, cfg, args, final_diag)
    plotted = maybe_plot(args.plot, params, cfg)
    print(json.dumps(final_diag, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    if plotted:
        print(f"wrote {args.plot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
