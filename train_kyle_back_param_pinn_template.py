#!/usr/bin/env python3
"""Parameter-conditioned stationary Kyle--Back PINN.

This script is a template-style rewrite of the stationary Kyle--Back PINN: it
samples parameter tuples during training and minimizes the stationary residuals
of the chapter system directly.  The layout follows the cleaner
``train_stationary_param_pinn.py`` style: one-time lag fields, adjoint fields,
parameter-conditioned MLPs, validation batches, and exported browser-friendly
weights.

The mathematical fields are for the two strategic trader, one public order-flow
channel case with primitive Brownian coordinates

    W = (fundamental, order-flow noise, trader-1 signal noise, trader-2 signal noise).

The one-time fields are

    D1, D2                  noise-state policy kernels;
    calD1, calD2            primitive/raw projected demand kernels;
    vtilde0,1,2             observer-specific unresolved value kernels;
    dtot_tilde0,1,2         observer-specific unresolved aggregate-demand kernels.

The adjoint fields are

    H01, H21    value to trader 1 of market-maker / trader-2 noise-state shifts;
    H02, H12    value to trader 2 of market-maker / trader-1 noise-state shifts.

The local FOC is enforced in primitive coordinates:

    Lambda * (calD_i - E_i[std_z * ctilde0]) = E_i[calH_i].

Notes:
  * ``Lambda`` is computed endogenously from the learned order-flow readout. It
    is not a sampled parameter and is not used to normalize the filter.
  * ``private_signal_gauge=literal`` uses the chapter signal drift V-P, whose
    primitive kernel is the market-maker residual value vtilde0.
  * ``private_signal_gauge=public_adjusted`` uses the equivalent public-adjusted
    signal with drift V, since P is public/order-flow-measurable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import optax


D_W = 4
EV = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
EZ = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
EY1 = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
EY2 = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
EYE = jnp.eye(D_W, dtype=jnp.float32)

A_FIELDS = [
    "D1",
    "D2",
    "calD1",
    "calD2",
    "vtilde0",
    "vtilde1",
    "vtilde2",
    "dtot_tilde0",
    "dtot_tilde1",
    "dtot_tilde2",
]
H_FIELDS = ["H01", "H21", "H02", "H12"]

METRIC_NAMES = [
    "filter0",
    "filter1",
    "filter2",
    "policy_projection",
    "H01",
    "H21",
    "H02",
    "H12",
    "foc",
    "policy_target_stabilizer",
    "tail",
    "oscillation",
    "regularization",
]

FILTER_ONLY_METRIC_NAMES = [
    "filter0",
    "filter1",
    "filter2",
    "tail",
    "oscillation",
    "regularization",
]


# ---------------------------------------------------------------------------
# Basic neural-network utilities
# ---------------------------------------------------------------------------


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

    # Simpson 1/3 on the first intervals-3 intervals and 3/8 on the tail.
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
    params = []
    keys = jax.random.split(key, len(dims) - 1)
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
    return h @ params[-1]["W"] + params[-1]["b"]


def mse(x):
    return jnp.mean(jnp.square(x))


def vec_outer(a, b):
    return a[..., :, None] * b[..., None, :]


def split_a(y):
    chunks = jnp.split(y, len(A_FIELDS), axis=-1)
    return {name: chunk for name, chunk in zip(A_FIELDS, chunks)}


def split_H(y):
    out = {}
    offset = 0
    for name in H_FIELDS:
        out[name] = y[..., offset : offset + D_W * D_W].reshape(y.shape[:-1] + (D_W, D_W))
        offset += D_W * D_W
    return out


def encode_params(p, ranges):
    names = ["gamma1", "gamma2", "rho"]
    mins = jnp.log(jnp.array([ranges[f"{n}_min"] for n in names], dtype=jnp.float32))
    maxs = jnp.log(jnp.array([ranges[f"{n}_max"] for n in names], dtype=jnp.float32))
    center = 0.5 * (mins + maxs)
    scale = 0.5 * (maxs - mins)
    return (jnp.log(jnp.maximum(p, 1e-8)) - center) / scale


def concat_inputs(p_enc, coord):
    coord = jnp.asarray(coord, dtype=jnp.float32)
    flat = coord.reshape((-1, coord.shape[-1]))
    p_tiled = jnp.broadcast_to(p_enc, (flat.shape[0], p_enc.shape[0]))
    return jnp.concatenate([p_tiled, flat], axis=1)


def unpack_p(p):
    return {
        "gamma1": jnp.maximum(p[0], 1e-8),
        "gamma2": jnp.maximum(p[1], 1e-8),
        "rho": jnp.maximum(p[2], 1e-8),
    }


# ---------------------------------------------------------------------------
# Parameter-conditioned fields
# ---------------------------------------------------------------------------


def coord_lag(ell, cfg):
    return (2.0 * jnp.asarray(ell, dtype=jnp.float32) / cfg["L"] - 1.0)[..., None]


def baseline_a(p, ell, cfg):
    ell = jnp.asarray(ell, dtype=jnp.float32)
    par = unpack_p(p)
    tail_v = jnp.exp(-cfg["filter_decay"] * ell)[..., None]
    tail_d = jnp.exp(-cfg["demand_decay"] * ell)[..., None]
    zero = jnp.zeros(ell.shape + (D_W,), dtype=jnp.float32)

    D1 = zero
    D2 = zero
    calD1 = zero
    calD2 = zero
    v0 = cfg["std_v"] * tail_v * EV
    # Private signals make trader filters faster when gamma is large.
    v1 = cfg["std_v"] * jnp.exp(-(cfg["filter_decay"] + 0.1 * par["gamma1"]) * ell)[..., None] * EV
    v2 = cfg["std_v"] * jnp.exp(-(cfg["filter_decay"] + 0.1 * par["gamma2"]) * ell)[..., None] * EV
    dt0 = zero
    dt1 = zero
    dt2 = zero
    return jnp.concatenate([D1, D2, calD1, calD2, v0, v1, v2, dt0, dt1, dt2], axis=-1)


def fixed_policy_kernels(ell, cfg):
    """Preset exponential noise-state demand kernels for filter-only tests."""
    ell = jnp.asarray(ell, dtype=jnp.float32)
    trace = (1.0 - jnp.exp(-cfg["fixed_policy_ramp"] * ell))[..., None]
    tail = jnp.exp(-cfg["fixed_policy_decay"] * ell)[..., None]
    d1 = (
        cfg["fixed_d1_v"] * EV
        + cfg["fixed_d1_z"] * EZ
        + cfg["fixed_d1_y"] * EY1
    )
    d2 = (
        cfg["fixed_d2_v"] * EV
        + cfg["fixed_d2_z"] * EZ
        + cfg["fixed_d2_y"] * EY2
    )
    return trace * tail * d1, trace * tail * d2


def eval_a(params, p, ell, cfg):
    p_enc = encode_params(p, cfg["ranges"])
    ell = jnp.asarray(ell, dtype=jnp.float32)
    raw = mlp(params["a"], concat_inputs(p_enc, coord_lag(ell, cfg))).reshape(ell.shape + (len(A_FIELDS) * D_W,))
    tail = jnp.exp(-cfg["tail_decay"] * ell)[..., None]
    y = baseline_a(p, ell, cfg) + cfg["output_scale_a"] * tail * raw
    A = split_a(y)

    # Predictable zero trace for control/risk-bearing demand objects.  The
    # unresolved observation-drift kernels are not forced to zero at lag zero:
    # they can contain initial V-estimation errors in private-signal rows and
    # initial D_tot-estimation errors in order-flow rows.
    trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell))[..., None]
    for name in ["D1", "D2", "calD1", "calD2"]:
        A[name] = trace * A[name]

    # Value residual at age zero is the primitive value shock because value noise
    # is orthogonal to order-flow and private signal noise.
    zero = jnp.isclose(ell, 0.0, atol=1e-7)[..., None]
    v_zero = cfg["std_v"] * EV
    for name in ["vtilde0", "vtilde1", "vtilde2"]:
        A[name] = jnp.where(zero, v_zero, A[name])

    return jnp.concatenate([A[name] for name in A_FIELDS], axis=-1)


def eval_H(params, p, ell, cfg):
    p_enc = encode_params(p, cfg["ranges"])
    ell = jnp.asarray(ell, dtype=jnp.float32)
    raw = mlp(params["h"], concat_inputs(p_enc, coord_lag(ell, cfg))).reshape(ell.shape + (len(H_FIELDS) * D_W * D_W,))
    tail = jnp.exp(-cfg["tail_decay"] * ell)[..., None]
    return cfg["output_scale_h"] * tail * raw


def value_and_deriv_a(params, p, ell, cfg):
    return jax.jvp(lambda ee: eval_a(params, p, ee, cfg), (ell,), (jnp.ones_like(ell),))


def value_first_second_deriv_a(params, p, ell, cfg):
    value, first = value_and_deriv_a(params, p, ell, cfg)
    _, second = jax.jvp(
        lambda ee: value_and_deriv_a(params, p, ee, cfg)[1],
        (ell,),
        (jnp.ones_like(ell),),
    )
    return value, first, second


def value_and_deriv_H(params, p, ell, cfg):
    return jax.jvp(lambda ee: eval_H(params, p, ee, cfg), (ell,), (jnp.ones_like(ell),))


# ---------------------------------------------------------------------------
# Observer rows, filter kernels, and projections
# ---------------------------------------------------------------------------


def observer_rows(observer: int):
    if observer == 0:
        return jnp.stack([EZ], axis=0)
    if observer == 1:
        return jnp.stack([EZ, EY1], axis=0)
    if observer == 2:
        return jnp.stack([EZ, EY2], axis=0)
    raise ValueError(f"unknown observer {observer}")


def observer_pi(observer: int):
    rows = observer_rows(observer)
    return rows.T @ rows


def model_at(params, p, ell_eval, cfg, train_grids=None):
    par = unpack_p(p)
    A = split_a(eval_a(params, p, ell_eval, cfg))
    if cfg["fixed_policy"]:
        D1, D2 = fixed_policy_kernels(ell_eval, cfg)
        inner_grid = cfg["inner_grid"] if train_grids is None else train_grids["inner_grid"]
        D1_inner, D2_inner = fixed_policy_kernels(inner_grid, cfg)
        calD1 = project_at(params, p, 1, D1, D1_inner, ell_eval, cfg, train_grids)
        calD2 = project_at(params, p, 2, D2, D2_inner, ell_eval, cfg, train_grids)
    else:
        D1, D2 = A["D1"], A["D2"]
        calD1, calD2 = A["calD1"], A["calD2"]
    Dtot = calD1 + calD2
    primitive_V = jnp.broadcast_to(cfg["std_v"] * EV, Dtot.shape)
    c0 = Dtot / cfg["std_z"]

    if cfg["private_signal_gauge"] == "literal":
        # Chapter convention dY^j = gamma_j (V-P) dt + sigma_Y dW.
        # The primitive kernel of V-P is the market-maker residual value.
        cY1 = (par["gamma1"] / cfg["std_y1"]) * A["vtilde0"]
        cY2 = (par["gamma2"] / cfg["std_y2"]) * A["vtilde0"]
    else:
        # Public-adjusted gauge: since P is public/order-flow measurable, observing
        # Y^j is filtration-equivalent to observing Y^j + int gamma_j P dt, whose
        # drift is gamma_j V.
        cY1 = (par["gamma1"] / cfg["std_y1"]) * primitive_V
        cY2 = (par["gamma2"] / cfg["std_y2"]) * primitive_V

    return {
        **A,
        "D1": D1,
        "D2": D2,
        "calD1": calD1,
        "calD2": calD2,
        "Dtot": Dtot,
        "primitive_V": primitive_V,
        "c0": c0,
        "cY1": cY1,
        "cY2": cY2,
    }


def eval_ctilde(params, p, ell, cfg):
    par = unpack_p(p)
    A = split_a(eval_a(params, p, ell, cfg))
    ce0 = A["dtot_tilde0"] / cfg["std_z"]
    ce1_z = A["dtot_tilde1"] / cfg["std_z"]
    ce2_z = A["dtot_tilde2"] / cfg["std_z"]
    ce1_y = (par["gamma1"] / cfg["std_y1"]) * A["vtilde1"]
    ce2_y = (par["gamma2"] / cfg["std_y2"]) * A["vtilde2"]
    return {
        "ce0": ce0[..., None, :],
        "ce1": jnp.stack([ce1_z, ce1_y], axis=-2),
        "ce2": jnp.stack([ce2_z, ce2_y], axis=-2),
        "ce1_z": ce1_z,
        "ce1_y": ce1_y,
        "ce2_z": ce2_z,
        "ce2_y": ce2_y,
    }


def ctilde_rows(params, p, observer: int, ell, cfg):
    ce = eval_ctilde(params, p, ell, cfg)
    if observer == 0:
        return ce["ce0"]
    if observer == 1:
        return ce["ce1"]
    if observer == 2:
        return ce["ce2"]
    raise ValueError(f"unknown observer {observer}")


def c_rows_for_observer(model, observer: int):
    if observer == 0:
        return model["c0"][..., None, :]
    if observer == 1:
        return jnp.stack([model["c0"], model["cY1"]], axis=-2)
    if observer == 2:
        return jnp.stack([model["c0"], model["cY2"]], axis=-2)
    raise ValueError(f"unknown observer {observer}")


def filter_kernel_pairs(params, p, observer: int, first_lag, second_lag, cfg, train_grids=None):
    """Characteristic solution for F^observer(first_lag, second_lag).

    Birth side traces are included when one lag is zero relative to the other;
    equal positive lags have no density mass from the diagonal atom.
    """
    first_lag = jnp.asarray(first_lag, dtype=jnp.float32)
    second_lag = jnp.asarray(second_lag, dtype=jnp.float32)
    first_ge_second = first_lag >= second_lag
    first_gt_second = first_lag > second_lag + 1e-7
    second_gt_first = second_lag > first_lag + 1e-7
    base = jnp.abs(first_lag - second_lag)
    width = jnp.minimum(first_lag, second_lag)
    rows = observer_rows(observer)

    c_base = ctilde_rows(params, p, observer, base, cfg)
    boundary_right = jnp.einsum("...rd,re->...de", c_base, rows)
    boundary_left = jnp.einsum("rd,...re->...de", rows, c_base)
    boundary = jnp.where(
        first_gt_second[..., None, None],
        boundary_right,
        jnp.where(second_gt_first[..., None, None], boundary_left, jnp.zeros_like(boundary_right)),
    )

    filter_nodes = cfg["filter_nodes"] if train_grids is None else train_grids["filter_nodes"]
    filter_weights = cfg["filter_weights"] if train_grids is None else train_grids["filter_weights"]
    tau = 0.5 * width[..., None] * (filter_nodes + 1.0)
    quad_w = 0.5 * width[..., None] * filter_weights
    c_first = ctilde_rows(
        params,
        p,
        observer,
        jnp.where(first_ge_second[..., None], base[..., None] + tau, tau),
        cfg,
    )
    c_second = ctilde_rows(
        params,
        p,
        observer,
        jnp.where(first_ge_second[..., None], tau, base[..., None] + tau),
        cfg,
    )
    integral = jnp.einsum("...q,...qrd,...qre->...de", quad_w, c_first, c_second)
    return boundary + integral


def project_at(params, p, observer: int, kernel_eval, kernel_grid, ell_eval, cfg, train_grids=None):
    rows = observer_rows(observer)
    direct = jnp.einsum("nd,rd,re->ne", kernel_eval, rows, rows)
    inner_grid = cfg["inner_grid"] if train_grids is None else train_grids["inner_grid"]
    inner_weights = cfg["inner_weights"] if train_grids is None else train_grids["inner_weights"]
    first = jnp.broadcast_to(inner_grid[:, None], (inner_grid.shape[0], ell_eval.shape[0]))
    second = jnp.broadcast_to(ell_eval[None, :], first.shape)
    f = filter_kernel_pairs(params, p, observer, first, second, cfg, train_grids)
    indirect = jnp.einsum("a,ad,alde->le", inner_weights, kernel_grid, f)
    # Lag zero is the newborn observed innovation atom.  Do not let old-history
    # density terms leak into the instantaneous primitive-shock boundary.
    indirect = jnp.where((ell_eval > 1e-7)[..., None], indirect, jnp.zeros_like(indirect))
    return direct + indirect


def closure_residual(params, p, observer: int, model_grid, cfg, ell_eval=None, train_grids=None):
    grid = cfg["grid"] if ell_eval is None else jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    c_rows = c_rows_for_observer(model_grid, observer)
    quad_grid = cfg["quad_grid"] if train_grids is None else train_grids["quad_grid"]
    quad_weights = cfg["quad_weights"] if train_grids is None else train_grids["quad_weights"]
    c_quad = c_rows_for_observer(model_at(params, p, quad_grid, cfg, train_grids), observer)
    ce_rows = ctilde_rows(params, p, observer, grid, cfg)
    pi = observer_pi(observer)
    direct = jnp.einsum("lrd,de->lre", c_rows, EYE - pi)
    first = jnp.broadcast_to(quad_grid[:, None], (quad_grid.shape[0], grid.shape[0]))
    second = jnp.broadcast_to(grid[None, :], first.shape)
    f = filter_kernel_pairs(params, p, observer, first, second, cfg, train_grids)
    integral = jnp.einsum("a,ard,alde->lre", quad_weights, c_quad, f)
    # At zero lag there is no old history to infer from.  The observer only has
    # the instantaneous directly observed innovation, represented by Pi above;
    # F contributes through birth side traces for positive lags, not as an
    # indirect history integral at ell=0.
    integral = jnp.where((grid > 1e-7)[..., None, None], integral, jnp.zeros_like(integral))
    return ce_rows - (direct - integral)


def vtilde_projection_residual(params, p, observer: int, cfg, ell_eval=None, train_grids=None):
    grid = cfg["grid"] if ell_eval is None else jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    A = split_a(eval_a(params, p, grid, cfg))
    vt = A[f"vtilde{observer}"]
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (grid.shape[0], D_W))
    inner_grid = cfg["inner_grid"] if train_grids is None else train_grids["inner_grid"]
    v_inner = jnp.broadcast_to(cfg["std_v"] * EV, (inner_grid.shape[0], D_W))
    v_proj = project_at(params, p, observer, v_eval, v_inner, grid, cfg, train_grids)
    return vt - (v_eval - v_proj)


def filter_residual(params, p, observer: int, model_grid, cfg, ell_eval=None, train_grids=None):
    return jnp.concatenate(
        [
            closure_residual(params, p, observer, model_grid, cfg, ell_eval, train_grids).reshape(-1),
            vtilde_projection_residual(params, p, observer, cfg, ell_eval, train_grids).reshape(-1),
        ]
    )


def oscillation_penalty(params, p, cfg, grid=None, include_policy=False, include_H=False):
    """Curvature penalty for high-frequency lag wiggles.

    This is deliberately not a monotonicity penalty.  It penalizes squared
    second lag derivatives of the learned curves, so smooth humps remain
    possible while grid-scale oscillations become expensive.
    """
    if grid is None:
        grid = cfg["quad_grid"]
    _, _, d2a = value_first_second_deriv_a(params, p, grid, cfg)
    A2 = split_a(d2a)
    names = ["vtilde0", "vtilde1", "vtilde2", "dtot_tilde0", "dtot_tilde1", "dtot_tilde2"]
    if include_policy and not cfg["fixed_policy"]:
        names += ["D1", "D2", "calD1", "calD2"]
    penalty = sum(mse(A2[name]) for name in names)
    if include_H:
        _, d2H = jax.jvp(
            lambda ee: value_and_deriv_H(params, p, ee, cfg)[1],
            (grid,),
            (jnp.ones_like(grid),),
        )
        penalty = penalty + mse(d2H)
    return penalty / float(len(names) + (1 if include_H else 0))


def random_filter_loss_grid(key, cfg):
    """Jittered collocation grid for filter-only training.

    We keep the old-history quadrature grids deterministic, but randomize where
    the closure/projection residual is evaluated.  Stratification avoids leaving
    large lag intervals untested in a given step, while still preventing the net
    from fitting a fixed set of lag points.
    """
    if not cfg["random_filter_grid"]:
        return cfg["grid"]
    n = int(cfg["filter_loss_points"])
    if n <= 2:
        return jnp.asarray([0.0, cfg["L"]], dtype=cfg["grid"].dtype)
    interior_n = n - 2
    dtype = cfg["grid"].dtype
    u = (jnp.arange(interior_n, dtype=dtype) + jax.random.uniform(key, (interior_n,), dtype=dtype)) / interior_n
    interior = cfg["L"] * jnp.sort(u)
    return jnp.concatenate(
        [
            jnp.asarray([0.0], dtype=dtype),
            interior,
            jnp.asarray([cfg["L"]], dtype=dtype),
        ]
    )


def random_stratified_quadrature(key, n: int, lo: float, hi: float, dtype):
    u = (jnp.arange(n, dtype=dtype) + jax.random.uniform(key, (n,), dtype=dtype)) / n
    nodes = lo + (hi - lo) * jnp.sort(u)
    weights = jnp.full((n,), (hi - lo) / n, dtype=dtype)
    return nodes, weights


def random_tail_grid(key, cfg):
    dtype = cfg["grid"].dtype
    if not cfg["random_filter_tail"]:
        return jnp.asarray([cfg["L"]], dtype=dtype)
    n = int(cfg["filter_tail_points"])
    lo = cfg["L"] * (1.0 - cfg["filter_tail_fraction"])
    nodes, _ = random_stratified_quadrature(key, n, lo, cfg["L"], dtype)
    return nodes


def random_filter_training_grids(key, cfg):
    k_loss, k_quad, k_inner, k_filter, k_tail = jax.random.split(key, 5)
    dtype = cfg["grid"].dtype
    loss_grid = random_filter_loss_grid(k_loss, cfg)
    if cfg["random_filter_integrals"]:
        quad_grid, quad_weights = random_stratified_quadrature(
            k_quad, int(cfg["filter_quad_loss_points"]), 0.0, cfg["L"], dtype
        )
        inner_grid, inner_weights = random_stratified_quadrature(
            k_inner, int(cfg["filter_inner_loss_points"]), 0.0, cfg["L"], dtype
        )
        filter_nodes, filter_weights = random_stratified_quadrature(
            k_filter, int(cfg["filter_kernel_loss_points"]), -1.0, 1.0, dtype
        )
    else:
        quad_grid, quad_weights = cfg["quad_grid"], cfg["quad_weights"]
        inner_grid, inner_weights = cfg["inner_grid"], cfg["inner_weights"]
        filter_nodes, filter_weights = cfg["filter_nodes"], cfg["filter_weights"]
    return {
        "loss_grid": loss_grid,
        "quad_grid": quad_grid,
        "quad_weights": quad_weights,
        "inner_grid": inner_grid,
        "inner_weights": inner_weights,
        "filter_nodes": filter_nodes,
        "filter_weights": filter_weights,
        "tail_grid": random_tail_grid(k_tail, cfg),
    }


def row_rms(residual):
    return jnp.sqrt(jnp.mean(jnp.square(residual), axis=(0, 2)))


def zero_unobserved_residual(params, p, observer: int, model_grid, cfg, ell_eval=None):
    pi = observer_pi(observer)
    unobs = jnp.diag(jnp.eye(D_W, dtype=jnp.float32) - pi)
    return closure_residual(params, p, observer, model_grid, cfg, ell_eval)[0] * unobs[None, :]


def zero_birth_residual(params, p, observer: int, model_grid, cfg, ell_eval=None):
    obs = jnp.diag(observer_pi(observer))
    return closure_residual(params, p, observer, model_grid, cfg, ell_eval)[0] * obs[None, :]


def observer_projection_matrix(params, p, observer: int, ell_eval, cfg):
    """IRF matrix for an observer noise state from primitive Brownian shocks."""
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    eye = jnp.eye(D_W, dtype=jnp.float32)
    mats = []
    for ch in range(D_W):
        kernel_eval = jnp.broadcast_to(eye[ch], (ell_eval.shape[0], D_W))
        kernel_grid = jnp.broadcast_to(eye[ch], (cfg["inner_grid"].shape[0], D_W))
        mats.append(project_at(params, p, observer, kernel_eval, kernel_grid, ell_eval, cfg))
    return jnp.stack(mats, axis=1)


def value_irfs(params, p, ell_eval, cfg):
    """Price and trader value-estimate IRFs implied by unresolved V kernels."""
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    A = split_a(eval_a(params, p, ell_eval, cfg))
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (ell_eval.shape[0], D_W))
    v_grid = jnp.broadcast_to(cfg["std_v"] * EV, (cfg["inner_grid"].shape[0], D_W))
    price = v_eval - A["vtilde0"]
    trader1 = v_eval - A["vtilde1"]
    trader2 = v_eval - A["vtilde2"]
    price_from_F = project_at(params, p, 0, v_eval, v_grid, ell_eval, cfg)
    trader1_from_F = project_at(params, p, 1, v_eval, v_grid, ell_eval, cfg)
    trader2_from_F = project_at(params, p, 2, v_eval, v_grid, ell_eval, cfg)
    return {
        "true_value": v_eval,
        "price": price,
        "trader1_estimate": trader1,
        "trader2_estimate": trader2,
        "mispricing_gap1": trader1 - price,
        "mispricing_gap2": trader2 - price,
        "price_residual_implied": price,
        "trader1_estimate_residual_implied": trader1,
        "trader2_estimate_residual_implied": trader2,
        "price_from_F_projection": price_from_F,
        "trader1_estimate_from_F_projection": trader1_from_F,
        "trader2_estimate_from_F_projection": trader2_from_F,
    }


# ---------------------------------------------------------------------------
# Adjoints and weak stationary readout
# ---------------------------------------------------------------------------


def alpha_terms(params, p, cfg):
    w = cfg["quad_weights"]
    H = split_H(eval_H(params, p, cfg["quad_grid"], cfg))
    H0 = split_H(eval_H(params, p, jnp.array([0.0], dtype=jnp.float32), cfg))
    ce = eval_ctilde(params, p, cfg["quad_grid"], cfg)
    rows0 = observer_rows(0)
    rows1 = observer_rows(1)
    rows2 = observer_rows(2)

    a01_z = jnp.einsum("n,nij,nj->i", w, H["H01"], ce["ce0"][:, 0, :]) + H0["H01"][0] @ rows0[0]
    a02_z = jnp.einsum("n,nij,nj->i", w, H["H02"], ce["ce0"][:, 0, :]) + H0["H02"][0] @ rows0[0]
    a21_z = jnp.einsum("n,nij,nj->i", w, H["H21"], ce["ce2"][:, 0, :]) + H0["H21"][0] @ rows2[0]
    a21_y = jnp.einsum("n,nij,nj->i", w, H["H21"], ce["ce2"][:, 1, :]) + H0["H21"][0] @ rows2[1]
    a12_z = jnp.einsum("n,nij,nj->i", w, H["H12"], ce["ce1"][:, 0, :]) + H0["H12"][0] @ rows1[0]
    a12_y = jnp.einsum("n,nij,nj->i", w, H["H12"], ce["ce1"][:, 1, :]) + H0["H12"][0] @ rows1[1]
    return {
        "a01_z": a01_z,
        "a02_z": a02_z,
        "a21_z": a21_z,
        "a21_y": a21_y,
        "a12_z": a12_z,
        "a12_y": a12_y,
        "beta1": a01_z + a21_z,
        "beta2": a02_z + a12_z,
    }


def source_blocks(params, p, ell, alphas, cfg):
    model = model_at(params, p, ell, cfg)
    v0 = cfg["std_v"] * EV
    src01 = vec_outer(model["calD1"], v0) - vec_outer(alphas["a01_z"], model["c0"])
    src02 = vec_outer(model["calD2"], v0) - vec_outer(alphas["a02_z"], model["c0"])
    src21 = (
        vec_outer(alphas["beta1"], model["D2"] / cfg["std_z"])
        - vec_outer(alphas["a21_z"], model["c0"])
        - vec_outer(alphas["a21_y"], model["cY2"])
    )
    src12 = (
        vec_outer(alphas["beta2"], model["D1"] / cfg["std_z"])
        - vec_outer(alphas["a12_z"], model["c0"])
        - vec_outer(alphas["a12_y"], model["cY1"])
    )
    return {"H01": src01, "H21": src21, "H02": src02, "H12": src12}


def lambda_value(params, p, cfg):
    ce0 = eval_ctilde(params, p, cfg["quad_grid"], cfg)["ce0"][:, 0, :]
    return (cfg["std_v"] / cfg["std_z"]) * jnp.einsum("n,nd,d->", cfg["quad_weights"], ce0, EV)


def readout_for_deviator(params, p, H_name0, H_name_opp, opp: int, alphas, cfg):
    par = unpack_p(p)
    grid = cfg["grid"]
    quad_grid = cfg["quad_grid"]
    inner_grid = cfg["inner_grid"]
    weights = cfg["quad_weights"]

    H = split_H(eval_H(params, p, grid, cfg))
    Hq = split_H(eval_H(params, p, quad_grid, cfg))
    Hi = split_H(eval_H(params, p, inner_grid, cfg))
    H0val = split_H(eval_H(params, p, jnp.array([0.0], dtype=jnp.float32), cfg))

    src = source_blocks(params, p, grid, alphas, cfg)
    src_q = source_blocks(params, p, quad_grid, alphas, cfg)
    src_i = source_blocks(params, p, inner_grid, alphas, cfg)
    model_q = model_at(params, p, quad_grid, cfg)
    model_i = model_at(params, p, inner_grid, cfg)
    ce = eval_ctilde(params, p, grid, cfg)
    ce_q = eval_ctilde(params, p, quad_grid, cfg)
    ce_i = eval_ctilde(params, p, inner_grid, cfg)
    ce_zero = eval_ctilde(params, p, jnp.array([0.0], dtype=jnp.float32), cfg)

    if opp == 1:
        ce_opp_z, ce_opp_y = ce["ce1"][:, 0, :], ce["ce1"][:, 1, :]
        ce_opp_z_q, ce_opp_y_q = ce_q["ce1"][:, 0, :], ce_q["ce1"][:, 1, :]
        ce_opp_z_i, ce_opp_y_i = ce_i["ce1"][:, 0, :], ce_i["ce1"][:, 1, :]
        c_opp_y_q = model_q["cY1"]
        D_opp_q = model_q["D1"]
        ey = EY1
        ce_opp_zero_z = ce_zero["ce1"][0, 0, :]
    else:
        ce_opp_z, ce_opp_y = ce["ce2"][:, 0, :], ce["ce2"][:, 1, :]
        ce_opp_z_q, ce_opp_y_q = ce_q["ce2"][:, 0, :], ce_q["ce2"][:, 1, :]
        ce_opp_z_i, ce_opp_y_i = ce_i["ce2"][:, 0, :], ce_i["ce2"][:, 1, :]
        c_opp_y_q = model_q["cY2"]
        D_opp_q = model_q["D2"]
        ey = EY2
        ce_opp_zero_z = ce_zero["ce2"][0, 0, :]

    phi0 = ce["ce0"][:, 0, :] / cfg["std_z"]
    phi_opp = ce_opp_z / cfg["std_z"]
    phi0_q = ce_q["ce0"][:, 0, :] / cfg["std_z"]
    phi_opp_q = ce_opp_z_q / cfg["std_z"]
    phi0_i = ce_i["ce0"][:, 0, :] / cfg["std_z"]
    phi_opp_i = ce_opp_z_i / cfg["std_z"]
    phi0_zero = ce_zero["ce0"][0, 0, :] / cfg["std_z"]
    phi_opp_zero = ce_opp_zero_z / cfg["std_z"]
    psi = EZ / cfg["std_z"]

    R = jnp.einsum("n,nd,nd->", weights, D_opp_q, phi_opp_q)

    bracket0 = jnp.einsum("n,nd,nd->", weights, model_q["c0"], phi0_q) - R / cfg["std_z"]
    bracket_opp_z = jnp.einsum("n,nd,nd->", weights, model_q["c0"], phi_opp_q) - R / cfg["std_z"]
    bracket_opp_y = jnp.einsum("n,nd,nd->", weights, c_opp_y_q, phi_opp_q)

    q0 = ce["ce0"][:, 0, :] * bracket0
    q_opp = ce_opp_z * bracket_opp_z + ce_opp_y * bracket_opp_y
    q0_q = ce_q["ce0"][:, 0, :] * bracket0
    q_opp_q = ce_opp_z_q * bracket_opp_z + ce_opp_y_q * bracket_opp_y
    q0_i = ce_i["ce0"][:, 0, :] * bracket0
    q_opp_i = ce_opp_z_i * bracket_opp_z + ce_opp_y_i * bracket_opp_y
    q0_diag = EZ * bracket0
    q_opp_diag = EZ * bracket_opp_z + ey * bracket_opp_y

    density0 = jnp.einsum("nij,nj->ni", src[H_name0], phi0) + jnp.einsum("nij,nj->ni", H[H_name0], q0)
    density_opp = jnp.einsum("nij,nj->ni", src[H_name_opp], phi_opp) + jnp.einsum("nij,nj->ni", H[H_name_opp], q_opp)
    density0_q = jnp.einsum("nij,nj->ni", src_q[H_name0], phi0_q) + jnp.einsum("nij,nj->ni", Hq[H_name0], q0_q)
    density_opp_q = jnp.einsum("nij,nj->ni", src_q[H_name_opp], phi_opp_q) + jnp.einsum("nij,nj->ni", Hq[H_name_opp], q_opp_q)
    density0_i = jnp.einsum("nij,nj->ni", src_i[H_name0], phi0_i) + jnp.einsum("nij,nj->ni", Hi[H_name0], q0_i)
    density_opp_i = jnp.einsum("nij,nj->ni", src_i[H_name_opp], phi_opp_i) + jnp.einsum("nij,nj->ni", Hi[H_name_opp], q_opp_i)

    old0 = jnp.einsum("n,nd->d", weights, density0_q)
    old_opp = jnp.einsum("n,nd->d", weights, density_opp_q)
    diag0 = H0val[H_name0][0] @ (par["rho"] * psi - phi0_zero + q0_diag)
    diag_opp = H0val[H_name_opp][0] @ (par["rho"] * psi - phi_opp_zero + q_opp_diag)

    return {
        "density_total": density0 + density_opp,
        "density_total_quad": density0_q + density_opp_q,
        "density_total_inner": density0_i + density_opp_i,
        "total": old0 + old_opp + diag0 + diag_opp,
        "old_market": old0,
        "old_opponent": old_opp,
        "diag_market": diag0,
        "diag_opponent": diag_opp,
        "R": R,
    }


def equilibrium_targets(params, p, cfg):
    grid = cfg["grid"]
    H_grid = eval_H(params, p, grid, cfg)
    alphas = alpha_terms(params, p, cfg)
    ro1 = readout_for_deviator(params, p, "H01", "H21", 2, alphas, cfg)
    ro2 = readout_for_deviator(params, p, "H02", "H12", 1, alphas, cfg)
    lam = lambda_value(params, p, cfg)
    inv_lam = 1.0 / jnp.where(jnp.abs(lam) > 1e-6, lam, jnp.inf)
    ce0 = eval_ctilde(params, p, grid, cfg)["ce0"][:, 0, :]
    ce0_i = eval_ctilde(params, p, cfg["inner_grid"], cfg)["ce0"][:, 0, :]
    model = model_at(params, p, grid, cfg)

    local_info = cfg["std_z"] * ce0
    local_info_i = cfg["std_z"] * ce0_i
    mispricing1 = project_at(params, p, 1, local_info, local_info_i, grid, cfg)
    mispricing2 = project_at(params, p, 2, local_info, local_info_i, grid, cfg)

    calH1 = project_at(params, p, 1, ro1["density_total"], ro1["density_total_inner"], grid, cfg)
    calH2 = project_at(params, p, 2, ro2["density_total"], ro2["density_total_inner"], grid, cfg)

    foc_lhs1 = lam * (model["calD1"] - mispricing1)
    foc_lhs2 = lam * (model["calD2"] - mispricing2)
    return {
        "target1": mispricing1 + calH1 * inv_lam,
        "target2": mispricing2 + calH2 * inv_lam,
        "lambda": lam,
        "ro1": ro1,
        "ro2": ro2,
        "mispricing_demand1": mispricing1,
        "mispricing_demand2": mispricing2,
        "calH1": calH1,
        "calH2": calH2,
        "foc_lhs1": foc_lhs1,
        "foc_lhs2": foc_lhs2,
        "foc_mismatch1": foc_lhs1 - calH1,
        "foc_mismatch2": foc_lhs2 - calH2,
    }


# ---------------------------------------------------------------------------
# Residuals, diagnostics, and training
# ---------------------------------------------------------------------------


def single_residual_metrics(params, p, cfg, train_grids=None):
    grid = cfg["grid"] if train_grids is None else train_grids["loss_grid"]
    model = model_at(params, p, grid, cfg, train_grids)

    if cfg["filter_only"]:
        A = split_a(eval_a(params, p, grid, cfg))
        tail_grid = jnp.array([cfg["L"]], dtype=jnp.float32) if train_grids is None else train_grids["tail_grid"]
        rough_grid = cfg["quad_grid"] if train_grids is None else train_grids["quad_grid"]
        tail_A = split_a(eval_a(params, p, tail_grid, cfg))
        tilde_names = ["vtilde0", "vtilde1", "vtilde2", "dtot_tilde0", "dtot_tilde1", "dtot_tilde2"]
        tilde_tail = sum(mse(tail_A[name]) for name in tilde_names)
        tilde_reg = sum(mse(A[name]) for name in tilde_names)
        rough = oscillation_penalty(params, p, cfg, rough_grid)
        return jnp.stack(
            [
                cfg["w_filter"] * mse(filter_residual(params, p, 0, model, cfg, grid, train_grids)),
                cfg["w_filter"] * mse(filter_residual(params, p, 1, model, cfg, grid, train_grids)),
                cfg["w_filter"] * mse(filter_residual(params, p, 2, model, cfg, grid, train_grids)),
                cfg["w_tail"] * tilde_tail,
                cfg["w_oscillation"] * rough,
                cfg["w_reg"] * tilde_reg,
            ]
        )

    A = split_a(eval_a(params, p, grid, cfg))
    A_inner = split_a(eval_a(params, p, cfg["inner_grid"], cfg))
    D1_proj = project_at(params, p, 1, A["D1"], A_inner["D1"], grid, cfg)
    D2_proj = project_at(params, p, 2, A["D2"], A_inner["D2"], grid, cfg)

    H_val, H_der = value_and_deriv_H(params, p, cfg["ode_grid"], cfg)
    H = split_H(H_val)
    dH = split_H(H_der)
    alphas = alpha_terms(params, p, cfg)
    src = source_blocks(params, p, cfg["ode_grid"], alphas, cfg)
    targets = equilibrium_targets(params, p, cfg)
    par = unpack_p(p)

    tail_model = model_at(params, p, jnp.array([cfg["L"]], dtype=jnp.float32), cfg)
    tail_H = eval_H(params, p, jnp.array([cfg["L"]], dtype=jnp.float32), cfg)

    return jnp.stack(
        [
            cfg["w_filter"] * mse(filter_residual(params, p, 0, model, cfg)),
            cfg["w_filter"] * mse(filter_residual(params, p, 1, model, cfg)),
            cfg["w_filter"] * mse(filter_residual(params, p, 2, model, cfg)),
            cfg["w_policy_proj"] * (mse(A["calD1"] - D1_proj) + mse(A["calD2"] - D2_proj)),
            cfg["w_adjoint"] * mse(par["rho"] * H["H01"] - dH["H01"] - src["H01"]),
            cfg["w_adjoint"] * mse(par["rho"] * H["H21"] - dH["H21"] - src["H21"]),
            cfg["w_adjoint"] * mse(par["rho"] * H["H02"] - dH["H02"] - src["H02"]),
            cfg["w_adjoint"] * mse(par["rho"] * H["H12"] - dH["H12"] - src["H12"]),
            cfg["w_foc"] * (mse(targets["foc_mismatch1"]) + mse(targets["foc_mismatch2"])),
            cfg["w_policy_target"] * (mse(A["calD1"] - targets["target1"]) + mse(A["calD2"] - targets["target2"])),
            cfg["w_tail"]
            * (
                mse(tail_model["D1"])
                + mse(tail_model["D2"])
                + mse(tail_model["calD1"])
                + mse(tail_model["calD2"])
                + mse(tail_model["vtilde0"])
                + mse(tail_model["vtilde1"])
                + mse(tail_model["vtilde2"])
                + mse(tail_model["dtot_tilde0"])
                + mse(tail_model["dtot_tilde1"])
                + mse(tail_model["dtot_tilde2"])
                + mse(tail_H)
            ),
            cfg["w_oscillation"] * oscillation_penalty(
                params,
                p,
                cfg,
                cfg["quad_grid"],
                include_policy=True,
                include_H=True,
            ),
            cfg["w_reg"]
            * (
                mse(eval_a(params, p, grid, cfg))
                + mse(eval_H(params, p, grid, cfg))
            ),
        ]
    )


def batch_metrics(params, p_batch, cfg, train_grids=None):
    return jnp.mean(jax.vmap(lambda pp: single_residual_metrics(params, pp, cfg, train_grids))(p_batch), axis=0)


def sample_params(key, batch_size: int, ranges: Dict[str, float], fixed_p=None):
    if fixed_p is not None:
        return jnp.broadcast_to(fixed_p, (batch_size, fixed_p.shape[0]))
    names = ["gamma1", "gamma2", "rho"]
    lo = jnp.log(jnp.array([ranges[f"{n}_min"] for n in names], dtype=jnp.float32))
    hi = jnp.log(jnp.array([ranges[f"{n}_max"] for n in names], dtype=jnp.float32))
    u = jax.random.uniform(key, (batch_size, len(names)), dtype=jnp.float32)
    return jnp.exp(lo + u * (hi - lo))


def diagnostics_for_p(params, p, cfg):
    grid = cfg["grid"]
    model = model_at(params, p, grid, cfg)
    targets = equilibrium_targets(params, p, cfg)
    Dtot = model["calD1"] + model["calD2"]
    Drel = model["calD1"] - model["calD2"]
    back_gap = model["dtot_tilde0"] - Dtot
    zero = jnp.array([0.0], dtype=jnp.float32)
    A0 = split_a(eval_a(params, p, zero, cfg))
    Dtot_inner = model_at(params, p, cfg["inner_grid"], cfg)["Dtot"]
    mm_proj_Dtot = project_at(params, p, 0, Dtot, Dtot_inner, grid, cfg)
    return {
        "lambda": float(targets["lambda"]),
        "foc_rms": {
            "player1": float(jnp.sqrt(mse(targets["foc_mismatch1"]))),
            "player2": float(jnp.sqrt(mse(targets["foc_mismatch2"]))),
        },
        "mode_norms": {
            "aggregate_Dtot_rms": float(jnp.sqrt(mse(Dtot))),
            "relative_D1_minus_D2_rms": float(jnp.sqrt(mse(Drel))),
            "back_like_gap_dtot_tilde0_minus_Dtot_rms": float(jnp.sqrt(mse(back_gap))),
            "market_maker_projection_Dtot_rms": float(jnp.sqrt(mse(mm_proj_Dtot))),
        },
        "zero_trace": {
            "D1_zero_rms": float(jnp.sqrt(mse(A0["D1"]))),
            "D2_zero_rms": float(jnp.sqrt(mse(A0["D2"]))),
            "calD1_zero_rms": float(jnp.sqrt(mse(A0["calD1"]))),
            "calD2_zero_rms": float(jnp.sqrt(mse(A0["calD2"]))),
        },
        "readout_components": {
            "player1_R": float(targets["ro1"]["R"]),
            "player2_R": float(targets["ro2"]["R"]),
            "player1_old_market_rms": float(jnp.sqrt(mse(targets["ro1"]["old_market"]))),
            "player1_old_opponent_rms": float(jnp.sqrt(mse(targets["ro1"]["old_opponent"]))),
            "player1_diag_market_rms": float(jnp.sqrt(mse(targets["ro1"]["diag_market"]))),
            "player1_diag_opponent_rms": float(jnp.sqrt(mse(targets["ro1"]["diag_opponent"]))),
        },
    }


def filter_diagnostics_for_p(params, p, cfg):
    grid = cfg["grid"]
    model = model_at(params, p, grid, cfg)
    targets_lambda = lambda_value(params, p, cfg)
    Dtot = model["calD1"] + model["calD2"]
    Drel = model["calD1"] - model["calD2"]
    zero = jnp.array([0.0], dtype=jnp.float32)
    A0 = split_a(eval_a(params, p, zero, cfg))
    return {
        "lambda": float(targets_lambda),
        "closure_rms": {
            f"observer{b}": float(jnp.sqrt(mse(filter_residual(params, p, b, model, cfg))))
            for b in (0, 1, 2)
        },
        "mode_norms": {
            "aggregate_Dtot_rms": float(jnp.sqrt(mse(Dtot))),
            "relative_D1_minus_D2_rms": float(jnp.sqrt(mse(Drel))),
            "back_like_gap_dtot_tilde0_minus_Dtot_rms": float(jnp.sqrt(mse(model["dtot_tilde0"] - Dtot))),
        },
        "zero_trace": {
            "D1_zero_rms": float(jnp.sqrt(mse(model_at(params, p, zero, cfg)["D1"]))),
            "D2_zero_rms": float(jnp.sqrt(mse(model_at(params, p, zero, cfg)["D2"]))),
            "calD1_zero_rms": float(jnp.sqrt(mse(model_at(params, p, zero, cfg)["calD1"]))),
            "calD2_zero_rms": float(jnp.sqrt(mse(model_at(params, p, zero, cfg)["calD2"]))),
        },
        "zero_lag_unresolved_rows": {
            "dtot_tilde0_rms": float(jnp.sqrt(mse(A0["dtot_tilde0"]))),
            "dtot_tilde1_rms": float(jnp.sqrt(mse(A0["dtot_tilde1"]))),
            "dtot_tilde2_rms": float(jnp.sqrt(mse(A0["dtot_tilde2"]))),
        },
    }


def diagnostics(params, cfg, args, step, elapsed, loss, metrics, grad_norm, val_loss=None, val_metrics=None):
    metric_names = cfg["metric_names"]
    block_rms = {name: float(math.sqrt(max(v, 0.0))) for name, v in zip(metric_names, np.asarray(metrics))}
    center_p = cfg["fixed_param"]
    if center_p is None:
        center_p = jnp.array(
            [
                math.sqrt(args.gamma1_min * args.gamma1_max),
                math.sqrt(args.gamma2_min * args.gamma2_max),
                math.sqrt(args.rho_min * args.rho_max),
            ],
            dtype=jnp.float32,
        )
    out = {
        "step": int(step),
        "elapsed_s": float(elapsed),
        "loss": float(loss),
        "train_rms": float(math.sqrt(max(float(loss) / len(metric_names), 0.0))),
        "grad_norm": float(grad_norm),
        "block_rms": block_rms,
        "n_params": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))),
        "random_parameter_training": not bool(args.fixed_params),
        "fixed_parameter_training": bool(args.fixed_params),
        "filter_only_training": bool(args.filter_only),
        "fixed_policy": bool(args.fixed_policy),
        "fixed_parameter": {
            "gamma1": float(center_p[0]),
            "gamma2": float(center_p[1]),
            "rho": float(center_p[2]),
        },
        "private_signal_gauge": cfg["private_signal_gauge"],
        "lambda_is_parameter": False,
        "representative_selection": (
            "D_i is a fixed exponential noise-state strategy; calD_i is its primitive projection"
            if args.fixed_policy
            else "calD_i is projected primitive demand; D_i is learned noise-state representative"
        ),
        "sample_diagnostics_center": (
            filter_diagnostics_for_p(params, center_p, cfg)
            if args.filter_only
            else diagnostics_for_p(params, center_p, cfg)
        ),
    }
    if val_loss is not None and val_metrics is not None:
        out["val_loss"] = float(val_loss)
        out["val_rms"] = float(math.sqrt(max(float(val_loss) / len(metric_names), 0.0)))
        out["val_block_rms"] = {name: float(math.sqrt(max(v, 0.0))) for name, v in zip(metric_names, np.asarray(val_metrics))}
    return out


def tree_to_jsonable(params):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32).tolist(), params)


def export_parameter(args, cfg):
    if cfg["fixed_param"] is not None:
        return cfg["fixed_param"]
    return jnp.array(
        [
            math.sqrt(args.gamma1_min * args.gamma1_max),
            math.sqrt(args.gamma2_min * args.gamma2_max),
            math.sqrt(args.rho_min * args.rho_max),
        ],
        dtype=jnp.float64 if args.x64 else jnp.float32,
    )


def as_json_array(value):
    return np.asarray(value, dtype=float).tolist()


def export_grid(cfg):
    n = int(cfg.get("export_N", 0) or 0)
    if n <= 0 or n == int(cfg["N"]):
        return cfg["grid"]
    return jnp.linspace(0.0, float(cfg["L"]), n, dtype=cfg["grid"].dtype)


def save_json(path: str, params, args, cfg, diag):
    grid = export_grid(cfg)
    p_export = export_parameter(args, cfg)
    par = unpack_p(p_export)
    model = model_at(params, p_export, grid, cfg)
    model_inner = model_at(params, p_export, cfg["inner_grid"], cfg)
    A = split_a(eval_a(params, p_export, grid, cfg))
    A_inner = split_a(eval_a(params, p_export, cfg["inner_grid"], cfg))
    ce = eval_ctilde(params, p_export, grid, cfg)
    H = split_H(eval_H(params, p_export, grid, cfg))
    targets = equilibrium_targets(params, p_export, cfg)
    val = value_irfs(params, p_export, grid, cfg)
    Dtot = model["calD1"] + model["calD2"]
    Drel = model["calD1"] - model["calD2"]
    Dmean = 0.5 * Dtot
    Dtot_proj0 = project_at(params, p_export, 0, Dtot, model_inner["Dtot"], grid, cfg)
    if cfg["fixed_policy"]:
        D1_inner, D2_inner = fixed_policy_kernels(cfg["inner_grid"], cfg)
    else:
        D1_inner, D2_inner = A_inner["D1"], A_inner["D2"]
    D1_proj = project_at(params, p_export, 1, model["D1"], D1_inner, grid, cfg)
    D2_proj = project_at(params, p_export, 2, model["D2"], D2_inner, grid, cfg)
    filter_residuals = {
        f"observer{b}": as_json_array(closure_residual(params, p_export, b, model, cfg, grid))
        for b in (0, 1, 2)
    }
    vtilde_residuals = {
        f"observer{b}": as_json_array(vtilde_projection_residual(params, p_export, b, cfg, grid))
        for b in (0, 1, 2)
    }
    zero_filter_residuals = {
        "unobserved": {
            f"observer{b}": as_json_array(zero_unobserved_residual(params, p_export, b, model, cfg, grid))
            for b in (0, 1, 2)
        },
        "birth_observed": {
            f"observer{b}": as_json_array(zero_birth_residual(params, p_export, b, model, cfg, grid))
            for b in (0, 1, 2)
        },
    }
    noise_state_irfs = {
        "market_maker": as_json_array(observer_projection_matrix(params, p_export, 0, grid, cfg)),
        "trader1": as_json_array(observer_projection_matrix(params, p_export, 1, grid, cfg)),
        "trader2": as_json_array(observer_projection_matrix(params, p_export, 2, grid, cfg)),
    }
    row_names = {
        0: ["order_flow_tilde"],
        1: ["order_flow_tilde", "private_signal_V_residual"],
        2: ["order_flow_tilde", "private_signal_V_residual"],
    }
    closure_row_rms = {
        f"observer{b}": {
            name: float(value)
            for name, value in zip(
                row_names[b],
                np.asarray(row_rms(closure_residual(params, p_export, b, model, cfg, grid)), dtype=float),
            )
        }
        for b in (0, 1, 2)
    }
    zero = jnp.array([0.0], dtype=jnp.float32)
    A0 = split_a(eval_a(params, p_export, zero, cfg))
    ce0 = eval_ctilde(params, p_export, zero, cfg)
    value_err1 = np.abs(np.asarray(model["vtilde1"], dtype=float)[:, 0])
    value_err2 = np.abs(np.asarray(model["vtilde2"], dtype=float)[:, 0])
    if float(par["gamma2"]) > float(par["gamma1"]):
        info_violation = np.maximum(value_err2 - value_err1, 0.0)
        higher_gamma = "trader2"
    elif float(par["gamma1"]) > float(par["gamma2"]):
        info_violation = np.maximum(value_err1 - value_err2, 0.0)
        higher_gamma = "trader1"
    else:
        info_violation = value_err1 - value_err2
        higher_gamma = "symmetric"

    export_diag = dict(diag)
    export_diag.pop("best", None)
    export_diag["rms"] = float(export_diag.get("val_rms", export_diag.get("train_rms", 0.0)))
    export_diag["lambda"] = float(targets["lambda"])
    export_diag["lambda_abs"] = float(jnp.abs(targets["lambda"]))
    export_diag["lambda_sign"] = float(jnp.sign(targets["lambda"]))
    block_rms = dict(export_diag.get("val_block_rms") or export_diag.get("block_rms") or {})
    if "foc" in block_rms and "foc_mismatch" not in block_rms:
        block_rms["foc_mismatch"] = block_rms["foc"]
    if "policy_target_stabilizer" in block_rms and "policy_target" not in block_rms:
        block_rms["policy_target"] = block_rms["policy_target_stabilizer"]
    export_diag["block_rms"] = block_rms
    export_diag["closure_rms_unweighted"] = {
        f"observer{b}": float(jnp.sqrt(mse(filter_residual(params, p_export, b, model, cfg, grid))))
        for b in (0, 1, 2)
    }
    export_diag["closure_row_rms_unweighted"] = closure_row_rms
    export_diag["zero_unobserved_rms_unweighted"] = {
        f"observer{b}": float(jnp.sqrt(mse(zero_unobserved_residual(params, p_export, b, model, cfg, grid))))
        for b in (0, 1, 2)
    }
    export_diag["zero_birth_rms_unweighted"] = {
        f"observer{b}": float(jnp.sqrt(mse(zero_birth_residual(params, p_export, b, model, cfg, grid))))
        for b in (0, 1, 2)
    }
    export_diag["policy_projection_rms_unweighted"] = {
        "player1": float(jnp.sqrt(mse(model["calD1"] - D1_proj))),
        "player2": float(jnp.sqrt(mse(model["calD2"] - D2_proj))),
    }
    export_diag["representative_gap_rms"] = {
        "player1": float(jnp.sqrt(mse(model["D1"] - model["calD1"]))),
        "player2": float(jnp.sqrt(mse(model["D2"] - model["calD2"]))),
    }
    export_diag["mode_norms"] = {
        "aggregate_Dtot_rms": float(jnp.sqrt(mse(Dtot))),
        "relative_D1_minus_D2_rms": float(jnp.sqrt(mse(Drel))),
        "player1_share_gap_rms": float(jnp.sqrt(mse(model["calD1"] - Dmean))),
        "player2_share_gap_rms": float(jnp.sqrt(mse(model["calD2"] - Dmean))),
    }
    export_diag["trader_information_order"] = {
        "higher_gamma": higher_gamma,
        "violation_count": int(np.sum(info_violation > 1e-6)),
        "violation_rms": float(np.sqrt(np.mean(np.square(info_violation)))),
        "avg_abs_value_error_player1": float(np.mean(value_err1)),
        "avg_abs_value_error_player2": float(np.mean(value_err2)),
    }
    export_diag["back_like_gap_rms"] = float(jnp.sqrt(mse(model["dtot_tilde0"] - Dtot)))
    export_diag["market_maker_Dtot_projection_rms"] = float(jnp.sqrt(mse(Dtot_proj0)))
    export_diag["zero_trace_rms"] = {
        "D1": float(jnp.sqrt(mse(A0["D1"]))),
        "D2": float(jnp.sqrt(mse(A0["D2"]))),
        "calD1": float(jnp.sqrt(mse(A0["calD1"]))),
        "calD2": float(jnp.sqrt(mse(A0["calD2"]))),
    }
    export_diag["zero_lag_unresolved_rows"] = {
        "dtot_tilde0": as_json_array(A0["dtot_tilde0"][0]),
        "dtot_tilde1": as_json_array(A0["dtot_tilde1"][0]),
        "dtot_tilde2": as_json_array(A0["dtot_tilde2"][0]),
        "vtilde0": as_json_array(A0["vtilde0"][0]),
        "vtilde1": as_json_array(A0["vtilde1"][0]),
        "vtilde2": as_json_array(A0["vtilde2"][0]),
    }
    target_model = model if grid.shape[0] == cfg["grid"].shape[0] else model_at(params, p_export, cfg["grid"], cfg)
    export_diag["policy_target_rms"] = {
        "player1": float(jnp.sqrt(mse(target_model["calD1"] - targets["target1"]))),
        "player2": float(jnp.sqrt(mse(target_model["calD2"] - targets["target2"]))),
    }
    export_diag["foc_mismatch_rms_unweighted"] = {
        "player1": float(jnp.sqrt(mse(targets["foc_mismatch1"]))),
        "player2": float(jnp.sqrt(mse(targets["foc_mismatch2"]))),
    }

    fixed_parameter = {
        "gamma1": float(par["gamma1"]),
        "gamma2": float(par["gamma2"]),
        "rho": float(par["rho"]),
    }
    private_signal_text = (
        "literal chapter signal gamma_j * (V - P), implemented as gamma_j * vtilde0 / sigma_Yj"
        if cfg["private_signal_gauge"] == "literal"
        else "public-adjusted signal with drift gamma_j * V; filtration-equivalent because P is public"
    )
    kind = (
        "kyle_back_fixed_policy_filter_pinn_template"
        if cfg["filter_only"]
        else "kyle_back_stationary_param_pinn_template"
    )
    description = (
        "Fixed-exponential-policy Kyle-Back filter PINN template, exported at the displayed fixed parameter."
        if cfg["filter_only"]
        else "Parameter-conditioned stationary Kyle-Back PINN template, exported at the displayed fixed parameter."
    )
    payload = {
        "version": 1,
        "kind": kind,
        "description": description,
        "assumptions": {
            "scope": "coupled stationary residual solve on truncated lag window",
            "equations": "clean stationary Kyle-Back template using observer-specific unresolved value and aggregate-demand kernels",
            "primitive_coordinates": ["fundamental", "order_flow_noise", "private_signal_1", "private_signal_2"],
            "normalization": "no exogenous lambda target; Lambda is computed endogenously and is not a filter input",
            "fixed_policy": bool(cfg["fixed_policy"]),
            "filter_only": bool(cfg["filter_only"]),
        },
        "mathematical_conventions": {
            "private_signal": private_signal_text,
            "filter_unresolved_private_row": "trader j unresolved row is gamma_j * vtilde_j / sigma_Yj",
            "policy_zero_trace": "D_i(0)=calD_i(0)=0 enforced by a lag-zero predictability ramp in eval_a",
            "filter_zero_lag_boundary": (
                "zero-lag indirect history integrals are zero; direct measurement uses Pi=E^T E over observed "
                "Z/Y rows only, never V. Unresolved drift rows may still have zero-lag V or Dtot error terms."
            ),
            "policy_foc_object": (
                "not active in filter-only training; exported FOC fields are diagnostics only"
                if cfg["filter_only"]
                else "FOC residual uses primitive projected demand calD_i, not the unprojected representative D_i"
            ),
            "fixed_policy_strategy": (
                "D_i are preset exponential noise-state demand kernels and calD_i are projected through each trader filter"
                if cfg["fixed_policy"]
                else "D_i/calD_i are learned jointly"
            ),
            "filter_loss_collocation": (
                f"filter-only training uses {cfg['filter_loss_points']} jittered stratified lag points, "
                f"{cfg['filter_quad_loss_points']} old-history quadrature points, "
                f"{cfg['filter_inner_loss_points']} projection points, and "
                f"{cfg['filter_kernel_loss_points']} characteristic-kernel points per step; "
                "validation and export still use deterministic grids"
                if cfg["random_filter_grid"] and cfg["filter_only"]
                else "filter residuals are evaluated on the fixed training grid"
            ),
            "lambda": "computed endogenously from the market-maker unresolved order-flow row; not an input to either network",
            "filter_projector_residual_active": False,
            "oscillation_penalty": (
                "active squared second-lag-derivative roughness penalty on unresolved filter kernels; "
                "it discourages high-frequency wiggles but does not impose monotonic IRFs"
            ),
            "separate_response_network": "none; displayed value readouts are V - vtilde^observer with F-projection diagnostics",
            "jax_x64": bool(jax.config.read("jax_enable_x64")),
        },
        "nets": tree_to_jsonable(params),
        "architecture": {
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": "tanh",
            "net_roles": {
                "a": "one-time policy, primitive demand, unresolved value, and unresolved aggregate-demand kernels",
                "h": "adjoint/readout kernels for future-profit effects",
            },
            "n_params": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))),
        },
        "input": {
            "param_names": ["gamma1", "gamma2", "rho"],
            "coordinate_scaling": {"ell": "2*ell/L-1"},
            "ranges": cfg["ranges"],
            "fixed_parameter_training": bool(args.fixed_params),
            "fixed_parameter": fixed_parameter if cfg["fixed_param"] is not None else None,
            "displayed_parameter": fixed_parameter,
        },
        "output": {
            "a_fields": A_FIELDS,
            "h_fields": H_FIELDS,
            "private_signal_gauge": cfg["private_signal_gauge"],
            "lambda": "endogenous FOC/readout scale, not a trained input parameter",
        },
        "lag": np.asarray(cfg["grid"]).astype(float).tolist(),
        "grid": as_json_array(grid),
        "params": {
            "N": cfg["N"],
            "export_N": int(grid.shape[0]),
            "L": cfg["L"],
            "gamma1": float(par["gamma1"]),
            "gamma2": float(par["gamma2"]),
            "rho": float(par["rho"]),
            "std_v": cfg["std_v"],
            "std_z": cfg["std_z"],
            "std_y1": cfg["std_y1"],
            "std_y2": cfg["std_y2"],
            "quadrature": args.quadrature,
            "integral_quad": int(np.asarray(cfg["quad_grid"]).shape[0]),
            "inner_quad": int(np.asarray(cfg["inner_grid"]).shape[0]),
            "ode_n": int(np.asarray(cfg["ode_grid"]).shape[0]),
            "filter_quad": int(np.asarray(cfg["filter_nodes"]).shape[0]),
            "fixed_policy": bool(cfg["fixed_policy"]),
            "filter_only": bool(cfg["filter_only"]),
            "random_filter_grid": bool(cfg["random_filter_grid"]),
            "random_filter_integrals": bool(cfg["random_filter_integrals"]),
            "random_filter_tail": bool(cfg["random_filter_tail"]),
            "filter_loss_points": int(cfg["filter_loss_points"]),
            "filter_quad_loss_points": int(cfg["filter_quad_loss_points"]),
            "filter_inner_loss_points": int(cfg["filter_inner_loss_points"]),
            "filter_kernel_loss_points": int(cfg["filter_kernel_loss_points"]),
            "filter_tail_points": int(cfg["filter_tail_points"]),
            "filter_tail_fraction": float(cfg["filter_tail_fraction"]),
            "w_oscillation": float(cfg["w_oscillation"]),
            "fixed_policy_decay": cfg["fixed_policy_decay"],
            "fixed_d1_v": cfg["fixed_d1_v"],
            "fixed_d1_z": cfg["fixed_d1_z"],
            "fixed_d1_y": cfg["fixed_d1_y"],
            "fixed_d2_v": cfg["fixed_d2_v"],
            "fixed_d2_z": cfg["fixed_d2_z"],
            "fixed_d2_y": cfg["fixed_d2_y"],
        },
        "derived_params": {
            "inner_quad": int(np.asarray(cfg["inner_grid"]).shape[0]),
            "training_N": int(cfg["N"]),
            "export_N": int(grid.shape[0]),
            "displayed_parameter": fixed_parameter,
        },
        "fields": {
            "policy_noise_state": ["D1", "D2"],
            "primitive_demand": ["calD1", "calD2"],
            "noise_state_irfs": "lag x primitive_shock x observer_estimate_coordinate",
            "foc_terms": ["estimated_mispricing_demand", "estimated_calH", "foc_lhs", "foc_mismatch"],
            "filter_tilde": ["vtilde0", "vtilde1", "vtilde2", "dtot_tilde0", "dtot_tilde1", "dtot_tilde2"],
            "response_readouts": ["price", "trader1_estimate", "trader2_estimate"],
            "filter_observation_rows": {
                "market_maker": ["dtot_tilde0 / std_z"],
                "trader1": ["dtot_tilde1 / std_z", "gamma1 * vtilde1 / std_y1"],
                "trader2": ["dtot_tilde2 / std_z", "gamma2 * vtilde2 / std_y2"],
            },
            "raw_private_signal_rows": ["gamma1 * vtilde0 / std_y1", "gamma2 * vtilde0 / std_y2"],
            "ctilde": ["ce0", "ce1_z", "ce1_y", "ce2_z", "ce2_y"],
            "H_blocks": H_FIELDS,
        },
        "policy_noise_state": {
            "player1": as_json_array(model["D1"]),
            "player2": as_json_array(model["D2"]),
        },
        "primitive_demand": {
            "player1": as_json_array(model["calD1"]),
            "player2": as_json_array(model["calD2"]),
        },
        "calD": {
            "player1": as_json_array(model["calD1"]),
            "player2": as_json_array(model["calD2"]),
        },
        "mode_decomposition": {
            "Dtot": as_json_array(Dtot),
            "Drel": as_json_array(Drel),
            "Dmean": as_json_array(Dmean),
            "player1_share_gap": as_json_array(model["calD1"] - Dmean),
            "player2_share_gap": as_json_array(model["calD2"] - Dmean),
        },
        "noise_state_irfs": noise_state_irfs,
        "value_irfs": {name: as_json_array(value) for name, value in val.items()},
        "foc_terms": {
            "estimated_mispricing_demand": {
                "player1": as_json_array(targets["mispricing_demand1"]),
                "player2": as_json_array(targets["mispricing_demand2"]),
            },
            "estimated_calH": {
                "player1": as_json_array(targets["calH1"]),
                "player2": as_json_array(targets["calH2"]),
            },
            "foc_lhs": {
                "player1": as_json_array(targets["foc_lhs1"]),
                "player2": as_json_array(targets["foc_lhs2"]),
            },
            "foc_mismatch": {
                "player1": as_json_array(targets["foc_mismatch1"]),
                "player2": as_json_array(targets["foc_mismatch2"]),
            },
            "raw_future_profit_effect": {
                "player1": as_json_array(targets["ro1"]["density_total"]),
                "player2": as_json_array(targets["ro2"]["density_total"]),
            },
            "readout_components": {
                "player1": {
                    "old_market": as_json_array(targets["ro1"]["old_market"]),
                    "old_opponent": as_json_array(targets["ro1"]["old_opponent"]),
                    "diag_market": as_json_array(targets["ro1"]["diag_market"]),
                    "diag_opponent": as_json_array(targets["ro1"]["diag_opponent"]),
                    "R": float(targets["ro1"]["R"]),
                },
                "player2": {
                    "old_market": as_json_array(targets["ro2"]["old_market"]),
                    "old_opponent": as_json_array(targets["ro2"]["old_opponent"]),
                    "diag_market": as_json_array(targets["ro2"]["diag_market"]),
                    "diag_opponent": as_json_array(targets["ro2"]["diag_opponent"]),
                    "R": float(targets["ro2"]["R"]),
                },
            },
        },
        "filter_closure_residuals": filter_residuals,
        "vtilde_projection_residuals": vtilde_residuals,
        "zero_filter_residuals": zero_filter_residuals,
        "filter_tilde": {
            name: as_json_array(model[name])
            for name in ["vtilde0", "vtilde1", "vtilde2", "dtot_tilde0", "dtot_tilde1", "dtot_tilde2"]
        },
        "unresolved_value": {
            "market_maker": as_json_array(model["vtilde0"]),
            "trader1": as_json_array(model["vtilde1"]),
            "trader2": as_json_array(model["vtilde2"]),
        },
        "unresolved_total_demand": {
            "market_maker": as_json_array(model["dtot_tilde0"]),
            "trader1": as_json_array(model["dtot_tilde1"]),
            "trader2": as_json_array(model["dtot_tilde2"]),
        },
        "ctilde": {
            "ce0": as_json_array(ce["ce0"][:, 0, :]),
            "ce1_z": as_json_array(ce["ce1_z"]),
            "ce1_y": as_json_array(ce["ce1_y"]),
            "ce2_z": as_json_array(ce["ce2_z"]),
            "ce2_y": as_json_array(ce["ce2_y"]),
        },
        "model": {
            "primitive_V": as_json_array(model["primitive_V"]),
            "c0": as_json_array(model["c0"]),
            "cY1": as_json_array(model["cY1"]),
            "cY2": as_json_array(model["cY2"]),
        },
        "H": {name: as_json_array(H[name]) for name in H_FIELDS},
        "diagnostics": export_diag,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def make_cfg(args):
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    L = float(args.L)
    n = int(args.N)
    grid = np.linspace(0.0, L, n, dtype=np.float64)
    h = L / max(n - 1, 1)
    if args.quadrature == "simpson":
        weights = simpson_weights(n, h)
    else:
        weights = np.full(n, h, dtype=np.float32)
        weights[0] *= 0.5
        weights[-1] *= 0.5

    integral_quad = args.integral_quad if args.integral_quad > 0 else max(n, 32)
    if integral_quad % 2 == 1:
        integral_quad += 1
    q_nodes, q_weights = np.polynomial.legendre.leggauss(integral_quad)
    quad_grid = 0.5 * L * (q_nodes + 1.0)
    quad_weights = 0.5 * L * q_weights

    inner_quad = integral_quad + 2
    if inner_quad % 2 == 1:
        inner_quad += 1
    i_nodes, i_weights = np.polynomial.legendre.leggauss(inner_quad)
    inner_grid = 0.5 * L * (i_nodes + 1.0)
    inner_weights = 0.5 * L * i_weights

    if args.ode_grid == "chebyshev":
        theta = np.linspace(0.0, math.pi, args.ode_n if args.ode_n > 0 else n, dtype=np.float64)
        ode_grid = 0.5 * L * (1.0 - np.cos(theta))
    else:
        ode_grid = np.linspace(0.0, L, args.ode_n if args.ode_n > 0 else n, dtype=np.float64)

    f_nodes, f_weights = np.polynomial.legendre.leggauss(args.filter_quad)
    dtype = np.float64 if args.x64 else np.float32
    return {
        "N": n,
        "L": L,
        "export_N": int(args.export_N),
        "grid": jnp.asarray(grid.astype(dtype)),
        "weights": jnp.asarray(weights.astype(dtype)),
        "quad_grid": jnp.asarray(quad_grid.astype(dtype)),
        "quad_weights": jnp.asarray(quad_weights.astype(dtype)),
        "inner_grid": jnp.asarray(inner_grid.astype(dtype)),
        "inner_weights": jnp.asarray(inner_weights.astype(dtype)),
        "ode_grid": jnp.asarray(ode_grid.astype(dtype)),
        "filter_nodes": jnp.asarray(f_nodes.astype(dtype)),
        "filter_weights": jnp.asarray(f_weights.astype(dtype)),
        "std_v": float(args.std_v),
        "std_z": float(args.std_z),
        "std_y1": float(args.std_y1),
        "std_y2": float(args.std_y2),
        "filter_decay": float(args.filter_decay),
        "demand_decay": float(args.demand_decay),
        "diag_ramp": float(args.diag_ramp),
        "tail_decay": float(args.tail_decay),
        "output_scale_a": float(args.output_scale_a),
        "output_scale_h": float(args.output_scale_h),
        "private_signal_gauge": args.private_signal_gauge,
        "filter_only": bool(args.filter_only),
        "fixed_policy": bool(args.fixed_policy),
        "random_filter_grid": bool(args.random_filter_grid),
        "random_filter_integrals": bool(args.random_filter_integrals),
        "random_filter_tail": bool(args.random_filter_tail),
        "filter_loss_points": int(args.filter_loss_points if args.filter_loss_points > 0 else n),
        "filter_quad_loss_points": int(args.filter_quad_loss_points if args.filter_quad_loss_points > 0 else integral_quad),
        "filter_inner_loss_points": int(args.filter_inner_loss_points if args.filter_inner_loss_points > 0 else inner_quad),
        "filter_kernel_loss_points": int(args.filter_kernel_loss_points if args.filter_kernel_loss_points > 0 else args.filter_quad),
        "filter_tail_points": max(int(args.filter_tail_points), 1),
        "filter_tail_fraction": min(max(float(args.filter_tail_fraction), 1e-6), 1.0),
        "metric_names": FILTER_ONLY_METRIC_NAMES if args.filter_only else METRIC_NAMES,
        "fixed_policy_decay": float(args.fixed_policy_decay),
        "fixed_policy_ramp": float(args.fixed_policy_ramp),
        "fixed_d1_v": float(args.fixed_d1_v),
        "fixed_d1_z": float(args.fixed_d1_z),
        "fixed_d1_y": float(args.fixed_d1_y),
        "fixed_d2_v": float(args.fixed_d2_v),
        "fixed_d2_z": float(args.fixed_d2_z),
        "fixed_d2_y": float(args.fixed_d2_y),
        "ranges": {
            "gamma1_min": float(args.gamma1_min),
            "gamma1_max": float(args.gamma1_max),
            "gamma2_min": float(args.gamma2_min),
            "gamma2_max": float(args.gamma2_max),
            "rho_min": float(args.rho_min),
            "rho_max": float(args.rho_max),
        },
        "w_filter": float(args.w_filter),
        "w_policy_proj": float(args.w_policy_proj),
        "w_adjoint": float(args.w_adjoint),
        "w_foc": float(args.w_foc),
        "w_policy_target": float(args.w_policy_target),
        "w_tail": float(args.w_tail),
        "w_oscillation": float(args.w_oscillation),
        "w_reg": float(args.w_reg),
        "fixed_param": None
        if not args.fixed_params
        else jnp.asarray(
            [args.gamma1, args.gamma2, args.rho],
            dtype=jnp.float64 if args.x64 else jnp.float32,
        ),
    }


def init_params(args):
    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key, 2)
    return {
        "a": init_mlp(k1, [4] + [args.hidden] * args.depth + [len(A_FIELDS) * D_W], args.last_scale),
        "h": init_mlp(k2, [4] + [args.hidden] * args.depth + [len(H_FIELDS) * D_W * D_W], args.last_scale),
    }


def load_params(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    nets = payload.get("nets")
    if not isinstance(nets, dict):
        raise ValueError(f"{path} does not contain exported PINN nets")

    def legacy_lambda_encoding():
        params_payload = payload.get("params") or {}
        fixed_payload = payload.get("input", {}).get("fixed_parameter") or {}
        displayed_payload = payload.get("input", {}).get("displayed_parameter") or {}
        ranges_payload = payload.get("input", {}).get("ranges") or {}
        lam = (
            params_payload.get("lambda_target")
            or fixed_payload.get("lambda_target")
            or displayed_payload.get("lambda_target")
            or 0.4
        )
        lo = float(ranges_payload.get("lambda_target_min", 0.15))
        hi = float(ranges_payload.get("lambda_target_max", 0.8))
        center = 0.5 * (math.log(lo) + math.log(hi))
        scale = 0.5 * (math.log(hi) - math.log(lo))
        if abs(scale) < 1e-12:
            return 0.0
        return (math.log(max(float(lam), 1e-8)) - center) / scale

    legacy_lam_enc = legacy_lambda_encoding()

    def load_layers(layers):
        out = []
        for layer_idx, layer in enumerate(layers):
            W = jnp.asarray(layer["W"], dtype=jnp.float32)
            b = jnp.asarray(layer["b"], dtype=jnp.float32)
            if layer_idx == 0 and W.shape[0] == 5:
                # Backward compatibility with checkpoints that used
                # (gamma1,gamma2,rho,lambda_target,ell) as network inputs.
                # Fold the saved lambda input into the bias, then drop that
                # column so lambda is no longer a live conditioning variable.
                b = b + jnp.asarray(legacy_lam_enc, dtype=jnp.float32) * W[3]
                W = jnp.concatenate([W[:3], W[4:5]], axis=0)
            if layer_idx == 0 and W.shape[0] != 4:
                raise ValueError(f"expected first layer input dimension 4 after lambda removal, got {W.shape[0]}")
            out.append({"W": W, "b": b})
        return out

    return {name: load_layers(layers) for name, layers in nets.items()}


def parse_args():
    ap = argparse.ArgumentParser(description="Parameter-conditioned stationary Kyle-Back PINN template.")
    ap.add_argument("--out", default="data/kyle_back_stationary_param_pinn_template.json")
    ap.add_argument("--checkpoint", default="data/kyle_back_stationary_param_pinn_template_checkpoint.json")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--val-batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-decay", type=float, default=0.65)
    ap.add_argument("--weight-decay", type=float, default=1e-7)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--init", default="")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--N", type=int, default=41)
    ap.add_argument("--export-N", type=int, default=0,
                    help="Number of lag points to export for plotting; 0 uses the training grid.")
    ap.add_argument("--ode-n", type=int, default=41)
    ap.add_argument("--ode-grid", choices=["uniform", "chebyshev"], default="chebyshev")
    ap.add_argument("--integral-quad", type=int, default=0)
    ap.add_argument("--filter-quad", type=int, default=13)
    ap.add_argument("--quadrature", choices=["trapezoid", "simpson"], default="simpson")
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--std-v", type=float, default=1.0)
    ap.add_argument("--std-z", type=float, default=1.0)
    ap.add_argument("--std-y1", type=float, default=1.0)
    ap.add_argument("--std-y2", type=float, default=1.0)
    ap.add_argument("--gamma1-min", type=float, default=0.5)
    ap.add_argument("--gamma1-max", type=float, default=10.0)
    ap.add_argument("--gamma2-min", type=float, default=0.5)
    ap.add_argument("--gamma2-max", type=float, default=10.0)
    ap.add_argument("--rho-min", type=float, default=0.03)
    ap.add_argument("--rho-max", type=float, default=0.5)
    ap.add_argument("--fixed-params", action="store_true",
                    help="Train only one parameter tuple instead of sampling from ranges.")
    ap.add_argument("--gamma1", type=float, default=3.0)
    ap.add_argument("--gamma2", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--lambda-target-min", type=float, default=0.15, help=argparse.SUPPRESS)
    ap.add_argument("--lambda-target-max", type=float, default=0.8, help=argparse.SUPPRESS)
    ap.add_argument("--lambda-target", type=float, default=0.4, help=argparse.SUPPRESS)
    ap.add_argument("--private-signal-gauge", choices=["literal", "public_adjusted"], default="literal")
    ap.add_argument("--filter-only", action="store_true",
                    help="Train only the filtering residual system; adjoints and FOC become export diagnostics.")
    ap.add_argument("--fixed-policy", action="store_true",
                    help="Use preset exponential demand strategies D_i and project them through the filters.")
    ap.add_argument("--random-filter-grid", action=argparse.BooleanOptionalAction, default=True,
                    help="Use jittered stratified lag collocation points for filter-only training.")
    ap.add_argument("--random-filter-integrals", action=argparse.BooleanOptionalAction, default=True,
                    help="Use jittered stratified quadrature for filter-only closure/projection/kernel integrals.")
    ap.add_argument("--random-filter-tail", action=argparse.BooleanOptionalAction, default=True,
                    help="Use jittered tail-band points instead of only L for filter-only tail regularization.")
    ap.add_argument("--filter-loss-points", type=int, default=0,
                    help="Number of jittered lag collocation points per filter-only step; 0 uses N.")
    ap.add_argument("--filter-quad-loss-points", type=int, default=0,
                    help="Number of jittered old-history quadrature points per filter-only step; 0 uses integral-quad.")
    ap.add_argument("--filter-inner-loss-points", type=int, default=0,
                    help="Number of jittered projection quadrature points per filter-only step; 0 uses inner quad.")
    ap.add_argument("--filter-kernel-loss-points", type=int, default=0,
                    help="Number of jittered characteristic-kernel quadrature points per filter-only step; 0 uses filter-quad.")
    ap.add_argument("--filter-tail-points", type=int, default=4,
                    help="Number of jittered tail-band regularization points.")
    ap.add_argument("--filter-tail-fraction", type=float, default=0.2,
                    help="Fraction of the lag window used for randomized tail regularization.")
    ap.add_argument("--fixed-policy-decay", type=float, default=0.95)
    ap.add_argument("--fixed-policy-ramp", type=float, default=4.0)
    ap.add_argument("--fixed-d1-v", type=float, default=0.65)
    ap.add_argument("--fixed-d1-z", type=float, default=-0.10)
    ap.add_argument("--fixed-d1-y", type=float, default=0.30)
    ap.add_argument("--fixed-d2-v", type=float, default=0.55)
    ap.add_argument("--fixed-d2-z", type=float, default=-0.08)
    ap.add_argument("--fixed-d2-y", type=float, default=0.28)
    ap.add_argument("--filter-decay", type=float, default=0.8)
    ap.add_argument("--demand-decay", type=float, default=0.8)
    ap.add_argument("--diag-ramp", type=float, default=4.0)
    ap.add_argument("--tail-decay", type=float, default=0.05)
    ap.add_argument("--output-scale-a", type=float, default=0.2)
    ap.add_argument("--output-scale-h", type=float, default=0.2)
    ap.add_argument("--last-scale", type=float, default=1e-3)
    ap.add_argument("--w-filter", type=float, default=1.0)
    ap.add_argument("--w-policy-proj", type=float, default=0.5)
    ap.add_argument("--w-adjoint", type=float, default=1.0)
    ap.add_argument("--w-foc", type=float, default=1.0)
    ap.add_argument("--w-policy-target", type=float, default=0.1)
    ap.add_argument("--w-lambda", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--w-tail", type=float, default=0.05)
    ap.add_argument("--w-oscillation", type=float, default=1e-4,
                    help="Weight on squared second-lag-derivative roughness; discourages grid-scale oscillations without imposing monotonicity.")
    ap.add_argument("--w-reg", type=float, default=1e-6)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=2000)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.filter_only and not args.fixed_policy:
        args.fixed_policy = True
    cfg = make_cfg(args)
    metric_names = cfg["metric_names"]
    params = load_params(args.init) if args.init else init_params(args)
    n_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))
    print(
        f"Kyle-Back parameter PINN params={n_params} N={args.N} batch={args.batch} "
        f"gauge={args.private_signal_gauge} filter_only={args.filter_only} fixed_policy={args.fixed_policy}",
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
    rng = jax.random.PRNGKey(args.seed + 1)
    val_key = jax.random.PRNGKey(args.seed + 2)
    val_batch = sample_params(val_key, args.val_batch, cfg["ranges"], cfg["fixed_param"])

    def loss_fn(params, p_batch, train_grids):
        metrics = batch_metrics(params, p_batch, cfg, train_grids)
        return jnp.sum(metrics), metrics

    @jax.jit
    def train_step(params, opt_state, rng):
        rng, sub_p, sub_grid = jax.random.split(rng, 3)
        p_batch = sample_params(sub_p, args.batch, cfg["ranges"], cfg["fixed_param"])
        train_grids = random_filter_training_grids(sub_grid, cfg) if cfg["filter_only"] else None
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, p_batch, train_grids)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, rng, loss, metrics, optax.global_norm(grads)

    @jax.jit
    def validate(params):
        metrics = batch_metrics(params, val_batch, cfg, None)
        return jnp.sum(metrics), metrics

    if args.steps == 0:
        val_loss, val_metrics = validate(params)
        jax.block_until_ready(val_loss)
        diag = diagnostics(
            params,
            cfg,
            args,
            0,
            0.0,
            val_loss,
            val_metrics,
            jnp.array(0.0, dtype=jnp.float32),
            val_loss,
            val_metrics,
        )
        diag["best_val_rms"] = diag.get("val_rms")
        diag["best"] = None
        save_json(args.out, params, args, cfg, diag)
        print(json.dumps(diag, indent=2, sort_keys=True), flush=True)
        print(f"wrote {args.out}", flush=True)
        return 0

    t0 = time.perf_counter()
    best_val = float("inf")
    best_payload = None
    last_diag = {}

    # Trigger compilation.
    params, opt_state, rng, loss, metrics, grad_norm = train_step(params, opt_state, rng)
    jax.block_until_ready(loss)

    for step in range(1, args.steps + 1):
        if step > 1:
            params, opt_state, rng, loss, metrics, grad_norm = train_step(params, opt_state, rng)
        if step == 1 or step % args.log_every == 0:
            val_loss, val_metrics = validate(params)
            jax.block_until_ready(val_loss)
            elapsed = time.perf_counter() - t0
            train_rms = float(jnp.sqrt(loss / len(metric_names)))
            val_rms = float(jnp.sqrt(val_loss / len(metric_names)))
            diag = diagnostics(params, cfg, args, step, elapsed, loss, metrics, grad_norm, val_loss, val_metrics)
            last_diag = diag
            if val_rms < best_val:
                best_val = val_rms
                best_payload = diag.copy()
            worst = sorted(diag["val_block_rms"].items(), key=lambda kv: kv[1], reverse=True)[:4]
            print(
                f"step={step} train_rms={train_rms:.4e} val_rms={val_rms:.4e} "
                f"grad={float(grad_norm):.3e} elapsed={elapsed:.1f}s "
                + " ".join(f"{k}={v:.2e}" for k, v in worst),
                flush=True,
            )
        if step % args.save_every == 0:
            payload_diag = dict(last_diag)
            payload_diag["best_val_rms"] = best_val
            payload_diag["best"] = best_payload
            save_json(args.checkpoint, params, args, cfg, payload_diag)
            save_json(args.out, params, args, cfg, payload_diag)

    val_loss, val_metrics = validate(params)
    jax.block_until_ready(val_loss)
    final_diag = diagnostics(params, cfg, args, args.steps, time.perf_counter() - t0, loss, metrics, grad_norm, val_loss, val_metrics)
    final_diag["best_val_rms"] = best_val
    final_diag["best"] = best_payload
    save_json(args.out, params, args, cfg, final_diag)
    print(json.dumps(final_diag, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
