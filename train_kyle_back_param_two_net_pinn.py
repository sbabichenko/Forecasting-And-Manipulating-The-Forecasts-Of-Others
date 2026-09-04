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

The script intentionally uses two neural networks:

    filter net:  vtilde0,1,2 and dtot_tilde0,1,2;
    policy net:  D1,D2,calD1,calD2 plus the adjoint blocks H.

The filter net can be trained separately from the policy/adjoint net with
``--train-mode filter`` and ``--train-mode policy``.

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

POLICY_FIELDS = ["D1", "D2", "calD1", "calD2"]
FILTER_FIELDS = [
    "vtilde0",
    "vtilde1",
    "vtilde2",
    "dtot_tilde0",
    "dtot_tilde1",
    "dtot_tilde2",
]
A_FIELDS = POLICY_FIELDS + FILTER_FIELDS
H_FIELDS = ["H01", "H21", "H02", "H12"]
POLICY_OUT_DIM = len(POLICY_FIELDS) * D_W
FILTER_OUT_DIM = len(FILTER_FIELDS) * D_W
H_OUT_DIM = len(H_FIELDS) * D_W * D_W

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
    "regularization",
]

FILTER_METRIC_NAMES = {
    "filter0",
    "filter1",
    "filter2",
    "tail",
    "regularization",
}
POLICY_METRIC_NAMES = {
    "policy_projection",
    "H01",
    "H21",
    "H02",
    "H12",
    "foc",
    "policy_target_stabilizer",
    "tail",
    "regularization",
}


def metric_mask(train_mode: str):
    if train_mode == "joint":
        keep = set(METRIC_NAMES)
    elif train_mode == "filter":
        keep = FILTER_METRIC_NAMES
    elif train_mode == "policy":
        keep = POLICY_METRIC_NAMES
    else:
        raise ValueError(f"unknown train_mode {train_mode}")
    return jnp.asarray([1.0 if name in keep else 0.0 for name in METRIC_NAMES], dtype=jnp.float32)


def freeze_grads_for_mode(grads, train_mode: str):
    if train_mode == "joint":
        return grads
    if train_mode == "filter":
        return {
            "filter": grads["filter"],
            "policy": jax.tree_util.tree_map(jnp.zeros_like, grads["policy"]),
        }
    if train_mode == "policy":
        return {
            "filter": jax.tree_util.tree_map(jnp.zeros_like, grads["filter"]),
            "policy": grads["policy"],
        }
    raise ValueError(f"unknown train_mode {train_mode}")


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


def baseline_policy(p, ell, cfg):
    ell = jnp.asarray(ell, dtype=jnp.float32)
    zero = jnp.zeros(ell.shape + (D_W,), dtype=jnp.float32)
    return {"D1": zero, "D2": zero, "calD1": zero, "calD2": zero}


def baseline_filter(p, ell, cfg):
    ell = jnp.asarray(ell, dtype=jnp.float32)
    par = unpack_p(p)
    tail_v = jnp.exp(-cfg["filter_decay"] * ell)[..., None]
    zero = jnp.zeros(ell.shape + (D_W,), dtype=jnp.float32)
    v0 = cfg["std_v"] * tail_v * EV
    # Private signals make trader filters faster when gamma is large.
    v1 = cfg["std_v"] * jnp.exp(-(cfg["filter_decay"] + 0.1 * par["gamma1"]) * ell)[..., None] * EV
    v2 = cfg["std_v"] * jnp.exp(-(cfg["filter_decay"] + 0.1 * par["gamma2"]) * ell)[..., None] * EV
    return {
        "vtilde0": v0,
        "vtilde1": v1,
        "vtilde2": v2,
        "dtot_tilde0": zero,
        "dtot_tilde1": zero,
        "dtot_tilde2": zero,
    }


def eval_filter_fields(params, p, ell, cfg):
    p_enc = encode_params(p, cfg["ranges"])
    ell = jnp.asarray(ell, dtype=jnp.float32)
    raw = mlp(params["filter"], concat_inputs(p_enc, coord_lag(ell, cfg))).reshape(ell.shape + (FILTER_OUT_DIM,))
    tail = jnp.exp(-cfg["tail_decay"] * ell)[..., None]
    raw_fields = {name: raw[..., i * D_W : (i + 1) * D_W] for i, name in enumerate(FILTER_FIELDS)}
    base = baseline_filter(p, ell, cfg)
    out = {name: base[name] + cfg["output_scale_filter"] * tail * raw_fields[name] for name in FILTER_FIELDS}

    # Demand residuals are old-history densities, hence zero at the diagonal.
    trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell))[..., None]
    for name in ["dtot_tilde0", "dtot_tilde1", "dtot_tilde2"]:
        out[name] = trace * out[name]

    # Value residual at age zero is the primitive value shock because value noise
    # is orthogonal to order-flow and private-signal noise.
    zero = jnp.isclose(ell, 0.0, atol=1e-7)[..., None]
    v_zero = cfg["std_v"] * EV
    for name in ["vtilde0", "vtilde1", "vtilde2"]:
        out[name] = jnp.where(zero, v_zero, out[name])
    return out


def eval_policy_and_H_fields(params, p, ell, cfg):
    p_enc = encode_params(p, cfg["ranges"])
    ell = jnp.asarray(ell, dtype=jnp.float32)
    raw = mlp(params["policy"], concat_inputs(p_enc, coord_lag(ell, cfg))).reshape(
        ell.shape + (POLICY_OUT_DIM + H_OUT_DIM,)
    )
    raw_policy = raw[..., :POLICY_OUT_DIM]
    raw_H = raw[..., POLICY_OUT_DIM:]
    tail = jnp.exp(-cfg["tail_decay"] * ell)[..., None]
    raw_policy_fields = {name: raw_policy[..., i * D_W : (i + 1) * D_W] for i, name in enumerate(POLICY_FIELDS)}
    base = baseline_policy(p, ell, cfg)
    policy = {name: base[name] + cfg["output_scale_policy"] * tail * raw_policy_fields[name] for name in POLICY_FIELDS}

    # Predictable zero trace for the policy representative and primitive demand.
    trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell))[..., None]
    for name in POLICY_FIELDS:
        policy[name] = trace * policy[name]
    H = cfg["output_scale_h"] * tail * raw_H
    return policy, H


def eval_a(params, p, ell, cfg):
    policy, _ = eval_policy_and_H_fields(params, p, ell, cfg)
    filt = eval_filter_fields(params, p, ell, cfg)
    merged = {**policy, **filt}
    return jnp.concatenate([merged[name] for name in A_FIELDS], axis=-1)


def eval_H(params, p, ell, cfg):
    _, H = eval_policy_and_H_fields(params, p, ell, cfg)
    return H


def value_and_deriv_a(params, p, ell, cfg):
    return jax.jvp(lambda ee: eval_a(params, p, ee, cfg), (ell,), (jnp.ones_like(ell),))


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


def model_at(params, p, ell_eval, cfg):
    par = unpack_p(p)
    A = split_a(eval_a(params, p, ell_eval, cfg))
    Dtot = A["calD1"] + A["calD2"]
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


def filter_matrix(params, p, observer: int, first_lag, second_lag, cfg):
    """LQG-PINN-style characteristic filter matrix F^observer(a,b).

    This mirrors ``train_stationary_param_pinn.py::filter_matrix``: the
    observed birth/direct trace is explicit, and the old-history term is the
    min-lag integral of unresolved observation rows.  We keep one smooth
    stationary representation rather than separate positive/negative side H
    networks.
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

    tau = 0.5 * width[..., None] * (cfg["filter_nodes"] + 1.0)
    quad_w = 0.5 * width[..., None] * cfg["filter_weights"]
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


# Backward-compatible name for older helper calls.
filter_kernel_pairs = filter_matrix


def split_interval_nodes(ell_eval, cfg):
    ell_eval = jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    nodes = cfg["inner_nodes"]
    weights = cfg["inner_base_weights"]
    left_width = ell_eval[..., None]
    right_width = cfg["L"] - ell_eval[..., None]
    left = 0.5 * left_width * (nodes + 1.0)
    right = ell_eval[..., None] + 0.5 * right_width * (nodes + 1.0)
    left_w = 0.5 * left_width * weights
    right_w = 0.5 * right_width * weights
    return left, left_w, right, right_w


def split_project_integral(params, p, observer: int, ell_eval, kernel_fn, cfg):
    left, left_w, right, right_w = split_interval_nodes(ell_eval, cfg)
    second_left = jnp.broadcast_to(ell_eval[..., None], left.shape)
    second_right = jnp.broadcast_to(ell_eval[..., None], right.shape)
    k_left = kernel_fn(left)
    k_right = kernel_fn(right)
    f_left = filter_matrix(params, p, observer, left, second_left, cfg)
    f_right = filter_matrix(params, p, observer, right, second_right, cfg)
    return (
        jnp.einsum("mq,mqd,mqde->me", left_w, k_left, f_left)
        + jnp.einsum("mq,mqd,mqde->me", right_w, k_right, f_right)
    )


def split_closure_integral(params, p, observer: int, ell_eval, c_rows_fn, cfg):
    left, left_w, right, right_w = split_interval_nodes(ell_eval, cfg)
    second_left = jnp.broadcast_to(ell_eval[..., None], left.shape)
    second_right = jnp.broadcast_to(ell_eval[..., None], right.shape)
    c_left = c_rows_fn(left)
    c_right = c_rows_fn(right)
    f_left = filter_matrix(params, p, observer, left, second_left, cfg)
    f_right = filter_matrix(params, p, observer, right, second_right, cfg)
    return (
        jnp.einsum("mq,mqrd,mqde->mre", left_w, c_left, f_left)
        + jnp.einsum("mq,mqrd,mqde->mre", right_w, c_right, f_right)
    )


def project_from_history(params, p, observer: int, kernel_eval, kernel_history, history_grid, history_weights, ell_eval, cfg):
    rows = observer_rows(observer)
    direct = jnp.einsum("nd,rd,re->ne", kernel_eval, rows, rows)
    first = jnp.broadcast_to(history_grid[:, None], (history_grid.shape[0], ell_eval.shape[0]))
    second = jnp.broadcast_to(ell_eval[None, :], first.shape)
    f = filter_matrix(params, p, observer, first, second, cfg)
    indirect = jnp.einsum("a,ad,alde->le", history_weights, kernel_history, f)
    indirect = jnp.where((ell_eval > 1e-7)[..., None], indirect, jnp.zeros_like(indirect))
    return direct + indirect


def project_from_interior_history(params, p, observer: int, kernel_eval, kernel_fn, ell_eval, cfg):
    history_grid = cfg["quad_grid"]
    history_weights = cfg["quad_weights"]
    return project_from_history(
        params,
        p,
        observer,
        kernel_eval,
        kernel_fn(history_grid),
        history_grid,
        history_weights,
        ell_eval,
        cfg,
    )


def project_at(params, p, observer: int, kernel_eval, kernel_grid, ell_eval, cfg, kernel_fn=None):
    if kernel_fn is None:
        return project_from_history(
            params, p, observer, kernel_eval, kernel_grid, cfg["inner_grid"], cfg["inner_weights"], ell_eval, cfg
        )
    rows = observer_rows(observer)
    direct = jnp.einsum("nd,rd,re->ne", kernel_eval, rows, rows)
    if cfg["split_dense_projection"]:
        # Split the history integral at the moving birth boundary u = ell.
        # Otherwise a fixed quadrature grid aliases the side-trace kink into
        # high-frequency oscillations in exported noise-state IRFs.
        indirect = split_project_integral(params, p, observer, ell_eval, kernel_fn, cfg)
    else:
        kernel_history = kernel_fn(cfg["inner_grid"])
        return project_from_history(
            params, p, observer, kernel_eval, kernel_history, cfg["inner_grid"], cfg["inner_weights"], ell_eval, cfg
        )
    # The lag-zero observed innovation is an atom.  Do not let old-history
    # density terms leak into the instantaneous primitive-shock boundary.
    indirect = jnp.where((ell_eval > 1e-7)[..., None], indirect, jnp.zeros_like(indirect))
    return direct + indirect


def closure_residual(params, p, observer: int, model_grid, cfg, ell_eval=None):
    grid = cfg["grid"] if ell_eval is None else jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    c_rows = c_rows_for_observer(model_grid, observer)
    ce_rows = ctilde_rows(params, p, observer, grid, cfg)
    pi = observer_pi(observer)
    direct = jnp.einsum("lrd,de->lre", c_rows, EYE - pi)
    if ell_eval is None:
        # Old-history density integral.  Use endpoint-free quadrature so the
        # birth/side trace at lag zero is not counted as ordinary density mass.
        hist = cfg["quad_grid"]
        hist_w = cfg["quad_weights"]
        c_hist = c_rows_for_observer(model_at(params, p, hist, cfg), observer)
        first = jnp.broadcast_to(hist[:, None], (hist.shape[0], grid.shape[0]))
        second = jnp.broadcast_to(grid[None, :], first.shape)
        f = filter_matrix(params, p, observer, first, second, cfg)
        integral = jnp.einsum("a,ard,alde->lre", hist_w, c_hist, f)
    else:
        integral = split_closure_integral(
            params,
            p,
            observer,
            grid,
            lambda ee: c_rows_for_observer(model_at(params, p, ee, cfg), observer),
            cfg,
        )
    return ce_rows - (direct - integral)


def vtilde_projection_residual(params, p, observer: int, cfg, ell_eval=None):
    grid = cfg["grid"] if ell_eval is None else jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    A = split_a(eval_a(params, p, grid, cfg))
    vt = A[f"vtilde{observer}"]
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (grid.shape[0], D_W))
    if ell_eval is None:
        v_proj = project_from_interior_history(
            params,
            p,
            observer,
            v_eval,
            lambda ee: jnp.broadcast_to(cfg["std_v"] * EV, ee.shape + (D_W,)),
            grid,
            cfg,
        )
    else:
        v_inner = jnp.broadcast_to(cfg["std_v"] * EV, (cfg["inner_grid"].shape[0], D_W))
        v_proj = project_at(
            params,
            p,
            observer,
            v_eval,
            v_inner,
            grid,
            cfg,
            lambda ee: jnp.broadcast_to(cfg["std_v"] * EV, ee.shape + (D_W,)),
        )
    return vt - (v_eval - v_proj)


def filter_residual(params, p, observer: int, model_grid, cfg, ell_eval=None):
    return jnp.concatenate(
        [
            closure_residual(params, p, observer, model_grid, cfg, ell_eval).reshape(-1),
            vtilde_projection_residual(params, p, observer, cfg, ell_eval).reshape(-1),
        ]
    )


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
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    eye = jnp.eye(D_W, dtype=jnp.float32)
    mats = []
    for ch in range(D_W):
        kernel_eval = jnp.broadcast_to(eye[ch], (ell_eval.shape[0], D_W))
        kernel_grid = jnp.broadcast_to(eye[ch], (cfg["inner_grid"].shape[0], D_W))
        mats.append(
            project_at(
                params,
                p,
                observer,
                kernel_eval,
                kernel_grid,
                ell_eval,
                cfg,
                lambda ee, ch=ch: jnp.broadcast_to(eye[ch], ee.shape + (D_W,)),
            )
        )
    return jnp.stack(mats, axis=1)


def value_irfs(params, p, ell_eval, cfg):
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    A = split_a(eval_a(params, p, ell_eval, cfg))
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (ell_eval.shape[0], D_W))
    v_grid = jnp.broadcast_to(cfg["std_v"] * EV, (cfg["inner_grid"].shape[0], D_W))
    price = v_eval - A["vtilde0"]
    trader1 = v_eval - A["vtilde1"]
    trader2 = v_eval - A["vtilde2"]
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
        "price_from_F_projection": project_at(
            params,
            p,
            0,
            v_eval,
            v_grid,
            ell_eval,
            cfg,
            lambda ee: jnp.broadcast_to(cfg["std_v"] * EV, ee.shape + (D_W,)),
        ),
        "trader1_estimate_from_F_projection": project_at(
            params,
            p,
            1,
            v_eval,
            v_grid,
            ell_eval,
            cfg,
            lambda ee: jnp.broadcast_to(cfg["std_v"] * EV, ee.shape + (D_W,)),
        ),
        "trader2_estimate_from_F_projection": project_at(
            params,
            p,
            2,
            v_eval,
            v_grid,
            ell_eval,
            cfg,
            lambda ee: jnp.broadcast_to(cfg["std_v"] * EV, ee.shape + (D_W,)),
        ),
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


def readout_for_deviator(params, p, H_name0, H_name_opp, opp: int, alphas, cfg, ell_eval=None):
    par = unpack_p(p)
    grid_in = cfg["grid"] if ell_eval is None else jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    grid_shape = grid_in.shape
    grid = grid_in.reshape(-1)
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

    def restore_density(x):
        return x.reshape(grid_shape + (D_W,))

    return {
        "density_total": restore_density(density0 + density_opp),
        "density_total_quad": density0_q + density_opp_q,
        "density_total_inner": density0_i + density_opp_i,
        "total": old0 + old_opp + diag0 + diag_opp,
        "old_market": old0,
        "old_opponent": old_opp,
        "diag_market": diag0,
        "diag_opponent": diag_opp,
        "R": R,
    }


def equilibrium_targets(params, p, cfg, ell_eval=None):
    grid = cfg["grid"] if ell_eval is None else jnp.asarray(ell_eval, dtype=cfg["grid"].dtype)
    alphas = alpha_terms(params, p, cfg)
    ro1 = readout_for_deviator(params, p, "H01", "H21", 2, alphas, cfg, grid)
    ro2 = readout_for_deviator(params, p, "H02", "H12", 1, alphas, cfg, grid)
    lam = lambda_value(params, p, cfg)
    inv_lam = 1.0 / jnp.where(jnp.abs(lam) > 1e-6, lam, jnp.inf)
    ce0 = eval_ctilde(params, p, grid, cfg)["ce0"][:, 0, :]
    model = model_at(params, p, grid, cfg)

    local_info = cfg["std_z"] * ce0
    if ell_eval is None:
        local_info_fn = lambda ee: cfg["std_z"] * eval_ctilde(params, p, ee, cfg)["ce0"][..., 0, :]
        ro1_density_fn = lambda ee: readout_for_deviator(
            params, p, "H01", "H21", 2, alphas, cfg, ee
        )["density_total"]
        ro2_density_fn = lambda ee: readout_for_deviator(
            params, p, "H02", "H12", 1, alphas, cfg, ee
        )["density_total"]
        mispricing1 = project_from_interior_history(params, p, 1, local_info, local_info_fn, grid, cfg)
        mispricing2 = project_from_interior_history(params, p, 2, local_info, local_info_fn, grid, cfg)
        calH1 = project_from_interior_history(params, p, 1, ro1["density_total"], ro1_density_fn, grid, cfg)
        calH2 = project_from_interior_history(params, p, 2, ro2["density_total"], ro2_density_fn, grid, cfg)
    else:
        ce0_i = eval_ctilde(params, p, cfg["inner_grid"], cfg)["ce0"][:, 0, :]
        local_info_i = cfg["std_z"] * ce0_i
        local_info_fn = lambda ee: cfg["std_z"] * eval_ctilde(params, p, ee, cfg)["ce0"][..., 0, :]
        mispricing1 = project_at(params, p, 1, local_info, local_info_i, grid, cfg, local_info_fn)
        mispricing2 = project_at(params, p, 2, local_info, local_info_i, grid, cfg, local_info_fn)

        ro1_density_fn = lambda ee: readout_for_deviator(params, p, "H01", "H21", 2, alphas, cfg, ee)["density_total"]
        ro2_density_fn = lambda ee: readout_for_deviator(params, p, "H02", "H12", 1, alphas, cfg, ee)["density_total"]
        calH1 = project_at(params, p, 1, ro1["density_total"], ro1["density_total_inner"], grid, cfg, ro1_density_fn)
        calH2 = project_at(params, p, 2, ro2["density_total"], ro2["density_total_inner"], grid, cfg, ro2_density_fn)

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


def single_residual_metrics(params, p, cfg):
    grid = cfg["grid"]
    model = model_at(params, p, grid, cfg)
    A = split_a(eval_a(params, p, grid, cfg))
    D1_proj = project_from_interior_history(
        params,
        p,
        1,
        A["D1"],
        lambda ee: split_a(eval_a(params, p, ee, cfg))["D1"],
        grid,
        cfg,
    )
    D2_proj = project_from_interior_history(
        params,
        p,
        2,
        A["D2"],
        lambda ee: split_a(eval_a(params, p, ee, cfg))["D2"],
        grid,
        cfg,
    )

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
            cfg["w_reg"]
            * (
                mse(eval_a(params, p, grid, cfg))
                + mse(eval_H(params, p, grid, cfg))
            ),
        ]
    )


def batch_metrics(params, p_batch, cfg):
    return jnp.mean(jax.vmap(lambda pp: single_residual_metrics(params, pp, cfg))(p_batch), axis=0)


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
    mm_proj_Dtot = project_from_interior_history(
        params,
        p,
        0,
        Dtot,
        lambda ee: model_at(params, p, ee, cfg)["Dtot"],
        grid,
        cfg,
    )
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


def diagnostics(params, cfg, args, step, elapsed, loss, metrics, grad_norm, val_loss=None, val_metrics=None):
    block_rms = {name: float(math.sqrt(max(v, 0.0))) for name, v in zip(METRIC_NAMES, np.asarray(metrics))}
    active_count = max(float(np.asarray(cfg["metric_mask"]).sum()), 1.0)
    center_p = cfg["fixed_param"]
    if center_p is None:
        center_p = jnp.array(
            [
                math.sqrt(args.gamma1_min * args.gamma1_max),
                math.sqrt(args.gamma2_min * args.gamma2_max),
                math.sqrt(args.rho_min * args.rho_max),
            ],
            dtype=cfg["grid"].dtype,
        )
    out = {
        "step": int(step),
        "elapsed_s": float(elapsed),
        "loss": float(loss),
        "train_rms": float(math.sqrt(max(float(loss) / active_count, 0.0))),
        "grad_norm": float(grad_norm),
        "block_rms": block_rms,
        "n_params": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))),
        "random_parameter_training": not bool(args.fixed_params),
        "fixed_parameter_training": bool(args.fixed_params),
        "fixed_parameter": {
            "gamma1": float(center_p[0]),
            "gamma2": float(center_p[1]),
            "rho": float(center_p[2]),
        },
        "private_signal_gauge": cfg["private_signal_gauge"],
        "lambda_is_parameter": False,
        "representative_selection": "calD_i is projected primitive demand; D_i is learned noise-state representative",
        "sample_diagnostics_center": diagnostics_for_p(params, center_p, cfg),
    }
    if val_loss is not None and val_metrics is not None:
        out["val_loss"] = float(val_loss)
        out["val_rms"] = float(math.sqrt(max(float(val_loss) / active_count, 0.0)))
        out["val_block_rms"] = {name: float(math.sqrt(max(v, 0.0))) for name, v in zip(METRIC_NAMES, np.asarray(val_metrics))}
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
        dtype=cfg["grid"].dtype,
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
    targets = equilibrium_targets(params, p_export, cfg, grid)
    val = value_irfs(params, p_export, grid, cfg)
    Dtot = model["calD1"] + model["calD2"]
    Drel = model["calD1"] - model["calD2"]
    Dmean = 0.5 * Dtot
    Dtot_proj0 = project_at(
        params,
        p_export,
        0,
        Dtot,
        model_inner["Dtot"],
        grid,
        cfg,
        lambda ee: model_at(params, p_export, ee, cfg)["Dtot"],
    )
    D1_proj = project_at(
        params,
        p_export,
        1,
        model["D1"],
        A_inner["D1"],
        grid,
        cfg,
        lambda ee: split_a(eval_a(params, p_export, ee, cfg))["D1"],
    )
    D2_proj = project_at(
        params,
        p_export,
        2,
        model["D2"],
        A_inner["D2"],
        grid,
        cfg,
        lambda ee: split_a(eval_a(params, p_export, ee, cfg))["D2"],
    )
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
    zero = jnp.array([0.0], dtype=cfg["grid"].dtype)
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
        "dtot_tilde0": float(jnp.sqrt(mse(A0["dtot_tilde0"]))),
        "dtot_tilde1": float(jnp.sqrt(mse(A0["dtot_tilde1"]))),
        "dtot_tilde2": float(jnp.sqrt(mse(A0["dtot_tilde2"]))),
        "ce0": float(jnp.sqrt(mse(ce0["ce0"]))),
        "ce1_z": float(jnp.sqrt(mse(ce0["ce1_z"]))),
        "ce2_z": float(jnp.sqrt(mse(ce0["ce2_z"]))),
    }
    export_diag["policy_target_rms"] = {
        "player1": float(jnp.sqrt(mse(model["calD1"] - targets["target1"]))),
        "player2": float(jnp.sqrt(mse(model["calD2"] - targets["target2"]))),
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
    filter_only = cfg["train_mode"] == "filter"
    payload = {
        "version": 1,
        "kind": "kyle_back_stationary_param_two_net_pinn",
        "description": "Two-network parameter-conditioned stationary Kyle-Back PINN: filter net for unresolved residual kernels, policy net for policy and adjoint kernels.",
        "assumptions": {
            "scope": "coupled stationary residual solve on truncated lag window",
            "equations": "observer-specific unresolved value and aggregate-demand kernels, with separate filter and policy networks",
            "primitive_coordinates": ["fundamental", "order_flow_noise", "private_signal_1", "private_signal_2"],
            "normalization": "no exogenous lambda target; Lambda is computed endogenously and is not a filter input",
            "fixed_policy": False,
            "filter_only": bool(filter_only),
        },
        "mathematical_conventions": {
            "private_signal": private_signal_text,
            "filter_unresolved_private_row": "trader j unresolved row is gamma_j * vtilde_j / sigma_Yj",
            "policy_zero_trace": "D_i(0)=calD_i(0)=0 enforced by a lag-zero predictability ramp",
            "policy_foc_object": "FOC residual uses primitive projected demand calD_i, not the unprojected representative D_i",
            "lambda": "computed endogenously from the market-maker unresolved order-flow row; not an input to either network",
            "separate_networks": "filter net learns vtilde and dtot_tilde; policy net learns D, calD, and H",
            "filter_projection": "training uses an LQG-PINN-style filter_matrix(a,b); old-history density integrals use endpoint-free Gauss quadrature so birth traces are not counted as density mass; dense exports split the projection at u=ell to avoid quadrature aliasing",
            "separate_positive_negative_side_eval": False,
            "jax_x64": bool(jax.config.read("jax_enable_x64")),
        },
        "nets": tree_to_jsonable(params),
        "architecture": {
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": "tanh",
            "net_roles": {
                "filter": "observer-specific unresolved value and aggregate-demand kernels",
                "policy": "noise-state policy kernels, primitive demand kernels, and adjoint/readout kernels",
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
            "policy_fields": POLICY_FIELDS,
            "filter_fields": FILTER_FIELDS,
            "h_fields": H_FIELDS,
            "net_roles": {"filter": "vtilde and dtot_tilde", "policy": "D, calD, and H adjoints"},
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
            "tail_decay": cfg["tail_decay"],
            "filter_decay": cfg["filter_decay"],
            "diag_ramp": cfg["diag_ramp"],
            "quadrature": args.quadrature,
            "integral_quad": int(np.asarray(cfg["quad_grid"]).shape[0]),
            "inner_quad": int(np.asarray(cfg["inner_grid"]).shape[0]),
            "ode_n": int(np.asarray(cfg["ode_grid"]).shape[0]),
            "filter_quad": int(np.asarray(cfg["filter_nodes"]).shape[0]),
            "train_mode": cfg["train_mode"],
            "split_dense_projection": bool(cfg["split_dense_projection"]),
            "fixed_policy": False,
            "filter_only": bool(filter_only),
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
            "filter_tilde": FILTER_FIELDS,
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
    fixed_param = None
    if args.fixed_params:
        fixed_param = jnp.asarray(
            [args.gamma1, args.gamma2, args.rho],
            dtype=dtype,
        )
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
        "inner_nodes": jnp.asarray(i_nodes.astype(dtype)),
        "inner_base_weights": jnp.asarray(i_weights.astype(dtype)),
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
        "output_scale_policy": float(args.output_scale_policy),
        "output_scale_filter": float(args.output_scale_filter),
        "output_scale_h": float(args.output_scale_h),
        "private_signal_gauge": args.private_signal_gauge,
        "split_dense_projection": not bool(args.no_split_dense_projection),
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
        "w_reg": float(args.w_reg),
        "train_mode": args.train_mode,
        "metric_mask": metric_mask(args.train_mode),
        "fixed_param": fixed_param,
    }


def init_params(args):
    key = jax.random.PRNGKey(args.seed)
    k_filter, k_policy = jax.random.split(key, 2)
    return {
        "filter": init_mlp(k_filter, [4] + [args.hidden] * args.depth + [FILTER_OUT_DIM], args.last_scale),
        "policy": init_mlp(k_policy, [4] + [args.hidden] * args.depth + [POLICY_OUT_DIM + H_OUT_DIM], args.last_scale),
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

    loaded = {name: load_layers(layers) for name, layers in nets.items()}
    if "filter" in loaded and "policy" in loaded:
        return {"filter": loaded["filter"], "policy": loaded["policy"]}
    raise ValueError(
        "checkpoint must contain separate 'filter' and 'policy' nets; "
        "use this two-stage script's own checkpoints to warm-start"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Two-network parameter-conditioned stationary Kyle-Back PINN.")
    ap.add_argument("--out", default="data/kyle_back_stationary_param_two_net_pinn.json")
    ap.add_argument("--checkpoint", default="data/kyle_back_stationary_param_two_net_pinn_checkpoint.json")
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
    ap.add_argument("--export-N", type=int, default=400, help="dense lag grid length exported for browser plots; <=0 uses training grid")
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
    ap.add_argument("--fixed-params", action="store_true", help="train and export one displayed parameter tuple instead of random parameter draws")
    ap.add_argument("--gamma1", type=float, default=3.0)
    ap.add_argument("--gamma2", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--lambda-target-min", type=float, default=0.15, help=argparse.SUPPRESS)
    ap.add_argument("--lambda-target-max", type=float, default=0.8, help=argparse.SUPPRESS)
    ap.add_argument("--lambda-target", type=float, default=0.4, help=argparse.SUPPRESS)
    ap.add_argument("--private-signal-gauge", choices=["literal", "public_adjusted"], default="literal")
    ap.add_argument(
        "--no-split-dense-projection",
        action="store_true",
        help="disable the dense-export split at u=ell; training still uses the LQG-style filter_matrix path",
    )
    ap.add_argument("--train-mode", choices=["filter", "policy", "joint"], default="joint",
                    help="filter: update only unresolved filter net; policy: update only policy/adjoint net; joint: update both")
    ap.add_argument("--filter-decay", type=float, default=0.8)
    ap.add_argument("--demand-decay", type=float, default=0.8)
    ap.add_argument("--diag-ramp", type=float, default=4.0)
    ap.add_argument("--tail-decay", type=float, default=0.05)
    ap.add_argument("--output-scale-policy", type=float, default=0.2)
    ap.add_argument("--output-scale-filter", type=float, default=0.2)
    ap.add_argument("--output-scale-h", type=float, default=0.2)
    ap.add_argument("--last-scale", type=float, default=1e-3)
    ap.add_argument("--w-filter", type=float, default=1.0)
    ap.add_argument("--w-policy-proj", type=float, default=0.5)
    ap.add_argument("--w-adjoint", type=float, default=1.0)
    ap.add_argument("--w-foc", type=float, default=1.0)
    ap.add_argument("--w-policy-target", type=float, default=0.1)
    ap.add_argument("--w-lambda", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--w-tail", type=float, default=0.05)
    ap.add_argument("--w-reg", type=float, default=1e-6)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=2000)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = make_cfg(args)
    params = load_params(args.init) if args.init else init_params(args)
    n_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))
    print(
        f"Kyle-Back two-net PINN params={n_params} N={args.N} batch={args.batch} "
        f"gauge={args.private_signal_gauge} train_mode={args.train_mode}",
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

    def loss_fn(params, p_batch):
        metrics = batch_metrics(params, p_batch, cfg)
        return jnp.sum(metrics * cfg["metric_mask"]), metrics

    @jax.jit
    def train_step(params, opt_state, rng):
        rng, sub = jax.random.split(rng)
        p_batch = sample_params(sub, args.batch, cfg["ranges"], cfg["fixed_param"])
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, p_batch)
        grads = freeze_grads_for_mode(grads, cfg["train_mode"])
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, rng, loss, metrics, optax.global_norm(grads)

    @jax.jit
    def validate(params):
        metrics = batch_metrics(params, val_batch, cfg)
        return jnp.sum(metrics * cfg["metric_mask"]), metrics

    t0 = time.perf_counter()
    if args.steps == 0:
        val_loss, val_metrics = validate(params)
        jax.block_until_ready(val_loss)
        final_diag = diagnostics(
            params,
            cfg,
            args,
            0,
            time.perf_counter() - t0,
            val_loss,
            val_metrics,
            0.0,
            val_loss,
            val_metrics,
        )
        save_json(args.out, params, args, cfg, final_diag)
        print(json.dumps(final_diag, indent=2, sort_keys=True), flush=True)
        print(f"wrote {args.out}", flush=True)
        return 0

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
            active_count = jnp.maximum(jnp.sum(cfg["metric_mask"]), 1.0)
            train_rms = float(jnp.sqrt(loss / active_count))
            val_rms = float(jnp.sqrt(val_loss / active_count))
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
