#!/usr/bin/env python3
"""Train a direct parameter-conditioned PINN for stationary equations.

This is different from train_stationary_param_nn.py. It does not pre-solve
parameter tuples and regress on cached targets. Instead, it samples parameter
tuples inside the training loop and minimizes the stationary residuals directly.

The browser still only needs exported dense-layer weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Iterable, List, Tuple

# NVIDIA 580.142 reports a two-part kernel driver version. Current XLA logs
# that harmless parse failure as an error even though CUDA execution works.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import optax


D_W = 3
E0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
E1 = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
E2 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
PI1 = jnp.diag(jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
PI2 = jnp.diag(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
EYE3 = jnp.eye(3, dtype=jnp.float32)

A_FIELDS = ["x", "d1", "d2", "calD1", "calD2", "xtilde1", "xtilde2"]
B_FIELDS = ["hx1", "hx2", "wedge1", "wedge2"]
AB_FIELDS = ["H1", "H2"]
METRIC_NAMES = [
    "state",
    "filter1_xtilde",
    "filter1_control",
    "filter1_zero_xtilde",
    "filter1_zero_control",
    "filter2_xtilde",
    "filter2_control",
    "filter2_zero_xtilde",
    "filter2_zero_control",
    "hx1",
    "hx2",
    "H1",
    "H2",
    "H1_interface",
    "H2_interface",
    "wedge1",
    "wedge2",
    "policy1",
    "policy2",
    "x0",
    "control1_zero_unobserved",
    "control2_zero_unobserved",
    "x_tail",
    "hx1_right",
    "hx2_right",
    "H1_a_tail",
    "H2_a_tail",
    "H1_b_right",
    "H2_b_right",
    "left_tail",
]


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
        W = scale * jax.random.normal(k, (din, dout), dtype=jnp.float32)
        b = jnp.zeros((dout,), dtype=jnp.float32)
        params.append({"W": W, "b": b})
    return params


def mlp(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def encode_params(p, ranges):
    mins = jnp.log(jnp.array([ranges["p_min"], ranges["p_min"], ranges["r_min"], ranges["r_min"]], dtype=jnp.float32))
    maxs = jnp.log(jnp.array([ranges["p_max"], ranges["p_max"], ranges["r_max"], ranges["r_max"]], dtype=jnp.float32))
    center = 0.5 * (mins + maxs)
    scale = 0.5 * (maxs - mins)
    return (jnp.log(jnp.maximum(p, 1e-8)) - center) / scale


def concat_inputs(p_enc, coord):
    coord = jnp.asarray(coord, dtype=jnp.float32)
    flat = coord.reshape((-1, coord.shape[-1]))
    p_tiled = jnp.broadcast_to(p_enc, (flat.shape[0], p_enc.shape[0]))
    return jnp.concatenate([p_tiled, flat], axis=1)


def split_a(y):
    return {
        "x": y[..., 0:3],
        "d1": y[..., 3:6],
        "d2": y[..., 6:9],
        "calD1": y[..., 9:12],
        "calD2": y[..., 12:15],
        "xtilde1": y[..., 15:18],
        "xtilde2": y[..., 18:21],
    }


def split_b(y):
    return {
        "hx1": y[..., 0:3],
        "hx2": y[..., 3:6],
        "wedge1": y[..., 6:9],
        "wedge2": y[..., 9:12],
    }


def split_ab(y):
    return {
        "H1": y[..., 0:9].reshape(y.shape[:-1] + (3, 3)),
        "H2": y[..., 9:18].reshape(y.shape[:-1] + (3, 3)),
    }


def baseline_a(p, a, sigma):
    r1 = jnp.maximum(p[2], 1e-6)
    r2 = jnp.maximum(p[3], 1e-6)
    inv_sum = 1.0 / r1 + 1.0 / r2
    s = jnp.sqrt(1.0 / jnp.maximum(inv_sum, 1e-8))
    decay = inv_sum * s
    x = sigma * jnp.exp(-decay * a)[..., None] * E0
    d1 = -(s / r1) * x
    d2 = -(s / r2) * x
    calD1 = d1 @ PI1.T
    calD2 = d2 @ PI2.T
    xt1 = x - x @ PI1.T
    xt2 = x - x @ PI2.T
    return jnp.concatenate([x, d1, d2, calD1, calD2, xt1, xt2], axis=-1)


def baseline_b(p, b, sigma):
    r1 = jnp.maximum(p[2], 1e-6)
    r2 = jnp.maximum(p[3], 1e-6)
    inv_sum = 1.0 / r1 + 1.0 / r2
    s = jnp.sqrt(1.0 / jnp.maximum(inv_sum, 1e-8))
    decay = inv_sum * s
    b_pos = jnp.maximum(b, 0.0)
    mask = (b >= 0.0)[..., None].astype(jnp.float32)
    x_pos = sigma * jnp.exp(-decay * b_pos)[..., None] * E0
    hx = mask * s * x_pos
    zero = jnp.zeros_like(hx)
    return jnp.concatenate([hx, hx, zero, zero], axis=-1)


def eval_a(params, p, a, cfg):
    p_enc = encode_params(p, cfg["ranges"])
    a = jnp.asarray(a, dtype=jnp.float32)
    coord = (2.0 * a / cfg["L"] - 1.0)[..., None]
    corr = cfg["output_scale_a"] * mlp(params["a"], concat_inputs(p_enc, coord)).reshape(a.shape + (21,))
    return baseline_a(p, a, cfg["sigma"]) + corr


def eval_b(params, p, b, cfg):
    p_enc = encode_params(p, cfg["ranges"])
    b = jnp.asarray(b, dtype=jnp.float32)
    coord = (b / cfg["L"])[..., None]
    corr = cfg["output_scale_b"] * mlp(params["b"], concat_inputs(p_enc, coord)).reshape(b.shape + (12,))
    return baseline_b(p, b, cfg["sigma"]) + corr


def eval_ab(params, p, a, b, cfg):
    return eval_ab_auto(params, p, a, b, cfg)


def eval_ab_side(params, p, a, b, cfg, side: str):
    p_enc = encode_params(p, cfg["ranges"])
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    coord = jnp.stack([2.0 * a / cfg["L"] - 1.0, b / cfg["L"]], axis=-1)
    if "ab_neg" in params and "ab_pos" in params:
        net_name = "ab_neg" if side == "neg" else "ab_pos"
    else:
        net_name = "ab"
    return cfg["output_scale_ab"] * mlp(params[net_name], concat_inputs(p_enc, coord)).reshape(a.shape + (18,))


def eval_ab_auto(params, p, a, b, cfg):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    if "ab_neg" not in params or "ab_pos" not in params:
        return eval_ab_side(params, p, a, b, cfg, "pos")
    neg = eval_ab_side(params, p, a, b, cfg, "neg")
    pos = eval_ab_side(params, p, a, b, cfg, "pos")
    return jnp.where((b < 0.0)[..., None], neg, pos)


def value_and_deriv_a(params, p, a, cfg):
    return jax.jvp(lambda aa: eval_a(params, p, aa, cfg), (a,), (jnp.ones_like(a),))


def value_and_deriv_b(params, p, b, cfg):
    return jax.jvp(lambda bb: eval_b(params, p, bb, cfg), (b,), (jnp.ones_like(b),))


def value_and_deriv_ab(params, p, a, b, cfg):
    val, da = jax.jvp(lambda aa: eval_ab(params, p, aa, b, cfg), (a,), (jnp.ones_like(a),))
    _, db = jax.jvp(lambda bb: eval_ab(params, p, a, bb, cfg), (b,), (jnp.ones_like(b),))
    return val, da, db


def value_and_deriv_ab_side(params, p, a, b, cfg, side: str):
    val, da = jax.jvp(lambda aa: eval_ab_side(params, p, aa, b, cfg, side), (a,), (jnp.ones_like(a),))
    _, db = jax.jvp(lambda bb: eval_ab_side(params, p, a, bb, cfg, side), (b,), (jnp.ones_like(b),))
    return val, da, db


def field_a(params, p, points, name, cfg):
    return split_a(eval_a(params, p, points, cfg))[name]


def field_b(params, p, points, name, cfg):
    return split_b(eval_b(params, p, points, cfg))[name]


def field_ab(params, p, a, b, name, cfg):
    return split_ab(eval_ab(params, p, a, b, cfg))[name]


def field_ab_side(params, p, a, b, name, cfg, side: str):
    return split_ab(eval_ab_side(params, p, a, b, cfg, side))[name]


def filter_matrix(params, p, xtilde_name, obs_gain, obs_idx, precision, a_pts, b_pts, cfg):
    e = jnp.eye(3, dtype=jnp.float32)[obs_idx]
    diff = a_pts[:, None] - b_pts[None, :]
    abs_diff = jnp.abs(diff)
    vals = field_a(params, p, abs_diff.reshape(-1), xtilde_name, cfg).reshape(diff.shape + (3,))
    gt = diff > 1e-7
    lt = diff < -1e-7
    direct_gt = vals[..., :, None] * e[None, None, None, :]
    direct_lt = e[None, None, :, None] * vals[..., None, :]
    direct = obs_gain * (
        jnp.where(gt[..., None, None], direct_gt, 0.0)
        + jnp.where(lt[..., None, None], direct_lt, 0.0)
    )

    upper = jnp.maximum(jnp.minimum(a_pts[:, None], b_pts[None, :]), 0.0)
    c_nodes = 0.5 * upper[..., None] * (cfg["inner_nodes"][None, None, :] + 1.0)
    c_weights = 0.5 * upper[..., None] * cfg["inner_weights"][None, None, :]
    left_points = a_pts[:, None, None] - c_nodes
    right_points = b_pts[None, :, None] - c_nodes
    left = field_a(params, p, left_points.reshape(-1), xtilde_name, cfg).reshape(
        c_nodes.shape + (3,)
    )
    right = field_a(params, p, right_points.reshape(-1), xtilde_name, cfg).reshape(
        c_nodes.shape + (3,)
    )
    integral = precision * jnp.einsum("abq,abqr,abqs->abrs", c_weights, left, right)
    return direct + integral


def project_filter(params, p, x, d, xtilde_name, obs_gain, obs_idx, precision, pi, a_pts, weights_a, cfg):
    f = filter_matrix(params, p, xtilde_name, obs_gain, obs_idx, precision, a_pts, a_pts, cfg)
    xhat = x @ pi.T + jnp.einsum("a,abrj,ar->bj", weights_a, f, x)
    chat = d @ pi.T + jnp.einsum("a,abrj,ar->bj", weights_a, f, d)
    zero_mask = jnp.isclose(a_pts, 0.0, atol=1e-7)
    chat = jnp.where(zero_mask[:, None], chat @ pi.T, chat)
    return xhat, x - xhat, chat


def project_filter_at_zero(params, p, x, d, x0, d0, xtilde_name, obs_gain, obs_idx, precision, pi, a_pts, weights_a, cfg):
    zero = jnp.array([0.0], dtype=jnp.float32)
    f = filter_matrix(params, p, xtilde_name, obs_gain, obs_idx, precision, a_pts, zero, cfg)[:, 0]
    xb = pi @ x0 + jnp.einsum("a,arj,ar->j", weights_a, f, x)
    cb = pi @ d0 + jnp.einsum("a,arj,ar->j", weights_a, f, d)
    cb = pi @ cb
    return xb, x0 - xb, cb


def wedge_from_H(params, p, H, xtilde_k, H_name, obs_gain, obs_idx, precision, b_pts, weights_a, cfg):
    e = jnp.eye(3, dtype=jnp.float32)[obs_idx]
    integral = jnp.einsum("a,abrs,ar->bs", weights_a, H, xtilde_k)
    # Use the positive-side trace at b=0 for the innovation-birth term.
    # The H residual itself is collocated one-sided away from the interface.
    H0b = field_ab(params, p, jnp.zeros_like(b_pts), b_pts, H_name, cfg)
    diagonal = obs_gain * jnp.einsum("brs,r->bs", H0b, e)
    return diagonal + precision * integral


def H_side_residual(params, p, H_name, side, d_name, hx_name, wedge_name, cfg):
    a_h = cfg["a_h"]
    b_h = cfg["b_h_neg"] if side == "neg" else cfg["b_h_pos"]
    aa, bb = jnp.meshgrid(a_h, b_h, indexing="ij")
    flat_a = aa.reshape(-1)
    flat_b = bb.reshape(-1)
    yab, yab_da, yab_db = value_and_deriv_ab_side(params, p, flat_a, flat_b, cfg, side)
    H_da = split_ab(yab_da)[H_name].reshape((a_h.size, b_h.size, 3, 3))
    H_db = split_ab(yab_db)[H_name].reshape((a_h.size, b_h.size, 3, 3))
    x_a = field_a(params, p, a_h, "x", cfg)
    d_a = field_a(params, p, a_h, d_name, cfg)
    hx_b = field_b(params, p, b_h, hx_name, cfg)
    wedge_b = field_b(params, p, b_h, wedge_name, cfg)
    return (
        H_da
        + H_db
        + d_a[:, None, :, None] * hx_b[None, :, None, :]
        - x_a[:, None, :, None] * wedge_b[None, :, None, :]
    )


def H_interface_gap(params, p, H_name, cfg):
    a_h = cfg["a_h"]
    zero = jnp.zeros_like(a_h)
    H_neg = field_ab_side(params, p, a_h, zero, H_name, cfg, "neg")
    H_pos = field_ab_side(params, p, a_h, zero, H_name, cfg, "pos")
    return H_neg - H_pos


def mse(x):
    return jnp.mean(jnp.square(x))


def single_residual_metrics(params, p, cfg):
    a_pts = cfg["a"]
    b_pts = cfg["b"]
    weights_a = cfg["weights_a"]
    aa, bb = jnp.meshgrid(a_pts, b_pts, indexing="ij")
    flat_a = aa.reshape(-1)
    flat_b = bb.reshape(-1)

    ya, ya_da = value_and_deriv_a(params, p, a_pts, cfg)
    ya_ode, ya_ode_da = value_and_deriv_a(params, p, cfg["a_ode"], cfg)
    yb, yb_db = value_and_deriv_b(params, p, b_pts, cfg)
    yab = eval_ab(params, p, flat_a, flat_b, cfg)
    A = split_a(ya)
    A_da = split_a(ya_da)
    A_ode = split_a(ya_ode)
    A_ode_da = split_a(ya_ode_da)
    B = split_b(yb)
    B_db = split_b(yb_db)
    AB = {k: v.reshape((a_pts.size, b_pts.size, 3, 3)) for k, v in split_ab(yab).items()}

    p1 = jnp.maximum(p[0], 0.0)
    p2 = jnp.maximum(p[1], 0.0)
    g1 = jnp.sqrt(p1)
    g2 = jnp.sqrt(p2)
    r1 = jnp.maximum(p[2], 1e-6)
    r2 = jnp.maximum(p[3], 1e-6)

    x = A["x"]
    d1 = A["d1"]
    d2 = A["d2"]
    c1 = A["calD1"]
    c2 = A["calD2"]
    xt1 = A["xtilde1"]
    xt2 = A["xtilde2"]
    hx1 = B["hx1"]
    hx2 = B["hx2"]
    w1 = B["wedge1"]
    w2 = B["wedge2"]
    H1 = AB["H1"]
    H2 = AB["H2"]

    x_ext = field_a(params, p, jnp.maximum(b_pts, 0.0), "x", cfg) * (b_pts[:, None] >= 0.0)
    hx1_pos = field_b(params, p, cfg["a_ode"], "hx1", cfg)
    hx2_pos = field_b(params, p, cfg["a_ode"], "hx2", cfg)

    y0 = split_a(eval_a(params, p, jnp.array([0.0], dtype=jnp.float32), cfg))
    ytail = split_a(eval_a(params, p, jnp.array([cfg["L"]], dtype=jnp.float32), cfg))
    b_left = split_b(eval_b(params, p, jnp.array([-cfg["L"]], dtype=jnp.float32), cfg))
    b_right = split_b(eval_b(params, p, jnp.array([cfg["L"]], dtype=jnp.float32), cfg))
    H1_a_tail = field_ab(params, p, jnp.full_like(b_pts, cfg["L"]), b_pts, "H1", cfg)
    H2_a_tail = field_ab(params, p, jnp.full_like(b_pts, cfg["L"]), b_pts, "H2", cfg)
    H1_b_right = field_ab(params, p, a_pts, jnp.full_like(a_pts, cfg["L"]), "H1", cfg)
    H2_b_right = field_ab(params, p, a_pts, jnp.full_like(a_pts, cfg["L"]), "H2", cfg)

    _, xt1_proj, c1_proj = project_filter(params, p, x, d1, "xtilde1", g1, 1, p1, PI1, a_pts, weights_a, cfg)
    _, xt2_proj, c2_proj = project_filter(params, p, x, d2, "xtilde2", g2, 2, p2, PI2, a_pts, weights_a, cfg)
    _, xt1_zero_proj, c1_zero_proj = project_filter_at_zero(
        params, p, x, d1, y0["x"][0], y0["d1"][0], "xtilde1", g1, 1, p1, PI1, a_pts, weights_a, cfg
    )
    _, xt2_zero_proj, c2_zero_proj = project_filter_at_zero(
        params, p, x, d2, y0["x"][0], y0["d2"][0], "xtilde2", g2, 2, p2, PI2, a_pts, weights_a, cfg
    )

    w1_proj = wedge_from_H(params, p, H1, xt2, "H1", g2, 2, p2, b_pts, weights_a, cfg)
    w2_proj = wedge_from_H(params, p, H2, xt1, "H2", g1, 1, p1, b_pts, weights_a, cfg)

    H1_neg_res = H_side_residual(params, p, "H1", "neg", "d2", "hx1", "wedge1", cfg)
    H1_pos_res = H_side_residual(params, p, "H1", "pos", "d2", "hx1", "wedge1", cfg)
    H2_neg_res = H_side_residual(params, p, "H2", "neg", "d1", "hx2", "wedge2", cfg)
    H2_pos_res = H_side_residual(params, p, "H2", "pos", "d1", "hx2", "wedge2", cfg)
    H1_interface = H_interface_gap(params, p, "H1", cfg)
    H2_interface = H_interface_gap(params, p, "H2", cfg)

    blocks = [
        cfg["w_state"] * mse(A_ode_da["x"] - cfg["A"] * A_ode["x"] - A_ode["calD1"] - A_ode["calD2"]),
        cfg["w_filter"] * mse(xt1 - xt1_proj),
        cfg["w_filter"] * mse(c1 - c1_proj),
        cfg["w_filter"] * mse(y0["xtilde1"][0] - xt1_zero_proj),
        cfg["w_filter"] * mse(y0["calD1"][0] - c1_zero_proj),
        cfg["w_filter"] * mse(xt2 - xt2_proj),
        cfg["w_filter"] * mse(c2 - c2_proj),
        cfg["w_filter"] * mse(y0["xtilde2"][0] - xt2_zero_proj),
        cfg["w_filter"] * mse(y0["calD2"][0] - c2_zero_proj),
        cfg["w_hx"] * mse(-B_db["hx1"] - x_ext - cfg["A"] * hx1 - w1),
        cfg["w_hx"] * mse(-B_db["hx2"] - x_ext - cfg["A"] * hx2 - w2),
        cfg["w_H"] * 0.5 * (mse(H1_neg_res) + mse(H1_pos_res)),
        cfg["w_H"] * 0.5 * (mse(H2_neg_res) + mse(H2_pos_res)),
        cfg["w_H_interface"] * mse(H1_interface),
        cfg["w_H_interface"] * mse(H2_interface),
        cfg["w_wedge"] * mse(w1 - w1_proj),
        cfg["w_wedge"] * mse(w2 - w2_proj),
        cfg["w_policy"] * mse(A_ode["d1"] + hx1_pos / r1),
        cfg["w_policy"] * mse(A_ode["d2"] + hx2_pos / r2),
        cfg["w_boundary"] * mse(y0["x"][0] - cfg["sigma"] * E0),
        cfg["w_boundary"] * mse((EYE3 - PI1) @ y0["calD1"][0]),
        cfg["w_boundary"] * mse((EYE3 - PI2) @ y0["calD2"][0]),
        cfg["w_boundary"] * mse(ytail["x"][0]),
        cfg["w_boundary"] * mse(b_right["hx1"][0]),
        cfg["w_boundary"] * mse(b_right["hx2"][0]),
        cfg["w_boundary"] * mse(H1_a_tail),
        cfg["w_boundary"] * mse(H2_a_tail),
        cfg["w_boundary"] * mse(H1_b_right),
        cfg["w_boundary"] * mse(H2_b_right),
        cfg["w_tail_left"] * mse(jnp.stack([b_left["hx1"][0], b_left["hx2"][0]], axis=0)),
    ]
    return jnp.stack(blocks)


def batch_metrics(params, p_batch, cfg):
    metrics = jax.vmap(lambda p: single_residual_metrics(params, p, cfg))(p_batch)
    return jnp.mean(metrics, axis=0)


def sample_params(key, batch_size: int, ranges: Dict[str, float]):
    lo = jnp.log(jnp.array([ranges["p_min"], ranges["p_min"], ranges["r_min"], ranges["r_min"]], dtype=jnp.float32))
    hi = jnp.log(jnp.array([ranges["p_max"], ranges["p_max"], ranges["r_max"], ranges["r_max"]], dtype=jnp.float32))
    u = jax.random.uniform(key, (batch_size, 4), dtype=jnp.float32)
    return jnp.exp(lo + u * (hi - lo))


def make_cfg(args) -> Dict[str, object]:
    L = float(args.L)
    n = int(args.N)
    a = np.linspace(0.0, L, n, dtype=np.float32)
    b = np.linspace(-L, L, 2 * n - 1, dtype=np.float32)
    ode_n = int(args.ode_n) if int(args.ode_n) > 0 else n
    if args.ode_grid == "chebyshev":
        theta = np.linspace(0.0, math.pi, ode_n, dtype=np.float64)
        a_ode = 0.5 * L * (1.0 - np.cos(theta))
    else:
        a_ode = np.linspace(0.0, L, ode_n, dtype=np.float64)
    h_a_n = int(args.h_a_n) if int(args.h_a_n) > 0 else n
    if args.h_grid == "chebyshev":
        theta = np.linspace(0.0, math.pi, h_a_n, dtype=np.float64)
        a_h = 0.5 * L * (1.0 - np.cos(theta))
    else:
        a_h = np.linspace(0.0, L, h_a_n, dtype=np.float64)
    h_b_n = int(args.h_b_n) if int(args.h_b_n) > 0 else n - 1
    h_b_n = max(1, h_b_n)
    if args.h_grid == "chebyshev":
        theta = np.linspace(0.0, math.pi, h_b_n + 1, dtype=np.float64)
        b_pos = 0.5 * L * (1.0 - np.cos(theta))[1:]
    else:
        b_pos = np.linspace(L / h_b_n, L, h_b_n, dtype=np.float64)
    b_neg = -b_pos
    h = L / max(n - 1, 1)
    weights = simpson_weights(n, h) if args.quadrature == "simpson" else np.full(n, h, dtype=np.float32)
    if args.quadrature != "simpson":
        weights[0] *= 0.5
        weights[-1] *= 0.5
    inner_nodes, inner_weights = np.polynomial.legendre.leggauss(args.inner_quad)
    return {
        "L": L,
        "N": n,
        "a": jnp.asarray(a),
        "a_ode": jnp.asarray(a_ode.astype(np.float32)),
        "a_h": jnp.asarray(a_h.astype(np.float32)),
        "b": jnp.asarray(b),
        "b_h_neg": jnp.asarray(b_neg.astype(np.float32)),
        "b_h_pos": jnp.asarray(b_pos.astype(np.float32)),
        "weights_a": jnp.asarray(weights),
        "inner_nodes": jnp.asarray(inner_nodes.astype(np.float32)),
        "inner_weights": jnp.asarray(inner_weights.astype(np.float32)),
        "sigma": float(args.sigma),
        "A": float(args.A),
        "ranges": {
            "p_min": float(args.p_min),
            "p_max": float(args.p_max),
            "r_min": float(args.r_min),
            "r_max": float(args.r_max),
        },
        "output_scale_a": float(args.output_scale_a),
        "output_scale_b": float(args.output_scale_b),
        "output_scale_ab": float(args.output_scale_ab),
        "w_state": float(args.w_state),
        "w_filter": float(args.w_filter),
        "w_hx": float(args.w_hx),
        "w_H": float(args.w_H),
        "w_H_interface": float(args.w_H_interface),
        "w_wedge": float(args.w_wedge),
        "w_policy": float(args.w_policy),
        "w_boundary": float(args.w_boundary),
        "w_tail_left": float(args.w_tail_left),
    }


def init_params(args):
    key = jax.random.PRNGKey(args.seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "a": init_mlp(k1, [5] + [args.hidden] * args.depth + [21], args.last_scale),
        "b": init_mlp(k2, [5] + [args.hidden] * args.depth + [12], args.last_scale),
        "ab_neg": init_mlp(k3, [6] + [args.hidden] * args.depth + [18], args.last_scale),
        "ab_pos": init_mlp(k4, [6] + [args.hidden] * args.depth + [18], args.last_scale),
    }


def load_params(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    nets = payload.get("nets")
    if not isinstance(nets, dict):
        raise ValueError(f"{path} does not contain exported PINN nets")
    params = {
        name: [
            {
                "W": jnp.asarray(layer["W"], dtype=jnp.float32),
                "b": jnp.asarray(layer["b"], dtype=jnp.float32),
            }
            for layer in layers
        ]
        for name, layers in nets.items()
    }
    if "ab" in params and ("ab_neg" not in params or "ab_pos" not in params):
        params["ab_neg"] = [
            {"W": layer["W"] + jnp.zeros_like(layer["W"]), "b": layer["b"] + jnp.zeros_like(layer["b"])}
            for layer in params["ab"]
        ]
        params["ab_pos"] = [
            {"W": layer["W"] + jnp.zeros_like(layer["W"]), "b": layer["b"] + jnp.zeros_like(layer["b"])}
            for layer in params["ab"]
        ]
    return {name: params[name] for name in ["a", "b", "ab_neg", "ab_pos"] if name in params}


def assert_architecture(params, args):
    expected = init_params(args)
    for net_name in ["a", "b", "ab_neg", "ab_pos"]:
        if net_name not in params:
            raise ValueError(f"missing net {net_name} in initial checkpoint")
        if len(params[net_name]) != len(expected[net_name]):
            raise ValueError(f"net {net_name} depth does not match --depth")
        for idx, (got, exp) in enumerate(zip(params[net_name], expected[net_name])):
            if got["W"].shape != exp["W"].shape or got["b"].shape != exp["b"].shape:
                raise ValueError(
                    f"net {net_name} layer {idx} shape mismatch: "
                    f"got W{got['W'].shape}/b{got['b'].shape}, "
                    f"expected W{exp['W'].shape}/b{exp['b'].shape}"
                )


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(x.shape) for x in leaves))


def tree_to_jsonable(params):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32).tolist(), params)


def save_json(path: str, params, args, cfg, diagnostics):
    payload = {
        "version": 1,
        "kind": "stationary_param_pinn",
        "description": "Browser-side MLP weights trained directly on stationary residuals over random parameters.",
        "nets": tree_to_jsonable(params),
        "architecture": {
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": "tanh",
            "n_params": count_params(params),
        },
        "input": {
            "param_names": ["p1", "p2", "r1", "r2"],
            "coordinate_scaling": {"a": "2*a/L-1", "b": "b/L"},
            "ranges": cfg["ranges"],
        },
        "output": {
            "a_fields": A_FIELDS,
            "b_fields": B_FIELDS,
            "ab_fields": AB_FIELDS,
            "h_decomposition": "split_by_b_sign",
            "h_zero_trace": "positive",
            "uses_baseline": True,
            "output_scale_a": cfg["output_scale_a"],
            "output_scale_b": cfg["output_scale_b"],
            "output_scale_ab": cfg["output_scale_ab"],
        },
        "lag": np.asarray(cfg["a"]).astype(float).tolist(),
        "b_lag": np.asarray(cfg["b"]).astype(float).tolist(),
        "params": {
            "N": cfg["N"],
            "L": cfg["L"],
            "sigma": cfg["sigma"],
            "A": cfg["A"],
            "quadrature": args.quadrature,
            "inner_quad": args.inner_quad,
            "ode_n": int(np.asarray(cfg["a_ode"]).shape[0]),
            "ode_grid": args.ode_grid,
            "h_a_n": int(np.asarray(cfg["a_h"]).shape[0]),
            "h_b_n": int(np.asarray(cfg["b_h_pos"]).shape[0]),
            "h_grid": args.h_grid,
        },
        "diagnostics": diagnostics,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def parse_args():
    ap = argparse.ArgumentParser(description="Train a direct stationary parameter PINN.")
    ap.add_argument("--out", default="data/stationary_param_pinn.json")
    ap.add_argument("--checkpoint", default="data/stationary_param_pinn_checkpoint.json")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-decay", type=float, default=0.6)
    ap.add_argument("--weight-decay", type=float, default=1e-7)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260520)
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--L", type=float, default=3.0)
    ap.add_argument("--inner-quad", type=int, default=7)
    ap.add_argument("--quadrature", choices=["trapezoid", "simpson"], default="simpson")
    ap.add_argument("--ode-n", type=int, default=0, help="Extra collocation count for local ODE/policy residuals; 0 uses N.")
    ap.add_argument("--ode-grid", choices=["uniform", "chebyshev"], default="uniform")
    ap.add_argument("--h-a-n", type=int, default=0, help="One-sided H residual collocation count in the a direction; 0 uses N.")
    ap.add_argument("--h-b-n", type=int, default=0, help="One-sided H residual collocation count per b side; 0 uses N-1.")
    ap.add_argument("--h-grid", choices=["uniform", "chebyshev"], default="uniform")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--A", type=float, default=0.0)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--p-max", type=float, default=40.0)
    ap.add_argument("--r-min", type=float, default=0.05)
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--output-scale-a", type=float, default=1.0)
    ap.add_argument("--output-scale-b", type=float, default=1.0)
    ap.add_argument("--output-scale-ab", type=float, default=1.0)
    ap.add_argument("--last-scale", type=float, default=1e-3)
    ap.add_argument("--w-state", type=float, default=1.0)
    ap.add_argument("--w-filter", type=float, default=1.0)
    ap.add_argument("--w-hx", type=float, default=1.0)
    ap.add_argument("--w-H", type=float, default=0.25)
    ap.add_argument("--w-H-interface", type=float, default=1.0)
    ap.add_argument("--w-wedge", type=float, default=1.0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-boundary", type=float, default=3.0)
    ap.add_argument("--w-tail-left", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-batch", type=int, default=64)
    ap.add_argument("--init", default="", help="Optional exported stationary_param_pinn JSON to warm-start from.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = make_cfg(args)
    if args.init:
        params = load_params(args.init)
        assert_architecture(params, args)
    else:
        params = init_params(args)
    n_params = count_params(params)
    init_msg = f" init={args.init}" if args.init else ""
    print(f"parameter PINN params={n_params} N={args.N} batch={args.batch}{init_msg}", flush=True)

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
    val_batch = sample_params(val_key, args.val_batch, cfg["ranges"])

    def loss_fn(params, p_batch):
        metrics = batch_metrics(params, p_batch, cfg)
        return jnp.sum(metrics), metrics

    @jax.jit
    def train_step(params, opt_state, rng):
        rng, sub = jax.random.split(rng)
        p_batch = sample_params(sub, args.batch, cfg["ranges"])

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, p_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return params, opt_state, rng, loss, metrics, grad_norm

    @jax.jit
    def validate(params):
        metrics = batch_metrics(params, val_batch, cfg)
        return jnp.sum(metrics), metrics

    t0 = time.perf_counter()
    best_val = float("inf")
    best_payload = None
    last_diag = {}

    # Trigger compilation before timing logs.
    params, opt_state, rng, loss, metrics, grad_norm = train_step(params, opt_state, rng)
    jax.block_until_ready(loss)

    for step in range(1, args.steps + 1):
        if step > 1:
            params, opt_state, rng, loss, metrics, grad_norm = train_step(params, opt_state, rng)

        if step == 1 or step % args.log_every == 0:
            val_loss, val_metrics = validate(params)
            jax.block_until_ready(val_loss)
            elapsed = time.perf_counter() - t0
            train_rms = float(jnp.sqrt(loss / len(METRIC_NAMES)))
            val_rms = float(jnp.sqrt(val_loss / len(METRIC_NAMES)))
            block_rms = {
                name: float(math.sqrt(max(v, 0.0)))
                for name, v in zip(METRIC_NAMES, np.asarray(val_metrics))
            }
            last_diag = {
                "step": step,
                "elapsed_s": elapsed,
                "loss": float(loss),
                "train_rms": train_rms,
                "val_loss": float(val_loss),
                "val_rms": val_rms,
                "grad_norm": float(grad_norm),
                "block_rms": block_rms,
                "n_params": n_params,
                "random_parameter_training": True,
                "true_residual_pinn": True,
            }
            if val_rms < best_val:
                best_val = val_rms
                best_payload = last_diag.copy()
            worst = sorted(block_rms.items(), key=lambda kv: kv[1], reverse=True)[:4]
            worst_s = " ".join(f"{k}={v:.2e}" for k, v in worst)
            print(
                f"step={step} train_rms={train_rms:.4e} val_rms={val_rms:.4e} "
                f"grad={float(grad_norm):.3e} elapsed={elapsed:.1f}s {worst_s}",
                flush=True,
            )

        if step % args.save_every == 0:
            diagnostics = dict(last_diag)
            diagnostics["best_val_rms"] = best_val
            diagnostics["best"] = best_payload
            save_json(args.checkpoint, params, args, cfg, diagnostics)
            save_json(args.out, params, args, cfg, diagnostics)

    val_loss, val_metrics = validate(params)
    jax.block_until_ready(val_loss)
    final_diag = dict(last_diag)
    final_diag.update(
        {
            "step": args.steps,
            "elapsed_s": time.perf_counter() - t0,
            "val_loss": float(val_loss),
            "val_rms": float(jnp.sqrt(val_loss / len(METRIC_NAMES))),
            "best_val_rms": best_val,
            "best": best_payload,
            "block_rms": {
                name: float(math.sqrt(max(v, 0.0)))
                for name, v in zip(METRIC_NAMES, np.asarray(val_metrics))
            },
        }
    )
    save_json(args.out, params, args, cfg, final_diag)
    print(json.dumps(final_diag, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
