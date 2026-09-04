#!/usr/bin/env python3
"""Coupled stationary Kyle-Back equilibrium PINN.

This follows ``pinn_stationary_lag_system.pdf``: policy kernels, unresolved
filter drifts, filter kernels, adjoints, and the local weak optimality residual
are solved in one coupled system on a stationary lag grid.
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

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import optax

import train_kyle_back_pinn as kb


D = kb.D
EV = kb.EV
EZ = kb.EZ
EY1 = kb.EY1
EY2 = kb.EY2
BLOCK_NAMES = kb.BLOCK_NAMES
FILTER_ROW_NAMES = {
    0: ["order_flow_tilde"],
    1: ["order_flow_tilde", "private_signal_V_residual"],
    2: ["order_flow_tilde", "private_signal_V_residual"],
}
METRIC_NAMES = [
    "filter_closure_0",
    "filter_closure_1",
    "filter_closure_2",
    "filter_projector",
    "policy_projection",
    "policy_zero_trace",
    "adjoint_H01",
    "adjoint_H21",
    "adjoint_H02",
    "adjoint_H12",
    "foc_mismatch",
    "policy_target",
    "response_consistency",
    "lambda_normalization",
    "market_maker_value_projection",
    "tail",
    "regularization",
]


def init_mlp(key, dims: List[int], last_scale: float):
    return kb.init_mlp(key, dims, last_scale)


def mlp(params, x):
    return kb.mlp(params, x)


def mse(x):
    return jnp.mean(jnp.square(x))


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


def coord(ell, cfg):
    return (2.0 * jnp.asarray(ell, dtype=jnp.float32) / cfg["L"] - 1.0)[..., None]


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


def eval_policy(params, ell, cfg):
    """Noise-state policy kernels D_i(ell), i=1,2."""
    ell = jnp.asarray(ell, dtype=jnp.float32)
    base = kb.candidate_kernels(ell, cfg)
    raw = mlp(params["D"], coord(ell, cfg).reshape((-1, 1))).reshape(ell.shape + (2, D))
    tail = jnp.exp(-cfg["tail_decay"] * ell)[..., None, None]
    trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell))[..., None, None]
    unramped = jnp.stack([base["D1"], base["D2"]], axis=-2) + cfg["policy_scale"] * tail * raw
    out = trace * unramped
    return out[..., 0, :], out[..., 1, :]


def eval_filter_tilde(params, ell, cfg):
    """Unresolved kernels used to build the Kyle-Back filter.

    This mirrors the LQG PINN: F is generated from unresolved loadings, and the
    loss enforces those loadings as projection residuals.  For Kyle-Back the
    relevant rows are the residual value kernel Vtilde^b and the residual total
    order-flow drift Dtot_tilde^b for each observer b.
    """
    ell = jnp.asarray(ell, dtype=jnp.float32)
    base = kb.candidate_kernels(ell, cfg)
    raw = cfg["tilde_scale"] * jnp.exp(-cfg["tail_decay"] * ell)[..., None, None] * mlp(
        params["U"], coord(ell, cfg).reshape((-1, 1))
    ).reshape(ell.shape + (6, D))
    ff = jnp.exp(-cfg["filter_decay"] * ell)[..., None]
    demand_trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell))[..., None]

    v0_base = ff * cfg["std_v"] * EV
    v1_base = jnp.exp(-jnp.maximum(cfg["filter_decay"], cfg["gamma1"]) * ell)[..., None] * cfg["std_v"] * EV
    v2_base = jnp.exp(-jnp.maximum(cfg["filter_decay"], cfg["gamma2"]) * ell)[..., None] * cfg["std_v"] * EV
    dt0_base = demand_trace * cfg["std_z"] * base["ce0"]
    dt1_base = demand_trace * cfg["std_z"] * base["ce1_z"]
    dt2_base = demand_trace * cfg["std_z"] * base["ce2_z"]

    v0 = v0_base + raw[..., 0, :]
    v1 = v1_base + raw[..., 1, :]
    v2 = v2_base + raw[..., 2, :]
    dt0 = dt0_base + raw[..., 3, :]
    dt1 = dt1_base + raw[..., 4, :]
    dt2 = dt2_base + raw[..., 5, :]

    zero = jnp.isclose(ell, 0.0, atol=1e-7)[..., None]
    v_zero = cfg["std_v"] * EV
    z_zero = jnp.zeros((D,), dtype=jnp.float32)
    v0 = jnp.where(zero, v_zero, v0)
    v1 = jnp.where(zero, v_zero, v1)
    v2 = jnp.where(zero, v_zero, v2)
    dt0 = jnp.where(zero, z_zero, dt0)
    dt1 = jnp.where(zero, z_zero, dt1)
    dt2 = jnp.where(zero, z_zero, dt2)
    return {
        "vtilde0": v0,
        "vtilde1": v1,
        "vtilde2": v2,
        "dtot_tilde0": dt0,
        "dtot_tilde1": dt1,
        "dtot_tilde2": dt2,
    }


def eval_ctilde(params, ell, cfg):
    """Unresolved observation-drift rows \tilde c^b(ell).

    These are derived from Vtilde and Dtot_tilde, not learned independently:
      market maker:        Dtot_tilde^0 / sigma_Z
      trader j Z row:      Dtot_tilde^j / sigma_Z
      trader j Y_j row:    gamma_j Vtilde^j / sigma_Yj
    """
    u = eval_filter_tilde(params, ell, cfg)
    ce0 = u["dtot_tilde0"] / cfg["std_z"]
    ce1_z = u["dtot_tilde1"] / cfg["std_z"]
    ce2_z = u["dtot_tilde2"] / cfg["std_z"]
    ce1_y = (cfg["gamma1"] / cfg["std_y1"]) * u["vtilde1"]
    ce2_y = (cfg["gamma2"] / cfg["std_y2"]) * u["vtilde2"]
    return {
        "ce0": ce0[..., None, :],
        "ce1": jnp.stack([ce1_z, ce1_y], axis=-2),
        "ce2": jnp.stack([ce2_z, ce2_y], axis=-2),
        "ce1_z": ce1_z,
        "ce1_y": ce1_y,
        "ce2_z": ce2_z,
        "ce2_y": ce2_y,
    }


def eval_H(params, ell, cfg):
    return kb.eval_H(params, ell, cfg)


def value_and_deriv_H(params, ell, cfg):
    return kb.value_and_deriv_H(params, ell, cfg)


def split_H(y):
    return kb.split_H(y)


def ctilde_rows(params, observer: int, ell, cfg):
    ce = eval_ctilde(params, ell, cfg)
    if observer == 0:
        return ce["ce0"]
    if observer == 1:
        return ce["ce1"]
    if observer == 2:
        return ce["ce2"]
    raise ValueError(f"unknown observer {observer}")


def filter_kernel_pairs(params, observer: int, first_lag, second_lag, cfg):
    """Characteristic solution of equations (15)-(17) for a given ctilde.

    The row index is the observation residual row: one row for the market
    maker, and two rows for each trader.  Both the birth terms and the interior
    PDE sum over those rows.
    """
    first_lag = jnp.asarray(first_lag, dtype=jnp.float32)
    second_lag = jnp.asarray(second_lag, dtype=jnp.float32)
    first_ge_second = first_lag >= second_lag
    first_gt_second = first_lag > second_lag + 1e-7
    second_gt_first = second_lag > first_lag + 1e-7
    base = jnp.abs(first_lag - second_lag)
    width = jnp.minimum(first_lag, second_lag)
    rows = observer_rows(observer)
    nodes = cfg["filter_nodes"]
    weights = cfg["filter_weights"]

    c_base = ctilde_rows(params, observer, base, cfg)
    boundary_right = jnp.einsum("...rd,re->...de", c_base, rows)
    boundary_left = jnp.einsum("rd,...re->...de", rows, c_base)
    # Appendix A.3 birth boundaries are side traces f(ell,0), f(0,a).
    # At equal positive lags the birth is a boundary/atom, not density mass.
    boundary = jnp.where(
        first_gt_second[..., None, None],
        boundary_right,
        jnp.where(second_gt_first[..., None, None], boundary_left, jnp.zeros_like(boundary_right)),
    )

    tau = 0.5 * width[..., None] * (nodes + 1.0)
    quad_w = 0.5 * width[..., None] * weights
    c_first = ctilde_rows(
        params,
        observer,
        jnp.where(first_ge_second[..., None], base[..., None] + tau, tau),
        cfg,
    )
    c_second = ctilde_rows(
        params,
        observer,
        jnp.where(first_ge_second[..., None], tau, base[..., None] + tau),
        cfg,
    )
    integral = jnp.einsum("...q,...qrd,...qre->...de", quad_w, c_first, c_second)
    return boundary + integral


def row_rms(residual):
    return jnp.sqrt(jnp.mean(jnp.square(residual), axis=(0, 2)))


def project_at(params, observer: int, kernel_eval, kernel_grid, ell_eval, cfg):
    rows = observer_rows(observer)
    direct = jnp.einsum("nd,rd,re->ne", kernel_eval, rows, rows)
    first = jnp.broadcast_to(cfg["inner_grid"][:, None], (cfg["inner_grid"].shape[0], ell_eval.shape[0]))
    second = jnp.broadcast_to(ell_eval[None, :], first.shape)
    f = filter_kernel_pairs(params, observer, first, second, cfg)
    indirect = jnp.einsum("a,ad,alde->le", cfg["inner_weights"], kernel_grid, f)
    # The lag-zero observed innovation is an atom.  Old-history density terms
    # should not be integrated into the newborn primitive shock response.
    indirect = jnp.where((ell_eval > 1e-7)[..., None], indirect, jnp.zeros_like(indirect))
    return direct + indirect


def m0_at(params, ell_eval, cfg):
    return eval_filter_tilde(params, ell_eval, cfg)["vtilde0"]


def residual_implied_value_response(params, ell_eval, cfg):
    """Value readouts implied by the learned unresolved value kernels."""
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, ell_eval.shape + (D,))
    u_eval = eval_filter_tilde(params, ell_eval, cfg)
    price = v_eval - u_eval["vtilde0"]
    vhat1 = v_eval - u_eval["vtilde1"]
    vhat2 = v_eval - u_eval["vtilde2"]
    return {
        "true_value": v_eval,
        "price": price,
        "trader1_estimate": vhat1,
        "trader2_estimate": vhat2,
    }


def eval_value_response(params, ell_eval, cfg):
    """Separate response/readout network for price and value-estimate IRFs.

    The U network estimates unresolved kernels Vtilde^b.  The R network
    estimates displayed readout responses, with a consistency loss tying it to
    the identity P_b V = V - Vtilde^b.
    """
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    implied = residual_implied_value_response(params, ell_eval, cfg)
    base = jnp.stack(
        [implied["price"], implied["trader1_estimate"], implied["trader2_estimate"]],
        axis=-2,
    )
    raw = mlp(params["R"], coord(ell_eval, cfg).reshape((-1, 1))).reshape(ell_eval.shape + (3, D))
    trace = (1.0 - jnp.exp(-cfg["diag_ramp"] * ell_eval))[..., None, None]
    tail = jnp.exp(-cfg["response_tail_decay"] * ell_eval)[..., None, None]
    out = base + cfg["response_scale"] * trace * tail * raw
    return {
        "true_value": implied["true_value"],
        "price": out[..., 0, :],
        "trader1_estimate": out[..., 1, :],
        "trader2_estimate": out[..., 2, :],
        "price_residual_implied": implied["price"],
        "trader1_estimate_residual_implied": implied["trader1_estimate"],
        "trader2_estimate_residual_implied": implied["trader2_estimate"],
    }


def response_consistency_residual(params, cfg):
    """Keep the separate response net consistent with V - Vtilde identities."""
    grid = cfg["grid"]
    response = eval_value_response(params, grid, cfg)
    implied = residual_implied_value_response(params, grid, cfg)
    return jnp.concatenate(
        [
            (response["price"] - implied["price"]).reshape(-1),
            (response["trader1_estimate"] - implied["trader1_estimate"]).reshape(-1),
            (response["trader2_estimate"] - implied["trader2_estimate"]).reshape(-1),
        ]
    )


def model_at(params, ell_eval, cfg):
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    D1_eval, D2_eval = eval_policy(params, ell_eval, cfg)
    D1_inner, D2_inner = eval_policy(params, cfg["inner_grid"], cfg)
    d1 = project_at(params, 1, D1_eval, D1_inner, ell_eval, cfg)
    d2 = project_at(params, 2, D2_eval, D2_inner, ell_eval, cfg)
    m0 = m0_at(params, ell_eval, cfg)
    u = eval_filter_tilde(params, ell_eval, cfg)
    primitive_V = jnp.broadcast_to(cfg["std_v"] * EV, ell_eval.shape + (D,))
    c0 = (d1 + d2) / cfg["std_z"]
    # Literal chapter signal: dY^j has drift Gamma_j (V - P) dt.
    # In primitive lag coordinates V - P is the market-maker residual Vtilde^0.
    cY1 = (cfg["gamma1"] / cfg["std_y1"]) * m0
    cY2 = (cfg["gamma2"] / cfg["std_y2"]) * m0
    return {
        "D1": D1_eval,
        "D2": D2_eval,
        "d1": d1,
        "d2": d2,
        "Dtot": d1 + d2,
        "m0": m0,
        "primitive_V": primitive_V,
        "primitive_Vtilde0": m0,
        "primitive_Vtilde1": u["vtilde1"],
        "primitive_Vtilde2": u["vtilde2"],
        "c0": c0,
        "cY1": cY1,
        "cY2": cY2,
    }


def c_rows_for_observer(model, observer: int):
    if observer == 0:
        return model["c0"][:, None, :]
    if observer == 1:
        return jnp.stack([model["c0"], model["cY1"]], axis=1)
    if observer == 2:
        return jnp.stack([model["c0"], model["cY2"]], axis=1)
    raise ValueError(f"unknown observer {observer}")


def raw_c_rows_for_observer(model, observer: int, cfg):
    """Raw observation-drift rows used to define unresolved filtering rows.

    The private signal row is the chapter's mispricing signal V-P.  The
    observer-specific unresolved row in eval_ctilde is still gamma_j Vtilde^j:
    projecting V-P through trader j's larger filtration removes the public P.
    """
    if observer == 0:
        return model["c0"][:, None, :]
    if observer == 1:
        return jnp.stack([model["c0"], model["cY1"]], axis=1)
    if observer == 2:
        return jnp.stack([model["c0"], model["cY2"]], axis=1)
    raise ValueError(f"unknown observer {observer}")


def closure_residual(params, observer: int, model_grid, cfg):
    c_rows = raw_c_rows_for_observer(model_grid, observer, cfg)
    c_quad = raw_c_rows_for_observer(model_at(params, cfg["quad_grid"], cfg), observer, cfg)
    ce_rows = ctilde_rows(params, observer, cfg["grid"], cfg)
    pi = observer_pi(observer)
    direct = jnp.einsum("lrd,de->lre", c_rows, jnp.eye(D, dtype=jnp.float32) - pi)
    first = jnp.broadcast_to(cfg["quad_grid"][:, None], (cfg["quad_grid"].shape[0], cfg["grid"].shape[0]))
    second = jnp.broadcast_to(cfg["grid"][None, :], first.shape)
    f = filter_kernel_pairs(params, observer, first, second, cfg)
    # The diagonal innovation birth is a boundary/atom, not a density mass.
    # Use interior quadrature so ctilde(0) is not spuriously integrated.
    integral = jnp.einsum("a,ard,alde->lre", cfg["quad_weights"], c_quad, f)
    return ce_rows - (direct - integral)


def vtilde_projection_residual(params, observer: int, cfg):
    """Enforce Vtilde^b = V - P_b V, the LQG-style filter residual."""
    grid = cfg["grid"]
    u = eval_filter_tilde(params, grid, cfg)
    vt = u[f"vtilde{observer}"]
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (grid.shape[0], D))
    v_inner = jnp.broadcast_to(cfg["std_v"] * EV, (cfg["inner_grid"].shape[0], D))
    v_proj = project_at(params, observer, v_eval, v_inner, grid, cfg)
    return vt - (v_eval - v_proj)


def market_maker_price_projection_response(params, cfg):
    """Redundant F^0-projection diagnostic for P_0 V."""
    grid = cfg["grid"]
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (grid.shape[0], D))
    v_inner = jnp.broadcast_to(cfg["std_v"] * EV, (cfg["inner_grid"].shape[0], D))
    return project_at(params, 0, v_eval, v_inner, grid, cfg)[:, 0]


def market_maker_price_response(params, cfg):
    """Displayed market-maker value estimate from the response/readout net."""
    grid = cfg["grid"]
    return eval_value_response(params, grid, cfg)["price"][:, 0]


def market_maker_price_shape_residual(params, cfg):
    """Keep the scalar price IRF moving monotonically toward the true value."""
    price = market_maker_price_response(params, cfg)
    target = cfg["std_v"]
    decreases = jax.nn.relu(price[:-1] - price[1:])
    overshoot = jax.nn.relu(price - target)
    below_zero = jax.nn.relu(-price)
    terminal_shortfall = jax.nn.relu(target - price[-1:])
    return jnp.concatenate([5.0 * decreases, overshoot, below_zero, 5.0 * terminal_shortfall])


def trader_information_order_residual(params, cfg):
    """Soft check: the higher-gamma trader should have weakly smaller V residual."""
    u = eval_filter_tilde(params, cfg["grid"], cfg)
    err1 = jnp.abs(u["vtilde1"][:, 0])
    err2 = jnp.abs(u["vtilde2"][:, 0])
    if float(cfg["gamma2"]) > float(cfg["gamma1"]):
        return jax.nn.relu(err2 - err1)
    if float(cfg["gamma1"]) > float(cfg["gamma2"]):
        return jax.nn.relu(err1 - err2)
    return err1 - err2


def filter_residual(params, observer: int, model_grid, cfg):
    """Combined filter residual for the learned unresolved kernels.

    The closure residual enforces Dtot_tilde^b and the private-signal row
    gamma_j Vtilde^j through the observation equations.  The extra Vtilde
    residual pins the market-maker value residual Vtilde^0 as well.
    """
    return jnp.concatenate(
        [
            closure_residual(params, observer, model_grid, cfg).reshape(-1),
            vtilde_projection_residual(params, observer, cfg).reshape(-1),
        ]
    )


def zero_unobserved_residual(params, observer: int, model_grid, cfg):
    """Zero-lag closure on coordinates outside the observed innovation birth.

    The birth conditions in the notes are boundary conditions for f^b(0,a) and
    f^b(ell,0).  They are already built into filter_kernel_pairs.  There is no
    separate direct-only condition ce^b(0)=c^b(0)(I-Pi_b); at ell=0 we should
    still enforce the full algebraic closure equation, including the filter
    projection integral.
    """
    pi = observer_pi(observer)
    unobs = jnp.diag(jnp.eye(D, dtype=jnp.float32) - pi)
    return closure_residual(params, observer, model_grid, cfg)[0] * unobs[None, :]


def zero_birth_residual(params, observer: int, model_grid, cfg):
    """Diagnostic zero-lag closure on directly observed innovation-birth coordinates."""
    obs = jnp.diag(observer_pi(observer))
    return closure_residual(params, observer, model_grid, cfg)[0] * obs[None, :]


def policy_zero_trace_residual(params, cfg):
    """Predictability condition: controls cannot load on the newborn lag-zero shock."""
    d1_zero, d2_zero = eval_policy(params, jnp.array([0.0], dtype=jnp.float32), cfg)
    return jnp.concatenate([d1_zero.reshape(-1), d2_zero.reshape(-1)])


def filter_projection_matrix(params, observer: int, cfg):
    """Finite-grid conditional-expectation projection induced by f^b.

    Appendix A derives the filter as a conditional-expectation projection.  The
    PDE/closure equations are the kernel form of that projection, but a weak
    collocation fit can otherwise learn an expansive non-projector.  This matrix
    is the quadrature discretization of (P_b x)(ell)=x(ell)Pi_b+int x(a)f(a,ell)da.
    """
    # Use the interior quadrature grid used by the closure equations.  The
    # zero-lag innovation birth is an atom, so it should not be counted as
    # ordinary density mass in this projection diagnostic.
    grid = cfg["quad_grid"]
    weights = cfg["quad_weights"]
    n = grid.shape[0]
    pi = observer_pi(observer)
    first = jnp.broadcast_to(grid[:, None], (n, n))
    second = jnp.broadcast_to(grid[None, :], (n, n))
    f = filter_kernel_pairs(params, observer, first, second, cfg)
    integral = jnp.transpose(f, (1, 3, 0, 2)) * weights[None, None, :, None]
    direct = jnp.eye(n, dtype=jnp.float32)[:, None, :, None] * pi.T[None, :, None, :]
    return (direct + integral).reshape((n * D, n * D))


def filter_projector_residual(params, observer: int, cfg):
    """Idempotence and weighted self-adjointness of the CE projection."""
    M = filter_projection_matrix(params, observer, cfg)
    weights = jnp.repeat(cfg["quad_weights"], D)
    idempotence = M @ M - M
    weighted = weights[:, None] * M
    self_adjoint = weighted - weighted.T
    return jnp.concatenate([idempotence.reshape(-1), self_adjoint.reshape(-1)])


def alpha_terms(params, H_grid, cfg):
    w = cfg["quad_weights"]
    H = split_H(eval_H(params, cfg["quad_grid"], cfg))
    Hz = split_H(eval_H(params, jnp.array([0.0], dtype=jnp.float32), cfg))
    ce = eval_ctilde(params, cfg["quad_grid"], cfg)
    rows0 = observer_rows(0)
    rows1 = observer_rows(1)
    rows2 = observer_rows(2)

    a01_z = jnp.einsum("n,nij,nj->i", w, H["H01"], ce["ce0"][:, 0, :]) + Hz["H01"][0] @ rows0[0]
    a02_z = jnp.einsum("n,nij,nj->i", w, H["H02"], ce["ce0"][:, 0, :]) + Hz["H02"][0] @ rows0[0]
    a21_z = jnp.einsum("n,nij,nj->i", w, H["H21"], ce["ce2"][:, 0, :]) + Hz["H21"][0] @ rows2[0]
    a21_y = jnp.einsum("n,nij,nj->i", w, H["H21"], ce["ce2"][:, 1, :]) + Hz["H21"][0] @ rows2[1]
    a12_z = jnp.einsum("n,nij,nj->i", w, H["H12"], ce["ce1"][:, 0, :]) + Hz["H12"][0] @ rows1[0]
    a12_y = jnp.einsum("n,nij,nj->i", w, H["H12"], ce["ce1"][:, 1, :]) + Hz["H12"][0] @ rows1[1]
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


def source_blocks(params, ell, alphas, cfg):
    model = model_at(params, ell, cfg)
    v0 = cfg["std_v"] * EV
    src01 = kb.vec_outer(model["d1"], v0) - kb.vec_outer(alphas["a01_z"], model["c0"])
    src21 = (
        kb.vec_outer(alphas["beta1"], model["D2"] / cfg["std_z"])
        - kb.vec_outer(alphas["a21_z"], model["c0"])
        - kb.vec_outer(alphas["a21_y"], model["cY2"])
    )
    src02 = kb.vec_outer(model["d2"], v0) - kb.vec_outer(alphas["a02_z"], model["c0"])
    src12 = (
        kb.vec_outer(alphas["beta2"], model["D1"] / cfg["std_z"])
        - kb.vec_outer(alphas["a12_z"], model["c0"])
        - kb.vec_outer(alphas["a12_y"], model["cY1"])
    )
    return {"H01": src01, "H21": src21, "H02": src02, "H12": src12}


def lambda_value(params, cfg):
    ce0 = eval_ctilde(params, cfg["quad_grid"], cfg)["ce0"][:, 0, :]
    return cfg["std_v"] / cfg["std_z"] * jnp.einsum("n,nd,d->", cfg["quad_weights"], ce0, EV)


def observer_projection_matrix(params, observer: int, ell_eval, cfg):
    """IRF matrix for observer-b noise state from primitive Brownian shocks."""
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    eye = jnp.eye(D, dtype=jnp.float32)
    mats = []
    for ch in range(D):
        kernel_eval = jnp.broadcast_to(eye[ch], (ell_eval.shape[0], D))
        kernel_grid = jnp.broadcast_to(eye[ch], (cfg["inner_grid"].shape[0], D))
        mats.append(project_at(params, observer, kernel_eval, kernel_grid, ell_eval, cfg))
    return jnp.stack(mats, axis=1)


def value_irfs(params, ell_eval, cfg):
    """Market-maker price and trader value estimates for primitive V shocks."""
    ell_eval = jnp.asarray(ell_eval, dtype=jnp.float32)
    v_eval = jnp.broadcast_to(cfg["std_v"] * EV, (ell_eval.shape[0], D))
    v_grid = jnp.broadcast_to(cfg["std_v"] * EV, (cfg["inner_grid"].shape[0], D))
    response = eval_value_response(params, ell_eval, cfg)
    price_from_F = project_at(params, 0, v_eval, v_grid, ell_eval, cfg)
    vhat1_from_F = project_at(params, 1, v_eval, v_grid, ell_eval, cfg)
    vhat2_from_F = project_at(params, 2, v_eval, v_grid, ell_eval, cfg)
    return {
        "true_value": response["true_value"],
        "price": response["price"],
        "trader1_estimate": response["trader1_estimate"],
        "trader2_estimate": response["trader2_estimate"],
        "mispricing_gap1": response["trader1_estimate"] - response["price"],
        "mispricing_gap2": response["trader2_estimate"] - response["price"],
        "price_residual_implied": response["price_residual_implied"],
        "trader1_estimate_residual_implied": response["trader1_estimate_residual_implied"],
        "trader2_estimate_residual_implied": response["trader2_estimate_residual_implied"],
        "price_from_F_projection": price_from_F,
        "trader1_estimate_from_F_projection": vhat1_from_F,
        "trader2_estimate_from_F_projection": vhat2_from_F,
    }


def readout_for_deviator(params, H_name0, H_name_opp, opp: int, alphas, cfg):
    grid = cfg["grid"]
    quad_grid = cfg["quad_grid"]
    inner_grid = cfg["inner_grid"]
    weights = cfg["quad_weights"]
    H_grid, _ = value_and_deriv_H(params, grid, cfg)
    H_quad, _ = value_and_deriv_H(params, quad_grid, cfg)
    H_inner, _ = value_and_deriv_H(params, inner_grid, cfg)
    H = split_H(H_grid)
    Hq = split_H(H_quad)
    Hi = split_H(H_inner)
    src = source_blocks(params, grid, alphas, cfg)
    src_q = source_blocks(params, quad_grid, alphas, cfg)
    src_i = source_blocks(params, inner_grid, alphas, cfg)
    model = model_at(params, grid, cfg)
    model_q = model_at(params, quad_grid, cfg)
    model_i = model_at(params, inner_grid, cfg)
    ce = eval_ctilde(params, grid, cfg)
    ce_q = eval_ctilde(params, quad_grid, cfg)
    ce_i = eval_ctilde(params, inner_grid, cfg)

    if opp == 1:
        ce_opp_z = ce["ce1"][:, 0, :]
        ce_opp_y = ce["ce1"][:, 1, :]
        ce_opp_z_q = ce_q["ce1"][:, 0, :]
        ce_opp_y_q = ce_q["ce1"][:, 1, :]
        ce_opp_z_i = ce_i["ce1"][:, 0, :]
        ce_opp_y_i = ce_i["ce1"][:, 1, :]
        c_opp_y = model["cY1"]
        c_opp_y_q = model_q["cY1"]
        c_opp_y_i = model_i["cY1"]
        D_opp = model["D1"]
        D_opp_q = model_q["D1"]
        ey = EY1
    else:
        ce_opp_z = ce["ce2"][:, 0, :]
        ce_opp_y = ce["ce2"][:, 1, :]
        ce_opp_z_q = ce_q["ce2"][:, 0, :]
        ce_opp_y_q = ce_q["ce2"][:, 1, :]
        ce_opp_z_i = ce_i["ce2"][:, 0, :]
        ce_opp_y_i = ce_i["ce2"][:, 1, :]
        c_opp_y = model["cY2"]
        c_opp_y_q = model_q["cY2"]
        c_opp_y_i = model_i["cY2"]
        D_opp = model["D2"]
        D_opp_q = model_q["D2"]
        ey = EY2

    phi0 = ce["ce0"][:, 0, :] / cfg["std_z"]
    phi_opp = ce_opp_z / cfg["std_z"]
    phi0_q = ce_q["ce0"][:, 0, :] / cfg["std_z"]
    phi_opp_q = ce_opp_z_q / cfg["std_z"]
    phi0_i = ce_i["ce0"][:, 0, :] / cfg["std_z"]
    phi_opp_i = ce_opp_z_i / cfg["std_z"]
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

    H0 = H[H_name0]
    Hopp = H[H_name_opp]
    H0_q = Hq[H_name0]
    Hopp_q = Hq[H_name_opp]
    H0_i = Hi[H_name0]
    Hopp_i = Hi[H_name_opp]
    S0 = src[H_name0]
    Sopp = src[H_name_opp]
    S0_q = src_q[H_name0]
    Sopp_q = src_q[H_name_opp]
    S0_i = src_i[H_name0]
    Sopp_i = src_i[H_name_opp]
    H_zero = split_H(eval_H(params, jnp.array([0.0], dtype=jnp.float32), cfg))
    H0_zero = H_zero[H_name0][0]
    Hopp_zero = H_zero[H_name_opp][0]
    ce_zero = eval_ctilde(params, jnp.array([0.0], dtype=jnp.float32), cfg)
    phi0_zero = ce_zero["ce0"][0, 0, :] / cfg["std_z"]
    phi_opp_zero = (ce_zero["ce1"][0, 0, :] if opp == 1 else ce_zero["ce2"][0, 0, :]) / cfg["std_z"]

    density0 = jnp.einsum("nij,nj->ni", S0, phi0) + jnp.einsum("nij,nj->ni", H0, q0)
    density_opp = jnp.einsum("nij,nj->ni", Sopp, phi_opp) + jnp.einsum("nij,nj->ni", Hopp, q_opp)
    density0_q = jnp.einsum("nij,nj->ni", S0_q, phi0_q) + jnp.einsum("nij,nj->ni", H0_q, q0_q)
    density_opp_q = jnp.einsum("nij,nj->ni", Sopp_q, phi_opp_q) + jnp.einsum("nij,nj->ni", Hopp_q, q_opp_q)
    density0_i = jnp.einsum("nij,nj->ni", S0_i, phi0_i) + jnp.einsum("nij,nj->ni", H0_i, q0_i)
    density_opp_i = jnp.einsum("nij,nj->ni", Sopp_i, phi_opp_i) + jnp.einsum("nij,nj->ni", Hopp_i, q_opp_i)
    old0 = jnp.einsum("n,nd->d", weights, density0_q)
    old_opp = jnp.einsum("n,nd->d", weights, density_opp_q)
    diag0 = H0_zero @ (cfg["rho"] * psi - phi0_zero + q0_diag)
    diag_opp = Hopp_zero @ (cfg["rho"] * psi - phi_opp_zero + q_opp_diag)
    return {
        "density_market": density0,
        "density_opponent": density_opp,
        "density_total": density0 + density_opp,
        "density_market_quad": density0_q,
        "density_opponent_quad": density_opp_q,
        "density_total_quad": density0_q + density_opp_q,
        "density_market_inner": density0_i,
        "density_opponent_inner": density_opp_i,
        "density_total_inner": density0_i + density_opp_i,
        "total": old0 + old_opp + diag0 + diag_opp,
        "old_market": old0,
        "old_opponent": old_opp,
        "diag_market": diag0,
        "diag_opponent": diag_opp,
        "R": R,
    }


def equilibrium_targets(params, cfg):
    H_grid = eval_H(params, cfg["grid"], cfg)
    alphas = alpha_terms(params, H_grid, cfg)
    ro1 = readout_for_deviator(params, "H01", "H21", 2, alphas, cfg)
    ro2 = readout_for_deviator(params, "H02", "H12", 1, alphas, cfg)
    lam = lambda_value(params, cfg)
    inv_lam = 1.0 / jnp.where(jnp.abs(lam) > 1e-6, lam, jnp.inf)
    ce0 = eval_ctilde(params, cfg["grid"], cfg)["ce0"][:, 0, :]
    ce0_i = eval_ctilde(params, cfg["inner_grid"], cfg)["ce0"][:, 0, :]
    model = model_at(params, cfg["grid"], cfg)

    # Myopic mispricing-demand part from the local range equation, estimated
    # through each trader's noise state and expressed back in primitive shocks.
    local_info = cfg["std_z"] * ce0
    local_info_i = cfg["std_z"] * ce0_i
    mispricing1 = project_at(params, 1, local_info, local_info_i, cfg["grid"], cfg)
    mispricing2 = project_at(params, 2, local_info, local_info_i, cfg["grid"], cfg)

    # Future-profit price-impact term H^i, estimated by the deviating trader's
    # noise state and converted to primitive Brownian coordinates.
    calH1 = project_at(params, 1, ro1["density_total"], ro1["density_total_inner"], cfg["grid"], cfg)
    calH2 = project_at(params, 2, ro2["density_total"], ro2["density_total_inner"], cfg["grid"], cfg)

    foc_lhs1 = lam * (model["d1"] - mispricing1)
    foc_lhs2 = lam * (model["d2"] - mispricing2)
    foc_mismatch1 = foc_lhs1 - calH1
    foc_mismatch2 = foc_lhs2 - calH2
    target1 = mispricing1 + calH1 * inv_lam
    target2 = mispricing2 + calH2 * inv_lam
    return {
        "target1": target1,
        "target2": target2,
        "target1_raw": local_info + ro1["density_total"] * inv_lam,
        "target2_raw": local_info + ro2["density_total"] * inv_lam,
        "lambda": lam,
        "ro1": ro1,
        "ro2": ro2,
        "mispricing_demand1": mispricing1,
        "mispricing_demand2": mispricing2,
        "calH1": calH1,
        "calH2": calH2,
        "foc_lhs1": foc_lhs1,
        "foc_lhs2": foc_lhs2,
        "foc_mismatch1": foc_mismatch1,
        "foc_mismatch2": foc_mismatch2,
    }


def residual_metrics(params, cfg):
    grid = cfg["grid"]
    model_grid = model_at(params, grid, cfg)
    D1, D2 = model_grid["D1"], model_grid["D2"]
    d1, d2 = model_grid["d1"], model_grid["d2"]
    D1_inner, D2_inner = eval_policy(params, cfg["inner_grid"], cfg)
    D1_proj = project_at(params, 1, D1, D1_inner, grid, cfg)
    D2_proj = project_at(params, 2, D2, D2_inner, grid, cfg)

    H_grid = eval_H(params, grid, cfg)
    alphas = alpha_terms(params, H_grid, cfg)
    H_ode, dH_ode = value_and_deriv_H(params, cfg["ode_grid"], cfg)
    H = split_H(H_ode)
    dH = split_H(dH_ode)
    src = source_blocks(params, cfg["ode_grid"], alphas, cfg)
    targets = equilibrium_targets(params, cfg)

    tail_model = model_at(params, jnp.array([cfg["L"]], dtype=jnp.float32), cfg)
    tail_ce = eval_ctilde(params, jnp.array([cfg["L"]], dtype=jnp.float32), cfg)
    tail_u = eval_filter_tilde(params, jnp.array([cfg["L"]], dtype=jnp.float32), cfg)
    tail_H = eval_H(params, jnp.array([cfg["L"]], dtype=jnp.float32), cfg)

    blocks = [
        cfg["w_filter"] * mse(filter_residual(params, 0, model_grid, cfg)),
        cfg["w_filter"] * mse(filter_residual(params, 1, model_grid, cfg)),
        cfg["w_filter"] * mse(filter_residual(params, 2, model_grid, cfg)),
        cfg["w_filter_projector"]
        * (
            mse(filter_projector_residual(params, 0, cfg))
            + mse(filter_projector_residual(params, 1, cfg))
            + mse(filter_projector_residual(params, 2, cfg))
        ),
        cfg["w_policy_proj"] * (mse(D1 - D1_proj) + mse(D2 - D2_proj)),
        cfg["w_policy_zero"] * mse(policy_zero_trace_residual(params, cfg)),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H01"] - dH["H01"] - src["H01"]),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H21"] - dH["H21"] - src["H21"]),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H02"] - dH["H02"] - src["H02"]),
        cfg["w_adjoint"] * mse(cfg["rho"] * H["H12"] - dH["H12"] - src["H12"]),
        cfg["w_foc"] * (mse(targets["foc_mismatch1"]) + mse(targets["foc_mismatch2"])),
        cfg["w_optimality"] * (mse(d1 - targets["target1"]) + mse(d2 - targets["target2"])),
        cfg["w_response_consistency"] * mse(response_consistency_residual(params, cfg)),
        cfg["w_lambda"] * jnp.square(targets["lambda"] - cfg["lambda_target"]),
        cfg["w_value_proj"] * mse(vtilde_projection_residual(params, 0, cfg)),
        cfg["w_tail"]
        * (
            mse(tail_model["D1"])
            + mse(tail_model["D2"])
            + mse(tail_ce["ce0"])
            + mse(tail_ce["ce1"])
            + mse(tail_ce["ce2"])
            + mse(tail_u["vtilde0"])
            + mse(tail_u["vtilde1"])
            + mse(tail_u["vtilde2"])
            + mse(tail_u["dtot_tilde0"])
            + mse(tail_u["dtot_tilde1"])
            + mse(tail_u["dtot_tilde2"])
            + mse(tail_H)
        ),
        cfg["w_reg"]
        * (
            mse(H_grid)
            + mse(D1)
            + mse(D2)
            + mse(eval_filter_tilde(params, grid, cfg)["vtilde0"])
            + mse(response_consistency_residual(params, cfg))
        ),
    ]
    return jnp.stack(blocks)


def diagnostics(params, cfg, step, elapsed, loss, metrics, grad_norm):
    grid = cfg["grid"]
    model = model_at(params, grid, cfg)
    model_inner = model_at(params, cfg["inner_grid"], cfg)
    ce = eval_ctilde(params, grid, cfg)
    u = eval_filter_tilde(params, grid, cfg)
    targets = equilibrium_targets(params, cfg)
    D_gap1 = model["d1"] - targets["target1"]
    D_gap2 = model["d2"] - targets["target2"]
    rep_gap1 = model["D1"] - model["d1"]
    rep_gap2 = model["D2"] - model["d2"]
    Dtot = model["d1"] + model["d2"]
    Drel = model["d1"] - model["d2"]
    Dmean = 0.5 * Dtot
    Dtot_proj0 = project_at(params, 0, Dtot, model_inner["Dtot"], grid, cfg)
    back_gap = u["dtot_tilde0"] - Dtot
    closure = {
        f"observer{b}": float(jnp.sqrt(mse(filter_residual(params, b, model, cfg))))
        for b in (0, 1, 2)
    }
    price = np.asarray(market_maker_price_response(params, cfg), dtype=float)
    price_from_F = np.asarray(market_maker_price_projection_response(params, cfg), dtype=float)
    price_distance = np.abs(float(cfg["std_v"]) - price)
    val = value_irfs(params, grid, cfg)
    t1_v = np.asarray(val["trader1_estimate"], dtype=float)[:, 0]
    t2_v = np.asarray(val["trader2_estimate"], dtype=float)[:, 0]
    info_order = np.asarray(trader_information_order_residual(params, cfg), dtype=float)
    price_shape = {
        "min": float(np.min(price)),
        "max": float(np.max(price)),
        "last": float(price[-1]),
        "distance_increases": int(np.sum(price_distance[1:] > price_distance[:-1] + 1e-6)),
        "shape_rms": float(jnp.sqrt(mse(market_maker_price_shape_residual(params, cfg)))),
        "value_projection_rms": float(jnp.sqrt(mse(vtilde_projection_residual(params, 0, cfg)))),
        "F_projection_min": float(np.min(price_from_F)),
        "F_projection_max": float(np.max(price_from_F)),
        "F_projection_last": float(price_from_F[-1]),
    }
    closure_rows = {
        f"observer{b}": {
            name: float(value)
            for name, value in zip(
                FILTER_ROW_NAMES[b],
                np.asarray(row_rms(closure_residual(params, b, model, cfg)), dtype=float),
            )
        }
        for b in (0, 1, 2)
    }
    zero_unobs = {
        f"observer{b}": float(jnp.sqrt(mse(zero_unobserved_residual(params, b, model, cfg))))
        for b in (0, 1, 2)
    }
    zero_birth = {
        f"observer{b}": float(jnp.sqrt(mse(zero_birth_residual(params, b, model, cfg))))
        for b in (0, 1, 2)
    }
    projector = {
        f"observer{b}": float(jnp.sqrt(mse(filter_projector_residual(params, b, cfg))))
        for b in (0, 1, 2)
    }
    D1_zero, D2_zero = eval_policy(params, jnp.array([0.0], dtype=jnp.float32), cfg)
    ce_zero = eval_ctilde(params, jnp.array([0.0], dtype=jnp.float32), cfg)
    u_zero = eval_filter_tilde(params, jnp.array([0.0], dtype=jnp.float32), cfg)
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
        "lambda": float(targets["lambda"]),
        "lambda_abs": float(jnp.abs(targets["lambda"])),
        "lambda_sign": float(jnp.sign(targets["lambda"])),
        "closure_rms_unweighted": closure,
        "closure_row_rms_unweighted": closure_rows,
        "market_maker_price_shape": price_shape,
        "response_consistency_rms_unweighted": float(jnp.sqrt(mse(response_consistency_residual(params, cfg)))),
        "zero_unobserved_rms_unweighted": zero_unobs,
        "zero_birth_rms_unweighted": zero_birth,
        "filter_projector_rms_unweighted": projector,
        "policy_projection_rms_unweighted": {
            "player1": float(jnp.sqrt(mse(model["D1"] - project_at(params, 1, model["D1"], eval_policy(params, cfg["inner_grid"], cfg)[0], grid, cfg)))),
            "player2": float(jnp.sqrt(mse(model["D2"] - project_at(params, 2, model["D2"], eval_policy(params, cfg["inner_grid"], cfg)[1], grid, cfg)))),
        },
        "representative_gap_rms": {
            "player1": float(jnp.sqrt(mse(rep_gap1))),
            "player2": float(jnp.sqrt(mse(rep_gap2))),
        },
        "mode_norms": {
            "aggregate_Dtot_rms": float(jnp.sqrt(mse(Dtot))),
            "relative_D1_minus_D2_rms": float(jnp.sqrt(mse(Drel))),
            "player1_share_gap_rms": float(jnp.sqrt(mse(model["d1"] - Dmean))),
            "player2_share_gap_rms": float(jnp.sqrt(mse(model["d2"] - Dmean))),
        },
        "trader_information_order": {
            "higher_gamma": "trader2" if float(cfg["gamma2"]) > float(cfg["gamma1"]) else ("trader1" if float(cfg["gamma1"]) > float(cfg["gamma2"]) else "symmetric"),
            "violation_count": int(np.sum(info_order > 1e-6)),
            "violation_rms": float(jnp.sqrt(mse(trader_information_order_residual(params, cfg)))),
            "avg_abs_value_error_player1": float(np.mean(np.abs(float(cfg["std_v"]) - t1_v))),
            "avg_abs_value_error_player2": float(np.mean(np.abs(float(cfg["std_v"]) - t2_v))),
        },
        "back_like_gap_rms": float(jnp.sqrt(mse(back_gap))),
        "market_maker_Dtot_projection_rms": float(jnp.sqrt(mse(Dtot_proj0))),
        "zero_trace_rms": {
            "D1": float(jnp.sqrt(mse(D1_zero))),
            "D2": float(jnp.sqrt(mse(D2_zero))),
            "dtot_tilde0": float(jnp.sqrt(mse(u_zero["dtot_tilde0"]))),
            "dtot_tilde1": float(jnp.sqrt(mse(u_zero["dtot_tilde1"]))),
            "dtot_tilde2": float(jnp.sqrt(mse(u_zero["dtot_tilde2"]))),
            "ce0": float(jnp.sqrt(mse(ce_zero["ce0"]))),
            "ce1_z": float(jnp.sqrt(mse(ce_zero["ce1_z"]))),
            "ce2_z": float(jnp.sqrt(mse(ce_zero["ce2_z"]))),
        },
        "policy_target_rms": {
            "player1": float(jnp.sqrt(mse(D_gap1))),
            "player2": float(jnp.sqrt(mse(D_gap2))),
        },
        "foc_mismatch_rms_unweighted": {
            "player1": float(jnp.sqrt(mse(targets["foc_mismatch1"]))),
            "player2": float(jnp.sqrt(mse(targets["foc_mismatch2"]))),
        },
        "theorem_1_12_policy": {
            "formula": "stationary weak FOC: Lambda*(calD_i - E_i[Sigma_Z^{1/2} ctilde0]) = E_i[calH_i]; D_i is the learned noise-state representative and calD_i is its primitive projection",
            "lambda": float(targets["lambda"]),
            "noise_state_policy": {
                "player1": np.asarray(model["D1"], dtype=float).tolist(),
                "player2": np.asarray(model["D2"], dtype=float).tolist(),
            },
            "best_response": {
                "player1": np.asarray(model["d1"], dtype=float).tolist(),
                "player2": np.asarray(model["d2"], dtype=float).tolist(),
            },
            "policy_target_rms": {
                "player1": float(jnp.sqrt(mse(D_gap1))),
                "player2": float(jnp.sqrt(mse(D_gap2))),
            },
        },
    }


def save_json(path, params, cfg, args, diag):
    grid = cfg["grid"]
    model = model_at(params, grid, cfg)
    ce = eval_ctilde(params, grid, cfg)
    u = eval_filter_tilde(params, grid, cfg)
    H_grid = eval_H(params, grid, cfg)
    H = split_H(H_grid)
    targets = equilibrium_targets(params, cfg)
    val = value_irfs(params, grid, cfg)
    noise_state_irfs = {
        "market_maker": np.asarray(observer_projection_matrix(params, 0, grid, cfg), dtype=float).tolist(),
        "trader1": np.asarray(observer_projection_matrix(params, 1, grid, cfg), dtype=float).tolist(),
        "trader2": np.asarray(observer_projection_matrix(params, 2, grid, cfg), dtype=float).tolist(),
    }
    Dtot = model["d1"] + model["d2"]
    Drel = model["d1"] - model["d2"]
    Dmean = 0.5 * Dtot
    filter_residuals = {
        f"observer{b}": np.asarray(closure_residual(params, b, model, cfg), dtype=float).tolist()
        for b in (0, 1, 2)
    }
    vtilde_residuals = {
        f"observer{b}": np.asarray(vtilde_projection_residual(params, b, cfg), dtype=float).tolist()
        for b in (0, 1, 2)
    }
    zero_filter_residuals = {
        "unobserved": {
            f"observer{b}": np.asarray(zero_unobserved_residual(params, b, model, cfg), dtype=float).tolist()
            for b in (0, 1, 2)
        },
        "birth_observed": {
            f"observer{b}": np.asarray(zero_birth_residual(params, b, model, cfg), dtype=float).tolist()
            for b in (0, 1, 2)
        },
    }
    payload = {
        "version": 1,
        "kind": "kyle_back_equilibrium_pinn",
        "description": "Coupled stationary Kyle-Back lag-equation PINN.",
        "assumptions": {
            "scope": "coupled stationary residual solve on truncated lag window",
            "equations": "pinn_stationary_lag_system.pdf equations (8)-(17), (24)-(27), and local weak optimality (38)-(39)",
            "primitive_coordinates": ["fundamental", "order_flow_noise", "private_signal_1", "private_signal_2"],
            "normalization": "lambda target used to select a Back-like representative",
        },
        "mathematical_conventions": {
            "private_signal": "literal chapter signal gamma_j * (V - P), implemented as gamma_j * vtilde0 / sigma_Yj",
            "filter_unresolved_private_row": "trader j unresolved row is gamma_j * vtilde_j / sigma_Yj",
            "policy_zero_trace": "D_i(0)=0 enforced by a lag-zero predictability ramp in eval_policy",
            "policy_foc_object": "FOC residual uses primitive projected demand calD_i=model['d_i'], not the unprojected representative D_i",
            "lambda_normalization_target": float(args.lambda_target),
            "filter_projector_residual_active": float(args.w_filter_projector) != 0.0,
            "separate_response_network": "U estimates unresolved tildes; R estimates displayed value readouts and is tied to V - Vtilde by response_consistency",
            "jax_x64": bool(jax.config.read("jax_enable_x64")),
        },
        "nets": tree_to_jsonable(params),
        "architecture": {
            "policy_filter_adjoint_hidden": args.hidden,
            "policy_filter_adjoint_depth": args.depth,
            "response_hidden": args.response_hidden if args.response_hidden > 0 else args.hidden,
            "response_depth": args.response_depth,
            "activation": "tanh",
            "net_roles": {
                "D": "learned policy/noise-state demand responses",
                "U": "unresolved residual kernels: vtilde0/1/2 and Dtot_tilde0/1/2",
                "R": "displayed value response readouts: price and trader value estimates",
                "H": "adjoint/readout kernels for future-profit effects",
            },
            "n_params": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))),
        },
        "grid": np.asarray(grid, dtype=float).tolist(),
        "params": {
            k: float(v)
            for k, v in vars(args).items()
            if isinstance(v, (int, float)) and k not in {"steps", "log_every", "save_every", "seed"}
        },
        "derived_params": {
            "inner_quad": int(cfg["inner_quad"]),
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
            "H_blocks": BLOCK_NAMES,
        },
        "policy_noise_state": {
            "player1": np.asarray(model["D1"], dtype=float).tolist(),
            "player2": np.asarray(model["D2"], dtype=float).tolist(),
        },
        "primitive_demand": {
            "player1": np.asarray(model["d1"], dtype=float).tolist(),
            "player2": np.asarray(model["d2"], dtype=float).tolist(),
        },
        "calD": {
            "player1": np.asarray(model["d1"], dtype=float).tolist(),
            "player2": np.asarray(model["d2"], dtype=float).tolist(),
        },
        "mode_decomposition": {
            "Dtot": np.asarray(Dtot, dtype=float).tolist(),
            "Drel": np.asarray(Drel, dtype=float).tolist(),
            "Dmean": np.asarray(Dmean, dtype=float).tolist(),
            "player1_share_gap": np.asarray(model["d1"] - Dmean, dtype=float).tolist(),
            "player2_share_gap": np.asarray(model["d2"] - Dmean, dtype=float).tolist(),
        },
        "noise_state_irfs": noise_state_irfs,
        "value_irfs": {name: np.asarray(value, dtype=float).tolist() for name, value in val.items()},
        "foc_terms": {
            "estimated_mispricing_demand": {
                "player1": np.asarray(targets["mispricing_demand1"], dtype=float).tolist(),
                "player2": np.asarray(targets["mispricing_demand2"], dtype=float).tolist(),
            },
            "estimated_calH": {
                "player1": np.asarray(targets["calH1"], dtype=float).tolist(),
                "player2": np.asarray(targets["calH2"], dtype=float).tolist(),
            },
            "foc_lhs": {
                "player1": np.asarray(targets["foc_lhs1"], dtype=float).tolist(),
                "player2": np.asarray(targets["foc_lhs2"], dtype=float).tolist(),
            },
            "foc_mismatch": {
                "player1": np.asarray(targets["foc_mismatch1"], dtype=float).tolist(),
                "player2": np.asarray(targets["foc_mismatch2"], dtype=float).tolist(),
            },
            "raw_future_profit_effect": {
                "player1": np.asarray(targets["ro1"]["density_total"], dtype=float).tolist(),
                "player2": np.asarray(targets["ro2"]["density_total"], dtype=float).tolist(),
            },
            "readout_components": {
                "player1": {
                    "old_market": np.asarray(targets["ro1"]["old_market"], dtype=float).tolist(),
                    "old_opponent": np.asarray(targets["ro1"]["old_opponent"], dtype=float).tolist(),
                    "diag_market": np.asarray(targets["ro1"]["diag_market"], dtype=float).tolist(),
                    "diag_opponent": np.asarray(targets["ro1"]["diag_opponent"], dtype=float).tolist(),
                    "R": float(targets["ro1"]["R"]),
                },
                "player2": {
                    "old_market": np.asarray(targets["ro2"]["old_market"], dtype=float).tolist(),
                    "old_opponent": np.asarray(targets["ro2"]["old_opponent"], dtype=float).tolist(),
                    "diag_market": np.asarray(targets["ro2"]["diag_market"], dtype=float).tolist(),
                    "diag_opponent": np.asarray(targets["ro2"]["diag_opponent"], dtype=float).tolist(),
                    "R": float(targets["ro2"]["R"]),
                },
            },
        },
        "filter_closure_residuals": filter_residuals,
        "vtilde_projection_residuals": vtilde_residuals,
        "zero_filter_residuals": zero_filter_residuals,
        "filter_tilde": {name: np.asarray(value, dtype=float).tolist() for name, value in u.items()},
        "unresolved_value": {
            "market_maker": np.asarray(u["vtilde0"], dtype=float).tolist(),
            "trader1": np.asarray(u["vtilde1"], dtype=float).tolist(),
            "trader2": np.asarray(u["vtilde2"], dtype=float).tolist(),
        },
        "unresolved_total_demand": {
            "market_maker": np.asarray(u["dtot_tilde0"], dtype=float).tolist(),
            "trader1": np.asarray(u["dtot_tilde1"], dtype=float).tolist(),
            "trader2": np.asarray(u["dtot_tilde2"], dtype=float).tolist(),
        },
        "ctilde": {
            "ce0": np.asarray(ce["ce0"][:, 0, :], dtype=float).tolist(),
            "ce1_z": np.asarray(ce["ce1_z"], dtype=float).tolist(),
            "ce1_y": np.asarray(ce["ce1_y"], dtype=float).tolist(),
            "ce2_z": np.asarray(ce["ce2_z"], dtype=float).tolist(),
            "ce2_y": np.asarray(ce["ce2_y"], dtype=float).tolist(),
        },
        "model": {
            "m0": np.asarray(model["m0"], dtype=float).tolist(),
            "primitive_V": np.asarray(model["primitive_V"], dtype=float).tolist(),
            "primitive_Vtilde0": np.asarray(model["primitive_Vtilde0"], dtype=float).tolist(),
            "primitive_Vtilde1": np.asarray(model["primitive_Vtilde1"], dtype=float).tolist(),
            "primitive_Vtilde2": np.asarray(model["primitive_Vtilde2"], dtype=float).tolist(),
            "c0": np.asarray(model["c0"], dtype=float).tolist(),
            "cY1": np.asarray(model["cY1"], dtype=float).tolist(),
            "cY2": np.asarray(model["cY2"], dtype=float).tolist(),
        },
        "H": {name: np.asarray(H[name], dtype=float).tolist() for name in BLOCK_NAMES},
        "diagnostics": diag,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def make_cfg(args):
    grid = np.linspace(0.0, args.L, args.N, dtype=np.float64)
    h = args.L / max(args.N - 1, 1)
    weights = kb.simpson_weights(args.N, h)
    integral_quad = args.integral_quad if args.integral_quad > 0 else max(args.N, 32)
    if integral_quad % 2 == 1:
        integral_quad += 1
    quad_nodes, quad_weights = np.polynomial.legendre.leggauss(integral_quad)
    quad_grid = 0.5 * args.L * (quad_nodes + 1.0)
    quad_weights = 0.5 * args.L * quad_weights
    inner_quad = integral_quad + 2
    if inner_quad % 2 == 1:
        inner_quad += 1
    inner_nodes, inner_weights = np.polynomial.legendre.leggauss(inner_quad)
    inner_grid = 0.5 * args.L * (inner_nodes + 1.0)
    inner_weights = 0.5 * args.L * inner_weights
    theta = np.linspace(0.0, math.pi, args.ode_N if args.ode_N > 0 else args.N, dtype=np.float64)
    ode_grid = 0.5 * args.L * (1.0 - np.cos(theta))
    filter_nodes, filter_weights = np.polynomial.legendre.leggauss(args.filter_quad)
    cfg = vars(args).copy()
    cfg["integral_quad"] = int(integral_quad)
    cfg["inner_quad"] = int(inner_quad)
    cfg["grid"] = jnp.asarray(grid.astype(np.float32))
    cfg["weights"] = jnp.asarray(weights.astype(np.float32))
    cfg["quad_grid"] = jnp.asarray(quad_grid.astype(np.float32))
    cfg["quad_weights"] = jnp.asarray(quad_weights.astype(np.float32))
    cfg["inner_grid"] = jnp.asarray(inner_grid.astype(np.float32))
    cfg["inner_weights"] = jnp.asarray(inner_weights.astype(np.float32))
    cfg["ode_grid"] = jnp.asarray(ode_grid.astype(np.float32))
    cfg["filter_nodes"] = jnp.asarray(filter_nodes.astype(np.float32))
    cfg["filter_weights"] = jnp.asarray(filter_weights.astype(np.float32))
    return cfg


def init_params(args):
    key = jax.random.PRNGKey(args.seed)
    kD, kU, kR, kH = jax.random.split(key, 4)
    response_hidden = args.response_hidden if args.response_hidden > 0 else args.hidden
    if args.init_adjoint_from and os.path.exists(args.init_adjoint_from):
        h_params = load_params(args.init_adjoint_from)["H"]
    else:
        h_params = init_mlp(kH, [1] + [args.hidden] * args.depth + [len(BLOCK_NAMES) * D * D], args.last_scale)
    return {
        "D": init_mlp(kD, [1] + [args.hidden] * args.depth + [2 * D], args.last_scale),
        "U": init_mlp(kU, [1] + [args.hidden] * args.depth + [6 * D], args.last_scale),
        "R": init_mlp(kR, [1] + [response_hidden] * args.response_depth + [3 * D], args.response_last_scale),
        "H": h_params,
    }


def same_mlp_shapes(a, b):
    if len(a) != len(b):
        return False
    return all(
        layer_a["W"].shape == layer_b["W"].shape
        and layer_a["b"].shape == layer_b["b"].shape
        for layer_a, layer_b in zip(a, b)
    )


def migrate_c_to_u(c_layers, u_layers, args):
    """Initialize new Vtilde/Dtot_tilde net from legacy ctilde net."""
    if len(c_layers) != len(u_layers):
        return u_layers
    for c_layer, u_layer in zip(c_layers[:-1], u_layers[:-1]):
        if c_layer["W"].shape != u_layer["W"].shape or c_layer["b"].shape != u_layer["b"].shape:
            return u_layers
    c_last = c_layers[-1]
    u_last = u_layers[-1]
    if c_last["W"].shape[1] != 5 * D or u_last["W"].shape[1] != 6 * D:
        return u_layers

    out = [{"W": layer["W"], "b": layer["b"]} for layer in c_layers[:-1]]
    W = jnp.zeros_like(u_last["W"])
    b = jnp.zeros_like(u_last["b"])
    scale = float(args.ctilde_scale) / max(float(args.tilde_scale), 1e-12)
    mappings = [
        (1, 2, scale * float(args.std_y1) / max(float(args.gamma1), 1e-12)),  # vtilde1 <- ce1_y
        (2, 4, scale * float(args.std_y2) / max(float(args.gamma2), 1e-12)),  # vtilde2 <- ce2_y
        (3, 0, scale * float(args.std_z)),  # Dtot_tilde0 <- ce0
        (4, 1, scale * float(args.std_z)),  # Dtot_tilde1 <- ce1_z
        (5, 3, scale * float(args.std_z)),  # Dtot_tilde2 <- ce2_z
    ]
    for u_block, c_block, factor in mappings:
        u_slice = slice(u_block * D, (u_block + 1) * D)
        c_slice = slice(c_block * D, (c_block + 1) * D)
        W = W.at[:, u_slice].set(factor * c_last["W"][:, c_slice])
        b = b.at[u_slice].set(factor * c_last["b"][c_slice])
    out.append({"W": W, "b": b})
    return out


def load_or_init_params(args):
    params = init_params(args)
    if not args.init_from:
        return params
    loaded = load_params(args.init_from)
    for name in params:
        if name in loaded and same_mlp_shapes(params[name], loaded[name]):
            params[name] = loaded[name]
    if args.migrate_legacy_ctilde and "U" not in loaded and "C" in loaded:
        params["U"] = migrate_c_to_u(loaded["C"], params["U"], args)
    return params


def source_diagnostic_step(path):
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            diag = json.load(f).get("diagnostics", {})
        return int(diag.get("step", 0) or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Coupled stationary Kyle-Back equilibrium PINN.")
    ap.add_argument("--out", default="data/kyle_back_equilibrium_pinn.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--response-hidden", type=int, default=0,
                    help="Hidden width for the separate response/readout net; 0 reuses --hidden.")
    ap.add_argument("--response-depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-decay", type=float, default=0.7)
    ap.add_argument("--weight-decay", type=float, default=1e-7)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--init-from", default="")
    ap.add_argument("--init-adjoint-from", default="data/kyle_back_pinn.json")
    ap.add_argument("--migrate-legacy-ctilde", action="store_true",
                    help="Opt-in migration from old learned ctilde nets; usually unstable.")
    ap.add_argument("--N", type=int, default=81)
    ap.add_argument("--ode-N", type=int, default=81)
    ap.add_argument("--integral-quad", type=int, default=0)
    ap.add_argument("--filter-quad", type=int, default=13)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--std-v", type=float, default=1.0)
    ap.add_argument("--std-z", type=float, default=1.0)
    ap.add_argument("--std-y1", type=float, default=1.0)
    ap.add_argument("--std-y2", type=float, default=1.0)
    ap.add_argument("--gamma1", type=float, default=3.0)
    ap.add_argument("--gamma2", type=float, default=10.0)
    ap.add_argument("--lambda-target", type=float, default=0.4)
    ap.add_argument("--d1-v", type=float, default=0.65)
    ap.add_argument("--d2-v", type=float, default=0.55)
    ap.add_argument("--d1-z", type=float, default=-0.10)
    ap.add_argument("--d2-z", type=float, default=-0.08)
    ap.add_argument("--d1-y", type=float, default=0.30)
    ap.add_argument("--d2-y", type=float, default=0.28)
    ap.add_argument("--d-decay", type=float, default=0.95)
    ap.add_argument("--diag-ramp", type=float, default=4.0)
    ap.add_argument("--private-decay", type=float, default=1.20)
    ap.add_argument("--birth-baseline", type=float, default=0.0, help="Deprecated; zero-lag private-signal birth is imposed as an atom.")
    ap.add_argument("--birth-decay", type=float, default=-1.0, help="Deprecated; zero-lag private-signal birth is imposed as an atom.")
    ap.add_argument("--filter-decay", type=float, default=1.05)
    ap.add_argument("--cy-v", type=float, default=0.35)
    ap.add_argument("--ce0-v", type=float, default=0.42)
    ap.add_argument("--ce0-z", type=float, default=0.22)
    ap.add_argument("--ce0-y1", type=float, default=0.04)
    ap.add_argument("--ce0-y2", type=float, default=0.03)
    ap.add_argument("--ce-trader-z-private", type=float, default=0.08)
    ap.add_argument("--ce-y-v", type=float, default=0.30)
    ap.add_argument("--ce-y-private", type=float, default=0.55)
    ap.add_argument("--policy-scale", type=float, default=0.25)
    ap.add_argument("--tilde-scale", type=float, default=0.20)
    ap.add_argument("--response-scale", type=float, default=0.05)
    ap.add_argument("--response-tail-decay", type=float, default=0.08)
    ap.add_argument("--ctilde-scale", type=float, default=0.20, help="Deprecated; ctilde is derived from Vtilde and Dtot_tilde.")
    ap.add_argument("--tail-decay", type=float, default=0.08)
    ap.add_argument("--last-scale", type=float, default=1e-3)
    ap.add_argument("--response-last-scale", type=float, default=1e-3)
    ap.add_argument("--w-filter", type=float, default=1.0)
    ap.add_argument("--w-filter-zero", type=float, default=10.0, help="Deprecated/no-op; zero-lag checks are diagnostics only.")
    ap.add_argument("--w-filter-projector", type=float, default=0.02)
    ap.add_argument("--w-policy-proj", type=float, default=0.25)
    ap.add_argument("--w-policy-zero", type=float, default=1.0)
    ap.add_argument("--w-adjoint", type=float, default=1.0)
    ap.add_argument("--w-foc", type=float, default=1.0)
    ap.add_argument("--w-optimality", type=float, default=1.0)
    ap.add_argument("--w-response-consistency", type=float, default=1.0)
    ap.add_argument("--w-lambda", type=float, default=1.0)
    ap.add_argument("--w-value-proj", type=float, default=1.0)
    ap.add_argument("--w-mm-price-shape", type=float, default=0.0, help="Deprecated/no-op; price-shape checks are diagnostics only.")
    ap.add_argument("--w-tail", type=float, default=0.05)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = make_cfg(args)
    params = load_or_init_params(args)
    n_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))
    print(
        f"kyle-back equilibrium PINN params={n_params} N={args.N} L={args.L} "
        f"filter_quad={args.filter_quad}",
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
            worst = sorted(block_rms.items(), key=lambda kv: kv[1], reverse=True)[:4]
            print(
                f"step={step} loss={float(loss):.4e} rms={rms:.4e} "
                f"grad={float(grad_norm):.3e} elapsed={elapsed:.1f}s "
                + " ".join(f"{k}={v:.2e}" for k, v in worst),
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
    final_step = args.steps if args.steps > 0 else source_diagnostic_step(args.init_from)
    final_diag = diagnostics(params, cfg, final_step, time.perf_counter() - t0, loss, metrics, 0.0)
    final_diag["best"] = best
    save_json(args.out, params, cfg, args, final_diag)
    print(json.dumps(final_diag, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
