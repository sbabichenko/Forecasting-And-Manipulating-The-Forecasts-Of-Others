#!/usr/bin/env python3
"""Lag-reduced PINN for stationary Kyle--Back forward IRFs.

This is the LQG-style alternative to the full unresolved-kernel Kyle--Back PINN.
For a fixed scalar two-trader candidate environment, the network learns only
three one-lag residual readouts for each deviating trader:

    r0(s), rz(s), ry(s)

The two-lag old-history density and the diagonal atom are then reconstructed by
the stationary transport/birth formulas.  This keeps the direct Pi terms and the
zero-lag birth geometry out of the network's degrees of freedom.

This script is intentionally a forward-IRF diagnostic, not a full equilibrium
solver.  The candidate policies and unresolved observation rows come from
``train_kyle_back_pinn.candidate_kernels``.
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

import train_kyle_back_pinn as kb


METRIC_NAMES = [
    "readout_player1_spike",
    "readout_player2_spike",
    "initial_price_player1_spike",
    "initial_price_player2_spike",
    "tail_price",
    "regularization",
]


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
    return h @ params[-1]["W"] + params[-1]["b"]


def mse(x):
    return jnp.mean(jnp.square(x))


def tree_to_jsonable(params):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32).tolist(), params)


def load_params(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload["nets"]
    return {
        name: [
            {
                "W": jnp.asarray(layer["W"], dtype=jnp.float32),
                "b": jnp.asarray(layer["b"], dtype=jnp.float32),
            }
            for layer in layers
        ]
        for name, layers in raw.items()
    }


def net_name(dev: int) -> str:
    return f"dev{int(dev)}"


def lag_to_coord(s, cfg):
    s = jnp.asarray(s, dtype=jnp.float32)
    if cfg["coord_map"] == "rational":
        return 2.0 * s / (s + cfg["coord_scale"]) - 1.0
    return 2.0 * s / cfg["train_L"] - 1.0


def eval_residual(params, dev: int, s, cfg):
    s = jnp.asarray(s, dtype=jnp.float32)
    coord = lag_to_coord(s, cfg)[..., None]
    y = mlp(params[net_name(dev)], coord.reshape((-1, 1)))
    return cfg["output_scale"] * y.reshape(s.shape + (3,))


def opponent_spec(dev: int):
    if dev == 1:
        return {
            "D": "D2",
            "c_z": "c2_z",
            "c_y": "cY2",
            "ce_z": "ce2_z",
            "ce_y": "ce2_y",
            "std_y": "std_y2",
            "ey": kb.EY2,
            "opponent_label": "trader2",
        }
    return {
        "D": "D1",
        "c_z": "c1_z",
        "c_y": "cY1",
        "ce_z": "ce1_z",
        "ce_y": "ce1_y",
        "std_y": "std_y1",
        "ey": kb.EY1,
        "opponent_label": "trader1",
    }


def _reshape_kernel_block(block, target_shape):
    return block.reshape(target_shape + (kb.D,))


def response_density_pairs(params, dev: int, s_pair, a_pair, cfg):
    """Old-history density U(s,a) for paired future-lag/age arrays."""
    spec = opponent_spec(dev)
    s_pair = jnp.asarray(s_pair, dtype=jnp.float32)
    a_pair = jnp.asarray(a_pair, dtype=jnp.float32)
    unit = cfg["inner_nodes"]
    unit_w = cfg["inner_weights"]

    lo = jnp.maximum(0.0, s_pair - a_pair)
    hi = s_pair
    width = jnp.maximum(hi - lo, 0.0)
    u = lo[..., None] + 0.5 * width[..., None] * (unit + 1.0)
    w = 0.5 * width[..., None] * unit_w
    lag_arg = a_pair[..., None] - s_pair[..., None] + u
    lag_pos = jnp.maximum(lag_arg, 0.0)

    r_u = eval_residual(params, dev, u.reshape(-1), cfg).reshape(u.shape + (3,))
    k_lag = kb.candidate_kernels(lag_pos.reshape(-1), cfg)
    ce0 = _reshape_kernel_block(k_lag["ce0"], lag_pos.shape)
    ce_z = _reshape_kernel_block(k_lag[spec["ce_z"]], lag_pos.shape)
    ce_y = _reshape_kernel_block(k_lag[spec["ce_y"]], lag_pos.shape)

    int0 = jnp.einsum("...q,...q,...qd->...d", w, r_u[..., 0], ce0)
    int_opp = (
        jnp.einsum("...q,...q,...qd->...d", w, r_u[..., 1], ce_z)
        + jnp.einsum("...q,...q,...qd->...d", w, r_u[..., 2], ce_y)
    )

    initial_lag = jnp.maximum(a_pair - s_pair, 0.0)
    boundary_lag = jnp.maximum(s_pair - a_pair, 0.0)
    initial_mask = (a_pair >= s_pair).astype(jnp.float32)
    boundary_mask = 1.0 - initial_mask

    k_initial = kb.candidate_kernels(initial_lag.reshape(-1), cfg)
    phi0 = _reshape_kernel_block(k_initial["ce0"] / cfg["std_z"], initial_lag.shape)
    phi_opp = _reshape_kernel_block(k_initial[spec["ce_z"]] / cfg["std_z"], initial_lag.shape)

    r_boundary = eval_residual(params, dev, boundary_lag.reshape(-1), cfg).reshape(
        boundary_lag.shape + (3,)
    )
    birth0 = r_boundary[..., 0, None] * kb.EZ
    birth_opp = (
        r_boundary[..., 1, None] * kb.EZ
        + r_boundary[..., 2, None] * cfg[spec["std_y"]] * spec["ey"]
    )

    U0 = initial_mask[..., None] * phi0 + boundary_mask[..., None] * birth0 + int0
    Uopp = initial_mask[..., None] * phi_opp + boundary_mask[..., None] * birth_opp + int_opp
    return U0, Uopp


def atom_from_residual(params, dev: int, s_eval, cfg):
    """Diagonal atom A(s) from residual functions.

    There is an order-flow birth in the public row and in the opponent's
    order-flow row.  Value noise has no direct birth term.
    """
    spec = opponent_spec(dev)
    s_eval = jnp.asarray(s_eval, dtype=jnp.float32)
    unit = cfg["inner_nodes"]
    unit_w = cfg["inner_weights"]
    u_atom = 0.5 * s_eval[:, None] * (unit[None, :] + 1.0)
    atom_w = 0.5 * s_eval[:, None] * unit_w[None, :]

    r_atom = eval_residual(params, dev, u_atom.reshape(-1), cfg).reshape(u_atom.shape + (3,))
    k_u = kb.candidate_kernels(u_atom.reshape(-1), cfg)
    ce0_atom = _reshape_kernel_block(k_u["ce0"], u_atom.shape)
    cez_atom = _reshape_kernel_block(k_u[spec["ce_z"]], u_atom.shape)
    cey_atom = _reshape_kernel_block(k_u[spec["ce_y"]], u_atom.shape)

    A0 = kb.EZ / cfg["std_z"] + jnp.einsum("sq,sq,sqd->sd", atom_w, r_atom[..., 0], ce0_atom)
    Aopp = (
        kb.EZ / cfg["std_z"]
        + jnp.einsum("sq,sq,sqd->sd", atom_w, r_atom[..., 1], cez_atom)
        + jnp.einsum("sq,sq,sqd->sd", atom_w, r_atom[..., 2], cey_atom)
    )
    return A0, Aopp


def response_from_residual(params, dev: int, s_eval, a_eval, cfg):
    """Build U(s,a) and A(s) from residual functions."""
    s_eval = jnp.asarray(s_eval, dtype=jnp.float32)
    a_eval = jnp.asarray(a_eval, dtype=jnp.float32)
    s_pair = jnp.broadcast_to(s_eval[:, None], (s_eval.shape[0], a_eval.shape[0]))
    a_pair = jnp.broadcast_to(a_eval[None, :], s_pair.shape)
    U0, Uopp = response_density_pairs(params, dev, s_pair, a_pair, cfg)
    A0, Aopp = atom_from_residual(params, dev, s_eval, cfg)
    return U0, Uopp, A0, Aopp


def lambda_value(cfg):
    x = 0.5 * cfg["L"] * (cfg["readout_nodes"] + 1.0)
    w = 0.5 * cfg["L"] * cfg["readout_weights"]
    k_x = kb.candidate_kernels(x, cfg)
    return (cfg["std_v"] / cfg["std_z"]) * jnp.einsum("q,qd,d->", w, k_x["ce0"], kb.EV)


def readout(params, dev: int, s_eval, cfg, include_grid: bool = False):
    spec = opponent_spec(dev)
    s_eval = jnp.asarray(s_eval, dtype=jnp.float32)
    nodes = cfg["readout_nodes"]
    weights = cfg["readout_weights"]
    s_col = s_eval[:, None]

    # Split the age integral at a=s so the side-birth branch is not smeared
    # across a quadrature interval.
    left_width = jnp.maximum(s_col, 0.0)
    right_width = jnp.maximum(cfg["L"] - s_col, 0.0)
    a_left = 0.5 * left_width * (nodes[None, :] + 1.0)
    w_left = 0.5 * left_width * weights[None, :]
    a_right = s_col + 0.5 * right_width * (nodes[None, :] + 1.0)
    w_right = 0.5 * right_width * weights[None, :]
    a_pair = jnp.concatenate([a_left, a_right], axis=1)
    w_pair = jnp.concatenate([w_left, w_right], axis=1)
    s_pair = jnp.broadcast_to(s_col, a_pair.shape)

    U0_quad, Uopp_quad = response_density_pairs(params, dev, s_pair, a_pair, cfg)
    A0, Aopp = atom_from_residual(params, dev, s_eval, cfg)

    k_pair = kb.candidate_kernels(a_pair.reshape(-1), cfg)
    D_pair = _reshape_kernel_block(k_pair[spec["D"]], a_pair.shape)
    c0_pair = _reshape_kernel_block(k_pair["c0"], a_pair.shape)
    cz_pair = _reshape_kernel_block(k_pair[spec["c_z"]], a_pair.shape)
    cy_pair = _reshape_kernel_block(k_pair[spec["c_y"]], a_pair.shape)
    k_s = kb.candidate_kernels(s_eval, cfg)

    density_demand = jnp.einsum("sq,sqd,sqd->s", w_pair, Uopp_quad, D_pair)
    atom_demand = jnp.einsum("sd,sd->s", Aopp, k_s[spec["D"]])
    delta_demand = density_demand + atom_demand

    self0 = (
        jnp.einsum("sq,sqd,sqd->s", w_pair, U0_quad, c0_pair)
        + jnp.einsum("sd,sd->s", A0, k_s["c0"])
    )
    self_z = (
        jnp.einsum("sq,sqd,sqd->s", w_pair, Uopp_quad, cz_pair)
        + jnp.einsum("sd,sd->s", Aopp, k_s[spec["c_z"]])
    )
    self_y = (
        jnp.einsum("sq,sqd,sqd->s", w_pair, Uopp_quad, cy_pair)
        + jnp.einsum("sd,sd->s", Aopp, k_s[spec["c_y"]])
    )
    target = jnp.stack(
        [
            delta_demand / cfg["std_z"] - self0,
            delta_demand / cfg["std_z"] - self_z,
            -self_y,
        ],
        axis=-1,
    )
    r = eval_residual(params, dev, s_eval, cfg)

    density_price = cfg["std_v"] * jnp.einsum("sq,sqd,d->s", w_pair, U0_quad, kb.EV)
    atom_price = cfg["std_v"] * jnp.einsum("sd,d->s", A0, kb.EV)
    price = density_price + atom_price
    density_state0 = jnp.einsum("sq,sqd->sd", w_pair, U0_quad)
    density_state_opp = jnp.einsum("sq,sqd->sd", w_pair, Uopp_quad)
    state0 = density_state0 + A0
    state_opp = density_state_opp + Aopp

    U0, Uopp = U0_quad, Uopp_quad
    if include_grid:
        U0, Uopp, _, _ = response_from_residual(params, dev, s_eval, cfg["a_grid"], cfg)

    return {
        "residual": r,
        "target": target,
        "delta_demand": delta_demand,
        "price": price,
        "density_price": density_price,
        "atom_price": atom_price,
        "density_state0": density_state0,
        "density_state_opp": density_state_opp,
        "state0": state0,
        "state_opp": state_opp,
        "v_estimate0": cfg["std_v"] * state0[:, 0],
        "v_estimate_opp": cfg["std_v"] * state_opp[:, 0],
        "U0": U0,
        "Uopp": Uopp,
        "A0": A0,
        "Aopp": Aopp,
    }


def residual_metrics(params, cfg):
    s = cfg["s_colloc"]
    ro1 = readout(params, 1, s, cfg)
    ro2 = readout(params, 2, s, cfg)
    lam = lambda_value(cfg)
    blocks = [
        cfg["w_readout"] * mse(ro1["residual"] - ro1["target"]),
        cfg["w_readout"] * mse(ro2["residual"] - ro2["target"]),
        cfg["w_initial_price"] * jnp.square(ro1["price"][0] - lam),
        cfg["w_initial_price"] * jnp.square(ro2["price"][0] - lam),
        cfg["w_tail"] * (jnp.square(ro1["price"][-1]) + jnp.square(ro2["price"][-1])),
        cfg["w_reg"] * (mse(ro1["residual"]) + mse(ro2["residual"])),
    ]
    return jnp.stack(blocks)


def value_shock_irf(cfg, s_eval):
    """Candidate-profile belief/action curves for a unit V shock.

    These are not trained by the lag-reduced residual net; they are included so
    the browser can keep the same plot layout while this diagnostic solver is
    being used.
    """
    s_eval = jnp.asarray(s_eval, dtype=jnp.float32)
    true_v = cfg["std_v"] * jnp.ones_like(s_eval)
    price_speed = jnp.maximum(cfg["filter_decay"], 1e-4)
    trader1_speed = jnp.maximum(cfg["filter_decay"] + cfg["cy_v"] * cfg["gamma1"], 1e-4)
    trader2_speed = jnp.maximum(cfg["filter_decay"] + cfg["cy_v"] * cfg["gamma2"], 1e-4)
    price = true_v * (1.0 - jnp.exp(-price_speed * s_eval))
    trader1_estimate = true_v * (1.0 - jnp.exp(-trader1_speed * s_eval))
    trader2_estimate = true_v * (1.0 - jnp.exp(-trader2_speed * s_eval))
    k = kb.candidate_kernels(s_eval, cfg)
    return {
        "true_value": true_v[:, None] * kb.EV,
        "price": price[:, None] * kb.EV,
        "trader1_estimate": trader1_estimate[:, None] * kb.EV,
        "trader2_estimate": trader2_estimate[:, None] * kb.EV,
        "mispricing_gap1": (trader1_estimate - price)[:, None] * kb.EV,
        "mispricing_gap2": (trader2_estimate - price)[:, None] * kb.EV,
        "demand1": k["D1"],
        "demand2": k["D2"],
        "demand_source": "fixed candidate policy",
    }


def _source_block(label, state0, state1, state2, demand1, demand2):
    return {
        "label": label,
        "kind": "lag_reduced_forward_irf",
        "state_market_maker": state0,
        "state_trader1": state1,
        "state_trader2": state2,
        "price": state0[:, 0],
        "trader1_estimate": state1[:, 0],
        "trader2_estimate": state2[:, 0],
        "demand1": demand1,
        "demand2": demand2,
    }


def noise_source_irfs(cfg, s_eval, ro1, ro2, vshock):
    """Line-based primitive-source diagnostics for the browser UI."""
    s_eval = jnp.asarray(s_eval, dtype=jnp.float32)
    zeros = jnp.zeros((s_eval.shape[0], kb.D), dtype=jnp.float32)
    k = kb.candidate_kernels(s_eval, cfg)
    direct_z = jnp.exp(-cfg["filter_decay"] * s_eval)[:, None] * kb.EZ
    direct_y = jnp.exp(-cfg["private_decay"] * s_eval)[:, None]

    sources = {
        "v_noise": _source_block(
            "V noise",
            vshock["price"],
            vshock["trader1_estimate"],
            vshock["trader2_estimate"],
            k["D1"],
            k["D2"],
        ),
        "z_noise": _source_block(
            "Z noise",
            k["ce0"] + direct_z,
            k["ce1_z"] + direct_z,
            k["ce2_z"] + direct_z,
            k["D1"][:, 1],
            k["D2"][:, 1],
        ),
        "y1_noise": _source_block(
            "Y1 noise",
            zeros,
            k["ce1_y"] + direct_y * kb.EY1,
            zeros,
            k["D1"][:, 2],
            jnp.zeros_like(s_eval),
        ),
        "y2_noise": _source_block(
            "Y2 noise",
            zeros,
            zeros,
            k["ce2_y"] + direct_y * kb.EY2,
            jnp.zeros_like(s_eval),
            k["D2"][:, 3],
        ),
        "trader1_order": _source_block(
            "trader 1 deviation order",
            ro1["state0"],
            zeros,
            ro1["state_opp"],
            jnp.ones_like(s_eval),
            ro1["delta_demand"],
        ),
        "trader2_order": _source_block(
            "trader 2 deviation order",
            ro2["state0"],
            ro2["state_opp"],
            zeros,
            ro2["delta_demand"],
            jnp.ones_like(s_eval),
        ),
    }
    return sources


def noise_state_irf_matrices(sources):
    mats = {}
    for observer_key, source_key in [
        ("market_maker", "state_market_maker"),
        ("trader1", "state_trader1"),
        ("trader2", "state_trader2"),
    ]:
        mats[observer_key] = jnp.stack(
            [
                sources["v_noise"][source_key],
                sources["z_noise"][source_key],
                sources["y1_noise"][source_key],
                sources["y2_noise"][source_key],
            ],
            axis=1,
        )
    return mats


def diagnostics(params, cfg, step, elapsed, loss, metrics, grad_norm):
    s = cfg["eval_grid"]
    ro1 = readout(params, 1, s, cfg)
    ro2 = readout(params, 2, s, cfg)
    lam = lambda_value(cfg)
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
        "readout_rms_unweighted": {
            "player1_spike": float(jnp.sqrt(mse(ro1["residual"] - ro1["target"]))),
            "player2_spike": float(jnp.sqrt(mse(ro2["residual"] - ro2["target"]))),
        },
        "price_initial": {
            "player1_spike": float(ro1["price"][0]),
            "player2_spike": float(ro2["price"][0]),
        },
        "price_tail": {
            "player1_spike": float(ro1["price"][-1]),
            "player2_spike": float(ro2["price"][-1]),
        },
        "opponent_demand_initial": {
            "player1_spike": float(ro1["delta_demand"][0]),
            "player2_spike": float(ro2["delta_demand"][0]),
        },
    }


def simpson_or_trap_grid(n: int, L: float, quadrature: str):
    if quadrature == "gauss":
        x, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * L * (x + 1.0), 0.5 * L * w
    grid = np.linspace(0.0, L, n, dtype=np.float64)
    h = L / max(n - 1, 1)
    if quadrature == "simpson":
        weights = kb.simpson_weights(n, h).astype(np.float64)
    else:
        weights = np.full(n, h, dtype=np.float64)
        weights[0] *= 0.5
        weights[-1] *= 0.5
    return grid, weights


def make_cfg(args):
    plot_L = args.plot_L if args.plot_L > 0.0 else args.L
    train_L = args.train_L if args.train_L > 0.0 else min(args.L, plot_L)
    a, weights = simpson_or_trap_grid(args.N, args.L, args.quadrature)
    s = np.linspace(0.0, train_L, args.s_N, dtype=np.float64)
    inner_nodes, inner_weights = np.polynomial.legendre.leggauss(args.inner_quad)
    readout_nodes, readout_weights = np.polynomial.legendre.leggauss(args.readout_quad)
    eval_grid = np.linspace(0.0, plot_L, args.eval_N, dtype=np.float64)

    cfg = vars(args).copy()
    cfg["plot_L"] = float(plot_L)
    cfg["train_L"] = float(train_L)
    cfg["a_grid"] = jnp.asarray(a.astype(np.float32))
    cfg["a_weights"] = jnp.asarray(weights.astype(np.float32))
    cfg["s_colloc"] = jnp.asarray(s.astype(np.float32))
    cfg["inner_nodes"] = jnp.asarray(inner_nodes.astype(np.float32))
    cfg["inner_weights"] = jnp.asarray(inner_weights.astype(np.float32))
    cfg["readout_nodes"] = jnp.asarray(readout_nodes.astype(np.float32))
    cfg["readout_weights"] = jnp.asarray(readout_weights.astype(np.float32))
    cfg["eval_grid"] = jnp.asarray(eval_grid.astype(np.float32))
    return cfg


def init_params(args):
    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key, 2)
    dims = [1] + [args.hidden] * args.depth + [3]
    return {
        "dev1": init_mlp(k1, dims, args.last_scale),
        "dev2": init_mlp(k2, dims, args.last_scale),
    }


def as_json_array(value):
    return np.asarray(value, dtype=float).tolist()


def _jsonable_block(block):
    return {
        k: (v if isinstance(v, str) else as_json_array(v))
        for k, v in block.items()
    }


def save_json(path: str, params, cfg, args, diag):
    grid = cfg["eval_grid"]
    ro1 = readout(params, 1, grid, cfg, include_grid=True)
    ro2 = readout(params, 2, grid, cfg, include_grid=True)
    vshock = value_shock_irf(cfg, grid)
    nshocks = noise_source_irfs(cfg, grid, ro1, ro2, vshock)
    noise_mats = noise_state_irf_matrices(nshocks)
    k = kb.candidate_kernels(grid, cfg)

    payload = {
        "version": 1,
        "kind": "kyle_back_lag_irf_pinn",
        "description": "Lag-reduced Kyle-Back forward IRF PINN: learns scalar residual readouts and reconstructs transport/birth responses analytically.",
        "assumptions": {
            "scope": "fixed candidate profile; forward IRF diagnostic, not a full equilibrium solve",
            "filter_only": True,
            "primitive_coordinates": ["V", "Z", "Y1", "Y2"],
            "direct_observation_geometry": "Pi0=EZ'EZ, Pi1=EZ'EZ+EY1'EY1, Pi2=EZ'EZ+EY2'EY2",
            "value_noise_direct_birth": False,
            "transport_and_birth_hard_coded": True,
        },
        "mathematical_conventions": {
            "learned_objects": "r0(s), rz(s), ry(s) for each deviating trader",
            "old_history_density": "constructed by stationary Volterra/transport formula",
            "zero_lag_indirect_history": "zero by construction",
        },
        "nets": tree_to_jsonable(params),
        "architecture": {
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": "tanh",
            "n_params": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))),
        },
        "grid": as_json_array(grid),
        "params": {
            k0: float(v)
            for k0, v in vars(args).items()
            if isinstance(v, (int, float)) and k0 not in {"steps", "log_every", "save_every", "seed"}
        },
        "fields": {
            "residual_readouts": ["r0", "rz", "ry"],
            "policy_noise_state": ["D1", "D2"],
            "noise_state_irfs": ["market_maker", "trader1", "trader2"],
        },
        "policy_noise_state": {
            "player1": as_json_array(k["D1"]),
            "player2": as_json_array(k["D2"]),
        },
        "primitive_demand": {
            "player1": as_json_array(k["D1"]),
            "player2": as_json_array(k["D2"]),
        },
        "calD": {
            "player1": as_json_array(k["D1"]),
            "player2": as_json_array(k["D2"]),
        },
        "value_irfs": {
            name: (value if isinstance(value, str) else as_json_array(value))
            for name, value in vshock.items()
        },
        "noise_sources": {
            name: _jsonable_block(block)
            for name, block in nshocks.items()
        },
        "noise_state_irfs": {
            name: as_json_array(value)
            for name, value in noise_mats.items()
        },
        "irf": {
            "player1_spike": {
                "residual": as_json_array(ro1["residual"]),
                "target": as_json_array(ro1["target"]),
                "density_market_maker": as_json_array(ro1["U0"]),
                "density_opponent": as_json_array(ro1["Uopp"]),
                "atom_market_maker": as_json_array(ro1["A0"]),
                "atom_opponent": as_json_array(ro1["Aopp"]),
                "price": as_json_array(ro1["price"]),
                "density_price": as_json_array(ro1["density_price"]),
                "atom_price": as_json_array(ro1["atom_price"]),
                "state_market_maker": as_json_array(ro1["state0"]),
                "state_opponent": as_json_array(ro1["state_opp"]),
                "v_estimate_market_maker": as_json_array(ro1["v_estimate0"]),
                "v_estimate_opponent": as_json_array(ro1["v_estimate_opp"]),
                "opponent_demand": as_json_array(ro1["delta_demand"]),
            },
            "player2_spike": {
                "residual": as_json_array(ro2["residual"]),
                "target": as_json_array(ro2["target"]),
                "density_market_maker": as_json_array(ro2["U0"]),
                "density_opponent": as_json_array(ro2["Uopp"]),
                "atom_market_maker": as_json_array(ro2["A0"]),
                "atom_opponent": as_json_array(ro2["Aopp"]),
                "price": as_json_array(ro2["price"]),
                "density_price": as_json_array(ro2["density_price"]),
                "atom_price": as_json_array(ro2["atom_price"]),
                "state_market_maker": as_json_array(ro2["state0"]),
                "state_opponent": as_json_array(ro2["state_opp"]),
                "v_estimate_market_maker": as_json_array(ro2["v_estimate0"]),
                "v_estimate_opponent": as_json_array(ro2["v_estimate_opp"]),
                "opponent_demand": as_json_array(ro2["delta_demand"]),
            },
        },
        "diagnostics": diag,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def parse_args():
    ap = argparse.ArgumentParser(description="Lag-reduced Kyle-Back forward IRF PINN.")
    ap.add_argument("--out", default="data/kyle_back_irf_lag_pinn.json")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-decay", type=float, default=0.6)
    ap.add_argument("--weight-decay", type=float, default=1e-7)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--init-from", default="", help="Resume from a saved lag-IRF PINN JSON export.")
    ap.add_argument("--N", type=int, default=81)
    ap.add_argument("--s-N", type=int, default=81)
    ap.add_argument("--inner-quad", type=int, default=11)
    ap.add_argument("--readout-quad", type=int, default=21)
    ap.add_argument("--eval-N", type=int, default=121)
    ap.add_argument("--L", type=float, default=24.0)
    ap.add_argument("--plot-L", type=float, default=8.0)
    ap.add_argument("--train-L", type=float, default=0.0)
    ap.add_argument("--coord-map", choices=["rational", "linear"], default="rational")
    ap.add_argument("--coord-scale", type=float, default=4.0)
    ap.add_argument("--quadrature", choices=["trapezoid", "simpson", "gauss"], default="simpson")

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

    ap.add_argument("--output-scale", type=float, default=0.25)
    ap.add_argument("--last-scale", type=float, default=1e-3)
    ap.add_argument("--w-readout", type=float, default=1.0)
    ap.add_argument("--w-initial-price", type=float, default=5.0)
    ap.add_argument("--w-tail", type=float, default=0.0)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2500)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = make_cfg(args)
    params = load_params(args.init_from) if args.init_from else init_params(args)
    n_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))
    print(
        f"kyle-back lag IRF PINN params={n_params} N={args.N} s_N={args.s_N} "
        f"inner_quad={args.inner_quad} readout_quad={args.readout_quad} "
        f"L={args.L} train_L={cfg['train_L']} plot_L={cfg['plot_L']}",
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
    print(json.dumps(final_diag, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
