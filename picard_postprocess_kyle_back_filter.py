#!/usr/bin/env python3
"""Picard postprocess for the fixed-policy Kyle--Back filter checkpoint.

This is a diagnostic, not a replacement training solver.  It freezes the
exported fixed exponential policies, initializes the unresolved kernels from
the NN checkpoint, and applies the deterministic filter fixed-point map on the
exported lag grid.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np


D_W = 4
EV = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
EZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
EY1 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
EY2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
EYE = np.eye(D_W, dtype=np.float64)


def observer_rows(observer: int) -> np.ndarray:
    if observer == 0:
        return np.stack([EZ], axis=0)
    if observer == 1:
        return np.stack([EZ, EY1], axis=0)
    if observer == 2:
        return np.stack([EZ, EY2], axis=0)
    raise ValueError(observer)


def interp_field(grid: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1)
    out = np.empty((flat.size, values.shape[-1]), dtype=np.float64)
    for k in range(values.shape[-1]):
        out[:, k] = np.interp(flat, grid, values[:, k])
    return out.reshape(x.shape + (values.shape[-1],))


def fixed_policy(grid: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    trace = (1.0 - np.exp(-float(params.get("fixed_policy_ramp", 4.0)) * grid))[:, None]
    tail = np.exp(-float(params["fixed_policy_decay"]) * grid)[:, None]
    d1 = (
        float(params["fixed_d1_v"]) * EV
        + float(params["fixed_d1_z"]) * EZ
        + float(params["fixed_d1_y"]) * EY1
    )
    d2 = (
        float(params["fixed_d2_v"]) * EV
        + float(params["fixed_d2_z"]) * EZ
        + float(params["fixed_d2_y"]) * EY2
    )
    return trace * tail * d1, trace * tail * d2


def ctilde_rows(obs: int, grid: np.ndarray, fields: dict, x: np.ndarray, params: dict) -> np.ndarray:
    gamma1 = float(params["gamma1"])
    gamma2 = float(params["gamma2"])
    std_z = float(params["std_z"])
    std_y1 = float(params["std_y1"])
    std_y2 = float(params["std_y2"])
    if obs == 0:
        return (interp_field(grid, fields["dtot_tilde0"], x) / std_z)[..., None, :]
    if obs == 1:
        z = interp_field(grid, fields["dtot_tilde1"], x) / std_z
        y = gamma1 * interp_field(grid, fields["vtilde1"], x) / std_y1
        return np.stack([z, y], axis=-2)
    if obs == 2:
        z = interp_field(grid, fields["dtot_tilde2"], x) / std_z
        y = gamma2 * interp_field(grid, fields["vtilde2"], x) / std_y2
        return np.stack([z, y], axis=-2)
    raise ValueError(obs)


def filter_kernel(
    obs: int,
    first_lag: np.ndarray,
    second_lag: np.ndarray,
    grid: np.ndarray,
    fields: dict,
    params: dict,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    first_lag = np.asarray(first_lag, dtype=np.float64)
    second_lag = np.asarray(second_lag, dtype=np.float64)
    first_ge_second = first_lag >= second_lag
    first_gt_second = first_lag > second_lag + 1e-10
    second_gt_first = second_lag > first_lag + 1e-10
    base = np.abs(first_lag - second_lag)
    width = np.minimum(first_lag, second_lag)
    rows = observer_rows(obs)

    c_base = ctilde_rows(obs, grid, fields, base, params)
    boundary_right = np.einsum("...rd,re->...de", c_base, rows)
    boundary_left = np.einsum("rd,...re->...de", rows, c_base)
    boundary = np.where(
        first_gt_second[..., None, None],
        boundary_right,
        np.where(second_gt_first[..., None, None], boundary_left, np.zeros_like(boundary_right)),
    )

    tau = 0.5 * width[..., None] * (nodes + 1.0)
    quad_w = 0.5 * width[..., None] * weights
    c_first = ctilde_rows(
        obs,
        grid,
        fields,
        np.where(first_ge_second[..., None], base[..., None] + tau, tau),
        params,
    )
    c_second = ctilde_rows(
        obs,
        grid,
        fields,
        np.where(first_ge_second[..., None], tau, base[..., None] + tau),
        params,
    )
    integral = np.einsum("...q,...qrd,...qre->...de", quad_w, c_first, c_second)
    return boundary + integral


def project_kernel(
    obs: int,
    kernel_eval: np.ndarray,
    kernel_hist: np.ndarray,
    eval_grid: np.ndarray,
    hist_grid: np.ndarray,
    hist_weights: np.ndarray,
    base_grid: np.ndarray,
    fields: dict,
    params: dict,
    nodes: np.ndarray,
    node_weights: np.ndarray,
    chunk: int,
) -> np.ndarray:
    rows = observer_rows(obs)
    pi = rows.T @ rows
    out = kernel_eval @ pi
    for start in range(0, eval_grid.size, chunk):
        stop = min(start + chunk, eval_grid.size)
        ev = eval_grid[start:stop]
        first = np.broadcast_to(hist_grid[:, None], (hist_grid.size, ev.size))
        second = np.broadcast_to(ev[None, :], first.shape)
        f = filter_kernel(obs, first, second, base_grid, fields, params, nodes, node_weights)
        indirect = np.einsum("h,hd,hlde->le", hist_weights, kernel_hist, f)
        indirect = np.where((ev > 1e-10)[:, None], indirect, np.zeros_like(indirect))
        out[start:stop] += indirect
    return out


def compute_model(grid: np.ndarray, fields: dict, params: dict, nodes: np.ndarray, node_weights: np.ndarray, chunk: int):
    std_v = float(params["std_v"])
    std_z = float(params["std_z"])
    std_y1 = float(params["std_y1"])
    std_y2 = float(params["std_y2"])
    gamma1 = float(params["gamma1"])
    gamma2 = float(params["gamma2"])
    weights = history_weights(grid)
    d1, d2 = fixed_policy(grid, params)
    cald1 = project_kernel(1, d1, d1, grid, grid, weights, grid, fields, params, nodes, node_weights, chunk)
    cald2 = project_kernel(2, d2, d2, grid, grid, weights, grid, fields, params, nodes, node_weights, chunk)
    dtot = cald1 + cald2
    primitive_v = np.broadcast_to(std_v * EV, dtot.shape)
    c0 = dtot / std_z
    cy1 = gamma1 * fields["vtilde0"] / std_y1
    cy2 = gamma2 * fields["vtilde0"] / std_y2
    return {
        "D1": d1,
        "D2": d2,
        "calD1": cald1,
        "calD2": cald2,
        "Dtot": dtot,
        "primitive_V": primitive_v,
        "c0": c0,
        "cY1": cy1,
        "cY2": cy2,
    }


def raw_rows(obs: int, model: dict) -> np.ndarray:
    if obs == 0:
        return model["c0"][:, None, :]
    if obs == 1:
        return np.stack([model["c0"], model["cY1"]], axis=1)
    if obs == 2:
        return np.stack([model["c0"], model["cY2"]], axis=1)
    raise ValueError(obs)


def picard_map(grid: np.ndarray, fields: dict, params: dict, nodes: np.ndarray, node_weights: np.ndarray, chunk: int):
    weights = history_weights(grid)
    model = compute_model(grid, fields, params, nodes, node_weights, chunk)
    next_fields = {}
    std_v = float(params["std_v"])
    std_z = float(params["std_z"])
    for obs in (0, 1, 2):
        rows = observer_rows(obs)
        pi = rows.T @ rows
        raw = raw_rows(obs, model)
        ce_new = np.empty_like(raw)
        for start in range(0, grid.size, chunk):
            stop = min(start + chunk, grid.size)
            ev = grid[start:stop]
            first = np.broadcast_to(grid[:, None], (grid.size, ev.size))
            second = np.broadcast_to(ev[None, :], first.shape)
            f = filter_kernel(obs, first, second, grid, fields, params, nodes, node_weights)
            direct = np.einsum("lrd,de->lre", raw[start:stop], EYE - pi)
            integral = np.einsum("h,hrd,hlde->lre", weights, raw, f)
            # Zero lag has no old-history inference term; only direct observed
            # innovation coordinates are known instantaneously through Pi.
            integral = np.where((ev > 1e-10)[:, None, None], integral, np.zeros_like(integral))
            ce_new[start:stop] = direct - integral
        v_eval = np.broadcast_to(std_v * EV, (grid.size, D_W))
        v_proj = project_kernel(obs, v_eval, v_eval, grid, grid, weights, grid, fields, params, nodes, node_weights, chunk)
        next_fields[f"vtilde{obs}"] = v_eval - v_proj
        next_fields[f"dtot_tilde{obs}"] = std_z * ce_new[:, 0, :]

    for obs in (0, 1, 2):
        next_fields[f"vtilde{obs}"][0] = std_v * EV
    return next_fields, model


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    w = np.empty_like(grid)
    w[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    w[0] = 0.5 * (grid[1] - grid[0])
    w[-1] = 0.5 * (grid[-1] - grid[-2])
    return w


def history_weights(grid: np.ndarray) -> np.ndarray:
    """Quadrature weights for old-history density integrals.

    The zero-lag trace is a birth boundary, not ordinary history density.
    Giving it positive quadrature mass feeds the birth term back into the
    Picard map as if it were interior history.
    """
    w = trapezoid_weights(grid)
    w[0] = 0.0
    return w


def residual_diagnostics(grid: np.ndarray, fields: dict, mapped: dict, model: dict, params: dict) -> dict:
    out = {}
    total = []
    for obs in (0, 1, 2):
        gamma = 0.0 if obs == 0 else float(params[f"gamma{obs}"])
        std_y = 1.0 if obs == 0 else float(params[f"std_y{obs}"])
        current_rows = [fields[f"dtot_tilde{obs}"] / float(params["std_z"])]
        mapped_rows = [mapped[f"dtot_tilde{obs}"] / float(params["std_z"])]
        if obs:
            current_rows.append(gamma * fields[f"vtilde{obs}"] / std_y)
            mapped_rows.append(gamma * mapped[f"vtilde{obs}"] / std_y)
        closure_resid = np.stack(current_rows, axis=1) - np.stack(mapped_rows, axis=1)
        v_resid = fields[f"vtilde{obs}"] - mapped[f"vtilde{obs}"]
        resid = np.concatenate([closure_resid.reshape(-1), v_resid.reshape(-1)])
        out[f"observer{obs}"] = float(np.sqrt(np.mean(resid * resid)))
        total.append(resid)
    all_resid = np.concatenate(total)
    return {
        "rms": float(np.sqrt(np.mean(all_resid * all_resid))),
        "closure_rms_unweighted": out,
    }


def as_list(x):
    return np.asarray(x, dtype=float).tolist()


def build_payload(source: dict, grid: np.ndarray, fields: dict, model: dict, mapped: dict, params: dict, args, nodes, node_weights):
    payload = copy.deepcopy(source)
    weights = history_weights(grid)
    std_v = float(params["std_v"])
    v_eval = np.broadcast_to(std_v * EV, (grid.size, D_W))
    price = v_eval - fields["vtilde0"]
    trader1 = v_eval - fields["vtilde1"]
    trader2 = v_eval - fields["vtilde2"]

    noise_state_irfs = {}
    eye = np.eye(D_W)
    for obs_name, obs in [("market_maker", 0), ("trader1", 1), ("trader2", 2)]:
        mats = []
        for ch in range(D_W):
            kernel = np.broadcast_to(eye[ch], (grid.size, D_W))
            mats.append(project_kernel(obs, kernel, kernel, grid, grid, weights, grid, fields, params, nodes, node_weights, args.chunk))
        noise_state_irfs[obs_name] = as_list(np.stack(mats, axis=1))

    mode_dtot = model["calD1"] + model["calD2"]
    mode_drel = model["calD1"] - model["calD2"]
    mode_mean = 0.5 * mode_dtot
    diagnostics = residual_diagnostics(grid, fields, mapped, model, params)
    diagnostics.update(
        {
            "source_checkpoint_rms": source.get("diagnostics", {}).get("rms"),
            "picard_iterations": args.iters,
            "picard_damping": args.damping,
            "picard_kernel_quad": args.kernel_quad,
            "picard_chunk": args.chunk,
            "lambda": float((std_v / float(params["std_z"])) * np.einsum("n,nd,d->", weights, fields["dtot_tilde0"] / float(params["std_z"]), EV)),
        }
    )

    payload["kind"] = "kyle_back_fixed_policy_filter_picard_postprocess"
    payload["description"] = "Picard postprocess of the fixed-policy Kyle-Back filter checkpoint initialized from the NN unresolved kernels."
    payload.setdefault("assumptions", {})["filter_only"] = True
    payload.setdefault("assumptions", {})["fixed_policy"] = True
    payload.setdefault("mathematical_conventions", {})["picard_postprocess"] = (
        "Fixed policy held constant; unresolved filter kernels updated by deterministic Picard map from the NN initialization."
    )
    payload["grid"] = as_list(grid)
    payload["lag"] = as_list(grid)
    payload["params"]["picard_iterations"] = args.iters
    payload["params"]["picard_damping"] = args.damping
    payload["policy_noise_state"] = {"player1": as_list(model["D1"]), "player2": as_list(model["D2"])}
    payload["primitive_demand"] = {"player1": as_list(model["calD1"]), "player2": as_list(model["calD2"])}
    payload["calD"] = {"player1": as_list(model["calD1"]), "player2": as_list(model["calD2"])}
    payload["mode_decomposition"] = {
        "Dtot": as_list(mode_dtot),
        "Drel": as_list(mode_drel),
        "Dmean": as_list(mode_mean),
        "player1_share_gap": as_list(model["calD1"] - mode_mean),
        "player2_share_gap": as_list(model["calD2"] - mode_mean),
    }
    payload["noise_state_irfs"] = noise_state_irfs
    payload["value_irfs"] = {
        "true_value": as_list(v_eval),
        "price": as_list(price),
        "trader1_estimate": as_list(trader1),
        "trader2_estimate": as_list(trader2),
        "mispricing_gap1": as_list(trader1 - price),
        "mispricing_gap2": as_list(trader2 - price),
        "price_residual_implied": as_list(price),
        "trader1_estimate_residual_implied": as_list(trader1),
        "trader2_estimate_residual_implied": as_list(trader2),
    }
    payload["filter_tilde"] = {name: as_list(fields[name]) for name in fields}
    payload["unresolved_value"] = {
        "market_maker": as_list(fields["vtilde0"]),
        "trader1": as_list(fields["vtilde1"]),
        "trader2": as_list(fields["vtilde2"]),
    }
    payload["unresolved_total_demand"] = {
        "market_maker": as_list(fields["dtot_tilde0"]),
        "trader1": as_list(fields["dtot_tilde1"]),
        "trader2": as_list(fields["dtot_tilde2"]),
    }
    payload["ctilde"] = {
        "ce0": as_list(fields["dtot_tilde0"] / float(params["std_z"])),
        "ce1_z": as_list(fields["dtot_tilde1"] / float(params["std_z"])),
        "ce1_y": as_list(float(params["gamma1"]) * fields["vtilde1"] / float(params["std_y1"])),
        "ce2_z": as_list(fields["dtot_tilde2"] / float(params["std_z"])),
        "ce2_y": as_list(float(params["gamma2"]) * fields["vtilde2"] / float(params["std_y2"])),
    }
    payload["model"] = {
        "primitive_V": as_list(model["primitive_V"]),
        "c0": as_list(model["c0"]),
        "cY1": as_list(model["cY1"]),
        "cY2": as_list(model["cY2"]),
    }
    payload["diagnostics"] = diagnostics
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kyle_back_fixed_policy_filter_N41.json")
    ap.add_argument("--out", default="data/kyle_back_fixed_policy_filter_N41_picard.json")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--kernel-quad", type=int, default=17)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    source = json.loads(Path(args.input).read_text())
    params = source["params"]
    grid = np.asarray(source["grid"], dtype=np.float64)
    fields = {
        "vtilde0": np.asarray(source["filter_tilde"]["vtilde0"], dtype=np.float64),
        "vtilde1": np.asarray(source["filter_tilde"]["vtilde1"], dtype=np.float64),
        "vtilde2": np.asarray(source["filter_tilde"]["vtilde2"], dtype=np.float64),
        "dtot_tilde0": np.asarray(source["filter_tilde"]["dtot_tilde0"], dtype=np.float64),
        "dtot_tilde1": np.asarray(source["filter_tilde"]["dtot_tilde1"], dtype=np.float64),
        "dtot_tilde2": np.asarray(source["filter_tilde"]["dtot_tilde2"], dtype=np.float64),
    }
    nodes, node_weights = np.polynomial.legendre.leggauss(args.kernel_quad)

    last_mapped = None
    last_model = None
    for it in range(1, args.iters + 1):
        mapped, model = picard_map(grid, fields, params, nodes, node_weights, args.chunk)
        diff = np.sqrt(np.mean(np.concatenate([(mapped[k] - fields[k]).reshape(-1) for k in fields]) ** 2))
        for key in fields:
            fields[key] = (1.0 - args.damping) * fields[key] + args.damping * mapped[key]
        last_mapped, last_model = mapped, model
        print(f"iter={it} update_rms={diff:.6e}", flush=True)

    mapped, model = picard_map(grid, fields, params, nodes, node_weights, args.chunk)
    payload = build_payload(source, grid, fields, model, mapped, params, args, nodes, node_weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps(payload["diagnostics"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
