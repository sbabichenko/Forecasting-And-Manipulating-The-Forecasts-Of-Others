#!/usr/bin/env python3
"""Train a browser-side stationary parameter network.

The residual/Galerkin solver in stationary_neural_solver.py still solves one
parameter tuple at a time. This script builds a second-stage supervised MLP:

    (log p1, log p2, log r1, log r2) -> stationary lag-grid curves.

The exported JSON is intentionally simple so interactive.html can evaluate the
weights directly, without Python or JAX in the browser.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# NVIDIA 580.142 reports a two-part kernel driver version. Current XLA logs
# that harmless parse failure as an error even though CUDA execution works.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import optax

import stationary_neural_solver as sns


@dataclass(frozen=True)
class Sample:
    p1: float
    p2: float
    r1: float
    r2: float


FIELD_ORDER = [
    ("x", "lag"),
    ("xhat1", "lag"),
    ("xhat2", "lag"),
    ("xtilde1", "lag"),
    ("xtilde2", "lag"),
    ("d1", "lag"),
    ("d2", "lag"),
    ("calD1", "lag"),
    ("calD2", "lag"),
    ("hx1", "b_lag"),
    ("hx2", "b_lag"),
    ("wedge1", "b_lag"),
    ("wedge2", "b_lag"),
]

FIELD_SWAP = {
    "x": "x",
    "xhat1": "xhat2",
    "xhat2": "xhat1",
    "xtilde1": "xtilde2",
    "xtilde2": "xtilde1",
    "d1": "d2",
    "d2": "d1",
    "calD1": "calD2",
    "calD2": "calD1",
    "hx1": "hx2",
    "hx2": "hx1",
    "wedge1": "wedge2",
    "wedge2": "wedge1",
}


def random_samples(args: argparse.Namespace) -> Tuple[List[Sample], int]:
    rng = np.random.default_rng(args.sample_seed)
    total = args.n_train + args.n_val
    samples = [Sample(3.0, 3.0, 0.10, 0.10)]
    seen = {(3.0, 3.0, 0.10, 0.10)}
    while len(samples) < total:
        vals = np.exp(
            rng.uniform(
                low=np.log([args.p_min, args.p_min, args.r_min, args.r_min]),
                high=np.log([args.p_max, args.p_max, args.r_max, args.r_max]),
            )
        )
        # Round only the cache key scale, not to a hand-picked grid. This keeps
        # the training design random while making filenames stable/readable.
        s = Sample(*(float(f"{v:.6g}") for v in vals))
        key = (s.p1, s.p2, s.r1, s.r2)
        if key in seen:
            continue
        seen.add(key)
        samples.append(s)
    return samples, args.n_train


def cfg_for(sample: Sample, args: argparse.Namespace) -> sns.Config:
    return sns.Config(
        p1=sample.p1,
        p2=sample.p2,
        r1=sample.r1,
        r2=sample.r2,
        sigma=1.0,
        A=0.0,
        n=args.N,
        lag_max=args.L,
        quadrature="gauss",
        inner_quad=args.inner_quad or args.N,
        n_features=args.features,
        seed=args.seed,
        max_nfev=args.max_nfev,
        tol=1e-6,
        accept_rms=args.accept_rms,
        coef_penalty=1e-5,
        w_state=1.0,
        w_filter=1.0,
        w_hx=1.0,
        w_H=0.25,
        w_wedge=1.0,
        w_policy=1.0,
        w_boundary=3.0,
        w_tail_left=0.1,
    )


def sample_path(sample_dir: str, sample: Sample, args: argparse.Namespace) -> str:
    stem = f"p1_{sample.p1:g}_p2_{sample.p2:g}_r1_{sample.r1:g}_r2_{sample.r2:g}_N{args.N}_f{args.features}"
    return os.path.join(sample_dir, stem.replace(".", "p") + ".npz")


def solve_or_load(sample: Sample, args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    os.makedirs(args.sample_dir, exist_ok=True)
    path = sample_path(args.sample_dir, sample, args)
    legacy_path = path[:-4] + "pnpz.npz" if path.endswith(".npz") else path + ".npz"
    legacy_dir = os.path.join(os.path.dirname(args.sample_dir), "stationary_surrogate_samples")
    legacy_sample_path = sample_path(legacy_dir, sample, args)
    legacy_sample_pnpz = legacy_sample_path[:-4] + "pnpz.npz" if legacy_sample_path.endswith(".npz") else legacy_sample_path + ".npz"
    load_candidates = [path, legacy_path, legacy_sample_path, legacy_sample_pnpz]
    load_path = next((p for p in load_candidates if os.path.exists(p)), path)
    if os.path.exists(load_path) and not args.refresh:
        with np.load(load_path) as z:
            arrays = {k: np.asarray(z[k], dtype=float) for k, _ in FIELD_ORDER if k in z}
            meta = {k: float(z[k]) for k in ["residual_rms", "initial_rms", "nfev"] if k in z}
            arrays["lag"] = np.asarray(z["lag"], dtype=float)
            arrays["b_lag"] = np.asarray(z["b_lag"], dtype=float)
            return arrays, meta

    cfg = cfg_for(sample, args)
    basis, packer, result, initial_rms, final_rms, backend = sns.run_solver(
        cfg, verbose=False, init_path="", backend="jax"
    )
    raw = sns.solution_arrays(result.x, basis, packer)
    x = raw["x"]
    xt1 = raw["xt1"]
    xt2 = raw["xt2"]
    arrays = {
        "lag": basis.a,
        "b_lag": basis.b,
        "x": x,
        "xhat1": x - xt1,
        "xhat2": x - xt2,
        "xtilde1": xt1,
        "xtilde2": xt2,
        "d1": raw["d1"],
        "d2": raw["d2"],
        "calD1": raw["c1"],
        "calD2": raw["c2"],
        "hx1": raw["hx1"],
        "hx2": raw["hx2"],
        "wedge1": raw["w1"],
        "wedge2": raw["w2"],
    }
    meta = {
        "residual_rms": float(final_rms),
        "initial_rms": float(initial_rms),
        "nfev": float(result.nfev),
    }
    np.savez(
        path,
        p1=sample.p1,
        p2=sample.p2,
        r1=sample.r1,
        r2=sample.r2,
        residual_rms=final_rms,
        initial_rms=initial_rms,
        nfev=int(result.nfev),
        backend=backend,
        **arrays,
    )
    return arrays, meta


def flatten_arrays(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    pieces = []
    for name, _ in FIELD_ORDER:
        pieces.append(np.asarray(arrays[name], dtype=float).reshape(-1))
    return np.concatenate(pieces)


def swap_player_input_logs(x: np.ndarray) -> np.ndarray:
    return x[:, [1, 0, 3, 2]]


def swap_player_output_flat(y: np.ndarray, n_lag: int, n_blag: int) -> np.ndarray:
    specs = field_specs(n_lag, n_blag)
    by_name = {}
    for spec in specs:
        offset = int(spec["offset"])
        length = int(spec["length"])
        field = y[offset : offset + length].reshape(-1, 3)
        # Player relabeling swaps observation-shock channels W1 and W2.
        by_name[str(spec["name"])] = field[:, [0, 2, 1]]

    pieces = []
    for name, _ in FIELD_ORDER:
        pieces.append(by_name[FIELD_SWAP[name]].reshape(-1))
    return np.concatenate(pieces)


def augment_with_player_symmetry(
    x: np.ndarray,
    y: np.ndarray,
    n_lag: int,
    n_blag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y_swapped = np.vstack([swap_player_output_flat(row, n_lag, n_blag) for row in y])
    return np.vstack([x, swap_player_input_logs(x)]), np.vstack([y, y_swapped])


def init_mlp(rng: np.random.Generator, dims: List[int]):
    params = []
    for din, dout in zip(dims[:-1], dims[1:]):
        scale = math.sqrt(2.0 / max(din + dout, 1))
        params.append(
            {
                "W": rng.normal(0.0, scale, size=(din, dout)),
                "b": np.zeros(dout),
            }
        )
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), params)


def mlp(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def predict_mlp_np(
    params,
    x: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
) -> np.ndarray:
    h = (x - x_mean) / x_scale
    for layer in params[:-1]:
        h = np.tanh(h @ np.asarray(layer["W"]) + np.asarray(layer["b"]))
    last = params[-1]
    return (h @ np.asarray(last["W"]) + np.asarray(last["b"])) * y_scale + y_mean


def train_mlp(x: np.ndarray, y: np.ndarray, args: argparse.Namespace):
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0), 1e-6)
    y_mean = y.mean(axis=0)
    y_scale = np.maximum(y.std(axis=0), 1e-6)
    xs = (x - x_mean) / x_scale
    ys = (y - y_mean) / y_scale

    if args.fit == "ridge":
        rng = np.random.default_rng(args.train_seed)
        w1 = rng.normal(0.0, 1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], args.hidden))
        b1 = rng.normal(0.0, 0.35, size=args.hidden)
        h = np.tanh(xs @ w1 + b1)
        design = np.column_stack([np.ones(h.shape[0]), h])
        gram = design.T @ design
        rhs = design.T @ ys
        beta = np.linalg.solve(gram + args.ridge * np.eye(gram.shape[0]), rhs)
        params = [
            {"W": w1, "b": b1},
            {"W": beta[1:, :], "b": beta[0, :]},
        ]
        pred = h @ beta[1:, :] + beta[0, :]
        loss = float(np.mean((pred - ys) ** 2))
        pred_unscaled = pred * y_scale + y_mean
        rmse = float(np.sqrt(np.mean((pred_unscaled - y) ** 2)))
        return params, x_mean, x_scale, y_mean, y_scale, loss, rmse

    dims = [x.shape[1]] + [args.hidden] * args.depth + [y.shape[1]]
    params = init_mlp(np.random.default_rng(args.train_seed), dims)
    if args.lr_decay < 1.0:
        lr = optax.exponential_decay(
            args.lr,
            transition_steps=max(args.steps // 4, 1),
            decay_rate=args.lr_decay,
            staircase=False,
        )
    else:
        lr = args.lr
    opt = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(lr, weight_decay=args.weight_decay),
    )
    opt_state = opt.init(params)
    xj = jnp.asarray(xs, dtype=jnp.float64)
    yj = jnp.asarray(ys, dtype=jnp.float64)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            pred = mlp(p, xj)
            return jnp.mean((pred - yj) ** 2)

        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss = None
    for _ in range(args.steps):
        params, opt_state, loss = step(params, opt_state)

    pred = np.asarray(mlp(params, xj)) * y_scale + y_mean
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return params, x_mean, x_scale, y_mean, y_scale, float(loss), rmse


def field_specs(n_lag: int, n_blag: int) -> List[Dict[str, object]]:
    specs = []
    offset = 0
    for name, grid in FIELD_ORDER:
        count = (n_lag if grid == "lag" else n_blag) * 3
        specs.append({"name": name, "grid": grid, "offset": offset, "length": count})
        offset += count
    return specs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a parameter-conditioned stationary browser NN.")
    ap.add_argument("--out", default="data/stationary_param_nn.json")
    ap.add_argument("--sample-dir", default="data/stationary_param_nn_samples")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--L", type=float, default=3.0)
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--inner-quad", type=int, default=0)
    ap.add_argument("--max-nfev", type=int, default=12)
    ap.add_argument("--accept-rms", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=96)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--sample-seed", type=int, default=20260519)
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--p-max", type=float, default=40.0)
    ap.add_argument("--r-min", type=float, default=0.05)
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-decay", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--fit", choices=["ridge", "adam"], default="ridge")
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--train-seed", type=int, default=123)
    ap.add_argument("--no-augment-symmetry", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    samples, n_train = random_samples(args)
    inputs = []
    outputs = []
    diagnostics = []
    lag = None
    b_lag = None
    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] sample p1={sample.p1:g} p2={sample.p2:g} r1={sample.r1:g} r2={sample.r2:g}", flush=True)
        arrays, meta = solve_or_load(sample, args)
        lag = arrays["lag"]
        b_lag = arrays["b_lag"]
        inputs.append([math.log(sample.p1), math.log(sample.p2), math.log(sample.r1), math.log(sample.r2)])
        outputs.append(flatten_arrays(arrays))
        diagnostics.append({"p1": sample.p1, "p2": sample.p2, "r1": sample.r1, "r2": sample.r2, **meta})

    x_all = np.asarray(inputs, dtype=float)
    y_all = np.asarray(outputs, dtype=float)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    x_val = x_all[n_train:]
    y_val = y_all[n_train:]
    if not args.no_augment_symmetry:
        x_train, y_train = augment_with_player_symmetry(x_train, y_train, len(lag), len(b_lag))
        if x_val.size:
            x_val, y_val = augment_with_player_symmetry(x_val, y_val, len(lag), len(b_lag))
    params, x_mean, x_scale, y_mean, y_scale, final_loss, train_rmse = train_mlp(x_train, y_train, args)
    val_rmse = None
    if x_val.size:
        val_pred = predict_mlp_np(params, x_val, x_mean, x_scale, y_mean, y_scale)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    params_np = jax.tree_util.tree_map(lambda v: np.asarray(v).tolist(), params)
    ranges = {
        "p1": [min(s.p1 for s in samples), max(s.p1 for s in samples)],
        "p2": [min(s.p2 for s in samples), max(s.p2 for s in samples)],
        "r1": [min(s.r1 for s in samples), max(s.r1 for s in samples)],
        "r2": [min(s.r2 for s in samples), max(s.r2 for s in samples)],
    }
    payload = {
        "version": 1,
        "kind": "stationary_param_nn",
        "description": "Browser-side MLP weights trained from offline JAX residual stationary fits.",
        "input_names": ["log_p1", "log_p2", "log_r1", "log_r2"],
        "input_mean": x_mean.tolist(),
        "input_scale": x_scale.tolist(),
        "output_mean": y_mean.tolist(),
        "output_scale": y_scale.tolist(),
        "layers": params_np,
        "lag": np.asarray(lag, dtype=float).tolist(),
        "b_lag": np.asarray(b_lag, dtype=float).tolist(),
        "fields": field_specs(len(lag), len(b_lag)),
        "train_ranges": ranges,
        "diagnostics": {
            "n_samples": len(samples),
            "n_train": n_train,
            "n_val": len(samples) - n_train,
            "n_training_rows": int(x_train.shape[0]),
            "n_validation_rows": int(x_val.shape[0]) if x_val.size else 0,
            "sample_seed": args.sample_seed,
            "train_loss_standardized": final_loss,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "player_symmetry_augmented": not args.no_augment_symmetry,
            "samples": diagnostics,
            "N": args.N,
            "L": args.L,
            "features": args.features,
            "hidden": args.hidden,
            "depth": args.depth,
            "fit": args.fit,
            "max_nfev_per_sample": args.max_nfev,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(json.dumps(payload["diagnostics"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
