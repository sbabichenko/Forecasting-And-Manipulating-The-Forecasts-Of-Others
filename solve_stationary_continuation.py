#!/usr/bin/env python3
"""Grid continuation for solve_stationary: solve on a coarse lag grid, then
use each solution's policy kernels as the --init warm start for the next
finer grid.  Usage:

    solve_stationary_continuation.py OUT.json p1 p2 r1 r2 [solver flags...] --N 4000 [--start 500]

Grids double from --start up to --N.  All other arguments are passed through
to the solver unchanged and in order."""
import json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "build", "solve_stationary")

def main(argv):
    out = argv[0]; rest = argv[1:]
    def take(flag, default):
        if flag in rest:
            i = rest.index(flag); v = rest[i + 1]; del rest[i:i + 2]; return v
        return default
    N = int(take("--N", "4000")); start = int(take("--start", "500"))
    # everything else stays in the order given (positionals first)
    grids = []; n = start
    while n < N: grids.append(n); n *= 2
    grids.append(N)
    init = None; total = 0.0; last = None
    with tempfile.TemporaryDirectory() as tmp:
        for n in grids:
            cmd = [BIN] + rest + ["--N", str(n)] + (["--init", init] if init else [])
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            dt = time.time() - t0; total += dt
            sol = json.loads(res.stdout.strip().split("\n")[-1])
            last = sol
            print(f"N={n}: {sol['n_iters']} iterations, residual {sol['residual']:.3g}, "
                  f"{dt:.2f} s", file=sys.stderr)
            init = os.path.join(tmp, f"init_{n}.csv")
            with open(init, "w") as f:
                for i, lag in enumerate(sol["lag"]):
                    f.write(",".join(f"{v:.17g}" for v in [lag]
                        + [sol["d1"][c][i] for c in ("ch0", "ch1", "ch2")]
                        + [sol["d2"][c][i] for c in ("ch0", "ch1", "ch2")]) + "\n")
    print(f"total {total:.2f} s", file=sys.stderr)
    with open(out, "w") as f:
        json.dump(last, f)

if __name__ == "__main__":
    main(sys.argv[1:])
