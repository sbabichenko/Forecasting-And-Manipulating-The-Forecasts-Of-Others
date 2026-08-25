#!/usr/bin/env python3
"""Richardson extrapolation of the figure data.

The kernel scheme in lqg_solver.cpp is first order in the grid spacing
(verified: J and kernel values converge with observed order 1.04-1.06).
Two nested runs, N=40 and N=79 (same time points at every second index),
give the first-order extrapolation 2 f(N=79) - f(N=40), accurate to about
0.1% where the single N=40 run is accurate to about 3%.

usage: python3 richardson.py [coarse_dir fine_dir out_dir]
"""
import sys, os, numpy as np, pandas as pd

coarse, fine, out = (sys.argv[1:4] if len(sys.argv) >= 4 else ('data_N40', 'data_N79', 'data'))
os.makedirs(out, exist_ok=True)

NON_NUMERIC = {'config', 'r_config'}
NOT_EXTRAPOLATED = {'fig3_residuals.csv'}          # iteration counts, not grid functions
INDEX_COLS = {'fig10_asymmetric.csv': ('t', 'p2'), 'fig11_wedges.csv': ('t', 'p2'), 'fig12_costs.csv': ('p2',),
              'fig13_precision_allocation.csv': ('r1', 'r2', 'p1_prec', 'p2_prec'), 'fig10_response_p1_3_p2_10.csv': ('t', 'p1', 'p2')}
STACKED = {'fig10_asymmetric.csv', 'fig11_wedges.csv'}   # time series stacked by p2

def align(dc, df, name):
    """Return df rows matching the rows of dc: every second grid point for
    time-indexed tables, identical rows for parameter-indexed tables."""
    if 't' in dc.columns and len(df) != len(dc):
        # time-indexed, possibly stacked by a parameter column (p2): match on
        # nearest t within each block of equal size
        key = ['p2'] if name in STACKED else []                 # tables stacked by a parameter column
        if key:
            parts = []
            for kval, blk in dc.groupby(key[0], sort=False):
                fb = df[df[key[0]] == kval]
                idx = np.abs(fb['t'].values[None, :] - blk['t'].values[:, None]).argmin(axis=1)
                parts.append(fb.iloc[idx])
            fa = pd.concat(parts)
        else:
            idx = np.abs(df['t'].values[None, :] - dc['t'].values[:, None]).argmin(axis=1)
            fa = df.iloc[idx]
        assert np.allclose(fa['t'].values, dc['t'].values, atol=1e-9), 'grids are not nested'
        return fa.reset_index(drop=True)
    if all(c in dc.columns for c in ('t', 's')) and len(df) != len(dc):
        raise RuntimeError('2-D kernel tables are not extrapolated here')
    assert len(df) == len(dc), f'row mismatch {len(df)} vs {len(dc)}'
    return df.reset_index(drop=True)

for name in sorted(os.listdir(coarse)):
    if not name.endswith('.csv'):
        continue
    dc = pd.read_csv(os.path.join(coarse, name))
    if name in NOT_EXTRAPOLATED or ('s' in dc.columns and 't' in dc.columns) or name.startswith('fig7'):
        dc.to_csv(os.path.join(out, name), index=False)     # copied from the coarse run
        print(f'{name}: copied (not a grid function or 2-D kernel)')
        continue
    df = align(dc, pd.read_csv(os.path.join(fine, name)), name)
    res = dc.copy()
    changed = []
    for c in dc.columns:
        if c in NON_NUMERIC or c in INDEX_COLS.get(name, ('t',)):
            continue
        if not np.issubdtype(dc[c].dtype, np.number):
            continue
        res[c] = 2.0 * df[c].values - dc[c].values
        changed.append(c)
    res.to_csv(os.path.join(out, name), index=False)
    d = max((np.max(np.abs(res[c] - dc[c])) / max(1e-12, np.max(np.abs(res[c]))) for c in changed), default=0.0)
    print(f'{name}: extrapolated {len(changed)} columns; max relative change vs N={len(dc)}-row coarse run {100*d:.2f}%')
