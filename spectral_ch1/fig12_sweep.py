#!/usr/bin/env python3
"""Figure 12 (dissertation Fig 1.4) from the spectral solver: equilibrium costs, private
signals (p1 = 3, p2 varying) vs a pooled signal of precision p1 + p2, competitive targets
(1, -1) and common target (0, 0).  Total cost = variance part J_i + mean part Jbar_i; the
mean part is zero under the common target with x0 = 0.  Writes ../data/fig12_costs_spectral.csv
in the format of fig12_costs.csv, on a log-spaced p2 grid 0.1..1000.  usage: python3 fig12_sweep.py [Nt Nth m] (default 12 16 12)."""
import subprocess, re, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
Nt, Nth, m = (sys.argv[1:4] if len(sys.argv) >= 4 else ('12', '16', '12'))
P1 = 3.0
import numpy as np
p2s = [float(f'{v:.6g}') for v in np.logspace(-1, 3, 41)]   # 0.1 .. 1000, ten per decade
def run(p1, p2, pooled):
    cmd = [os.path.join(HERE, 'spec_ch1'), Nt, Nth, m, '--p1', str(p1), '--p2', str(p2), '--threads', '8'] + (['--pooled'] if pooled else [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    J = re.search(r'J1 = ([-\d.e+]+)\s+J2 = ([-\d.e+]+)', out)
    Jb = re.search(r'Jbar1 = ([-\d.e+]+)\s+Jbar2 = ([-\d.e+]+)', out)
    if not (J and Jb): raise RuntimeError(out[-2000:])
    return [float(J.group(1)), float(J.group(2)), float(Jb.group(1)), float(Jb.group(2))]
rows = []
t0 = time.time()
for p2 in p2s:
    a = run(P1, p2, False); c = run(P1 + p2, P1 + p2, True)
    rows.append((p2, a, c))
    print(f'p2={p2:5.2f}  priv J=({a[0]:.5f},{a[1]:.5f}) Jbar=({a[2]:.4f},{a[3]:.4f})  pool J={c[0]:.5f} Jbar={c[2]:.4f}  [{time.time()-t0:.0f}s]', flush=True)
out = os.path.join(HERE, '..', 'data', 'fig12_costs_spectral.csv')
with open(out, 'w') as f:
    f.write('config,p2,J1_priv,J2_priv,J1_pool,J2_pool\n')
    for p2, a, c in rows: f.write(f'competitive,{p2},{a[0]+a[2]:.10g},{a[1]+a[3]:.10g},{c[0]+c[2]:.10g},{c[1]+c[3]:.10g}\n')
    for p2, a, c in rows: f.write(f'cooperative,{p2},{a[0]:.10g},{a[1]:.10g},{c[0]:.10g},{c[1]:.10g}\n')
parts = os.path.join(HERE, '..', 'data', 'fig12_costs_spectral_parts.csv')
with open(parts, 'w') as f:   # variance and mean parts at theta = (1, -1); total at theta = (k, -k) is Jvar + k^2 Jbar
    f.write('p2,Jvar1_priv,Jvar2_priv,Jbar1_priv,Jbar2_priv,Jvar_pool,Jbar_pool\n')
    for p2, a, c in rows: f.write(f'{p2},{a[0]:.10g},{a[1]:.10g},{a[2]:.10g},{a[3]:.10g},{c[0]:.10g},{c[2]:.10g}\n')
print('wrote', out, parts)
