import re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm
lines = open('mean_sweep.txt').read().splitlines()
ps, D0, Jb = [], [], []
for l in lines:
    m = re.match(r'p=([0-9.]+) : .*Dbar1\(0\) = ([-0-9.e]+).*Jbar1 = ([-0-9.e]+).*perfect-info \(closed-loop\) Dbar1\(0\) = ([-0-9.e]+)', l)
    if m: ps.append(float(m.group(1))); D0.append(float(m.group(2))); Jb.append(float(m.group(3))); pi = float(m.group(4))
ps = np.array(ps); D0 = np.array(D0)
# open-loop Nash of the deterministic game (no information): lambda_i' = 2 (Xbar - b_i), Dbar_i = lambda_i/(2 r_i),
# Xbar' = Dbar_1 + Dbar_2, Xbar(0) = x0, lambda_i(T) = 0 -- a linear two-point BVP, solved by collocation on a fine grid
r1 = r2 = 0.1; b1, b2 = 1.0, -1.0; x0 = 0.0; T = 1.0; n = 2000; h = T / n
import numpy as np
N = 3 * (n + 1); A = np.zeros((N, N)); rhs = np.zeros(N)
X = lambda k: k; L1 = lambda k: n + 1 + k; L2 = lambda k: 2 * (n + 1) + k
A[X(0), X(0)] = 1; rhs[X(0)] = x0
for k in range(n):  # trapezoidal
    A[X(k + 1), X(k + 1)] += 1; A[X(k + 1), X(k)] -= 1
    for kk in (k, k + 1):
        A[X(k + 1), L1(kk)] -= h / 2 / (2 * r1); A[X(k + 1), L2(kk)] -= h / 2 / (2 * r2)
    A[L1(k), L1(k + 1)] += 1; A[L1(k), L1(k)] -= 1
    A[L2(k), L2(k + 1)] += 1; A[L2(k), L2(k)] -= 1
    for kk in (k, k + 1):
        A[L1(k), X(kk)] -= h / 2 * 2; A[L2(k), X(kk)] -= h / 2 * 2
    rhs[L1(k)] = -h * 2 * b1; rhs[L2(k)] = -h * 2 * b2
A[L1(n), L1(n)] = 1; A[L2(n), L2(n)] = 1
sol = np.linalg.solve(A, rhs); ol = sol[L1(0)] / (2 * r1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].semilogx(ps, D0, 'o-', color='C0', label=r'partial information, $\bar D^1_0$')
ax[0].axhline(pi, color='black', ls='--', lw=1.5, label=r'perfect information (closed-loop Nash), %.2f' % pi)
ax[0].axhline(ol, color='gray', ls=':', lw=1.5, label=r'no information (open-loop Nash), %.2f' % ol)
ax[0].set_xlabel('signal precision $p$ (both players)'); ax[0].set_ylabel(r'$\bar D^1_0$'); ax[0].legend(fontsize=8); ax[0].set_title('mean control at $t=0$ against precision')
cols = cm.viridis(np.linspace(0.05, 0.85, len(ps)))
for p, c in zip(ps, cols):
    d = np.loadtxt(f'mean_p{p:g}.txt')
    ax[1].plot(d[:, 0], d[:, 2], color=c, lw=1.4, label=f'$p={p:g}$')
d = np.loadtxt(f'mean_p{ps[-1]:g}.txt'); ax[1].plot(d[:, 0], d[:, 4], 'k--', lw=1.8, label='perfect information')
ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'$\bar D^1_t$'); ax[1].legend(fontsize=7, ncol=2); ax[1].set_title('mean control paths (spectral, 12x16)')
plt.tight_layout(); plt.savefig('mean_sweep.png', dpi=100); plt.savefig('mean_sweep.pdf')
print('p      D1(0)   Jbar1'); [print('%-6g %.4f  %.4f' % (p, d, j)) for p, d, j in zip(ps, D0, Jb)]
print('closed-loop %.4f, open-loop %.4f' % (pi, ol))
