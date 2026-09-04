#!/usr/bin/env python3
"""
Plot all paper figures from CSV data produced by the C++ solver.
Reads from data/, writes PDFs to figs/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

DATA_DIR = 'data'
FIGDIR = 'figs'
os.makedirs(FIGDIR, exist_ok=True)

T_VAL = 1
N = pd.read_csv(f'{DATA_DIR}/fig4_X.csv')['t'].nunique()   # grid size of the data in DATA_DIR
t_grid = np.linspace(0.0, T_VAL, N)
channel_labels = [r'$W^0$', r'$W^1$', r'$W^2$']
STATE_COLOR = '#333333'
PLAYER1_COLOR = '#1f77b4'
PLAYER2_COLOR = '#d62728'


def load_kernel2d(path):
    """Load a 2D kernel CSV into (N, N, 3) array."""
    df = pd.read_csv(path)
    K = np.zeros((N, N, 3))
    for _, row in df.iterrows():
        ti, si = int(row['t_idx']), int(row['s_idx'])
        K[ti, si] = [row['ch0'], row['ch1'], row['ch2']]
    return K


def make_3channel_fig(data, title_prefix, clabel, n_curves=15):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.8))
    cmap_loc = cm.viridis
    norm_loc = Normalize(vmin=0, vmax=T_VAL)
    for ch in range(3):
        ax = axes[ch]
        max_val = np.max(np.abs(data[:, :, ch]))
        for t_idx in range(0, N, max(1, N // n_curves)):
            color = cmap_loc(norm_loc(t_grid[t_idx]))
            ax.plot(t_grid[:t_idx+1], data[t_idx, :t_idx+1, ch], color=color, lw=1.2)
        ax.set_xlabel(r'$s$')
        ax.set_title(f'{title_prefix} channel {channel_labels[ch]}'
                     f'\nmax$|.|$={max_val:.3g}')
        ax.grid(alpha=0.2)
    fig.subplots_adjust(right=0.88, top=0.82)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    sm = cm.ScalarMappable(cmap=cmap_loc, norm=norm_loc)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=clabel)
    return fig


# ============================================================
# FIGURE 3: Picard residual
# ============================================================
print("Figure 3: Picard residual ...")
df = pd.read_csv(f'{DATA_DIR}/fig3_residuals.csv')
fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
ax.semilogy(df['iteration'], df['residual'], lw=2, color='C0')
ax.set_xlabel('Iteration')
ax.set_ylabel('Relative residual')
ax.set_title('Outer Picard Residual')
ax.grid(alpha=0.3)
fig.savefig(f'{FIGDIR}/fig3_picard_residual.pdf')
plt.close(fig)

# ============================================================
# FIGURE 4: State kernel X(t,s)
# ============================================================
print("Figure 4: State kernel X(t,s) ...")
X = load_kernel2d(f'{DATA_DIR}/fig4_X.csv')
fig = make_3channel_fig(X, r'$X$', r'$t$')
fig.savefig(f'{FIGDIR}/fig4_state_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 5: D1(t,s) feedback kernel
# ============================================================
print("Figure 5: D1(t,s) feedback kernel ...")
D1 = load_kernel2d(f'{DATA_DIR}/fig5_D1.csv')
fig = make_3channel_fig(D1, r'$D^1$', r'$t$')
fig.savefig(f'{FIGDIR}/fig5_D1_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 6: calD1(t,s)
# ============================================================
print("Figure 6: calD1(t,s) primitive-noise kernel ...")
calD1 = load_kernel2d(f'{DATA_DIR}/fig6_calD1.csv')
fig = make_3channel_fig(calD1, r'$\mathcal{D}^1$', r'$t$')
fig.savefig(f'{FIGDIR}/fig6_calD1_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 7: F1 at t=T
# ============================================================
print("Figure 7: Filtering kernel F1 at t=T ...")
df = pd.read_csv(f'{DATA_DIR}/fig7_F1_T.csv')
F1_T = np.zeros((N, N, 3, 3))
for _, row in df.iterrows():
    u, s = int(row['u_idx']), int(row['s_idx'])
    r, c = int(row['row']), int(row['col'])
    F1_T[u, s, r, c] = row['value']

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
cmap_f = cm.viridis
norm_u = Normalize(vmin=0, vmax=T_VAL)
row_labels = [r'to $W^0$', r'to $W^1$', r'to $W^2$']
col_labels = [r'from $W^0$', r'from $W^1$', r'from $W^2$']

for row in range(3):
    for col in range(3):
        ax = axes[row][col]
        max_val = np.max(np.abs(F1_T[:, :, row, col]))
        for u_idx in range(0, N, max(1, N // 12)):
            color = cmap_f(norm_u(t_grid[u_idx]))
            ax.plot(t_grid[:N], F1_T[u_idx, :N, row, col], color=color, lw=1.0, alpha=0.8)
        if max_val < 1e-10:
            ax.text(0.5, 0.5, r'$\approx 0$', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
        if row == 0: ax.set_title(f'{col_labels[col]}', fontsize=10)
        if col == 0: ax.set_ylabel(row_labels[row], fontsize=10)
        ax.set_xlabel(r'$s$', fontsize=9)
        ax.grid(alpha=0.2)

fig.subplots_adjust(right=0.88, top=0.92)
cax = fig.add_axes([0.90, 0.08, 0.015, 0.8])
sm = cm.ScalarMappable(cmap=cmap_f, norm=norm_u)
sm.set_array([])
fig.colorbar(sm, cax=cax, label=r'$u$')
fig.suptitle(r'$F^1$: entry slices of $F_t(u,s)$ at $t=T$ (color $= u$)', fontsize=14, y=0.97)
fig.savefig(f'{FIGDIR}/fig7_F1_kernel.pdf')
plt.close(fig)

# ============================================================
# FIGURE 8: Mean control barD1(t) vs precision p
# ============================================================
print("Figure 8: Mean control vs precision ...")
df = pd.read_csv(f'{DATA_DIR}/fig8_barD1.csv')
p_values = [1, 2, 3, 5, 10]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# distinct, print-safe colours: viridis with the pale yellow end removed
cols_p = cm.viridis(np.linspace(0.05, 0.8, len(p_values)))

for p, color in zip(p_values, cols_p):
    ax.plot(df['t'], df[f'p{p}'], lw=2.2, color=color, label=f'$p={p}$')
    ax.text(-0.012, df[f'p{p}'].iloc[0], f'{p}', ha='right', va='center', fontsize=9, color=color)

ax.plot(df['t'], df['perfect_info'], lw=2.4, ls='--', color='black', label='perfect information')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\bar{D}^1_t$')
ax.set_xlim(-0.03, 1.0)
ax.legend(loc='upper right', frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig8_barD1_vs_p.pdf')
plt.close(fig)

# ============================================================
# FIGURE 9: barH2 and R2
# ============================================================
print("Figure 9: Opponent adjoint barH2 and gain R2 ...")
barH2 = load_kernel2d(f'{DATA_DIR}/fig9_barH2.csv')
R2 = load_kernel2d(f'{DATA_DIR}/fig9_R2.csv')
norm9 = Normalize(vmin=0, vmax=T_VAL)

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))

for ch in range(3):
    ax = axes[0][ch]
    max_val = np.max(np.abs(barH2[:, :, ch]))
    for t_idx in range(0, N, max(1, N // 12)):
        color = cm.viridis(norm9(t_grid[t_idx]))
        ax.plot(t_grid[:t_idx+1], barH2[t_idx, :t_idx+1, ch], color=color, lw=1.0)
    ax.set_xlabel(r'$s$')
    ax.set_title(r'$\bar{H}^2$ channel ' + channel_labels[ch] + f'\nmax$|.|$={max_val:.3g}')
    ax.grid(alpha=0.2)

for ch in range(3):
    ax = axes[1][ch]
    max_val = np.max(np.abs(R2[:, :, ch]))
    for t_idx in range(0, N, max(1, N // 12)):
        color = cm.viridis(norm9(t_grid[t_idx]))
        ax.plot(t_grid[:t_idx+1], R2[t_idx, :t_idx+1, ch], color=color, lw=1.0)
    ax.set_xlabel(r'$s$')
    ax.set_title(r'$R^2$ channel ' + channel_labels[ch] + f'\nmax$|.|$={max_val:.3g}')
    ax.grid(alpha=0.2)

fig.subplots_adjust(right=0.88, top=0.88, hspace=0.45)
cax = fig.add_axes([0.90, 0.08, 0.015, 0.8])
sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm9)
sm.set_array([])
fig.colorbar(sm, cax=cax, label=r'$t$')
fig.suptitle(r'Opponent objects: $\bar{H}^2$ (top) and $R^2$ (bottom)', fontsize=13, y=0.97)
fig.savefig(f'{FIGDIR}/fig9_barH2_R2.pdf')
plt.close(fig)

# ============================================================
# FIGURE 10: Asymmetric equilibrium panels
# ============================================================
print("Figure 10: Asymmetric equilibrium panels ...")
df = pd.read_csv(f'{DATA_DIR}/fig10_asymmetric.csv')
df_pi = pd.read_csv(f'{DATA_DIR}/fig10_perfect_info.csv')
p2_values = sorted(df['p2'].unique())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
cmap_p2 = cm.viridis
norm_p2 = Normalize(vmin=min(p2_values), vmax=max(p2_values))

# Panel 1: mean controls
ax = axes[0]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'barD1'], color=c, lw=1.8, ls='-')
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'barD2'], color=c, lw=1.8, ls='--')
ax.plot([], [], color='gray', ls='-', lw=1.5, label=r'$\bar{D}^1$ (solid)')
ax.plot([], [], color='gray', ls='--', lw=1.5, label=r'$\bar{D}^2$ (dashed)')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$\bar{D}^i(t)$')
ax.set_title('Mean controls')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)

# Panel 2: mean state
ax = axes[1]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'barX'], color=c, lw=1.8,
            label=f'$p_2={int(p2v)}$')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$\bar{X}(t)$')
ax.set_title('Mean state path')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# Panel 3: aggregate effort
ax = axes[2]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    effort = np.abs(df.loc[mask, 'barD1'].values) + np.abs(df.loc[mask, 'barD2'].values)
    ax.plot(df.loc[mask, 't'], effort, color=c, lw=1.8)
effort_pi = 2 * np.abs(df_pi['barD1_pi'].values)
ax.plot(df_pi['t'], effort_pi, lw=1.8, ls='--', color='black', label='perfect information')
ax.set_xlabel(r'$t$'); ax.set_ylabel(r'$|\bar{D}^1|+|\bar{D}^2|$')
ax.set_title('Aggregate mean effort')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.subplots_adjust(right=0.90)
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = cm.ScalarMappable(cmap=cmap_p2, norm=norm_p2)
sm.set_array([])
fig.colorbar(sm, cax=cax, label=r'$p_2$')
fig.suptitle(r'Asymmetric equilibrium: $p_1=3$ fixed, $p_2$ varies ($r=0.1$, $T=1$)',
             fontsize=14, y=1.01)
fig.savefig(f'{FIGDIR}/fig10_asymmetric_panels.pdf')
plt.close(fig)

# ============================================================
# FIGURE 10b: Source-zero response slice (p1=3, p2=10)
# ============================================================
print("Figure 10b: Source-zero response slice, p1=3 p2=10 ...")
df_resp = pd.read_csv(f'{DATA_DIR}/fig10_response_p1_3_p2_10.csv')

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))

ax = axes[0]
ax.plot(df_resp['t'], df_resp['X'], lw=2.2, color=STATE_COLOR, label=r'$X(t,0)$')
ax.plot(df_resp['t'], df_resp['Xhat1'], lw=1.9, ls='--', color=PLAYER1_COLOR,
        label=r'$\hat{X}^1(t,0)$')
ax.plot(df_resp['t'], df_resp['Xhat2'], lw=1.9, ls='--', color=PLAYER2_COLOR,
        label=r'$\hat{X}^2(t,0)$')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel(r'$t$')
ax.set_ylabel('response')
ax.set_title(r'State and posterior estimates')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(df_resp['t'], df_resp['calD1'], lw=2.0, color=PLAYER1_COLOR,
        label=r'$\mathcal{D}^1(t,0)$')
ax.plot(df_resp['t'], df_resp['calD2'], lw=2.0, color=PLAYER2_COLOR,
        label=r'$\mathcal{D}^2(t,0)$')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel(r'$t$')
ax.set_title(r'Primitive-control responses')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

fig.suptitle(r'Source-$W^0$ responses for $p_1=3,\;p_2=10$', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig10_response_p1_3_p2_10.pdf')
plt.close(fig)

# ============================================================
# FIGURE 11: Information wedges
# ============================================================
print("Figure 11: Information wedges ...")
df = pd.read_csv(f'{DATA_DIR}/fig11_wedges.csv')
p2_values = sorted(df['p2'].unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap_p2 = cm.viridis
norm_p2 = Normalize(vmin=min(p2_values), vmax=max(p2_values))

ax = axes[0]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'V1'], lw=2.2, color=c,
            label=f'$p_2={int(p2v)}$')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$', fontsize=13); ax.set_ylabel(r'$\mathcal{V}^1(t)$', fontsize=13)
ax.set_title(r'Player 1 wedge $\mathcal{V}^1(t)$', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for p2v in p2_values:
    mask = df['p2'] == p2v
    c = cmap_p2(norm_p2(p2v))
    ax.plot(df.loc[mask, 't'], df.loc[mask, 'V2'], lw=2.2, color=c,
            label=f'$p_2={int(p2v)}$')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$t$', fontsize=13); ax.set_ylabel(r'$\mathcal{V}^2(t)$', fontsize=13)
ax.set_title(r'Player 2 wedge $\mathcal{V}^2(t)$', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle(r'Information wedges ($p_1=3$ fixed, $p_2$ varies)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig11_info_wedges.pdf')
plt.close(fig)

# ============================================================
# FIGURE 12: Player costs — private vs pooled
# ============================================================
print("Figure 12: Player costs, private vs pooled ...")
# Spectral-solver sweep (spectral_ch1/fig12_sweep.py): variance and mean parts at theta = (1, -1);
# the total at theta = (k, -k) is Jvar + k^2 Jbar, and the common target (0, 0) has no mean part.
dp = pd.read_csv(f'{DATA_DIR}/fig12_costs_spectral_parts.csv').sort_values('p2')
panels = [(1, r'Competitive ($\theta=\pm 1$)'), (5, r'Hypercompetitive ($\theta=\pm 5$)'), (0, r'Cooperative ($\theta=0,0$)')]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (k, label) in zip(axes, panels):
    J1p = dp['Jvar1_priv'] + k * k * dp['Jbar1_priv']
    J2p = dp['Jvar2_priv'] + k * k * dp['Jbar2_priv']
    Jpool = dp['Jvar_pool'] + k * k * dp['Jbar_pool']
    ax.plot(dp['p2'], J1p, '--', lw=2, color='C0', label=r'$J^1$ private')
    ax.plot(dp['p2'], J2p, '--', lw=2, color='C3', label=r'$J^2$ private')
    ax.plot(dp['p2'], Jpool, '-', lw=2, color='0.3', label=r'$J^1=J^2$ pooled')
    ax.axvline(3, color='0.6', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r'$p_2$', fontsize=13)
    ax.set_ylabel(r'Cost', fontsize=13)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle(r'Equilibrium costs: private vs pooled ($p_1=3$, $r=0.1$, spectral solution)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/fig12_costs_private_vs_pooled.pdf')
plt.close(fig)

# ============================================================
# FIGURE 13: Precision decomposition, asymmetric-cost row only
# ============================================================
print("Figure 13: Precision decomposition ...")
PBAR = 20.0
df14 = pd.read_csv(f'{DATA_DIR}/fig13_precision_allocation.csv')

sub = df14[(df14['config'] == 'competitive') &
           (df14['r_config'] == 'r0.05_0.2')].sort_values('p1_prec')
p1_frac = sub['p1_prec'].values / PBAR
r1_vals = sub['r1'].values
r2_vals = sub['r2'].values

fig, axes = plt.subplots(1, 3, figsize=(15, 4.1),
                         gridspec_kw={'wspace': 0.30})

# ── Col (a): Individual equilibrium costs J1, J2 ──
ax = axes[0]
ax.plot(p1_frac, sub['J1_eq'].values, lw=2.0, color='C0',
        label=r'$J^1$ (equilibrium)')
ax.plot(p1_frac, sub['J2_eq'].values, lw=2.0, color='C3',
        label=r'$J^2$ (equilibrium)')
ax.plot(p1_frac, sub['J1_fi'].values, lw=1.5, ls='--', color='C0', alpha=0.6,
        label=r'$J^1$ (full info)')
ax.plot(p1_frac, sub['J2_fi'].values, lw=1.5, ls='--', color='C3', alpha=0.6,
        label=r'$J^2$ (full info)')
ax.set_xlabel('Fraction of precision to player 1', fontsize=12)
ax.set_ylabel('Equilibrium cost', fontsize=12)
ax.set_title('Individual costs', fontsize=13)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Col (b): Total destructive effort ──
ax = axes[1]
total_eq = r1_vals * sub['barD1sq_eq'].values + r2_vals * sub['barD2sq_eq'].values
total_fi = r1_vals * sub['barD1sq_fi'].values + r2_vals * sub['barD2sq_fi'].values
ax.plot(p1_frac, total_eq, lw=2.0, color='C0', label='Equilibrium')
ax.plot(p1_frac, total_fi, lw=1.5, ls='--', color='C1', alpha=0.8, label='Full info')
ax.fill_between(p1_frac, total_eq, total_fi, alpha=0.27, color='C1')
ax.set_xlabel('Fraction of precision to player 1', fontsize=12)
ax.set_ylabel('Total destructive effort', fontsize=12)
ax.set_title('Total destructive effort', fontsize=13)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Col (c): Mean controls & mean state ──
ax = axes[2]
ax.plot(p1_frac, sub['barD1_avg_eq'].values, lw=2.0, color='C0',
        label=r'$\bar{D}_1$ (eq)')
ax.plot(p1_frac, sub['barD2_avg_eq'].values, lw=2.0, color='C3',
        label=r'$\bar{D}_2$ (eq)')
ax.plot(p1_frac, sub['barD1_avg_fi'].values, lw=1.5, ls=':', color='C0',
        label=r'$\bar{D}_1$ (CE)')
ax.plot(p1_frac, sub['barD2_avg_fi'].values, lw=1.5, ls=':', color='C3',
        label=r'$\bar{D}_2$ (CE)')
ax.plot(p1_frac, sub['barX_avg_eq'].values, lw=2.0, color='k',
        label=r'$\bar{X}$ (eq)')
ax.plot(p1_frac, sub['barX_avg_fi'].values, lw=1.5, ls=':', color='k',
        label=r'$\bar{X}$ (CE)')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.set_xlabel('Fraction of precision to player 1', fontsize=12)
ax.set_ylabel('Mean control / state', fontsize=12)
ax.set_title('Mean controls and state', fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

for ax in axes:
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=11)

fig.suptitle(r'Precision allocation with asymmetric costs ($r_1=0.05,\;r_2=0.2$)',
             fontsize=14, y=1.04)
fig.savefig(f'{FIGDIR}/fig13_precision_decomposition.pdf',
            bbox_inches='tight')
plt.close(fig)


print("\nAll figures saved to", FIGDIR)
