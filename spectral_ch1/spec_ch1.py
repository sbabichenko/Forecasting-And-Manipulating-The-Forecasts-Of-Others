"""Spectral-in-time prototype of the Chapter 1 finite-horizon two-player LQG game (variance part).

Kernels live on the triangle 0 <= s <= t <= T in Duffy coordinates (t, theta = s/t): Chebyshev-
Lobatto nodes in t and theta, barycentric interpolation, Clenshaw-Curtis / Gauss-Legendre
quadrature.  Player i's control is parametrized by its coefficient on the player's own
observation increments, g^i_t(u) (u <= t), so that the primitive-noise control kernel is

    calD^i_t(s) = sqrt(p_i) int_s^t g^i_t(u) X_u(s) du + g^i_t(s) e_i

without any projection, and the state kernel solves the LINEAR Volterra equation

    X_t(s) = sigma e_0 + int_s^t (calD^1_u(s) + calD^2_u(s)) du .

Costs (variance part): J_i = int_0^T dt int_0^t ds (|X_t(s)|^2 + r_i |calD^i_t(s)|^2).
Equilibrium: grad_{g^i} J_i = 0 for both players (reverse-mode AD), solved by Newton with a
forward-mode Jacobian.
"""
import sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

# ------------------------------------------------------------------ grids and quadrature
def lobatto(N, lo, hi):
    k = np.arange(N)
    return lo + 0.5 * (hi - lo) * (1.0 - np.cos(np.pi * k / (N - 1)))

def bary_w(N):
    w = np.ones(N); w[1::2] = -1.0; w[0] *= 0.5; w[-1] *= 0.5
    return w

def cc_weights(N, lo, hi):
    """Clenshaw-Curtis weights for the N Lobatto nodes on [lo, hi]."""
    n = N - 1
    w = np.zeros(N)
    for k in range(N):
        s = 0.0
        for j in range(1, n // 2 + 1):
            bj = 1.0 if 2 * j < n else 0.5
            s += bj / (4.0 * j * j - 1.0) * np.cos(2.0 * j * np.pi * k / n)
        ck = 1.0 if 0 < k < n else 0.5
        w[k] = ck * 2.0 / n * (1.0 - 2.0 * s)
    return w * 0.5 * (hi - lo)

def interp_matrix(nodes, w, x):
    """Barycentric interpolation matrix (len(x), N) from values at `nodes` to points x."""
    x = np.asarray(x, dtype=float).ravel()
    d = x[:, None] - nodes[None, :]
    exact = np.abs(d) < 1e-14
    d = np.where(exact, 1.0, d)
    num = w[None, :] / d
    hit = exact.any(axis=1)
    num = np.where(hit[:, None], exact.astype(float), num)
    return num / num.sum(axis=1)[:, None]

def gauss(m, lo, hi):
    """Gauss-Legendre nodes/weights on arrays of intervals [lo, hi] (broadcast); returns (..., m)."""
    x0, w0 = np.polynomial.legendre.leggauss(m)
    lo = np.asarray(lo, float)[..., None]; hi = np.asarray(hi, float)[..., None]
    return lo + (hi - lo) * 0.5 * (x0 + 1.0), 0.5 * (hi - lo) * w0

class Grid:
    def __init__(self, Nt, Nth, m, T=1.0):
        self.Nt, self.Nth, self.m, self.T = Nt, Nth, m, T
        self.tn = lobatto(Nt, 0.0, T); self.thn = lobatto(Nth, 0.0, 1.0)
        self.wt_b = bary_w(Nt); self.wth_b = bary_w(Nth)
        self.wt = cc_weights(Nt, 0.0, T); self.wth = cc_weights(Nth, 0.0, 1.0)
        self.S = self.tn[:, None] * self.thn[None, :]                       # s nodes (Nt, Nth)
        self.W = self.wt[:, None] * self.tn[:, None] * self.wth[None, :]    # triangle quadrature weights
        self._build()

    def interp2d(self, u, s):
        """Interpolation tensors (Lt, Lth) for points (u, s) with s <= u: F(u, s) = Lt . F . Lth."""
        u = np.asarray(u, float); s = np.asarray(s, float)
        phi = np.where(u > 0, s / np.where(u > 0, u, 1.0), 0.0)
        phi = np.clip(phi, 0.0, 1.0)
        Lt = interp_matrix(self.tn, self.wt_b, u.ravel()).reshape(u.shape + (self.Nt,))
        Lth = interp_matrix(self.thn, self.wth_b, phi.ravel()).reshape(u.shape + (self.Nth,))
        return Lt, Lth

    def _build(self):
        Nt, Nth, m = self.Nt, self.Nth, self.m
        tn, S = self.tn, self.S
        tt = np.broadcast_to(tn[:, None], (Nt, Nth))
        # (P1) double integral int_s^t du int_s^u dv: g at (u, v), X at (v, s)
        U, WU = gauss(m, S, tt)                                   # (Nt, Nth, m)
        V, WV = gauss(m, np.broadcast_to(S[..., None], U.shape), U)   # (Nt, Nth, m, m)
        Ub = np.broadcast_to(U[..., None], V.shape)
        Sb = np.broadcast_to(S[..., None, None], V.shape)
        self.P1_w = WU[..., None] * WV                            # (Nt, Nth, m, m)
        self.P1_gLt, self.P1_gLth = self.interp2d(Ub, V)          # g at (u, v)
        self.P1_xLt, self.P1_xLth = self.interp2d(V, Sb)          # X at (v, s)
        # (P2) source int_s^t g_u(s) du: g at (u, s)
        self.P2_w = WU
        self.P2_gLt, self.P2_gLth = self.interp2d(U, np.broadcast_to(S[..., None], U.shape))
        # (P3) calD at nodes: int_s^t g_t(v) X_v(s) dv with t = t_a: g by theta-interpolation in slice a
        Vc, WVc = gauss(m, S, tt)                                 # (Nt, Nth, m)
        self.P3_w = WVc
        phi = np.where(tt[..., None] > 0, Vc / np.where(tt[..., None] > 0, tt[..., None], 1.0), 0.0)
        self.P3_gLth = interp_matrix(self.thn, self.wth_b, phi.ravel()).reshape(Vc.shape + (Nth,))
        self.P3_xLt, self.P3_xLth = self.interp2d(Vc, np.broadcast_to(S[..., None], Vc.shape))
        # convert to jnp
        for k, v in list(self.__dict__.items()):
            if isinstance(v, np.ndarray) and k.startswith('P'):
                setattr(self, k, jnp.asarray(v))
        self.Wj = jnp.asarray(self.W)

# ------------------------------------------------------------------ model and forward map
class Model:
    def __init__(self, grid, p=(3.0, 3.0), r=(0.1, 0.1), sigma=1.0):
        self.g = grid; self.p = p; self.r = r; self.sigma = sigma

    def forward(self, g1, g2):
        """g_i: (Nt, Nth) nodal values of g^i_t(u) at u = t theta.  Returns X (Nt, Nth, 3), calD1, calD2."""
        G = self.g; Nt, Nth = G.Nt, G.Nth
        gs = (g1, g2); sp = [np.sqrt(self.p[0]), np.sqrt(self.p[1])]
        # Volterra operator A[(a,k),(a',k')]: X_t(s) <- sum_i sqrt(p_i) int_s^t du int_s^u dv g^i_u(v) X_v(s)
        A = jnp.zeros((Nt, Nth, Nt, Nth))
        b = jnp.zeros((Nt, Nth, 3)).at[:, :, 0].set(self.sigma)
        for i in range(2):
            gpts = jnp.einsum('akqrA,akqrK,AK->akqr', G.P1_gLt, G.P1_gLth, gs[i])   # g at (u, v)
            wg = sp[i] * G.P1_w * gpts
            A = A + jnp.einsum('akqr,akqrA,akqrK->akAK', wg, G.P1_xLt, G.P1_xLth)
            gsrc = jnp.einsum('akqA,akqK,AK->akq', G.P2_gLt, G.P2_gLth, gs[i])      # g at (u, s)
            b = b.at[:, :, i + 1].add(jnp.einsum('akq,akq->ak', G.P2_w, gsrc))
        n = Nt * Nth
        M = jnp.eye(n) - A.reshape(n, n)
        X = jnp.linalg.solve(M, b.reshape(n, 3)).reshape(Nt, Nth, 3)
        # controls at the nodes
        Xpts_t = jnp.einsum('akqA,akqK,AKc->akqc', G.P3_xLt, G.P3_xLth, X)          # X at (v, s)
        calD = []
        for i in range(2):
            gv = jnp.einsum('akqK,aK->akq', G.P3_gLth, gs[i])                          # g_t(v) in slice a
            cd = sp[i] * jnp.einsum('akq,akq,akqc->akc', G.P3_w, gv, Xpts_t)
            cd = cd.at[:, :, i + 1].add(gs[i])
            calD.append(cd)
        return X, calD[0], calD[1]

    def costs(self, g1, g2):
        X, c1, c2 = self.forward(g1, g2)
        JX = jnp.sum(self.g.Wj * jnp.sum(X * X, axis=-1))
        J1 = JX + self.r[0] * jnp.sum(self.g.Wj * jnp.sum(c1 * c1, axis=-1))
        J2 = JX + self.r[1] * jnp.sum(self.g.Wj * jnp.sum(c2 * c2, axis=-1))
        return J1, J2

# ------------------------------------------------------------------ equilibrium
def solve(Nt=16, Nth=16, m=12, p=(3.0, 3.0), r=(0.1, 0.1), verbose=True, z0=None, iters=30):
    G = Grid(Nt, Nth, m); M = Model(G, p, r)
    n = Nt * Nth
    def unpack(z): return z[:n].reshape(Nt, Nth), z[n:].reshape(Nt, Nth)
    def F(z):
        g1, g2 = unpack(z)
        f1 = jax.grad(lambda a: M.costs(a, g2)[0])(g1)
        f2 = jax.grad(lambda b: M.costs(g1, b)[1])(g2)
        return jnp.concatenate([f1.ravel(), f2.ravel()])
    Fj = jax.jit(F)
    # Jacobian by forward mode in chunks of tangent directions: a single jacfwd batches every
    # intermediate of the forward map over all 2n directions and exhausts the GPU above ~600 unknowns.
    chunk = 32
    jvp_batch = jax.jit(jax.vmap(lambda zz, v: jax.jvp(F, (zz,), (v,))[1], in_axes=(None, 0)))
    def Jf(zz):
        cols = []
        E = jnp.eye(2 * n)
        for c0 in range(0, 2 * n, chunk):
            cols.append(jvp_batch(zz, E[c0:c0 + chunk]))
        return jnp.concatenate(cols, axis=0).T
    z = jnp.zeros(2 * n) if z0 is None else jnp.asarray(z0)
    t0 = time.time()
    for it in range(iters):
        f = Fj(z); nf = float(jnp.linalg.norm(f))
        if verbose: print(f'  it {it}: |F| = {nf:.3e}  ({time.time() - t0:.1f}s)', flush=True)
        if nf < 1e-11: break
        J = Jf(z)
        dz = jnp.linalg.solve(J, -f)
        # backtracking on |F|
        lam = 1.0
        for _ in range(12):
            zn = z + lam * dz; fn = float(jnp.linalg.norm(Fj(zn)))
            if fn < nf: break
            lam *= 0.5
        z = zn
    g1, g2 = unpack(z)
    X, c1, c2 = M.forward(g1, g2)
    J1, J2 = M.costs(g1, g2)
    return dict(grid=G, model=M, z=np.asarray(z), g1=np.asarray(g1), g2=np.asarray(g2), X=np.asarray(X),
                calD1=np.asarray(c1), calD2=np.asarray(c2), J1=float(J1), J2=float(J2))

def eval_at(G, F, t, s):
    """Evaluate a nodal field F (Nt, Nth, ...) at points (t, s)."""
    Lt, Lth = G.interp2d(np.asarray(t, float), np.asarray(s, float))
    return np.einsum('pa,pk,ak...->p...', np.asarray(Lt), np.asarray(Lth), np.asarray(F))

if __name__ == '__main__':
    Nt = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    Nth = int(sys.argv[2]) if len(sys.argv) > 2 else Nt
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print(f'device {jax.devices()[0]}  Nt={Nt} Nth={Nth} m={m}')
    res = solve(Nt, Nth, m)
    print(f'J1 = {res["J1"]:.6f}  J2 = {res["J2"]:.6f}')
    G = res['grid']
    for s in [0.0, 0.1, 0.25, 0.4, 0.49]:
        cd = eval_at(G, res['calD1'], [0.5], [s])[0]; x = eval_at(G, res['X'], [0.5], [s])[0]
        print(f't=0.500 s={s:.3f}  calD1=({cd[0]:.4f} {cd[1]:.4f} {cd[2]:.4f})  X=({x[0]:.4f} {x[1]:.4f} {x[2]:.4f})')
    np.savez(f'/tmp/claude-1000/-home-sbabichenko/996a7341-219f-4d2c-850f-0ef3c55c0d61/scratchpad/spectral/sol_{Nt}_{Nth}_{m}.npz',
             z=res['z'], X=res['X'], calD1=res['calD1'], calD2=res['calD2'], tn=G.tn, thn=G.thn, J1=res['J1'], J2=res['J2'])
