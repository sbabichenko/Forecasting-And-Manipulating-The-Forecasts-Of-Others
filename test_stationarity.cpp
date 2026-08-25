// Stationarity test: the solver's fixed point must be a best response.  Finite differences of
// the discrete cost with respect to individual entries of player 1's kernel (Nash deviation:
// player 2's strategy and both mean controls held at their equilibrium values, player 1's own
// projection frozen, as in the control appendix) are compared with the same gradient at
// 0.9 x the equilibrium kernel.  The ratio must be small and shrink with N in the interior;
// the coordinate born at t and the band beside it are first-order in dt.
//   usage: test_stationarity N p1 p2 r [scale=1.0] [freeze_own_filter=1] [diag_cost=0]
#include "lqg_solver.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
// Per-row exact projection (copy of the solver's CERowFilter); `Xsrc` is the state kernel the
// filter is built from, which lets player 1's projection be frozen at the equilibrium state.
struct RowFilter {
    int n = 0, rank = 0, obs = 0; double g = 0.0; Eigen::MatrixXd V; Eigen::VectorXd h, v, coeff, Xvec, Dvec;
    void reset(int n_, int obs_, double g_) { n = n_; obs = obs_; g = g_; rank = 0; V = Eigen::MatrixXd::Zero(3 * n, n); h.setZero(3 * n); v.setZero(3 * n); Xvec.setZero(3 * n); Dvec.setZero(3 * n); coeff.setZero(n); }
    void row(int j, const Kernel2D& Xsrc, const Kernel2D& X, const Kernel2D& D, Kernel2D& Xt, Kernel2D& calD) {
        if (j == 0) { Xt[0][0] = X[0][0]; calD[0][0].setZero(); return; }
        const int active = 3 * (j + 1); auto Vact = V.topRows(active).leftCols(rank);
        h.head(active).setZero(); for (int z = 0; z <= j; ++z) h.segment<3>(3 * z) = g * g_dt * Xsrc[j][z]; h(3 * j + obs) += 1.0;
        v.head(active) = h.head(active); if (rank > 0) { auto c = coeff.head(rank); c.noalias() = Vact.transpose() * h.head(active); v.head(active).noalias() -= Vact * c; }
        const double vn = v.head(active).norm(); if (vn > 1e-15) { V.col(rank).head(active) = v.head(active) / vn; ++rank; }
        auto Vr = V.topRows(active).leftCols(rank); auto c = coeff.head(rank);
        for (int z = 0; z <= j; ++z) Xvec.segment<3>(3 * z) = X[j][z];
        if (rank > 0) { c.noalias() = Vr.transpose() * Xvec.head(active); Xvec.head(active).noalias() -= Vr * c; }
        for (int z = 0; z <= j; ++z) Xt[j][z] = Xvec.segment<3>(3 * z);
        for (int z = 0; z <= j; ++z) Dvec.segment<3>(3 * z) = D[j][z];
        if (rank > 0) { c.noalias() = Vr.transpose() * Dvec.head(active); Dvec.head(active).noalias() = Vr * c; } else Dvec.head(active).setZero();
        for (int z = 0; z <= j; ++z) calD[j][z] = Dvec.segment<3>(3 * z);
        calD[j][0](obs) = 0.0;
    }
};
// march; if Xfreeze1 != nullptr player 1's projection is built from it instead of the live state
static void march(const Kernel2D& D1, const Kernel2D& D2, double g1, double g2, const Kernel2D* Xfreeze1, Kernel2D& X, Kernel2D& Xt1, Kernel2D& Xt2, Kernel2D& c1, Kernel2D& c2) {
    RowFilter F1, F2; F1.reset(g_n, 1, g1); F2.reset(g_n, 2, g2); X.setZero(); c1.setZero(); c2.setZero();
    const Vec3 sigE0 = g_sigma * E0();
    for (int j = 0; j < g_n; ++j) {
        X[j][j] = sigE0; for (int s = 0; s < j; ++s) X[j][s] = X[j - 1][s] + g_dt * (c1[j - 1][s] + c2[j - 1][s]);
        F1.row(j, Xfreeze1 ? *Xfreeze1 : X, X, D1, Xt1, c1); F2.row(j, X, X, D2, Xt2, c2);
    }
}
int main(int argc, char** argv) {
    const int n = std::atoi(argv[1]); const double p1 = std::atof(argv[2]), p2 = std::atof(argv[3]), r = std::atof(argv[4]); const double scale = std::atof(argv[5]); const bool freeze = std::atoi(argv[6]); const bool diag_cost = argc > 7 && std::atoi(argv[7]);
    SolverContext ctx = SolverContext::capture_current(); ctx.n = n; ctx.T = 1.0; ctx.b1 = 1.0; ctx.b2 = -1.0; ctx.r1 = r; ctx.r2 = r; ctx.sigma = 1.0; ctx.terminal_weight = 0.0;
    ScopedSolverContext guard(ctx);
    const double g1 = std::sqrt(p1), g2 = std::sqrt(p2);
    auto eq = solve_equilibrium(g1, g2, false);
    Kernel2D D1 = eq.D1, D2 = eq.D2; for (auto& v : D1.data) v *= scale;
    Kernel2D Xeq = eq.env.X;                              // player 1's projection frozen at the equilibrium state
    const BarSolution bar0 = solve_bar_equilibrium(eq.env, eq.D1, eq.D2, p1, p2, 2000, 0.08, 1e-12);
    auto cost = [&](const Kernel2D& d1) {
        EnvironmentResult env; Kernel2D c1, c2; env.X.resize(); env.Xtilde1.resize(); env.Xtilde2.resize(); c1.resize(); c2.resize();
        march(d1, D2, g1, g2, freeze ? &Xeq : nullptr, env.X, env.Xtilde1, env.Xtilde2, c1, c2);
        if (!diag_cost) return compute_costs_general(env, c1, c2, bar0, r, r, 1.0, -1.0).J1;
        double J = 0.0;
        for (int j = 0; j < g_n; ++j) { double vx = 0, vd = 0; for (int s = 0; s <= j; ++s) { vx += g_dt * env.X[j][s].squaredNorm(); vd += g_dt * c1[j][s].squaredNorm(); } J += g_dt * (vx + r * vd); }
        return J;
    };
    const double eps = 1e-5;
    // per-entry: FD gradient at the fixed point (scale 1) and at 0.9 x equilibrium, ratio
    Kernel2D D1eq = eq.D1;
    std::printf("N=%d  entries (t,s,c): |dJ/dD| at fixed point | at 0.9 x eq | ratio\n", n);
    for (int t : {n / 2, n - 2}) for (int s : {0, 1, 2, t / 2, t - 2, t - 1, t}) for (int c : {0, 1, 2}) {
        if (s > t) continue;
        double g[2];
        for (int k = 0; k < 2; ++k) {
            D1 = D1eq; if (k == 1) for (auto& v : D1.data) v *= 0.9;
            Kernel2D Dp = D1, Dm = D1; Dp[t][s](c) += eps; Dm[t][s](c) -= eps;
            g[k] = (cost(Dp) - cost(Dm)) / (2 * eps);
        }
        std::printf("  t=%3d s=%3d c=%d | %10.3e | %10.3e | %6.3f\n", t, s, c, g[0], g[1], g[1] != 0 ? std::fabs(g[0] / g[1]) : 0.0);
    }
}
