// Spectral-in-time solver for the Chapter 1 finite-horizon two-player LQG game (variance part, then the mean part).
// C++/Eigen port of spec_ch1.py (same grids, quadrature and conventions; results agree to ~1e-10).
//
// Kernels on the triangle 0 <= s <= t <= T in Duffy coordinates (t, theta = s/t), Chebyshev-Lobatto
// nodes in both, barycentric interpolation, Clenshaw-Curtis / Gauss-Legendre quadrature.  Player i's
// control is parametrized by its coefficient on the player's own observation increments g^i_t(u):
//     calD^i_t(s) = sqrt(p_i) int_s^t g^i_t(u) X_u(s) du + g^i_t(s) e_i           (no projection)
//     X_t(s)      = sigma e_0 + int_s^t (calD^1_u(s) + calD^2_u(s)) du             (linear Volterra system)
//     J_i         = int_0^T dt int_0^t ds (|X_t(s)|^2 + r_i |calD^i_t(s)|^2)      (+ Tikhonov penalty)
// The first-order conditions grad_{g^i} J_i = 0 are formed by a hand-written reverse sweep (the
// adjoint of the three stages: operator assembly, linear solve, control quadrature) and solved by
// Newton with a finite-difference Jacobian (columns in parallel).  The t = 0 slice of g is tied to
// the first interior slice and a second-derivative penalty lambda (default 1e-7) removes the
// weakly determined corner modes (see README.md).
//
// After the kernels, the mean (bar) system is solved as a 3 Nt linear problem (see mean_system): the
// deterministic tug-of-war over the targets b_i with the opponent's naive response through its kernel,
// which is where the information wedge acts on the mean.  The closed-loop perfect-information mean path
// is computed alongside as the benchmark.
//
// Usage: spec_ch1 Nt Nth m [--p1 3 --p2 3 --r1 0.1 --r2 0.1 --sigma 1 --T 1 --lambda 1e-7 --tol 1e-10
//                          --b1 1 --b2 -1 --x0 0 --out file --out-mean file --threads t --verbose --dense --pooled]
//        (--pooled: both players observe one common signal, noise channel 1, with precisions p1 = p2)
//        (--dense: FD Jacobian instead of Newton-Krylov; --out-mean: t, Xbar, Dbar1, Dbar2, Dbar1 perfect-info)
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;
using Mat3c = Eigen::Matrix<double, Eigen::Dynamic, 3>;   // (nodes x channels)
using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

// ------------------------------------------------------------------ basics
VectorXd lobatto(int N, double lo, double hi) {
    VectorXd x(N);
    for (int k = 0; k < N; ++k) x[k] = lo + 0.5 * (hi - lo) * (1.0 - std::cos(M_PI * k / (N - 1)));
    return x;
}
VectorXd bary_w(int N) {
    VectorXd w = VectorXd::Ones(N);
    for (int k = 1; k < N; k += 2) w[k] = -1.0;
    w[0] *= 0.5; w[N - 1] *= 0.5;
    return w;
}
VectorXd cc_weights(int N, double lo, double hi) {
    const int n = N - 1; VectorXd w(N);
    for (int k = 0; k < N; ++k) {
        double s = 0.0;
        for (int j = 1; j <= n / 2; ++j) { const double bj = (2 * j < n) ? 1.0 : 0.5; s += bj / (4.0 * j * j - 1.0) * std::cos(2.0 * j * M_PI * k / n); }
        const double ck = (k > 0 && k < n) ? 1.0 : 0.5;
        w[k] = ck * 2.0 / n * (1.0 - 2.0 * s);
    }
    return w * 0.5 * (hi - lo);
}
// barycentric interpolation weights from `nodes` to one point x (row of the interpolation matrix)
void interp_row(const VectorXd& nodes, const VectorXd& w, double x, double* out) {
    const int N = nodes.size();
    for (int j = 0; j < N; ++j) if (std::abs(x - nodes[j]) < 1e-14) { for (int i = 0; i < N; ++i) out[i] = 0.0; out[j] = 1.0; return; }
    double sum = 0.0;
    for (int j = 0; j < N; ++j) { out[j] = w[j] / (x - nodes[j]); sum += out[j]; }
    for (int j = 0; j < N; ++j) out[j] /= sum;
}
// Gauss-Legendre on [-1, 1] (Newton on P_m)
void gauss_ref(int m, VectorXd& x, VectorXd& w) {
    x.resize(m); w.resize(m);
    for (int i = 0; i < m; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (m + 0.5)), pp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= m; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j; }
            pp = m * (z * p1 - p2) / (z * z - 1.0);
            const double dz = p1 / pp; z -= dz; if (std::abs(dz) < 1e-15) break;
        }
        x[i] = -z; w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
}
MatrixXd diffmat(const VectorXd& x) {
    const int N = x.size(); const VectorXd w = bary_w(N); MatrixXd D = MatrixXd::Zero(N, N);
    for (int i = 0; i < N; ++i) { for (int j = 0; j < N; ++j) if (i != j) D(i, j) = w[j] / w[i] / (x[i] - x[j]); D(i, i) = -D.row(i).sum(); }
    return D;
}

// ------------------------------------------------------------------ grid with precomputed quadrature/interpolation
struct Grid {
    int Nt, Nth, m, n; double T;
    VectorXd tn, thn, wt_b, wth_b, wt, wth, gx, gw;   // gx, gw: Gauss reference nodes/weights
    MatrixXd S, W;                                    // s nodes and triangle quadrature weights (Nt x Nth)
    MatrixXd Dt, Dth;                                 // differentiation matrices
    // point sets (row-major over nodes a, k; then quadrature index)
    // P1: (a, k, q, r): weights, g interpolation at (u_q, v_qr) [Lt (Nt), Lth (Nth)], X interpolation at (v_qr, s)
    std::vector<double> P1_w, P1_gLt, P1_gLth, P1_xLt, P1_xLth;
    // P2: (a, k, q): weights, g interpolation at (u_q, s)
    std::vector<double> P2_w, P2_gLt, P2_gLth;
    // P3: (a, k, q): weights, g theta-interpolation in slice a at v_q, X interpolation at (v_q, s)
    std::vector<double> P3_w, P3_gLth, P3_xLt, P3_xLth;

    Grid(int Nt_, int Nth_, int m_, double T_) : Nt(Nt_), Nth(Nth_), m(m_), n(Nt_ * Nth_), T(T_) {
        tn = lobatto(Nt, 0.0, T); thn = lobatto(Nth, 0.0, 1.0);
        wt_b = bary_w(Nt); wth_b = bary_w(Nth);
        wt = cc_weights(Nt, 0.0, T); wth = cc_weights(Nth, 0.0, 1.0);
        S = tn * thn.transpose();
        W.resize(Nt, Nth); for (int a = 0; a < Nt; ++a) for (int k = 0; k < Nth; ++k) W(a, k) = wt[a] * tn[a] * wth[k];
        Dt = diffmat(tn); Dth = diffmat(thn);
        gauss_ref(m, gx, gw);
        build();
    }
    int node(int a, int k) const { return a * Nth + k; }
    // interpolation factors for a point (u, s), s <= u
    void interp2d(double u, double s, double* Lt, double* Lth) const {
        double phi = u > 0.0 ? s / u : 0.0; phi = std::min(1.0, std::max(0.0, phi));
        interp_row(tn, wt_b, u, Lt); interp_row(thn, wth_b, phi, Lth);
    }
    void build() {
        const size_t np1 = static_cast<size_t>(n) * m * m, np = static_cast<size_t>(n) * m;
        P1_w.resize(np1); P1_gLt.resize(np1 * Nt); P1_gLth.resize(np1 * Nth); P1_xLt.resize(np1 * Nt); P1_xLth.resize(np1 * Nth);
        P2_w.resize(np); P2_gLt.resize(np * Nt); P2_gLth.resize(np * Nth);
        P3_w.resize(np); P3_gLth.resize(np * Nth); P3_xLt.resize(np * Nt); P3_xLth.resize(np * Nth);
        for (int a = 0; a < Nt; ++a) for (int k = 0; k < Nth; ++k) {
            const double t = tn[a], s = S(a, k); const size_t nd = node(a, k);
            for (int q = 0; q < m; ++q) {
                const double u = s + (t - s) * 0.5 * (gx[q] + 1.0), wu = 0.5 * (t - s) * gw[q];
                const size_t p2 = nd * m + q;
                P2_w[p2] = wu; interp2d(u, s, &P2_gLt[p2 * Nt], &P2_gLth[p2 * Nth]);
                // P3 uses the same nodes v_q = u_q on [s, t]
                P3_w[p2] = wu; interp_row(thn, wth_b, t > 0.0 ? u / t : 0.0, &P3_gLth[p2 * Nth]); interp2d(u, s, &P3_xLt[p2 * Nt], &P3_xLth[p2 * Nth]);
                for (int r = 0; r < m; ++r) {
                    const double v = s + (u - s) * 0.5 * (gx[r] + 1.0), wv = 0.5 * (u - s) * gw[r];
                    const size_t p1 = (nd * m + q) * m + r;
                    P1_w[p1] = wu * wv;
                    interp2d(u, v, &P1_gLt[p1 * Nt], &P1_gLth[p1 * Nth]);
                    interp2d(v, s, &P1_xLt[p1 * Nt], &P1_xLth[p1 * Nth]);
                }
            }
        }
    }
};

// ------------------------------------------------------------------ model: forward map, costs, gradients
struct Model {
    const Grid& G; double p[2], r[2], sigma, lambda; double sp[2]; int ch[2];   // ch[i]: noise channel of player i's observation (1, 2; pooled: both 1)
    Model(const Grid& g, double p1, double p2, double r1, double r2, double sig, double lam, bool pooled = false) : G(g), sigma(sig), lambda(lam) {
        p[0] = p1; p[1] = p2; r[0] = r1; r[1] = r2; sp[0] = std::sqrt(p1); sp[1] = std::sqrt(p2); ch[0] = 1; ch[1] = pooled ? 1 : 2;
    }
    // g_i: Nt x Nth nodal values.  Fills X, calD[2]; returns the LU of M = I - A (for the adjoint).
    struct Forward { Mat3c X, calD[2]; MatRM A; Eigen::PartialPivLU<MatrixXd> lu; std::vector<double> gpts1[2], gpts3[2]; std::vector<double> xpts3; };   // xpts3: X at the P3 points (3 per point)
    // value of a nodal field at a point given interpolation factors
    static double at(const MatrixXd& F, const double* Lt, const double* Lth, int Nt, int Nth) {
        double v = 0.0;
        for (int a = 0; a < Nt; ++a) { if (Lt[a] == 0.0) continue; double row = 0.0; for (int k = 0; k < Nth; ++k) row += Lth[k] * F(a, k); v += Lt[a] * row; }
        return v;
    }
    void forward(const MatrixXd g[2], Forward& fw) const {
        const int n = G.n, Nt = G.Nt, Nth = G.Nth, m = G.m;
        fw.A = MatRM::Zero(n, n);
        Mat3c b = Mat3c::Zero(n, 3); b.col(0).setConstant(sigma);
        for (int i = 0; i < 2; ++i) { fw.gpts1[i].resize(static_cast<size_t>(n) * m * m); fw.gpts3[i].resize(static_cast<size_t>(n) * m); }
        // operator A and source b
        #pragma omp parallel for schedule(dynamic, 4)
        for (int nd = 0; nd < n; ++nd) {
            std::vector<double> rowbuf(n);
            for (int i = 0; i < 2; ++i) {
                for (int q = 0; q < m; ++q) {
                    const size_t p2 = static_cast<size_t>(nd) * m + q;
                    b(nd, ch[i]) += G.P2_w[p2] * at(g[i], &G.P2_gLt[p2 * Nt], &G.P2_gLth[p2 * Nth], Nt, Nth);
                    for (int rr = 0; rr < m; ++rr) {
                        const size_t p1 = p2 * m + rr;
                        const double gp = at(g[i], &G.P1_gLt[p1 * Nt], &G.P1_gLth[p1 * Nth], Nt, Nth);
                        fw.gpts1[i][p1] = gp;
                        const double c = sp[i] * G.P1_w[p1] * gp;
                        if (c == 0.0) continue;
                        const double* Lt = &G.P1_xLt[p1 * Nt]; const double* Lth = &G.P1_xLth[p1 * Nth];
                        double* Arow = fw.A.data() + static_cast<size_t>(nd) * n;   // contiguous row
                        for (int a = 0; a < Nt; ++a) { if (Lt[a] == 0.0) continue; const double ca = c * Lt[a]; double* dst = Arow + a * Nth; for (int k = 0; k < Nth; ++k) dst[k] += ca * Lth[k]; }
                    }
                }
            }
        }
        MatrixXd M = MatrixXd::Identity(n, n) - fw.A;
        fw.lu.compute(M);
        fw.X = fw.lu.solve(b);
        // X at the P3 points (shared by the controls and the gradient)
        fw.xpts3.assign(static_cast<size_t>(n) * m * 3, 0.0);
        #pragma omp parallel for schedule(static)
        for (int nd = 0; nd < n; ++nd) for (int q = 0; q < m; ++q) {
            const size_t p3 = static_cast<size_t>(nd) * m + q; const double* Lt = &G.P3_xLt[p3 * Nt]; const double* Lth = &G.P3_xLth[p3 * Nth];
            double xp[3] = {0, 0, 0};
            for (int A = 0; A < Nt; ++A) { if (Lt[A] == 0.0) continue; for (int K = 0; K < Nth; ++K) { const double f = Lt[A] * Lth[K]; if (f == 0.0) continue; const int nn = A * Nth + K; for (int ch = 0; ch < 3; ++ch) xp[ch] += f * fw.X(nn, ch); } }
            for (int ch = 0; ch < 3; ++ch) fw.xpts3[p3 * 3 + ch] = xp[ch];
        }
        // controls at the nodes
        for (int i = 0; i < 2; ++i) fw.calD[i] = Mat3c::Zero(n, 3);
        #pragma omp parallel for schedule(static)
        for (int nd = 0; nd < n; ++nd) {
            const int a = nd / Nth;
            for (int i = 0; i < 2; ++i) {
                for (int q = 0; q < m; ++q) {
                    const size_t p3 = static_cast<size_t>(nd) * m + q;
                    double gv = 0.0; for (int K = 0; K < Nth; ++K) gv += G.P3_gLth[p3 * Nth + K] * g[i](a, K);
                    fw.gpts3[i][p3] = gv;
                    const double c = sp[i] * G.P3_w[p3] * gv;
                    for (int ch = 0; ch < 3; ++ch) fw.calD[i](nd, ch) += c * fw.xpts3[p3 * 3 + ch];
                }
                fw.calD[i](nd, ch[i]) += g[i](a, nd % Nth);
            }
        }
    }
    double penalty(const MatrixXd& gi) const {
        if (lambda == 0.0) return 0.0;
        const MatrixXd d2th = gi * G.Dth.transpose() * G.Dth.transpose(), d2t = G.Dt * G.Dt * gi;
        return lambda * ((G.W.array() * d2th.array().square()).sum() + (G.W.array() * d2t.array().square()).sum());
    }
    MatrixXd penalty_grad(const MatrixXd& gi) const {
        if (lambda == 0.0) return MatrixXd::Zero(G.Nt, G.Nth);
        const MatrixXd d2th = gi * G.Dth.transpose() * G.Dth.transpose(), d2t = G.Dt * G.Dt * gi;
        return 2.0 * lambda * ((G.W.array() * d2th.array()).matrix() * G.Dth * G.Dth + G.Dt.transpose() * G.Dt.transpose() * (G.W.array() * d2t.array()).matrix());
    }
    double cost(int i, const Forward& fw) const {
        double J = 0.0;
        for (int nd = 0; nd < G.n; ++nd) { const double w = G.W(nd / G.Nth, nd % G.Nth); J += w * (fw.X.row(nd).squaredNorm() + r[i] * fw.calD[i].row(nd).squaredNorm()); }
        return J;
    }
    // gradient of J_i (variance part, without penalty) with respect to the nodal values of g^i
    MatrixXd gradient(int i, const MatrixXd g[2], const Forward& fw) const {
        const int n = G.n, Nt = G.Nt, Nth = G.Nth, m = G.m;
        Mat3c Xbar(n, 3), cbar(n, 3);
        for (int nd = 0; nd < n; ++nd) { const double w = G.W(nd / Nth, nd % Nth); Xbar.row(nd) = 2.0 * w * fw.X.row(nd); cbar.row(nd) = 2.0 * r[i] * w * fw.calD[i].row(nd); }
        MatrixXd gbar = MatrixXd::Zero(Nt, Nth);
        // (3) calD_i[nd] = sp_i sum_q w3 g_i(slice a, v_q) X(v_q, s) + g_i[nd] e_i
        for (int nd = 0; nd < n; ++nd) {
            const int a = nd / Nth;
            gbar(a, nd % Nth) += cbar(nd, ch[i]);
            for (int q = 0; q < m; ++q) {
                const size_t p3 = static_cast<size_t>(nd) * m + q;
                const double* Lt = &G.P3_xLt[p3 * Nt]; const double* Lth = &G.P3_xLth[p3 * Nth];
                const double* xp = &fw.xpts3[p3 * 3];
                const double c = sp[i] * G.P3_w[p3];
                const double dot = c * (cbar(nd, 0) * xp[0] + cbar(nd, 1) * xp[1] + cbar(nd, 2) * xp[2]);   // d/d gv
                for (int K = 0; K < Nth; ++K) gbar(a, K) += dot * G.P3_gLth[p3 * Nth + K];
                const double cg = c * fw.gpts3[i][p3];
                if (cg != 0.0) for (int A = 0; A < Nt; ++A) { if (Lt[A] == 0.0) continue; for (int K = 0; K < Nth; ++K) { const double f = cg * Lt[A] * Lth[K]; if (f == 0.0) continue; const int nn = A * Nth + K; for (int ch = 0; ch < 3; ++ch) Xbar(nn, ch) += f * cbar(nd, ch); } }
            }
        }
        // (2) X = M^{-1} b:  lambda = M^{-T} Xbar;  bbar = lambda;  Abar = lambda X^T
        Mat3c lam = fw.lu.transpose().solve(Xbar);
        // (1) A and b as functions of g_i
        std::vector<MatrixXd> gpart(
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
            , MatrixXd::Zero(Nt, Nth));
        #pragma omp parallel for schedule(dynamic, 4)
        for (int nd = 0; nd < n; ++nd) {
#ifdef _OPENMP
            MatrixXd& gb = gpart[omp_get_thread_num()];
#else
            MatrixXd& gb = gpart[0];
#endif
            for (int q = 0; q < m; ++q) {
                const size_t p2 = static_cast<size_t>(nd) * m + q;
                const double cb = lam(nd, ch[i]) * G.P2_w[p2];
                if (cb != 0.0) { const double* Lt = &G.P2_gLt[p2 * Nt]; const double* Lth = &G.P2_gLth[p2 * Nth]; for (int A = 0; A < Nt; ++A) { if (Lt[A] == 0.0) continue; for (int K = 0; K < Nth; ++K) gb(A, K) += cb * Lt[A] * Lth[K]; } }
                for (int rr = 0; rr < m; ++rr) {
                    const size_t p1 = p2 * m + rr;
                    const double c = sp[i] * G.P1_w[p1]; if (c == 0.0) continue;
                    // q_p = sum_col Abar[nd, col] Lt_x[col_A] Lth_x[col_K] = sum_ch lam[nd,ch] * X(v, s)[ch]
                    const double* Lt = &G.P1_xLt[p1 * Nt]; const double* Lth = &G.P1_xLth[p1 * Nth];
                    double qp = 0.0;
                    for (int A = 0; A < Nt; ++A) { if (Lt[A] == 0.0) continue; for (int K = 0; K < Nth; ++K) { const double f = Lt[A] * Lth[K]; if (f == 0.0) continue; const int nn = A * Nth + K; qp += f * (lam(nd, 0) * fw.X(nn, 0) + lam(nd, 1) * fw.X(nn, 1) + lam(nd, 2) * fw.X(nn, 2)); } }
                    const double cg = c * qp; if (cg == 0.0) continue;
                    const double* Ltg = &G.P1_gLt[p1 * Nt]; const double* Lthg = &G.P1_gLth[p1 * Nth];
                    for (int A = 0; A < Nt; ++A) { if (Ltg[A] == 0.0) continue; const double f = cg * Ltg[A]; for (int K = 0; K < Nth; ++K) gb(A, K) += f * Lthg[K]; }
                }
            }
        }
        for (auto& gb : gpart) gbar += gb;
        return gbar;
    }
};

// ------------------------------------------------------------------ equilibrium: Newton on the tied unknowns
struct Solver {
    const Grid& G; const Model& M; int nz;   // unknowns per player: (Nt-1) Nth
    Solver(const Grid& g, const Model& m) : G(g), M(m), nz((g.Nt - 1) * g.Nth) {}
    MatrixXd expand(const double* z) const { MatrixXd g(G.Nt, G.Nth); for (int a = 1; a < G.Nt; ++a) for (int k = 0; k < G.Nth; ++k) g(a, k) = z[(a - 1) * G.Nth + k]; g.row(0) = g.row(1); return g; }
    void contract(const MatrixXd& gb, double* out) const { for (int a = 1; a < G.Nt; ++a) for (int k = 0; k < G.Nth; ++k) out[(a - 1) * G.Nth + k] = gb(a, k) + (a == 1 ? gb(0, k) : 0.0); }
    // F(z) = [grad_{g1} (J1 + pen); grad_{g2} (J2 + pen)]
    VectorXd F(const VectorXd& z, double* J = nullptr) const {
        MatrixXd g[2] = {expand(z.data()), expand(z.data() + nz)};
        Model::Forward fw; M.forward(g, fw);
        VectorXd f(2 * nz);
        for (int i = 0; i < 2; ++i) { MatrixXd gb = M.gradient(i, g, fw) + M.penalty_grad(g[i]); contract(gb, f.data() + i * nz); if (J) J[i] = M.cost(i, fw); }
        return f;
    }
    // GMRES(mk) on J dz = -f with finite-difference Jacobian-vector products; returns the step
    VectorXd gmres_step(const VectorXd& z, const VectorXd& f, double eta, int mk, bool verbose, int& evals) const {
        const int N = 2 * nz; const double nf = f.norm();
        auto Jv = [&](const VectorXd& v) { const double nv = v.norm(); if (nv == 0.0) return VectorXd(VectorXd::Zero(N)); const double h = 1e-7 * (1.0 + z.norm()) / nv; ++evals; return VectorXd(((F(z + h * v) - f) / h)); };
        VectorXd dz = VectorXd::Zero(N);
        for (int restart = 0; restart < 10; ++restart) {
            VectorXd r = -f - (restart == 0 ? VectorXd(VectorXd::Zero(N)) : Jv(dz));
            const double beta = r.norm(); if (beta <= eta * nf) break;
            MatrixXd V(N, mk + 1), H = MatrixXd::Zero(mk + 1, mk); VectorXd g = VectorXd::Zero(mk + 1); g[0] = beta;
            std::vector<double> cs(mk), sn(mk); V.col(0) = r / beta; int k = 0;
            for (; k < mk; ++k) {
                VectorXd w = Jv(V.col(k));
                for (int i = 0; i <= k; ++i) { H(i, k) = V.col(i).dot(w); w -= H(i, k) * V.col(i); }
                H(k + 1, k) = w.norm(); if (H(k + 1, k) > 1e-300) V.col(k + 1) = w / H(k + 1, k);
                for (int i = 0; i < k; ++i) { const double t = cs[i] * H(i, k) + sn[i] * H(i + 1, k); H(i + 1, k) = -sn[i] * H(i, k) + cs[i] * H(i + 1, k); H(i, k) = t; }
                const double den = std::hypot(H(k, k), H(k + 1, k)); cs[k] = H(k, k) / den; sn[k] = H(k + 1, k) / den;
                H(k, k) = den; H(k + 1, k) = 0.0; g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
                if (std::abs(g[k + 1]) <= eta * nf) { ++k; break; }
            }
            VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
            dz += V.leftCols(k) * y;
            if (verbose) std::fprintf(stderr, "    gmres restart %d: %d its, residual %.2e (target %.2e)\n", restart, k, std::abs(g[k]), eta * nf);
            if (std::abs(g[k]) <= eta * nf) break;
        }
        return dz;
    }
    VectorXd solve(double tol, bool verbose, bool dense, int max_it = 40) const {
        VectorXd z = VectorXd::Zero(2 * nz);
        const auto t0 = std::chrono::steady_clock::now();
        auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        for (int it = 0; it < max_it; ++it) {
            double J[2]; VectorXd f = F(z, J); const double nf = f.norm();
            if (verbose) std::fprintf(stderr, "  it %d: |F| = %.3e  J1 = %.7f  (%.1fs)\n", it, nf, J[0], secs());
            if (nf < tol) break;
            VectorXd dz;
            if (dense) {
                // finite-difference Jacobian, columns in parallel (each column one gradient evaluation)
                MatrixXd Jm(2 * nz, 2 * nz);
                #pragma omp parallel for schedule(dynamic, 1)
                for (int c = 0; c < 2 * nz; ++c) {
                    VectorXd zp = z; const double h = 1e-6 * (1.0 + std::abs(z[c])); zp[c] += h;
                    Jm.col(c) = (F(zp) - f) / h;
                }
                dz = Jm.partialPivLu().solve(-f);
            } else {
                // inexact Newton: forcing term eta = min(0.1, sqrt(|F|)) (Eisenstat-Walker style), GMRES(60)
                int evals = 0; const double eta = std::min(0.1, std::sqrt(nf));
                dz = gmres_step(z, f, eta, 60, verbose, evals);
                if (verbose) std::fprintf(stderr, "    %d Jacobian-vector products\n", evals);
            }
            double lam = 1.0; VectorXd zn;
            for (int k = 0; k < 12; ++k) { zn = z + lam * dz; if (F(zn).norm() < nf) break; lam *= 0.5; }
            z = zn;
        }
        return z;
    }
};

// evaluate a nodal field at (t, s)
double eval_at(const Grid& G, const MatrixXd& F, double t, double s) {
    std::vector<double> Lt(G.Nt), Lth(G.Nth); G.interp2d(t, s, Lt.data(), Lth.data());
    return Model::at(F, Lt.data(), Lth.data(), G.Nt, G.Nth);
}

// ------------------------------------------------------------------ mean (bar) system
// Given the equilibrium kernels, the mean paths solve a deterministic linear-quadratic game with a
// Volterra feedback: player i chooses its mean control freely, while the opponent j reacts to the mean
// state through its kernel on raw observations, delta Dbar^j_t = sqrt(p_j) int_0^t g^j_t(u) Xbar_u du
// (the naive response: j reads the mean movement as evidence about the shocks).  Pontryagin gives
//     Xbar' = Dbar^1 + Dbar^2,  Xbar(0) = x0,        Dbar^i = lambda^i / (2 r_i),
//     lambda^i'(u) = 2 (Xbar_u - b_i) - sqrt(p_j) int_u^T lambda^i_t g^j_t(u) dt,   lambda^i(T) = 0,
// a linear system in (Xbar, lambda^1, lambda^2) on the Chebyshev-Lobatto t-nodes.  Integration
// operators come from the differentiation matrix with one row replaced by the boundary condition.
struct MeanResult { VectorXd Xbar, D[2], lam[2]; double J[2]; };
MeanResult mean_system(const Grid& G, const Model& M, const MatrixXd g[2], double b1, double b2, double x0) {
    const int Nt = G.Nt;
    // I0 f = int_0^t f ;  IT f = int_t^T f
    MatrixXd D0 = G.Dt; D0.row(0).setZero(); D0(0, 0) = 1.0;
    MatrixXd I0 = D0.inverse(); I0.col(0).setZero();
    MatrixXd DT = G.Dt; DT.row(Nt - 1).setZero(); DT(Nt - 1, Nt - 1) = 1.0;
    MatrixXd IT = -DT.inverse(); IT.col(Nt - 1).setZero();
    // K^j: (K^j lam)(u_a) = int_{u_a}^T lam_t g^j_t(u_a) dt, Gauss quadrature in t with barycentric interpolation
    MatrixXd K[2] = {MatrixXd::Zero(Nt, Nt), MatrixXd::Zero(Nt, Nt)};
    std::vector<double> Lt(Nt), Lt2(Nt), Lth2(G.Nth);
    for (int a = 0; a < Nt; ++a) {
        const double u = G.tn[a]; if (u >= G.T) continue;
        for (int q = 0; q < G.m; ++q) {
            const double t = u + (G.T - u) * 0.5 * (G.gx[q] + 1.0), w = 0.5 * (G.T - u) * G.gw[q];
            interp_row(G.tn, G.wt_b, t, Lt.data());
            G.interp2d(t, u, Lt2.data(), Lth2.data());
            for (int j = 0; j < 2; ++j) {
                const double gv = Model::at(g[j], Lt2.data(), Lth2.data(), Nt, G.Nth);
                for (int c = 0; c < Nt; ++c) K[j](a, c) += w * gv * Lt[c];
            }
        }
    }
    const double b[2] = {b1, b2};
    MatrixXd A = MatrixXd::Zero(3 * Nt, 3 * Nt); VectorXd rhs = VectorXd::Zero(3 * Nt);
    const MatrixXd I = MatrixXd::Identity(Nt, Nt); const VectorXd one = VectorXd::Ones(Nt);
    A.block(0, 0, Nt, Nt) = I;
    for (int i = 0; i < 2; ++i) A.block(0, (i + 1) * Nt, Nt, Nt) = -I0 / (2.0 * M.r[i]);
    rhs.head(Nt) = x0 * one;
    for (int i = 0; i < 2; ++i) {
        const int j = 1 - i;
        // lambda^i_u = -int_u^T [2 (Xbar - b_i) - sqrt(p_j) K^j lambda^i] dt
        A.block((i + 1) * Nt, 0, Nt, Nt) = 2.0 * IT;
        A.block((i + 1) * Nt, (i + 1) * Nt, Nt, Nt) = I - M.sp[j] * IT * K[j];
        rhs.segment((i + 1) * Nt, Nt) = 2.0 * b[i] * (IT * one);
    }
    const VectorXd v = A.partialPivLu().solve(rhs);
    MeanResult R; R.Xbar = v.head(Nt);
    for (int i = 0; i < 2; ++i) {
        R.lam[i] = v.segment((i + 1) * Nt, Nt); R.D[i] = R.lam[i] / (2.0 * M.r[i]);
        R.J[i] = 0.0;
        for (int a = 0; a < Nt; ++a) R.J[i] += G.wt[a] * ((R.Xbar[a] - b[i]) * (R.Xbar[a] - b[i]) + M.r[i] * R.D[i][a] * R.D[i][a]);
    }
    return R;
}
// closed-loop (feedback) Nash mean path of the perfect-information game, RK4 on a fine grid:
// V_i = S_i X^2 + Q_i X + c_i,  D_i = -(S_i/r_i) X - Q_i/(2 r_i),
// -S_i' = 1 - S_i^2/r_i - 2 S_i S_j/r_j,   Q_i' = 2 b_i + S_i Q_j/r_j + Q_i (S_i/r_i + S_j/r_j),  S_i(T) = Q_i(T) = 0.
void perfect_info_mean(const Model& M, double T, double b1, double b2, double x0, int nfine, std::vector<double>& t, std::vector<double>& D1) {
    const double r1 = M.r[0], r2 = M.r[1]; const double h = T / nfine;
    std::vector<double> S1(nfine + 1), S2(nfine + 1), Q1(nfine + 1), Q2(nfine + 1);
    auto f = [&](const double y[4], double out[4]) {
        const double s1 = y[0], s2 = y[1], q1 = y[2], q2 = y[3];
        out[0] = -(1.0 - s1 * s1 / r1 - 2.0 * s1 * s2 / r2); out[1] = -(1.0 - s2 * s2 / r2 - 2.0 * s1 * s2 / r1);
        out[2] = 2.0 * b1 + s1 * q2 / r2 + q1 * (s1 / r1 + s2 / r2); out[3] = 2.0 * b2 + s2 * q1 / r1 + q2 * (s2 / r2 + s1 / r1);
    };
    double y[4] = {0, 0, 0, 0}; S1[nfine] = S2[nfine] = Q1[nfine] = Q2[nfine] = 0.0;
    for (int k = nfine; k > 0; --k) {   // backward with step -h
        double k1[4], k2[4], k3[4], k4[4], yt[4];
        f(y, k1); for (int c = 0; c < 4; ++c) yt[c] = y[c] - 0.5 * h * k1[c];
        f(yt, k2); for (int c = 0; c < 4; ++c) yt[c] = y[c] - 0.5 * h * k2[c];
        f(yt, k3); for (int c = 0; c < 4; ++c) yt[c] = y[c] - h * k3[c];
        f(yt, k4); for (int c = 0; c < 4; ++c) y[c] -= h / 6.0 * (k1[c] + 2 * k2[c] + 2 * k3[c] + k4[c]);
        S1[k - 1] = y[0]; S2[k - 1] = y[1]; Q1[k - 1] = y[2]; Q2[k - 1] = y[3];
    }
    t.resize(nfine + 1); D1.resize(nfine + 1); double X = x0;
    for (int k = 0; k <= nfine; ++k) {
        t[k] = k * h; const double d1 = -(S1[k] / r1) * X - Q1[k] / (2.0 * r1), d2 = -(S2[k] / r2) * X - Q2[k] / (2.0 * r2); D1[k] = d1;
        if (k < nfine) X += h * (d1 + d2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s Nt Nth m [--p1 --p2 --r1 --r2 --sigma --T --lambda --tol --b1 --b2 --x0 --out file --out-mean file --threads t --verbose --dense --pooled]\n", argv[0]); return 1; }
    const int Nt = std::atoi(argv[1]), Nth = std::atoi(argv[2]), m = std::atoi(argv[3]);
    double p1 = 3.0, p2 = 3.0, r1 = 0.1, r2 = 0.1, sigma = 1.0, T = 1.0, lambda = 1e-7, tol = 1e-10; std::string out, out_mean; bool verbose = false, dense = false, pooled = false;
    double b1 = 1.0, b2 = -1.0, x0 = 0.0;   // mean part: targets and initial state
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](double& d) { if (i + 1 < argc) d = std::atof(argv[++i]); };
        if (a == "--p1") val(p1); else if (a == "--p2") val(p2); else if (a == "--r1") val(r1); else if (a == "--r2") val(r2);
        else if (a == "--sigma") val(sigma); else if (a == "--T") val(T); else if (a == "--lambda") val(lambda); else if (a == "--tol") val(tol);
        else if (a == "--b1") val(b1); else if (a == "--b2") val(b2); else if (a == "--x0") val(x0);
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--out-mean" && i + 1 < argc) out_mean = argv[++i];
        else if (a == "--threads" && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(std::atoi(argv[++i]));
#else
            ++i;
#endif
        } else if (a == "--verbose") verbose = true; else if (a == "--dense") dense = true; else if (a == "--pooled") pooled = true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Grid G(Nt, Nth, m, T);
    Model M(G, p1, p2, r1, r2, sigma, lambda, pooled);
    Solver S(G, M);
    if (verbose) std::fprintf(stderr, "grid %dx%d m=%d: %d unknowns per player, setup %.2fs\n", Nt, Nth, m, S.nz, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (std::getenv("SPEC_GRADCHECK")) {   // hand adjoint vs central differences of J_i + penalty at a random point
        VectorXd z = VectorXd::Random(2 * S.nz) * 0.5; VectorXd f = S.F(z);
        auto Jtot = [&](const VectorXd& zz, int i) { MatrixXd g[2] = {S.expand(zz.data()), S.expand(zz.data() + S.nz)}; Model::Forward fw; M.forward(g, fw); return M.cost(i, fw) + M.penalty(g[i]); };
        double maxrel = 0.0;
        for (int c : {0, 3, S.nz / 2, S.nz - 1, S.nz, S.nz + 7, 2 * S.nz - 1}) {
            const int i = c < S.nz ? 0 : 1; const double h = 1e-5; VectorXd zp = z, zm = z; zp[c] += h; zm[c] -= h;
            const double fd = (Jtot(zp, i) - Jtot(zm, i)) / (2 * h);
            std::printf("  dJ/dz[%d]: adjoint %.10e  FD %.10e  rel %.1e\n", c, f[c], fd, std::abs(f[c] - fd) / (std::abs(fd) + 1e-12));
            maxrel = std::max(maxrel, std::abs(f[c] - fd) / (std::abs(fd) + 1e-12));
        }
        std::printf("gradient check max rel %.1e\n", maxrel); return 0;
    }
    VectorXd z = S.solve(tol, verbose, dense);
    MatrixXd g[2] = {S.expand(z.data()), S.expand(z.data() + S.nz)};
    Model::Forward fw; M.forward(g, fw);
    const double J1 = M.cost(0, fw), J2 = M.cost(1, fw);
    std::printf("J1 = %.7f  J2 = %.7f  (%.1fs)\n", J1, J2, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    // profile of calD1 at t = T/2
    const double tm = 0.5 * T;
    for (double lag : {0.0127, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4}) {
        double c[3], x[3];
        for (int ch = 0; ch < 3; ++ch) { MatrixXd F(Nt, Nth), Fx(Nt, Nth); for (int nd = 0; nd < G.n; ++nd) { F(nd / Nth, nd % Nth) = fw.calD[0](nd, ch); Fx(nd / Nth, nd % Nth) = fw.X(nd, ch); } c[ch] = eval_at(G, F, tm, tm - lag); x[ch] = eval_at(G, Fx, tm, tm - lag); }
        std::printf("t=%.3f lag=%.4f  calD1=(%.4f %.4f %.4f)  X=(%.4f %.4f %.4f)\n", tm, lag, c[0], c[1], c[2], x[0], x[1], x[2]);
    }
    // mean part
    {
        const MeanResult R = mean_system(G, M, g, b1, b2, x0);
        std::vector<double> tpi, D1pi; perfect_info_mean(M, T, b1, b2, x0, 4000, tpi, D1pi);
        std::vector<double> Lt(Nt);
        auto at_t = [&](const VectorXd& F, double t) { interp_row(G.tn, G.wt_b, t, Lt.data()); double v = 0.0; for (int a = 0; a < Nt; ++a) v += Lt[a] * F[a]; return v; };
        std::printf("mean: Dbar1(0) = %.6f  Dbar1(T/2) = %.6f  Dbar2(0) = %.6f  Xbar(T/2) = %.3e  Jbar1 = %.6f  Jbar2 = %.6f  | perfect-info (closed-loop) Dbar1(0) = %.6f\n",
                    R.D[0][0], at_t(R.D[0], 0.5 * T), R.D[1][0], at_t(R.Xbar, 0.5 * T), R.J[0], R.J[1], D1pi[0]);
        if (!out_mean.empty()) {
            FILE* f = std::fopen(out_mean.c_str(), "w");
            std::fprintf(f, "# p1 %.10g p2 %.10g r1 %.10g r2 %.10g sigma %.10g T %.10g b1 %.10g b2 %.10g x0 %.10g Nt %d Nth %d m %d lambda %.3g\n", p1, p2, r1, r2, sigma, T, b1, b2, x0, Nt, Nth, m, lambda);
            std::fprintf(f, "# Jbar1 %.12g Jbar2 %.12g Jvar1 %.12g Jvar2 %.12g\n", R.J[0], R.J[1], J1, J2);
            std::fprintf(f, "# t Xbar Dbar1 Dbar2 Dbar1_perfect_info\n");
            const int nu = 200;
            for (int k = 0; k <= nu; ++k) {
                const double t = T * k / nu; const int kp = static_cast<int>(std::lround(t / T * 4000));
                std::fprintf(f, "%.10g %.12g %.12g %.12g %.12g\n", t, at_t(R.Xbar, t), at_t(R.D[0], t), at_t(R.D[1], t), D1pi[kp]);
            }
            std::fclose(f);
        }
    }
    if (!out.empty()) {
        FILE* f = std::fopen(out.c_str(), "w");
        std::fprintf(f, "# a k t s g1 g2 X0 X1 X2 calD1_0 calD1_1 calD1_2 calD2_0 calD2_1 calD2_2\n");
        for (int nd = 0; nd < G.n; ++nd) { const int a = nd / Nth, k = nd % Nth; std::fprintf(f, "%d %d %.12g %.12g %.12g %.12g", a, k, G.tn[a], G.S(a, k), g[0](a, k), g[1](a, k)); for (int ch = 0; ch < 3; ++ch) std::fprintf(f, " %.12g", fw.X(nd, ch)); for (int i = 0; i < 2; ++i) for (int ch = 0; ch < 3; ++ch) std::fprintf(f, " %.12g", fw.calD[i](nd, ch)); std::fprintf(f, "\n"); }
        std::fclose(f);
    }
    return 0;
}
