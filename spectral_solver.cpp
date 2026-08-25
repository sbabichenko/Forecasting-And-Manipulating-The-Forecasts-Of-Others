// Spectral (piecewise-Chebyshev) solver for the stationary two-player LQG
// benchmark of Chapter 3 (average cost, A = 0, B^X_i = 1, G^{XX} = 1).
//
// Kernels on [0, L] are stored as values on N Chebyshev-Lobatto nodes; the
// two-sided adjoint and wedge on [-L, L] use two panels split at 0, where
// they have a kink.  Every operator is a dense matrix assembled from
// barycentric interpolation and Gauss-Legendre quadrature, so the
// discretization is spectrally accurate on functions smooth on each panel.
// The forward block uses the exact-projection formulation (x_hat = P x with
// P = Ht (H Ht)^{-1} H), the backward block solves the wedge as a linear
// system, and the outer equilibrium is Newton with a finite-difference
// Jacobian and a backtracking line search.
//
// Usage: solve_spectral p1 p2 r1 r2 [--N 24] [--L 3] [--tol 1e-12] [--pre 15]
//        [--relax 0.1] [--jac-ftol 1e-9] [--pre-ftol 1e-6] [--threads t]
//        [--uniform n] [--eval-only] [--verbose]
// Prints one JSON object with nodal values (and optionally values on a
// uniform lag grid of n points per side for comparison with the FD solver).

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// ---------------------------------------------------------------- basics

VectorXd cheb_nodes(int N, double lo, double hi) {
    VectorXd x(N);
    for (int k = 0; k < N; ++k)
        x[k] = lo + 0.5 * (hi - lo) * (1.0 - std::cos(M_PI * k / (N - 1)));
    return x;
}

VectorXd bary_weights(int N) {
    VectorXd w = VectorXd::Ones(N);
    for (int k = 1; k < N; k += 2) w[k] = -1.0;
    w[0] *= 0.5;
    w[N - 1] *= 0.5;
    return w;
}

// Gauss-Legendre nodes and weights on [lo, hi] (Golub-Welsch).
void gauss_legendre(int m, double lo, double hi, VectorXd& x, VectorXd& w) {
    static std::vector<std::pair<VectorXd, VectorXd>> cache;   // indexed by m
    if (static_cast<int>(cache.size()) <= m) cache.resize(m + 1);
    if (cache[m].first.size() == 0) {
        MatrixXd J = MatrixXd::Zero(m, m);
        for (int k = 1; k < m; ++k) {
            const double b = k / std::sqrt(4.0 * k * k - 1.0);
            J(k, k - 1) = J(k - 1, k) = b;
        }
        Eigen::SelfAdjointEigenSolver<MatrixXd> es(J);
        cache[m].first = es.eigenvalues();
        cache[m].second = 2.0 * es.eigenvectors().row(0).array().square();
    }
    x = lo + 0.5 * (hi - lo) * (cache[m].first.array() + 1.0);
    w = 0.5 * (hi - lo) * cache[m].second.array();
}

struct Panel {
    int N;
    double lo, hi;
    VectorXd x, w;
    Panel(int N_, double lo_, double hi_) : N(N_), lo(lo_), hi(hi_), x(cheb_nodes(N_, lo_, hi_)), w(bary_weights(N_)) {}
    // rows map nodal values to values at pts
    MatrixXd interp(const VectorXd& pts) const {
        MatrixXd M = MatrixXd::Zero(pts.size(), N);
        for (int r = 0; r < pts.size(); ++r) {
            const double p = pts[r];
            int hit = -1;
            for (int k = 0; k < N; ++k)
                if (std::abs(p - x[k]) < 1e-14 * std::max(1.0, std::abs(p))) { hit = k; break; }
            if (hit >= 0) { M(r, hit) = 1.0; continue; }
            double s = 0.0;
            for (int k = 0; k < N; ++k) { M(r, k) = w[k] / (p - x[k]); s += M(r, k); }
            M.row(r) /= s;
        }
        return M;
    }
};

// Two-sided panel pair on [-L, 0] and [0, L]; values stacked (neg, pos).
struct TwoSided {
    int N;
    double L;
    Panel neg, pos;
    VectorXd l;
    TwoSided(int N_, double L_) : N(N_), L(L_), neg(N_, -L_, 0.0), pos(N_, 0.0, L_), l(2 * N_) {
        l << neg.x, pos.x;
    }
    MatrixXd interp(const VectorXd& pts) const {
        MatrixXd M = MatrixXd::Zero(pts.size(), 2 * N);
        for (int r = 0; r < pts.size(); ++r) {
            VectorXd one(1); one << pts[r];
            if (pts[r] < 0) M.block(r, 0, 1, N) = neg.interp(one);
            else            M.block(r, N, 1, N) = pos.interp(one);
        }
        return M;
    }
};

// ------------------------------------------------------------- forward

struct Forward {
    int N, m;
    double L;
    Panel K;
    VectorXd e0;
    // tensors: HS[j] (m x N), TY[j] (m x N); HW, TW (N x m); HX, TX ((N m) x N)
    std::vector<MatrixXd> HS, TY;
    MatrixXd HW, TW, HX, TX, cumint, AH, AT;
    VectorXd wfull;

    Forward(int N_, double L_) : N(N_), m(N_ + 8), L(L_), K(N_, 0.0, L_), e0(3) {
        e0 << 1.0, 0.0, 0.0;
        HS.assign(N, MatrixXd::Zero(m, N));
        TY.assign(N, MatrixXd::Zero(m, N));
        HW = MatrixXd::Zero(N, m); TW = MatrixXd::Zero(N, m);
        VectorXd hu_all(N * m), tu_all(N * m);
        hu_all.setZero(); tu_all.setZero();
        VectorXd u, w;
        for (int j = 0; j < N; ++j) {
            const double a = K.x[j];
            if (L - a > 0) {
                gauss_legendre(m, 0.0, L - a, u, w);
                HW.row(j) = w; HS[j] = K.interp((u.array() + a).matrix());
                hu_all.segment(j * m, m) = u;
            }
            if (a > 0) {
                gauss_legendre(m, 0.0, a, u, w);
                TW.row(j) = w; TY[j] = K.interp(u);
                tu_all.segment(j * m, m) = (a - u.array()).matrix();
            }
        }
        HX = K.interp(hu_all);
        TX = K.interp(tu_all);
        // H and Ht are linear in x: precompute AH, AT (N x N^2) with
        //   H(j, ch*N + i)  = gain * sum_k x(k, ch) AH(k, j*N + i),
        //   Ht(ch*N + j, i) = gain * sum_k x(k, ch) AT(k, j*N + i).
        AH = MatrixXd::Zero(N, N * N); AT = MatrixXd::Zero(N, N * N);
        for (int j = 0; j < N; ++j)
            for (int q = 0; q < m; ++q) {
                AH.block(0, j * N, N, N).noalias() += HW(j, q) * HX.row(j * m + q).transpose() * HS[j].row(q);
                AT.block(0, j * N, N, N).noalias() += TW(j, q) * TX.row(j * m + q).transpose() * TY[j].row(q);
            }
        // cumulative integral (C f)(a_j) = int_0^{a_j} f
        cumint = MatrixXd::Zero(N, N);
        for (int j = 0; j < N; ++j) {
            if (K.x[j] > 0) {
                gauss_legendre(m, 0.0, K.x[j], u, w);
                cumint.row(j) = w.transpose() * K.interp(u);
            }
        }
        gauss_legendre(m, 0.0, L, u, w);
        wfull = (w.transpose() * K.interp(u)).transpose();
    }

    // H (N x 3N) and Ht (3N x N); flat vectors are channel-major.
    void observation_ops(const MatrixXd& x, double gain, int chan, MatrixXd& H, MatrixXd& Ht) const {
        H.resize(N, 3 * N);
        Ht.resize(3 * N, N);
        Eigen::RowVectorXd hrow(N * N), trow(N * N);
        for (int ch = 0; ch < 3; ++ch) {
            hrow.noalias() = gain * (x.col(ch).transpose() * AH);
            trow.noalias() = gain * (x.col(ch).transpose() * AT);
            for (int j = 0; j < N; ++j) {
                H.block(j, ch * N, 1, N) = hrow.segment(j * N, N);
                Ht.block(ch * N + j, 0, 1, N) = trow.segment(j * N, N);
            }
        }
        for (int j = 0; j < N; ++j) { H(j, chan * N + j) += 1.0; Ht(chan * N + j, j) += 1.0; }
    }

    static VectorXd flat(const MatrixXd& v) {   // (N,3) -> channel-major
        VectorXd s(v.rows() * 3);
        for (int c = 0; c < 3; ++c) s.segment(c * v.rows(), v.rows()) = v.col(c);
        return s;
    }
    static MatrixXd unflat(const VectorXd& s, int N) {
        MatrixXd v(N, 3);
        for (int c = 0; c < 3; ++c) v.col(c) = s.segment(c * N, N);
        return v;
    }

    MatrixXd state_from_controls(const MatrixXd& csum) const {
        MatrixXd x = MatrixXd::Zero(N, 3);
        x.col(0).setConstant(e0[0]);
        return x + cumint * csum;
    }

    struct Result { MatrixXd x, c1, c2, xhat1, xhat2, xtilde1, xtilde2; int iters; double res; };

    // Fixed point in x with Anderson acceleration (depth 4).
    Result solve(const MatrixXd& d1, const MatrixXd& d2, double g1, double g2,
                 const MatrixXd* x0, double tol = 1e-13, int maxit = 100) const {
        Result r;
        MatrixXd c1 = MatrixXd::Zero(N, 3), c2 = MatrixXd::Zero(N, 3);
        c1.col(1) = d1.col(1); c2.col(2) = d2.col(2);
        MatrixXd x = x0 ? *x0 : state_from_controls(c1 + c2);
        const int depth = 4;
        std::vector<VectorXd> X, F;
        // projection applied to the columns of V (3N x k): Ht (H Ht)^{-1} H V
        Eigen::PartialPivLU<MatrixXd> lu1, lu2;
        MatrixXd H1, Ht1, H2, Ht2;
        auto factor = [&](const MatrixXd& xx) {
            observation_ops(xx, g1, 1, H1, Ht1); lu1.compute(H1 * Ht1);
            observation_ops(xx, g2, 2, H2, Ht2); lu2.compute(H2 * Ht2);
        };
        auto apply1 = [&](const VectorXd& v) { return VectorXd(Ht1 * lu1.solve(H1 * v)); };
        auto apply2 = [&](const VectorXd& v) { return VectorXd(Ht2 * lu2.solve(H2 * v)); };
        const VectorXd d1f = flat(d1), d2f = flat(d2);
        int it = 0;
        for (; it < maxit; ++it) {
            factor(x);
            c1 = unflat(apply1(d1f), N); c2 = unflat(apply2(d2f), N);
            const MatrixXd gx = state_from_controls(c1 + c2);
            const VectorXd f = flat(gx - x);
            r.res = f.cwiseAbs().maxCoeff() / std::max(1.0, gx.cwiseAbs().maxCoeff());
            if (r.res < tol) { x = gx; break; }
            X.push_back(flat(x)); F.push_back(f);
            if (static_cast<int>(X.size()) > depth + 1) { X.erase(X.begin()); F.erase(F.begin()); }
            VectorXd xn;
            if (X.size() >= 2) {
                const int k = static_cast<int>(X.size()) - 1;
                MatrixXd dF(f.size(), k), dX(f.size(), k);
                for (int i = 0; i < k; ++i) { dF.col(i) = F[i + 1] - F[i]; dX.col(i) = X[i + 1] - X[i]; }
                const VectorXd th = dF.colPivHouseholderQr().solve(f);
                xn = flat(x) + f - (dX + dF) * th;
            } else {
                xn = flat(x) + f;
            }
            x = unflat(xn, N);
        }
        factor(x);
        r.c1 = unflat(apply1(d1f), N); r.c2 = unflat(apply2(d2f), N);
        r.xhat1 = unflat(apply1(flat(x)), N); r.xhat2 = unflat(apply2(flat(x)), N);
        r.xtilde1 = x - r.xhat1; r.xtilde2 = x - r.xhat2;
        r.x = x; r.iters = it + 1;
        return r;
    }
};

// ------------------------------------------------------------ backward

struct Backward {
    int N, m;
    double L;
    Panel K;
    TwoSided T;
    MatrixXd J;                                  // tail integral (2N x 2N)
    MatrixXd SW;                                 // (N x m)
    std::vector<MatrixXd> SPf, SPg;              // (m x N)
    // two-sided shifted-product layout: up to two pieces per node
    std::vector<std::vector<std::tuple<VectorXd, MatrixXd, MatrixXd>>> lay;   // (w, PB (m x N), PV (m x 2N))
    MatrixXd QT;                 // (N x 4N^2): shifted product is linear in the coefficient, T_flat = coef^T QT
    std::vector<MatrixXd> R;     // R[j] (N x N): shifted inner product is bilinear, A_j = sum_ch f_ch^T R[j] g_ch

    Backward(int N_, double L_) : N(N_), m(N_ + 8), L(L_), K(N_, 0.0, L_), T(N_, L_) {
        VectorXd u, w;
        // J
        J = MatrixXd::Zero(2 * N, 2 * N);
        gauss_legendre(m, 0.0, L, u, w);
        const Eigen::RowVectorXd pos_full = w.transpose() * T.pos.interp(u);
        for (int j = 0; j < 2 * N; ++j) {
            const double lj = T.l[j];
            if (lj >= 0) {
                if (L - lj > 0) { gauss_legendre(m, lj, L, u, w); J.block(j, N, 1, N) = w.transpose() * T.pos.interp(u); }
            } else {
                gauss_legendre(m, lj, 0.0, u, w);
                J.block(j, 0, 1, N) = w.transpose() * T.neg.interp(u);
                J.block(j, N, 1, N) = pos_full;
            }
        }
        // shifted inner products
        SW = MatrixXd::Zero(N, m);
        SPf.assign(N, MatrixXd::Zero(m, N)); SPg.assign(N, MatrixXd::Zero(m, N));
        for (int j = 0; j < N; ++j) {
            const double s = K.x[j];
            if (L - s > 0) {
                gauss_legendre(m, 0.0, L - s, u, w);
                SW.row(j) = w; SPf[j] = K.interp(u); SPg[j] = K.interp((u.array() + s).matrix());
            }
        }
        // shifted products
        lay.resize(2 * N);
        for (int j = 0; j < 2 * N; ++j) {
            const double lj = T.l[j];
            const double smax = std::min(L, L - lj);
            if (smax <= 0) continue;
            std::vector<double> br = {0.0};
            if (lj < 0) br.push_back(-lj);
            br.push_back(smax);
            for (size_t k = 0; k + 1 < br.size(); ++k) {
                if (br[k + 1] - br[k] <= 0) continue;
                gauss_legendre(m, br[k], br[k + 1], u, w);
                lay[j].emplace_back(w, K.interp(u), T.interp((u.array() + lj).matrix()));
            }
        }
        // linear map coef -> shifted product matrix (row-major flattening, j*2N + i)
        QT = MatrixXd::Zero(N, 4 * N * N);
        for (int j = 0; j < 2 * N; ++j)
            for (const auto& [w, PB, PV] : lay[j])
                for (int q = 0; q < w.size(); ++q)
                    QT.block(0, j * 2 * N, N, 2 * N).noalias() += w[q] * PB.row(q).transpose() * PV.row(q);
        // bilinear form for the shifted inner product
        R.assign(N, MatrixXd::Zero(N, N));
        for (int j = 0; j < N; ++j)
            for (int q = 0; q < m; ++q)
                if (SW(j, q) != 0.0) R[j].noalias() += SW(j, q) * SPf[j].row(q).transpose() * SPg[j].row(q);
    }

    VectorXd shifted_inner(const MatrixXd& f, const MatrixXd& g) const {
        VectorXd out(N);
        for (int j = 0; j < N; ++j) out[j] = (f.transpose() * R[j] * g).trace();
        return out;
    }

    MatrixXd shifted_product(const VectorXd& coef) const {
        const Eigen::RowVectorXd flatT = coef.transpose() * QT;
        return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(flatT.data(), 2 * N, 2 * N);
    }

    struct Result { MatrixXd hx, wedge, policy; };

    Result solve(const MatrixXd& x, const MatrixXd& xtilde_k, const MatrixXd& d_k,
                 double gain_k, int chan_k, double prec_k, double r_i) const {
        const VectorXd A = gain_k * d_k.col(chan_k) + prec_k * shifted_inner(xtilde_k, d_k);
        const VectorXd B = gain_k * x.col(chan_k) + prec_k * shifted_inner(xtilde_k, x);
        const MatrixXd TA = shifted_product(A), TB = shifted_product(B);
        MatrixXd xplus = MatrixXd::Zero(2 * N, 3);
        xplus.bottomRows(N) = x;
        const MatrixXd hx0 = J * xplus;
        const MatrixXd M = MatrixXd::Identity(2 * N, 2 * N) + TB - TA * J;
        Result r;
        r.wedge = M.partialPivLu().solve(TA * hx0);
        r.hx = hx0 + J * r.wedge;
        r.policy = -(1.0 / r_i) * r.hx.bottomRows(N);
        return r;
    }
};

// --------------------------------------------------------------- outer

struct Model {
    int N; double L, p1, p2, r1, r2, g1, g2;
    Forward fw; Backward bw;
    MatrixXd x_warm; bool have_warm = false;
    long evals = 0, fwd_iters = 0; double t_fwd = 0.0, t_bwd = 0.0;
    Forward::Result last_f; Backward::Result last_b1, last_b2;

    Model(int N_, double L_, double p1_, double p2_, double r1_, double r2_)
        : N(N_), L(L_), p1(p1_), p2(p2_), r1(r1_), r2(r2_), g1(std::sqrt(p1_)), g2(std::sqrt(p2_)), fw(N_, L_), bw(N_, L_) {}

    VectorXd pack(const MatrixXd& d1, const MatrixXd& d2) const {
        VectorXd z(6 * N); z << Forward::flat(d1), Forward::flat(d2); return z;
    }
    void unpack(const VectorXd& z, MatrixXd& d1, MatrixXd& d2) const {
        d1 = Forward::unflat(z.head(3 * N), N); d2 = Forward::unflat(z.tail(3 * N), N);
    }
    // r(z) = bestresponse(z) - z
    VectorXd residual(const VectorXd& z, bool keep = false, double ftol = 1e-13) {
        MatrixXd d1, d2; unpack(z, d1, d2);
        const auto ta = std::chrono::steady_clock::now();
        Forward::Result f = fw.solve(d1, d2, g1, g2, have_warm ? &x_warm : nullptr, ftol);
        if (keep) { x_warm = f.x; have_warm = true; }
        const auto tb = std::chrono::steady_clock::now();
        Backward::Result b1 = bw.solve(f.x, f.xtilde2, d2, g2, 2, p2, r1);
        Backward::Result b2 = bw.solve(f.x, f.xtilde1, d1, g1, 1, p1, r2);
        const auto tc = std::chrono::steady_clock::now();
        t_fwd += std::chrono::duration<double>(tb - ta).count(); t_bwd += std::chrono::duration<double>(tc - tb).count();
        fwd_iters += f.iters;
        ++evals;
        if (keep) { last_f = f; last_b1 = b1; last_b2 = b2; }
        return pack(b1.policy, b2.policy) - z;
    }
    // Thread-safe evaluation: no warm-start update, no counters.
    VectorXd residual_const(const VectorXd& z, double ftol) const {
        MatrixXd d1, d2; unpack(z, d1, d2);
        Forward::Result f = fw.solve(d1, d2, g1, g2, have_warm ? &x_warm : nullptr, ftol);
        Backward::Result b1 = bw.solve(f.x, f.xtilde2, d2, g2, 2, p2, r1);
        Backward::Result b2 = bw.solve(f.x, f.xtilde1, d1, g1, 1, p1, r2);
        return pack(b1.policy, b2.policy) - z;
    }
    VectorXd ce_start() const {
        const double inv = 1.0 / r1 + 1.0 / r2, S = std::sqrt(1.0 / inv), Kc = inv * S;
        MatrixXd d1 = MatrixXd::Zero(N, 3), d2 = MatrixXd::Zero(N, 3);
        for (int j = 0; j < N; ++j) {
            const double xpi = std::exp(-Kc * fw.K.x[j]);
            d1(j, 0) = -(S / r1) * xpi; d2(j, 0) = -(S / r2) * xpi;
        }
        return pack(d1, d2);
    }
    // Newton with finite-difference Jacobian and backtracking; damped
    // pre-phase from the CE start for globalization.
    struct Out { VectorXd z; double resid; int newton_steps; int pre_steps; bool ok; };
    double jac_ftol = 1e-13;   // forward tolerance for Jacobian-column evaluations
    double pre_ftol = 1e-13;   // forward tolerance during the damped pre-phase
    Out solve(VectorXd z, double tol, int pre = 40, double relax = 0.1, int maxnewton = 30, bool verbose = false) {
        Out o; o.pre_steps = pre; o.ok = false;
        for (int k = 0; k < pre; ++k) {
            const VectorXd rk = residual(z, true, pre_ftol);
            z += relax * rk;
        }
        VectorXd r = residual(z, true);
        double rn = r.cwiseAbs().maxCoeff();
        const int n = static_cast<int>(z.size());
        int step = 0;
        Eigen::PartialPivLU<MatrixXd> Jlu;
        bool have_J = false;
        double last_ratio = 0.0;
        for (; step < maxnewton && rn > tol; ++step) {
            // recompute the Jacobian when there is none or the chord step stalled
            if (!have_J || last_ratio > 0.3) {
                MatrixXd Jm(n, n);
                const double eps = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
#pragma omp parallel for schedule(dynamic, 4)
                for (int i = 0; i < n; ++i) {
                    VectorXd zp = z; zp[i] += eps;
                    Jm.col(i) = (residual_const(zp, jac_ftol) - r) / eps;
                }
                evals += n;
                Jlu.compute(Jm); have_J = true;
            }
            const VectorXd dz = Jlu.solve(-r);
            double lam = 1.0; bool acc = false;
            for (int ls = 0; ls < 8; ++ls) {
                const VectorXd zt = z + lam * dz;
                const double ftol_ls = std::max(1e-13, std::min(1e-6, 1e-4 * rn));
                VectorXd rt = residual(zt, true, ftol_ls);
                double rtn = rt.cwiseAbs().maxCoeff();
                if (rtn <= tol && ftol_ls > 1e-13) { rt = residual(zt, true); rtn = rt.cwiseAbs().maxCoeff(); }   // confirm at full accuracy
                if (std::isfinite(rtn) && rtn < (1.0 - 1e-4 * lam) * rn) { last_ratio = rtn / rn; z = zt; r = rt; rn = rtn; acc = true; break; }
                lam *= 0.5;
            }
            if (verbose) std::fprintf(stderr, "newton %d: |r| %.3e step %.3g%s\n", step, rn, lam, acc ? "" : " (rejected)");
            if (!acc) { if (last_ratio <= 0.3) { last_ratio = 1.0; continue; } break; }
        }
        residual(z, true);   // refresh last_* at the solution
        o.z = z; o.resid = rn; o.newton_steps = step; o.ok = rn <= tol;
        return o;
    }
};

// ---------------------------------------------------------------- json

void print_vec(const char* name, const VectorXd& v, bool last = false) {
    std::printf("\"%s\":[", name);
    for (int i = 0; i < v.size(); ++i) std::printf("%s%.15g", i ? "," : "", v[i]);
    std::printf("]%s", last ? "" : ",");
}
void print_kernel(const char* name, const MatrixXd& v, bool last = false) {
    std::printf("\"%s\":{", name);
    for (int c = 0; c < 3; ++c) { char nm[8]; std::snprintf(nm, sizeof nm, "ch%d", c); print_vec(nm, v.col(c), c == 2); }
    std::printf("}%s", last ? "" : ",");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s p1 p2 r1 r2 [--N 24] [--L 3] [--tol 1e-12] [--pre 40] [--relax 0.1] [--pre 15] [--relax 0.1] [--jac-ftol 1e-9] [--pre-ftol 1e-6] [--threads t] [--uniform n] [--eval-only] [--verbose]\n", argv[0]);
        return 1;
    }
    const double p1 = std::atof(argv[1]), p2 = std::atof(argv[2]), r1 = std::atof(argv[3]), r2 = std::atof(argv[4]);
    int N = 24, uniform = 0, pre = 15; double L = 3.0, tol = 1e-12, relax = 0.1, jac_ftol = 1e-9, pre_ftol = 1e-6; bool verbose = false, eval_only = false;
    for (int i = 5; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--N") && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--L") && i + 1 < argc) L = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--pre") && i + 1 < argc) pre = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--relax") && i + 1 < argc) relax = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--uniform") && i + 1 < argc) uniform = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--eval-only")) eval_only = true;
        else if (!std::strcmp(argv[i], "--jac-ftol") && i + 1 < argc) jac_ftol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--pre-ftol") && i + 1 < argc) pre_ftol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) { 
#ifdef _OPENMP
            omp_set_num_threads(std::atoi(argv[++i]));
#else
            ++i;
#endif
        }
    }
#ifdef _OPENMP
    // Jacobian columns parallelize; beyond the physical cores the extra
    // threads only spin.  8 matches the development machine.
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_num_procs()));
#endif
    const auto t0 = std::chrono::steady_clock::now();
    Model M(N, L, p1, p2, r1, r2);
    M.jac_ftol = jac_ftol; M.pre_ftol = pre_ftol;
    const auto t1 = std::chrono::steady_clock::now();
    Model::Out o;
    if (eval_only) {   // one residual evaluation at the CE start, for cross-checking
        o.z = M.ce_start(); const VectorXd r = M.residual(o.z, true);
        o.resid = r.cwiseAbs().maxCoeff(); o.newton_steps = 0; o.pre_steps = 0; o.ok = true;
        std::fprintf(stderr, "eval-only: |r| %.6e, forward iters %d, forward res %.2e\n", o.resid, M.last_f.iters, M.last_f.res);
    } else {
        o = M.solve(M.ce_start(), tol, pre, relax, 30, verbose);
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double t_setup = std::chrono::duration<double>(t1 - t0).count();
    const double t_solve = std::chrono::duration<double>(t2 - t1).count();
    if (verbose) std::fprintf(stderr, "setup %.4f s, solve %.4f s, %ld evaluations, %d newton steps, |r| %.2e | forward %.4f s (%.1f its/eval), backward %.4f s\n",
                              t_setup, t_solve, M.evals, o.newton_steps, o.resid, M.t_fwd, double(M.fwd_iters) / std::max(1L, M.evals), M.t_bwd);

    MatrixXd d1, d2; M.unpack(o.z, d1, d2);
    std::printf("{\"converged\":%s,\"residual\":%.6e,\"N\":%d,\"L\":%.15g,\"p1\":%.15g,\"p2\":%.15g,\"r1\":%.15g,\"r2\":%.15g,"
                "\"evaluations\":%ld,\"newton_steps\":%d,\"setup_seconds\":%.6f,\"solve_seconds\":%.6f,",
                o.ok ? "true" : "false", o.resid, N, L, p1, p2, r1, r2, M.evals, o.newton_steps, t_setup, t_solve);
    print_vec("lag", M.fw.K.x);
    print_vec("b_lag", M.bw.T.l);
    print_kernel("x", M.last_f.x); print_kernel("xhat1", M.last_f.xhat1); print_kernel("xhat2", M.last_f.xhat2);
    print_kernel("xtilde1", M.last_f.xtilde1); print_kernel("xtilde2", M.last_f.xtilde2);
    print_kernel("d1", d1); print_kernel("d2", d2);
    print_kernel("calD1", M.last_f.c1); print_kernel("calD2", M.last_f.c2);
    print_kernel("hx1", M.last_b1.hx); print_kernel("hx2", M.last_b2.hx);
    print_kernel("wedge1", M.last_b1.wedge); print_kernel("wedge2", M.last_b2.wedge, uniform == 0);
    if (uniform > 0) {
        // values on uniform grids: lag in [0, L] (n points) and b_lag in [-L, L] (2n-1 points)
        VectorXd ul(uniform), ub(2 * uniform - 1);
        for (int i = 0; i < uniform; ++i) ul[i] = L * i / (uniform - 1);
        for (int i = 0; i < 2 * uniform - 1; ++i) ub[i] = -L + L * i / (uniform - 1);
        const MatrixXd Iu = M.fw.K.interp(ul), Ib = M.bw.T.interp(ub);
        std::printf("\"uniform\":{");
        print_vec("lag", ul); print_vec("b_lag", ub);
        print_kernel("x", Iu * M.last_f.x); print_kernel("xtilde1", Iu * M.last_f.xtilde1); print_kernel("xtilde2", Iu * M.last_f.xtilde2);
        print_kernel("d1", Iu * d1); print_kernel("d2", Iu * d2);
        print_kernel("hx1", Ib * M.last_b1.hx); print_kernel("hx2", Ib * M.last_b2.hx);
        print_kernel("wedge1", Ib * M.last_b1.wedge); print_kernel("wedge2", Ib * M.last_b2.wedge, true);
        std::printf("}");
    }
    std::printf("}\n");
    return o.ok ? 0 : 2;
}
