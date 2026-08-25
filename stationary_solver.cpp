#include "stationary_solver.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <unsupported/Eigen/FFT>
#include <limits>

namespace {

struct StationaryGrid {
    int n = 0;
    int nb = 0;
    int offset = 0;
    double h = 0.0;
    std::vector<double> lag;
    std::vector<double> b_lag;
};

struct ForwardBlock {
    std::vector<Vec3> x, xhat1, xhat2, xtilde1, xtilde2, c1, c2;
    double residual = 0.0;
};

struct BackwardBlock {
    std::vector<Vec3> hx;
    std::vector<Vec3> wedge;
    std::vector<Vec3> policy;
    double residual = 0.0;
};

struct FilterProjection {
    std::vector<Vec3> xhat;
    std::vector<Vec3> xtilde;
    std::vector<Vec3> control;
};

static StationaryGrid make_grid(const StationaryParams& p) {
    StationaryGrid g;
    g.n = std::max(3, p.n_lag);
    g.nb = 2 * g.n - 1;
    g.offset = g.n - 1;
    g.h = p.lag_max / static_cast<double>(g.n - 1);
    g.lag.resize(g.n);
    g.b_lag.resize(g.nb);
    for (int i = 0; i < g.n; ++i)
        g.lag[i] = i * g.h;
    for (int i = 0; i < g.nb; ++i)
        g.b_lag[i] = (i - g.offset) * g.h;
    return g;
}

static Mat3 outer(const Vec3& a, const Vec3& b) {
    return a * b.transpose();
}

static double simpson_interval_weight(int idx, int intervals, double h) {
    if (intervals <= 0)
        return 0.0;
    if (intervals == 1)
        return 0.5 * h;
    if ((intervals % 2) == 0) {
        if (idx == 0 || idx == intervals)
            return h / 3.0;
        return ((idx % 2) ? 4.0 : 2.0) * h / 3.0;
    }

    // Composite Simpson for an odd number of intervals:
    // use 1/3 on the first intervals-3 intervals and 3/8 on the last three.
    const int tail0 = intervals - 3;
    double w = 0.0;
    if (tail0 > 0 && idx <= tail0) {
        if (idx == 0 || idx == tail0)
            w += h / 3.0;
        else
            w += ((idx % 2) ? 4.0 : 2.0) * h / 3.0;
    }
    if (idx >= tail0 && idx <= intervals)
        w += (idx == tail0 || idx == intervals) ? 3.0 * h / 8.0 : 9.0 * h / 8.0;
    return w;
}

static double trapezoid_weight(int idx, int n, double h) {
    return (idx == 0 || idx == n - 1) ? 0.5 * h : h;
}

static double quad_weight(int idx, int n, double h, bool use_simpson) {
    return use_simpson ? simpson_interval_weight(idx, n - 1, h)
                       : trapezoid_weight(idx, n, h);
}

static double prefix_quad_weight(int idx, int last_idx, double h, bool use_simpson) {
    if (!use_simpson)
        return (idx == 0 || idx == last_idx) ? 0.5 * h : h;
    return simpson_interval_weight(idx, last_idx, h);
}

struct GapMetrics {
    double absolute = 0.0;
    double relative = 0.0;
};

static double weighted_norm(const std::vector<Vec3>& v, double h, bool use_simpson) {
    double norm = 0.0;
    const int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i)
        norm += quad_weight(i, n, h, use_simpson) * v[i].squaredNorm();
    return std::sqrt(norm);
}

static GapMetrics weighted_gap(const std::vector<Vec3>& a,
                               const std::vector<Vec3>& b,
                               double h,
                               bool use_simpson) {
    double diff = 0.0;
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) {
        const Vec3 d = a[i] - b[i];
        diff += quad_weight(i, n, h, use_simpson) * d.squaredNorm();
    }
    GapMetrics out;
    out.absolute = std::sqrt(diff);
    out.relative = out.absolute / std::max(1.0, weighted_norm(b, h, use_simpson));
    return out;
}

static double relative_gap(const std::vector<Vec3>& a,
                           const std::vector<Vec3>& b,
                           double h,
                           bool use_simpson) {
    return weighted_gap(a, b, h, use_simpson).relative;
}

static bool finite_vecs(const std::vector<Vec3>& v) {
    for (const auto& x : v)
        if (!x.allFinite())
            return false;
    return true;
}

static bool finite_forward(const ForwardBlock& f) {
    return std::isfinite(f.residual)
        && finite_vecs(f.x) && finite_vecs(f.xhat1) && finite_vecs(f.xhat2)
        && finite_vecs(f.xtilde1) && finite_vecs(f.xtilde2)
        && finite_vecs(f.c1) && finite_vecs(f.c2);
}

static bool finite_backward(const BackwardBlock& b) {
    return std::isfinite(b.residual)
        && finite_vecs(b.hx) && finite_vecs(b.wedge) && finite_vecs(b.policy);
}

static void relax_into(std::vector<Vec3>& dst, const std::vector<Vec3>& src, double relax) {
    const int n = static_cast<int>(dst.size());
    for (int i = 0; i < n; ++i)
        dst[i] += relax * (src[i] - dst[i]);
}

static std::vector<Vec3> integrate_state_kernel(
    const StationaryGrid& grid, const StationaryParams& p,
    const std::vector<Vec3>& c1, const std::vector<Vec3>& c2) {

    std::vector<Vec3> x(grid.n, Vec3::Zero());
    x[0] = p.sigma * E0();
    const double lhs = 1.0 - 0.5 * grid.h * p.A;
    const double rhs = 1.0 + 0.5 * grid.h * p.A;
    for (int a = 1; a < grid.n; ++a) {
        const Vec3 u_prev = c1[a - 1] + c2[a - 1];
        const Vec3 u_curr = c1[a] + c2[a];
        x[a] = (rhs * x[a - 1] + 0.5 * grid.h * (u_prev + u_curr)) / lhs;
    }
    return x;
}

static Mat3 stationary_filter_kernel_at(
    int a, int b, const std::vector<Vec3>& xtilde,
    double obs_gain, int obs_idx, double precision, double h,
    bool use_simpson) {

    Vec3 e = Vec3::Zero();
    e(obs_idx) = 1.0;

    Mat3 f = Mat3::Zero();
    if (a > b)
        f += obs_gain * outer(xtilde[a - b], e);
    else if (b > a)
        f += obs_gain * outer(e, xtilde[b - a]);

    const int m = std::min(a, b);
    if (m > 0) {
        for (int c = 0; c <= m; ++c) {
            const double w = prefix_quad_weight(c, m, h, use_simpson);
            f += w * precision * outer(xtilde[a - c], xtilde[b - c]);
        }
    }

    return f;
}

static FilterProjection project_stationary_filter(
    const StationaryGrid& grid, const std::vector<Vec3>& x,
    const std::vector<Vec3>& xtilde, const std::vector<Vec3>& d,
    double obs_gain, int obs_idx, const Mat3& Pi, bool use_simpson) {

    const double precision = obs_gain * obs_gain;
    FilterProjection out;
    out.xhat.assign(grid.n, Vec3::Zero());
    out.xtilde.assign(grid.n, Vec3::Zero());
    out.control.assign(grid.n, Vec3::Zero());

    for (int b = 0; b < grid.n; ++b) {
        Vec3 xhat_b = Pi * x[b];
        Vec3 c_b = Pi * d[b];
        for (int a = 0; a < grid.n; ++a) {
            const Mat3 f = stationary_filter_kernel_at(
                a, b, xtilde, obs_gain, obs_idx, precision, grid.h, use_simpson);
            const double w = quad_weight(a, grid.n, grid.h, use_simpson);
            xhat_b += w * (f.transpose() * x[a]);
            c_b += w * (f.transpose() * d[a]);
        }
        out.xhat[b] = xhat_b;
        out.xtilde[b] = x[b] - xhat_b;
        out.control[b] = c_b;
    }
    // At zero lag, a player can react to the contemporaneously observed
    // innovation coordinate. Only unobserved primitive shock coordinates are
    // forced to have no instantaneous action load.
    out.control[0] = Pi * out.control[0];

    return out;
}

struct LLTCache {
    Eigen::LLT<Eigen::MatrixXd> llt;
    bool valid = false;
};

// FFT workspace for one observer's Toeplitz-structured projection solve.
// Correlations with the drift kernel xm are applied spectrally, and the
// Gram solve runs as PCG with a Strang circulant preconditioner.
struct SpectralObserver {
    int n = 0, M = 0;
    double gh = 0.0, ridge = 1e-12;
    int chan = 1;
    Eigen::FFT<double> fft;
    std::vector<Eigen::VectorXcd> Xm;     // per-channel fft of xm
    Eigen::VectorXd lam;                  // circulant eigenvalues (length n)

    void prepare(const std::vector<Vec3>& xm, double gain, double h, int channel,
                 const Eigen::VectorXd& cvec) {
        n = static_cast<int>(xm.size());
        M = 1; while (M < 2 * n) M <<= 1;
        gh = gain * h; chan = channel;
        Xm.assign(3, Eigen::VectorXcd());
        std::vector<double> buf(M, 0.0);
        std::vector<std::complex<double>> out(M);
        for (int ch = 0; ch < 3; ++ch) {
            std::fill(buf.begin(), buf.end(), 0.0);
            for (int i = 0; i + 1 < n; ++i) buf[i] = xm[i](ch);
            fft.fwd(out, buf);
            Xm[ch] = Eigen::Map<Eigen::VectorXcd>(out.data(), M);
        }
        // Strang circulant from the Toeplitz symbol t(delta) = G(0, delta)
        const double g2h2 = gh * gh;
        std::vector<double> t(n, 0.0);
        t[0] = 1.0 + ridge + g2h2 * cvec[0];
        for (int d = 1; d < n; ++d)
            t[d] = gh * xm[d - 1](chan) + g2h2 * cvec[d];
        std::vector<double> circ(n);
        circ[0] = t[0];
        for (int j = 1; j < n; ++j)
            circ[j] = (j <= n / 2) ? t[j] : t[n - j];
        std::vector<std::complex<double>> ce(n);
        fft.fwd(ce, circ);
        lam.resize(n);
        for (int j = 0; j < n; ++j)
            lam[j] = std::max(1e-8, ce[j].real());
    }

    // y(j) = s[3j+chan] + gh sum_ch corr(xm_ch, s_ch)(j+1)
    Eigen::VectorXd applyH(const Eigen::VectorXd& s3) {
        std::vector<double> buf(M, 0.0);
        std::vector<std::complex<double>> S(M);
        Eigen::VectorXcd acc = Eigen::VectorXcd::Zero(M);
        for (int ch = 0; ch < 3; ++ch) {
            std::fill(buf.begin(), buf.end(), 0.0);
            for (int k = 0; k < n; ++k) buf[k] = s3[3 * k + ch];
            fft.fwd(S, buf);
            acc.noalias() += Xm[ch].conjugate().cwiseProduct(
                Eigen::Map<Eigen::VectorXcd>(S.data(), M));
        }
        std::vector<std::complex<double>> a(acc.data(), acc.data() + M);
        std::vector<double> r(M);
        fft.inv(r, a);
        Eigen::VectorXd y(n);
        for (int j = 0; j < n; ++j)
            y[j] = s3[3 * j + chan] + gh * (j + 1 < M ? r[j + 1] : 0.0);
        return y;
    }

    // s_ch(k) = z(k) [ch==chan] + gh conv(z, xm_ch)(k-1)
    Eigen::VectorXd applyHt(const Eigen::VectorXd& z) {
        std::vector<double> buf(M, 0.0);
        for (int j = 0; j < n; ++j) buf[j] = z[j];
        std::vector<std::complex<double>> Z(M);
        fft.fwd(Z, buf);
        Eigen::Map<Eigen::VectorXcd> Zm(Z.data(), M);
        Eigen::VectorXd s3 = Eigen::VectorXd::Zero(3 * n);
        std::vector<std::complex<double>> prod(M);
        std::vector<double> c(M);
        for (int ch = 0; ch < 3; ++ch) {
            Eigen::Map<Eigen::VectorXcd>(prod.data(), M) =
                Xm[ch].cwiseProduct(Zm);
            fft.inv(c, prod);
            for (int k = 1; k < n; ++k)
                s3[3 * k + ch] = gh * c[k - 1];
        }
        for (int k = 0; k < n; ++k)
            s3[3 * k + chan] += z[k];
        return s3;
    }

    Eigen::VectorXd matvec(const Eigen::VectorXd& v) {
        return applyH(applyHt(v)) + ridge * v;
    }

    Eigen::VectorXd precond(const Eigen::VectorXd& r) {
        std::vector<double> buf(r.data(), r.data() + n);
        std::vector<std::complex<double>> R(n);
        fft.fwd(R, buf);
        for (int j = 0; j < n; ++j) R[j] /= lam[j];
        std::vector<double> z(n);
        fft.inv(z, R);
        return Eigen::Map<Eigen::VectorXd>(z.data(), n);
    }

    Eigen::VectorXd pcg(const Eigen::VectorXd& b, int maxit, double rtol) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd r = b;
        Eigen::VectorXd z = precond(r);
        Eigen::VectorXd pdir = z;
        double rz = r.dot(z);
        const double bnorm = b.norm() + 1e-300;
        for (int it = 0; it < maxit; ++it) {
            if (r.norm() < rtol * bnorm)
                break;
            const Eigen::VectorXd Ap = matvec(pdir);
            const double alpha = rz / (pdir.dot(Ap) + 1e-300);
            x += alpha * pdir;
            r -= alpha * Ap;
            z = precond(r);
            const double rz_new = r.dot(z);
            pdir = z + (rz_new / (rz + 1e-300)) * pdir;
            rz = rz_new;
        }
        return x;
    }
};

// Solve G Y = B with the cached Cholesky factor plus iterative refinement
// against the current G; refactor only when refinement cannot reach tolerance.
static Eigen::MatrixXd cached_spd_solve(LLTCache& c, const Eigen::MatrixXd& G,
                                        const Eigen::MatrixXd& B) {
    const double bn = B.norm() + 1e-300;
    if (c.valid) {
        Eigen::MatrixXd Y = c.llt.solve(B);
        double prev = std::numeric_limits<double>::infinity();
        for (int ref = 0; ref < 10; ++ref) {
            const Eigen::MatrixXd R = B - G * Y;
            const double rn = R.norm();
            if (rn < 1e-9 * bn)
                return Y;
            if (rn > 0.3 * prev)
                break;                      // stalled: factor too stale
            prev = rn;
            Y += c.llt.solve(R);
        }
    }
    c.llt.compute(G);
    c.valid = true;
    return c.llt.solve(B);
}

// Exact forward block: each observer's filter is computed as an exact
// orthogonal projection of the stacked shock vector onto the discrete signal
// window (one SPD solve), instead of Picard-iterating the quadratic filter
// map. Only the physical feedback loop x <-> projection is iterated, and it
// tolerates a full step. Observation rows use the Ito midpoint convention:
// the drift over a cell reads the state built from strictly older shocks.
static ForwardBlock solve_forward_block_exact(
    const StationaryGrid& grid, const StationaryParams& p,
    const std::vector<Vec3>& d1, const std::vector<Vec3>& d2,
    const std::vector<Vec3>* x_init = nullptr,
    LLTCache* fc1 = nullptr, LLTCache* fc2 = nullptr) {

    const int n = grid.n;
    const int N3 = 3 * n;
    const double h = grid.h;
    const double g1 = std::sqrt(std::max(0.0, p.p1));
    const double g2 = std::sqrt(std::max(0.0, p.p2));

    auto stack = [&](const std::vector<Vec3>& v) {
        Eigen::VectorXd s(N3);
        for (int k = 0; k < n; ++k) s.segment<3>(3 * k) = v[k];
        return s;
    };
    auto unstack = [&](const Eigen::VectorXd& s) {
        std::vector<Vec3> v(n);
        for (int k = 0; k < n; ++k) v[k] = s.segment<3>(3 * k);
        return v;
    };

    ForwardBlock fwd;
    fwd.c1.assign(n, Vec3::Zero());
    fwd.c2.assign(n, Vec3::Zero());
    for (int a = 0; a < n; ++a) {
        fwd.c1[a] = Pi1() * d1[a];
        fwd.c2[a] = Pi2() * d2[a];
    }
    if (x_init && static_cast<int>(x_init->size()) == n)
        fwd.x = *x_init;
    else
        fwd.x = integrate_state_kernel(grid, p, fwd.c1, fwd.c2);

    const Eigen::VectorXd d1s = stack(d1), d2s = stack(d2);
    Eigen::VectorXd xhat1s(N3), xhat2s(N3), c1s(N3), c2s(N3);
    LLTCache local1, local2;
    LLTCache& llt_c1 = fc1 ? *fc1 : local1;
    LLTCache& llt_c2 = fc2 ? *fc2 : local2;
    double x_gam = 1.0;
    double x_best = std::numeric_limits<double>::infinity();

    // Toeplitz-structured projection: the Gram matrix G = H H^T is Toeplitz
    // in |j - j'| up to a window-truncation correction with closed form, so
    // it is assembled in O(n^2) from prefix sums and H, H^T are applied as
    // correlations -- the observation matrix is never materialized.
    std::vector<Vec3> xm(n, Vec3::Zero());
    Eigen::MatrixXd C(p.circulant_cg ? 0 : n, p.circulant_cg ? 0 : n);   // C(delta, q) = sum_{m<=q} xm(m+delta).xm(m)
    auto gram = [&](double gain, int chan) {
        Eigen::MatrixXd G(n, n);
        const double gh = gain * h;
        const double g2h2 = gh * gh;
        for (int j = 0; j < n; ++j) {
            for (int jp = j; jp < n; ++jp) {
                const int delta = jp - j;
                const int q = n - 2 - jp;
                double v = g2h2 * (q >= 0 ? C(delta, q) : 0.0);
                if (delta == 0)
                    v += 1.0 + 1e-12;
                else
                    v += gh * xm[delta - 1](chan);
                G(j, jp) = v;
                G(jp, j) = v;
            }
        }
        return G;
    };
    auto apply_H = [&](double gain, int chan, const Eigen::VectorXd& s) {
        Eigen::VectorXd y(n);
        const double gh = gain * h;
        for (int j = 0; j < n; ++j) {
            double acc = s[3 * j + chan];
            for (int k = j + 1; k < n; ++k)
                acc += gh * (xm[k - j - 1](0) * s[3 * k] +
                             xm[k - j - 1](1) * s[3 * k + 1] +
                             xm[k - j - 1](2) * s[3 * k + 2]);
            y[j] = acc;
        }
        return y;
    };
    auto apply_Ht = [&](double gain, int chan, const Eigen::VectorXd& z) {
        Eigen::VectorXd s = Eigen::VectorXd::Zero(N3);
        const double gh = gain * h;
        for (int k = 0; k < n; ++k) {
            s[3 * k + chan] += z[k];
            Vec3 acc = Vec3::Zero();
            for (int j = 0; j < k; ++j)
                acc += z[j] * xm[k - j - 1];
            s.segment<3>(3 * k) += gh * acc;
        }
        return s;
    };

    for (int it = 0; it < p.forward_iters; ++it) {
        for (int i = 0; i + 1 < n; ++i)
            xm[i] = 0.5 * (fwd.x[i] + fwd.x[i + 1]);
        xm[n - 1].setZero();
        Eigen::VectorXd cvec = Eigen::VectorXd::Zero(n);
        if (p.circulant_cg) {
            // Only the full-range sums C(delta, n-2-delta) enter the
            // circulant preconditioner, so skip the n x n running-sum table.
            // Autocorrelation of xm over m = 0..n-2 by FFT.
            int M2 = 1; while (M2 < 2 * n) M2 <<= 1;
            Eigen::FFT<double> afft;
            Eigen::VectorXcd acc = Eigen::VectorXcd::Zero(M2);
            std::vector<double> abuf(M2, 0.0);
            std::vector<std::complex<double>> aout(M2);
            for (int ch = 0; ch < 3; ++ch) {
                std::fill(abuf.begin(), abuf.end(), 0.0);
                for (int m = 0; m <= n - 2; ++m) abuf[m] = xm[m](ch);
                afft.fwd(aout, abuf);
                Eigen::Map<Eigen::VectorXcd> A(aout.data(), M2);
                acc.array() += (A.conjugate().array() * A.array());
            }
            std::vector<std::complex<double>> ain(acc.data(), acc.data() + M2);
            std::vector<double> ares(M2);
            afft.inv(ares, ain);
            for (int delta = 0; delta < n; ++delta) cvec[delta] = ares[delta];
        } else {
            for (int delta = 0; delta < n; ++delta) {
                double run = 0.0;
                for (int m = 0; m < n; ++m) {
                    if (m + delta <= n - 2 && m <= n - 2)
                        run += xm[m + delta].dot(xm[m]);
                    C(delta, m) = run;
                }
            }
        }
        const Eigen::VectorXd xs = stack(fwd.x);
        if (p.circulant_cg) {
            // Four independent projection solves (two observers x two
            // right-hand sides), run concurrently.
            Eigen::VectorXd ys[4];
#pragma omp parallel for schedule(static) if (n >= 64)
            for (int task = 0; task < 4; ++task) {
                const int obs = task / 2;
                SpectralObserver so;
                so.prepare(xm, obs == 0 ? g1 : g2, h, obs == 0 ? 1 : 2, cvec);
                const Eigen::VectorXd& rhs = (task % 2 == 0) ? xs : (obs == 0 ? d1s : d2s);
                ys[task] = so.applyHt(so.pcg(so.applyH(rhs), 300, 1e-10));
            }
            xhat1s = ys[0]; c1s = ys[1]; xhat2s = ys[2]; c2s = ys[3];
        } else {
#pragma omp parallel sections if (n >= 64)
            {
#pragma omp section
                {
                    const Eigen::MatrixXd G1 = gram(g1, 1);
                    Eigen::MatrixXd B(n, 2);
                    B.col(0) = apply_H(g1, 1, xs);
                    B.col(1) = apply_H(g1, 1, d1s);
                    const Eigen::MatrixXd Y = cached_spd_solve(llt_c1, G1, B);
                    xhat1s = apply_Ht(g1, 1, Y.col(0));
                    c1s = apply_Ht(g1, 1, Y.col(1));
                }
#pragma omp section
                {
                    const Eigen::MatrixXd G2 = gram(g2, 2);
                    Eigen::MatrixXd B(n, 2);
                    B.col(0) = apply_H(g2, 2, xs);
                    B.col(1) = apply_H(g2, 2, d2s);
                    const Eigen::MatrixXd Y = cached_spd_solve(llt_c2, G2, B);
                    xhat2s = apply_Ht(g2, 2, Y.col(0));
                    c2s = apply_Ht(g2, 2, Y.col(1));
                }
            }
        }

        std::vector<Vec3> c1_new = unstack(c1s);
        std::vector<Vec3> c2_new = unstack(c2s);
        std::vector<Vec3> x_new = integrate_state_kernel(grid, p, c1_new, c2_new);

        fwd.residual = relative_gap(fwd.x, x_new, grid.h, p.use_simpson_quadrature);
        // Growth safeguard: far from equilibrium the x-map can be expansive;
        // halve the step whenever the residual stops improving geometrically.
        if (!std::isfinite(fwd.residual) || fwd.residual > 2.0 * x_best)
            x_gam = std::max(0.05, 0.5 * x_gam);
        if (std::isfinite(fwd.residual))
            x_best = std::min(x_best, fwd.residual);
        for (int a = 0; a < n; ++a)
            fwd.x[a] = (1.0 - x_gam) * fwd.x[a] + x_gam * x_new[a];
        fwd.c1 = c1_new;
        fwd.c2 = c2_new;
        if (!std::isfinite(fwd.residual) || fwd.residual < p.forward_tol)
            break;
    }

    fwd.xhat1 = unstack(xhat1s);
    fwd.xhat2 = unstack(xhat2s);
    fwd.xtilde1.assign(n, Vec3::Zero());
    fwd.xtilde2.assign(n, Vec3::Zero());
    for (int a = 0; a < n; ++a) {
        fwd.xtilde1[a] = fwd.x[a] - fwd.xhat1[a];
        fwd.xtilde2[a] = fwd.x[a] - fwd.xhat2[a];
    }
    return fwd;
}

static ForwardBlock solve_forward_block(
    const StationaryGrid& grid, const StationaryParams& p,
    const std::vector<Vec3>& d1, const std::vector<Vec3>& d2,
    const std::vector<Vec3>* x_init = nullptr,
    LLTCache* fc1 = nullptr, LLTCache* fc2 = nullptr) {

    if (p.exact_forward)
        return solve_forward_block_exact(grid, p, d1, d2, x_init, fc1, fc2);

    ForwardBlock fwd;
    fwd.x.assign(grid.n, Vec3::Zero());
    fwd.xhat1.assign(grid.n, Vec3::Zero());
    fwd.xhat2.assign(grid.n, Vec3::Zero());
    fwd.xtilde1.assign(grid.n, Vec3::Zero());
    fwd.xtilde2.assign(grid.n, Vec3::Zero());
    fwd.c1.assign(grid.n, Vec3::Zero());
    fwd.c2.assign(grid.n, Vec3::Zero());

    for (int a = 0; a < grid.n; ++a) {
        fwd.c1[a] = Pi1() * d1[a];
        fwd.c2[a] = Pi2() * d2[a];
    }
    fwd.x = integrate_state_kernel(grid, p, fwd.c1, fwd.c2);
    for (int a = 0; a < grid.n; ++a) {
        fwd.xhat1[a] = Pi1() * fwd.x[a];
        fwd.xhat2[a] = Pi2() * fwd.x[a];
        fwd.xtilde1[a] = fwd.x[a] - fwd.xhat1[a];
        fwd.xtilde2[a] = fwd.x[a] - fwd.xhat2[a];
    }

    const double g1 = std::sqrt(std::max(0.0, p.p1));
    const double g2 = std::sqrt(std::max(0.0, p.p2));

    for (int it = 0; it < p.forward_iters; ++it) {
        FilterProjection p1 = project_stationary_filter(
            grid, fwd.x, fwd.xtilde1, d1, g1, 1, Pi1(), p.use_simpson_quadrature);
        FilterProjection p2 = project_stationary_filter(
            grid, fwd.x, fwd.xtilde2, d2, g2, 2, Pi2(), p.use_simpson_quadrature);

        std::vector<Vec3> x_new = integrate_state_kernel(grid, p, p1.control, p2.control);

        const double r_x = relative_gap(fwd.x, x_new, grid.h, p.use_simpson_quadrature);
        const double r_1 = relative_gap(fwd.xtilde1, p1.xtilde, grid.h, p.use_simpson_quadrature);
        const double r_2 = relative_gap(fwd.xtilde2, p2.xtilde, grid.h, p.use_simpson_quadrature);
        const double r_c1 = relative_gap(fwd.c1, p1.control, grid.h, p.use_simpson_quadrature);
        const double r_c2 = relative_gap(fwd.c2, p2.control, grid.h, p.use_simpson_quadrature);
        fwd.residual = std::max({r_x, r_1, r_2, r_c1, r_c2});

        relax_into(fwd.x, x_new, p.forward_relax);
        relax_into(fwd.xhat1, p1.xhat, p.forward_relax);
        relax_into(fwd.xhat2, p2.xhat, p.forward_relax);
        relax_into(fwd.xtilde1, p1.xtilde, p.forward_relax);
        relax_into(fwd.xtilde2, p2.xtilde, p.forward_relax);
        relax_into(fwd.c1, p1.control, p.forward_relax);
        relax_into(fwd.c2, p2.control, p.forward_relax);

        if (!std::isfinite(fwd.residual) || fwd.residual < p.forward_tol)
            break;
    }

    return fwd;
}

static Vec3 causal_at_b(const StationaryGrid& grid, const std::vector<Vec3>& q, int b_idx) {
    if (b_idx < grid.offset)
        return Vec3::Zero();
    const int a = b_idx - grid.offset;
    if (a < 0 || a >= static_cast<int>(q.size()))
        return Vec3::Zero();
    return q[a];
}

// Reusable factorization cache: solves run against the last LU with
// iterative refinement on the current matrix, refactoring only when the
// refinement fails to reach tolerance (the system drifts slowly across
// outer iterations).
struct FactorCache {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu;
    bool valid = false;
};

static BackwardBlock solve_response_adjoint(
    const StationaryGrid& grid, const StationaryParams& p,
    const std::vector<Vec3>& x,
    const std::vector<Vec3>& xtilde_k,
    const std::vector<Vec3>& d_k,
    double opponent_precision,
    int opponent_obs_idx,
    double player_r,
    FactorCache* cache = nullptr) {

    const double obs_gain = std::sqrt(std::max(0.0, opponent_precision));
    Vec3 e_obs = Vec3::Zero();
    e_obs(opponent_obs_idx) = 1.0;

    BackwardBlock out;
    out.hx.assign(grid.nb, Vec3::Zero());
    out.wedge.assign(grid.nb, Vec3::Zero());
    out.policy.assign(grid.n, Vec3::Zero());

    auto pack = [&](const std::vector<Vec3>& v) {
        Eigen::VectorXd y(3 * grid.nb);
        for (int b = 0; b < grid.nb; ++b)
            y.segment<3>(3 * b) = v[b];
        return y;
    };
    auto unpack = [&](const Eigen::VectorXd& y) {
        std::vector<Vec3> v(grid.nb, Vec3::Zero());
        for (int b = 0; b < grid.nb; ++b)
            v[b] = y.segment<3>(3 * b);
        return v;
    };

    auto apply_wedge_map = [&](const std::vector<Vec3>& wedge,
                               std::vector<Vec3>* hx_out) {
        std::vector<Vec3> hx_new(grid.nb, Vec3::Zero());
        std::vector<Mat3> H(grid.n * grid.nb, Mat3::Zero());
        auto H_at = [&](int a, int b) -> Mat3& { return H[a * grid.nb + b]; };

        // Solve 0 = -h_X'(b) - x(b) - A h_X(b) - v(b)
        // by integrating backward from the positive truncation boundary.
        hx_new[grid.nb - 1].setZero();
        const double hx_lhs = 1.0 - 0.5 * grid.h * p.A;
        const double hx_rhs = 1.0 + 0.5 * grid.h * p.A;
        for (int b = grid.nb - 2; b >= 0; --b) {
            const int bp = b + 1;
            const Vec3 q_next = causal_at_b(grid, x, bp) + wedge[bp];
            const Vec3 q_curr = causal_at_b(grid, x, b) + wedge[b];
            hx_new[b] = (hx_rhs * hx_new[bp] + 0.5 * grid.h * (q_next + q_curr)) / hx_lhs;
        }

        for (int a = grid.n - 2; a >= 0; --a) {
            for (int b = grid.nb - 2; b >= 0; --b) {
                const int ap = a + 1;
                const int bp = b + 1;

                const Vec3 d_a = d_k[ap];
                const Vec3 hx_b = hx_new[bp];
                const Vec3 x_a = x[ap];
                const Vec3 v_b = wedge[bp];

                // Two-sided stationary H equation:
                // (partial_a + partial_b) H = -d_k(a) h_X(b)^T + x(a) v(b)^T.
                H_at(a, b) = H_at(ap, bp)
                    + grid.h * (outer(d_a, hx_b) - outer(x_a, v_b));
            }
        }

        std::vector<Vec3> wedge_next(grid.nb, Vec3::Zero());
        for (int b = 0; b < grid.nb; ++b) {
            Vec3 acc = Vec3::Zero();
            for (int a = 0; a < grid.n; ++a)
                acc += quad_weight(a, grid.n, grid.h, p.use_simpson_quadrature)
                    * (H_at(a, b).transpose() * xtilde_k[a]);

            // Diagonal innovation-birth term:
            // Gamma^T E H^k(0,b), from the newly born innovation coordinate.
            wedge_next[b] = obs_gain * (H_at(0, b).transpose() * e_obs)
                + opponent_precision * acc;
        }

        if (hx_out)
            *hx_out = std::move(hx_new);
        return wedge_next;
    };

    // The wedge map decouples per channel with shared scalar kernels:
    //   v_new(b) = sum_{s>=1} h [ A(s) hx[v](b+s) - B(s) v(b+s) ],
    //   A(s) = gain d_k(s)[obs] + prec sum_a w_a xtilde(a).d_k(a+s),
    //   B(s) = gain x(s)[obs]  + prec sum_a w_a xtilde(a).x(a+s),
    // and hx[v] is affine in v through the backward trapezoid recursion.
    // Assemble the nb x nb scalar system once and solve three channel RHS,
    // instead of applying the full map to 3*nb basis vectors.
    const int nb = grid.nb;
    std::vector<double> A_s(grid.n, 0.0), B_s(grid.n, 0.0);
    {
        // accA(s) = sum_a w_a xtilde(a).d_k(a+s), accB likewise with x:
        // cross-correlations of the weighted xtilde with d_k and x, by FFT.
        int M2 = 1; while (M2 < 2 * grid.n) M2 <<= 1;
        Eigen::FFT<double> cfft;
        std::vector<double> buf(M2, 0.0);
        std::vector<std::complex<double>> W(M2), D(M2), X(M2);
        Eigen::VectorXcd accA = Eigen::VectorXcd::Zero(M2), accB = Eigen::VectorXcd::Zero(M2);
        for (int ch = 0; ch < 3; ++ch) {
            std::fill(buf.begin(), buf.end(), 0.0);
            for (int a = 0; a < grid.n; ++a)
                buf[a] = quad_weight(a, grid.n, grid.h, p.use_simpson_quadrature) * xtilde_k[a](ch);
            cfft.fwd(W, buf);
            std::fill(buf.begin(), buf.end(), 0.0);
            for (int a = 0; a < grid.n; ++a) buf[a] = d_k[a](ch);
            cfft.fwd(D, buf);
            std::fill(buf.begin(), buf.end(), 0.0);
            for (int a = 0; a < grid.n; ++a) buf[a] = x[a](ch);
            cfft.fwd(X, buf);
            Eigen::Map<Eigen::VectorXcd> Wm(W.data(), M2), Dm(D.data(), M2), Xm_(X.data(), M2);
            accA.array() += Wm.conjugate().array() * Dm.array();
            accB.array() += Wm.conjugate().array() * Xm_.array();
        }
        std::vector<std::complex<double>> ta(accA.data(), accA.data() + M2), tb(accB.data(), accB.data() + M2);
        std::vector<double> ra(M2), rb(M2);
        cfft.inv(ra, ta);
        cfft.inv(rb, tb);
        for (int s_ = 1; s_ < grid.n; ++s_) {
            A_s[s_] = obs_gain * d_k[s_](opponent_obs_idx) + opponent_precision * ra[s_];
            B_s[s_] = obs_gain * x[s_](opponent_obs_idx) + opponent_precision * rb[s_];
        }
    }
    const double hx_lhs = 1.0 - 0.5 * grid.h * p.A;
    const double hx_rhs = 1.0 + 0.5 * grid.h * p.A;
    // hx0: recursion applied to the causal state kernel alone.
    std::vector<Vec3> hx0(nb, Vec3::Zero());
    for (int b = nb - 2; b >= 0; --b) {
        const Vec3 q_next = causal_at_b(grid, x, b + 1);
        const Vec3 q_curr = causal_at_b(grid, x, b);
        hx0[b] = (hx_rhs * hx0[b + 1] + 0.5 * grid.h * (q_next + q_curr)) / hx_lhs;
    }

    // Semi-separable assembly of sum_s h A(s) J(b+s, .): J has the closed
    // form J(r,c) = kappa (1+gamma) gamma^{c-r-1} for r < c <= nb-2,
    // J(r,r) = kappa, J(r,nb-1) = kappa gamma^{nb-2-r}, so the row-sums
    // reduce to one geometric prefix recursion G1(m) over m = c - b.
    const double gam_j = hx_rhs / hx_lhs;
    const double kap = 0.5 * grid.h / hx_lhs;
    std::vector<double> G1v(nb, 0.0);
    for (int m = 2; m < nb; ++m) {
        const double a_prev = (m - 1 <= grid.n - 1) ? A_s[m - 1] : 0.0;
        G1v[m] = gam_j * G1v[m - 1] + a_prev;
    }
    const double hk = grid.h * kap;
    const double hk1g = hk * (1.0 + gam_j);
    Eigen::MatrixXd rhs3 = Eigen::MatrixXd::Zero(nb, 3);
    {
        // rhs3(b) = h sum_{s>=1} A_s(s) hx0(b+s): a correlation, by FFT.
        int M2 = 1; while (M2 < 2 * nb) M2 <<= 1;
        Eigen::FFT<double> rfft;
        std::vector<double> buf(M2, 0.0);
        std::vector<std::complex<double>> Af(M2), Hf(M2);
        for (int s_ = 1; s_ < grid.n; ++s_) buf[s_] = grid.h * A_s[s_];
        rfft.fwd(Af, buf);
        for (int ch = 0; ch < 3; ++ch) {
            std::fill(buf.begin(), buf.end(), 0.0);
            for (int b = 0; b < nb; ++b) buf[b] = hx0[b](ch);
            rfft.fwd(Hf, buf);
            for (int i = 0; i < M2; ++i) Hf[i] *= std::conj(Af[i]);
            std::vector<double> out_(M2);
            rfft.inv(out_, Hf);
            for (int b = 0; b < nb; ++b) rhs3(b, ch) = out_[b];
        }
    }

    // The system is upper unit-triangular (the backward transport is
    // anti-causal: v(b) depends only on later lags), so back-substitution
    // solves it exactly -- rows are generated on the fly from the scalar
    // kernels, and the matrix is never materialized.
    const double rhs_norm = rhs3.norm() + 1e-300;
    Eigen::MatrixXd solved3 = rhs3;
    {
        // Rows 0..nb-2 form an upper unit-triangular Toeplitz system
        //   v(b) + sum_{m>=1} w(m) v(b+m) = rhs(b) - wl(b) v(nb-1),
        // with w depending on m = c - b only.  Reversing the index turns it
        // into a lower-triangular Toeplitz system, i.e. a truncated power
        // series division, solved by Newton inversion of the series and one
        // FFT convolution.
        const int K = nb - 1;
        std::vector<double> w(K, 0.0);
        w[0] = 1.0;
        for (int m = 1; m < K; ++m) {
            double v = -hk1g * G1v[m];
            if (m <= grid.n - 1)
                v += -hk * A_s[m] + grid.h * B_s[m];
            w[m] = v;
        }
        // right-hand side with the known last row moved over
        Eigen::MatrixXd r3(K, 3);
        for (int b = 0; b < K; ++b) {
            const int mm = nb - 1 - b;
            const double wl = -hk * G1v[mm] + (mm <= grid.n - 1 ? grid.h * B_s[mm] : 0.0);
            r3.row(b) = rhs3.row(b) - wl * rhs3.row(nb - 1);
        }
        int M2 = 1; while (M2 < 2 * K) M2 <<= 1;
        Eigen::FFT<double> tfft;
        auto conv_trunc = [&](const std::vector<double>& a, const std::vector<double>& b, int len) {
            // first len coefficients of a*b (both zero-padded to M2)
            std::vector<double> ba(M2, 0.0), bb(M2, 0.0);
            std::copy(a.begin(), a.begin() + std::min<int>(a.size(), len), ba.begin());
            std::copy(b.begin(), b.begin() + std::min<int>(b.size(), len), bb.begin());
            std::vector<std::complex<double>> A(M2), B(M2);
            tfft.fwd(A, ba); tfft.fwd(B, bb);
            for (int i = 0; i < M2; ++i) A[i] *= B[i];
            std::vector<double> c(M2);
            tfft.inv(c, A);
            c.resize(len);
            return c;
        };
        // Newton: g <- g (2 - w g), doubling precision in length each step
        std::vector<double> g(1, 1.0);
        int len = 1;
        while (len < K) {
            len = std::min(2 * len, K);
            std::vector<double> wg = conv_trunc(w, g, len);
            for (double& v : wg) v = -v;
            wg[0] += 2.0;
            g = conv_trunc(g, wg, len);
        }
        // u = r_reversed * g, then reverse back
        for (int ch = 0; ch < 3; ++ch) {
            std::vector<double> rr(K);
            for (int j = 0; j < K; ++j) rr[j] = r3(K - 1 - j, ch);
            std::vector<double> u = conv_trunc(rr, g, K);
            for (int j = 0; j < K; ++j) solved3(K - 1 - j, ch) = u[j];
        }
        // one step of iterative refinement against the exact triangular operator
        // (cheap: one Toeplitz matvec per RHS by FFT) to remove FFT rounding
        {
            std::vector<double> wrev(w);
            for (int ch = 0; ch < 3; ++ch) {
                std::vector<double> vv(K);
                for (int j = 0; j < K; ++j) vv[j] = solved3(K - 1 - j, ch);
                std::vector<double> Tv = conv_trunc(vv, wrev, K);   // lower-triangular Toeplitz apply
                std::vector<double> res(K);
                for (int j = 0; j < K; ++j) res[j] = r3(K - 1 - j, ch) - Tv[j];
                std::vector<double> corr = conv_trunc(res, g, K);
                for (int j = 0; j < K; ++j) solved3(K - 1 - j, ch) += corr[j];
            }
        }
    }
    out.wedge.assign(nb, Vec3::Zero());
    for (int b = 0; b < nb; ++b)
        out.wedge[b] = solved3.row(b).transpose();
    // hx from the backward trapezoid recursion; residual from the linear system.
    out.hx.assign(nb, Vec3::Zero());
    for (int b = nb - 2; b >= 0; --b) {
        const Vec3 q_next = causal_at_b(grid, x, b + 1) + out.wedge[b + 1];
        const Vec3 q_curr = causal_at_b(grid, x, b) + out.wedge[b];
        out.hx[b] = (hx_rhs * out.hx[b + 1] + 0.5 * grid.h * (q_next + q_curr)) / hx_lhs;
    }
    out.residual = 0.0;   // exact triangular solve
    (void)rhs_norm;

    for (int a = 0; a < grid.n; ++a)
        out.policy[a] = -(1.0 / player_r) * out.hx[grid.offset + a];

    return out;
}

static Vec3 interp_init(const std::vector<double>& xs,
                        const std::vector<Vec3>& ys, double x) {
    if (x <= xs.front()) return ys.front();
    if (x >= xs.back()) return ys.back();
    const auto it = std::upper_bound(xs.begin(), xs.end(), x);
    const int j = static_cast<int>(it - xs.begin());
    const double t = (x - xs[j - 1]) / (xs[j] - xs[j - 1]);
    return (1.0 - t) * ys[j - 1] + t * ys[j];
}

static void initialize_policy(
    const StationaryGrid& grid, const StationaryParams& p,
    std::vector<Vec3>& d1, std::vector<Vec3>& d2) {

    d1.assign(grid.n, Vec3::Zero());
    d2.assign(grid.n, Vec3::Zero());

    const bool warm = !p.init_lag.empty() &&
        p.init_d1.size() == p.init_lag.size() &&
        p.init_d2.size() == p.init_lag.size();
    if (warm) {
        for (int a = 0; a < grid.n; ++a) {
            d1[a] = interp_init(p.init_lag, p.init_d1, grid.lag[a]);
            d2[a] = interp_init(p.init_lag, p.init_d2, grid.lag[a]);
        }
        return;
    }

    const double inv_sum = 1.0 / p.r1 + 1.0 / p.r2;
    const double S = std::sqrt(1.0 / std::max(1e-12, inv_sum));
    const double K = inv_sum * S;
    for (int a = 0; a < grid.n; ++a) {
        const Vec3 x_pi = p.sigma * std::exp(-K * grid.lag[a]) * E0();
        d1[a] = -(S / p.r1) * x_pi;
        d2[a] = -(S / p.r2) * x_pi;
    }
}

} // namespace

StationarySolution solve_stationary(const StationaryParams& params, bool verbose) {
    StationaryParams p = params;
    p.n_lag = std::max(3, p.n_lag);
    p.lag_max = std::max(1e-6, p.lag_max);
    p.r1 = std::max(1e-8, p.r1);
    p.r2 = std::max(1e-8, p.r2);
    p.relax = std::clamp(p.relax, 1e-4, 1.0);
    p.forward_relax = std::clamp(p.forward_relax, 1e-4, 1.0);
    p.backward_relax = std::clamp(p.backward_relax, 1e-4, 1.0);

    const StationaryGrid grid = make_grid(p);

    std::vector<Vec3> d1, d2;
    initialize_policy(grid, p, d1, d2);

    StationarySolution sol;
    sol.params = p;
    sol.h = grid.h;
    sol.lag = grid.lag;
    sol.b_lag = grid.b_lag;

    ForwardBlock fwd;
    BackwardBlock bwd1, bwd2;
    std::vector<Vec3> last_good_d1 = d1;
    std::vector<Vec3> last_good_d2 = d2;
    double best_relative = std::numeric_limits<double>::infinity();
    double best_absolute = std::numeric_limits<double>::infinity();
    double best_score = std::numeric_limits<double>::infinity();

    std::vector<Vec3> x_warm;                     // forward warm start (exact mode)
    FactorCache bwd_cache1, bwd_cache2;
    LLTCache fwd_cache1, fwd_cache2;
    const int zdim = 6 * grid.n;                  // Anderson state: stacked (d1, d2)
    const int adepth = std::max(0, p.anderson_depth);
    Eigen::MatrixXd aa_dZ(zdim, std::max(1, adepth)), aa_dF(zdim, std::max(1, adepth));
    Eigen::VectorXd aa_z_prev(zdim), aa_f_prev(zdim);
    int aa_cols = 0;
    int aa_total = 0;
    bool aa_have_prev = false;
    auto stack_z = [&](const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
        Eigen::VectorXd z(zdim);
        for (int k = 0; k < grid.n; ++k) {
            z.segment<3>(3 * k) = a[k];
            z.segment<3>(3 * (grid.n + k)) = b[k];
        }
        return z;
    };
    auto unstack_z = [&](const Eigen::VectorXd& z,
                         std::vector<Vec3>& a, std::vector<Vec3>& b) {
        for (int k = 0; k < grid.n; ++k) {
            a[k] = z.segment<3>(3 * k);
            b[k] = z.segment<3>(3 * (grid.n + k));
        }
    };

    double t_fwd = 0.0, t_bwd = 0.0;
    const auto tick = [] { return std::chrono::steady_clock::now(); };
    const auto secs = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count(); };

    double outer_prev = std::numeric_limits<double>::infinity();
    for (int it = 0; it < p.max_iters; ++it) {
        const auto t0 = tick();
        // Inexact inner solve: the forward block only needs to be accurate
        // relative to the current outer residual.  Once the outer residual is
        // within a decade of tolerance, revert to the full forward tolerance
        // so the converged solution does not depend on this.
        StationaryParams pf = p;
        if (p.inexact_forward && std::isfinite(outer_prev) && outer_prev > 10.0 * p.tol)
            pf.forward_tol = std::max(p.forward_tol, std::min(1e-3, 1e-2 * outer_prev));
        fwd = solve_forward_block(grid, pf, d1, d2,
                                  x_warm.empty() ? nullptr : &x_warm,
                                  &fwd_cache1, &fwd_cache2);
        const auto t1 = tick();
#pragma omp parallel sections if (grid.n >= 64)
        {
#pragma omp section
            bwd1 = solve_response_adjoint(grid, p, fwd.x, fwd.xtilde2, d2,
                                          p.p2, 2, p.r1, &bwd_cache1);
#pragma omp section
            bwd2 = solve_response_adjoint(grid, p, fwd.x, fwd.xtilde1, d1,
                                          p.p1, 1, p.r2, &bwd_cache2);
        }
        const auto t2 = tick();
        t_fwd += secs(t0, t1);
        t_bwd += secs(t1, t2);

        if (!finite_forward(fwd) || !finite_backward(bwd1) || !finite_backward(bwd2)) {
            d1 = last_good_d1;
            d2 = last_good_d2;
            sol.relative_residual = best_relative;
            sol.absolute_residual = best_absolute;
            sol.residual = sol.relative_residual;
            break;
        }
        x_warm = fwd.x;

        const GapMetrics gap1 = weighted_gap(d1, bwd1.policy, grid.h, p.use_simpson_quadrature);
        const GapMetrics gap2 = weighted_gap(d2, bwd2.policy, grid.h, p.use_simpson_quadrature);
        sol.relative_residual = std::max(gap1.relative, gap2.relative);
        sol.absolute_residual = std::max(gap1.absolute, gap2.absolute);
        sol.residual = sol.relative_residual;
        outer_prev = sol.relative_residual;
        sol.forward_residual = fwd.residual;
        sol.backward_residual1 = bwd1.residual;
        sol.backward_residual2 = bwd2.residual;
        sol.residuals.push_back(sol.residual);
        sol.relative_residuals.push_back(sol.relative_residual);
        sol.absolute_residuals.push_back(sol.absolute_residual);

        const double score = std::max(sol.relative_residual / std::max(p.tol, 1e-16),
                                      sol.absolute_residual / std::max(p.abs_tol, 1e-16));
        if (std::isfinite(score) && score < best_score) {
            best_score = score;
            best_relative = sol.relative_residual;
            best_absolute = sol.absolute_residual;
            last_good_d1 = d1;
            last_good_d2 = d2;
        }

        if (verbose && (it < 5 || it % 25 == 0)) {
            std::cout << "stationary it=" << it
                      << " rel=" << sol.relative_residual
                      << " abs=" << sol.absolute_residual
                      << " forward=" << sol.forward_residual
                      << " backward=(" << sol.backward_residual1
                      << "," << sol.backward_residual2 << ")\n";
        }

        if (!std::isfinite(sol.relative_residual) || !std::isfinite(sol.absolute_residual))
            break;
        if (sol.relative_residual < p.tol && sol.absolute_residual < p.abs_tol) {
            sol.converged = true;
            break;
        }

        const bool use_aa = adepth > 0 && sol.relative_residual < 0.5;
        if (use_aa) {
            const Eigen::VectorXd z = stack_z(d1, d2);
            const Eigen::VectorXd g = stack_z(bwd1.policy, bwd2.policy);
            const Eigen::VectorXd f = g - z;
            if (aa_have_prev) {
                const int col = aa_total % adepth;
                aa_dZ.col(col) = z - aa_z_prev;
                aa_dF.col(col) = f - aa_f_prev;
                ++aa_total;
                aa_cols = std::min(aa_total, adepth);
            }
            aa_z_prev = z;
            aa_f_prev = f;
            aa_have_prev = true;
            const double beta = p.anderson_mixing;
            Eigen::VectorXd z_new;
            if (aa_cols == 0) {
                z_new = z + beta * f;
            } else {
                const auto dF = aa_dF.leftCols(aa_cols);
                const auto dZ = aa_dZ.leftCols(aa_cols);
                const Eigen::MatrixXd A =
                    dF.transpose() * dF +
                    1e-12 * Eigen::MatrixXd::Identity(aa_cols, aa_cols);
                const Eigen::VectorXd th = A.ldlt().solve(dF.transpose() * f);
                z_new = z + beta * f - (dZ + beta * dF) * th;
            }
            unstack_z(z_new, d1, d2);
        } else {
            aa_have_prev = false;
            aa_cols = 0;
            aa_total = 0;
            relax_into(d1, bwd1.policy, p.relax);
            relax_into(d2, bwd2.policy, p.relax);
        }
    }

    if (verbose)
        std::cerr << "timing: forward " << t_fwd << " s, backward " << t_bwd
                  << " s over " << sol.residuals.size() << " outer iterations\n";

    if (!sol.converged && std::isfinite(best_score)) {
        d1 = last_good_d1;
        d2 = last_good_d2;
        sol.relative_residual = best_relative;
        sol.absolute_residual = best_absolute;
        sol.residual = sol.relative_residual;
    }

    fwd = solve_forward_block(grid, p, d1, d2,
                              x_warm.empty() ? nullptr : &x_warm);
    bwd1 = solve_response_adjoint(grid, p, fwd.x, fwd.xtilde2, d2,
                                  p.p2, 2, p.r1);
    bwd2 = solve_response_adjoint(grid, p, fwd.x, fwd.xtilde1, d1,
                                  p.p1, 1, p.r2);

    if (finite_forward(fwd) && finite_backward(bwd1) && finite_backward(bwd2)) {
        const GapMetrics final_gap1 = weighted_gap(d1, bwd1.policy, grid.h, p.use_simpson_quadrature);
        const GapMetrics final_gap2 = weighted_gap(d2, bwd2.policy, grid.h, p.use_simpson_quadrature);
        sol.relative_residual = std::max(final_gap1.relative, final_gap2.relative);
        sol.absolute_residual = std::max(final_gap1.absolute, final_gap2.absolute);
        sol.residual = sol.relative_residual;
    }

    sol.forward_residual = fwd.residual;
    sol.backward_residual1 = bwd1.residual;
    sol.backward_residual2 = bwd2.residual;
    sol.x = std::move(fwd.x);
    sol.xhat1 = std::move(fwd.xhat1);
    sol.xhat2 = std::move(fwd.xhat2);
    sol.xtilde1 = std::move(fwd.xtilde1);
    sol.xtilde2 = std::move(fwd.xtilde2);
    sol.d1 = std::move(d1);
    sol.d2 = std::move(d2);
    sol.calD1 = std::move(fwd.c1);
    sol.calD2 = std::move(fwd.c2);
    sol.hx1 = std::move(bwd1.hx);
    sol.hx2 = std::move(bwd2.hx);
    sol.wedge1 = std::move(bwd1.wedge);
    sol.wedge2 = std::move(bwd2.wedge);

    return sol;
}
