// Spectral (Chebyshev collocation) solver for the stationary Kyle-Back
// system of Chapter 4: NT insiders with flowing private signals, a
// competitive market maker pricing from order flow, quadratic trading cost
// eps, discount rho, lag window [0, L].  Continuum form of
// numerics/kyleback/kb_final.py.
//
// Channels W = (W^V, W^Z, W^1..W^NT).  Kernels on ages a in [0, L] are
// stored as values on N Chebyshev-Lobatto nodes, (N x NC).
//   dV = sigma_V dW^V,   dZ = sum_i D^i dt + sigma_Z dW^Z,   dY^i = gamma_i (V - P) dt + dW^i,
//   P = E[V | flow history],   J^i = E int e^{-rho t} [D^i (V - P) - eps (D^i)^2] dt.
// Trader i's kernel c^i(a) in R^NC is its demand response to a shock of age a.
//
// Operators (dense, from barycentric interpolation + Gauss-Legendre):
//   observation  (O_k s)(j) = int_0^{L-j} k(m) . s(j+m) dm
//   Volterra     (A_k c)(a) = int_0^a k(a-j) c(j) dj              (price response to own trading)
//   adjoint      (A_k^T c)(j) = int_j^L e^{-rho(a-j)} k(a-j) c(a) da   (discounted transpose)
// All inner products are L2 on [0, L] with the exact mass matrix of the
// nodal basis, and the quadratic form of the price impact is assembled by
// Galerkin quadrature over the causal triangle, so it is symmetric by
// construction (a spurious indefiniteness otherwise appears in unresolved
// directions).  The market maker's price kernel is the mass-orthogonal
// projection of the value kernel onto the flow observations; trader i's
// best response solves the FOC in its observation coordinates,
//   Ht^T (Q_A + Q_A*_rho + 2 eps Mass) Ht y = Ht^T Mass a,  c = Ht y.
//
// Usage: kb_spectral N L eps rho gamma1[,gamma2,...] [--sigma-z s] [--eps-path e1,e2,...]
//        [--tol 1e-10] [--uniform n] [--threads t] [--verbose]
// Prints one JSON object.

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

VectorXd cheb_nodes(int N, double lo, double hi) {
    VectorXd x(N);
    for (int k = 0; k < N; ++k) x[k] = lo + 0.5 * (hi - lo) * (1.0 - std::cos(M_PI * k / (N - 1)));
    return x;
}
VectorXd bary_weights(int N) {
    VectorXd w = VectorXd::Ones(N);
    for (int k = 1; k < N; k += 2) w[k] = -1.0;
    w[0] *= 0.5; w[N - 1] *= 0.5;
    return w;
}
void gauss_legendre(int m, double lo, double hi, VectorXd& x, VectorXd& w) {
    static std::vector<std::pair<VectorXd, VectorXd>> cache;
    if (static_cast<int>(cache.size()) <= m) cache.resize(m + 1);
    if (cache[m].first.size() == 0) {
        MatrixXd J = MatrixXd::Zero(m, m);
        for (int k = 1; k < m; ++k) { const double b = k / std::sqrt(4.0 * k * k - 1.0); J(k, k - 1) = J(k - 1, k) = b; }
        Eigen::SelfAdjointEigenSolver<MatrixXd> es(J);
        cache[m].first = es.eigenvalues();
        cache[m].second = 2.0 * es.eigenvectors().row(0).array().square();
    }
    x = lo + 0.5 * (hi - lo) * (cache[m].first.array() + 1.0);
    w = 0.5 * (hi - lo) * cache[m].second.array();
}
struct Panel {
    int N; double lo, hi; VectorXd x, w;
    Panel(int N_, double lo_, double hi_) : N(N_), lo(lo_), hi(hi_), x(cheb_nodes(N_, lo_, hi_)), w(bary_weights(N_)) {}
    MatrixXd interp(const VectorXd& pts) const {
        MatrixXd M = MatrixXd::Zero(pts.size(), N);
        for (int r = 0; r < pts.size(); ++r) {
            const double p = pts[r]; int hit = -1;
            for (int k = 0; k < N; ++k) if (std::abs(p - x[k]) < 1e-14 * std::max(1.0, std::abs(p))) { hit = k; break; }
            if (hit >= 0) { M(r, hit) = 1.0; continue; }
            double s = 0.0;
            for (int k = 0; k < N; ++k) { M(r, k) = w[k] / (p - x[k]); s += M(r, k); }
            M.row(r) /= s;
        }
        return M;
    }
};

// ------------------------------------------------------------------ model

struct Model {
    int N, NC, NT, m;
    double L, eps, rho, sV, sZ;
    std::vector<double> gam;
    Panel K;
    VectorXd wq;                                   // integration weights on [0, L]
    // linear maps (kernel nodal values -> operator matrices), one per operator type:
    //   OBS[j]  (N x N):  row j of the observation operator for kernel k is k^T OBS[j] applied to s(j + .)  -> stored as (N_k x N_s)
    //   VOL[a]  (N x N):  row a of the Volterra operator:  (A_k c)(a) = k^T VOL[a] c
    //   ADJ[j]  (N x N):  row j of the discounted adjoint: (A_k^T c)(j) = k^T ADJ[j] c
    std::vector<MatrixXd> OBS, VOL, ADJ;
    MatrixXd Mass;                                 // exact L2 Gram of the nodal basis (N x N)
    std::vector<MatrixXd> QA, QAT;                 // Galerkin maps: Q_A = sum_k dP_k QA[k], Q_A*_rho = sum_k dP_k QAT[k]
    MatrixXd v;                                    // value kernel (N x NC)

    Model(int N_, double L_, double eps_, double rho_, std::vector<double> g, double sZ_)
        : N(N_), NC(2 + static_cast<int>(g.size())), NT(static_cast<int>(g.size())), m(N_ + 8),
          L(L_), eps(eps_), rho(rho_), sV(1.0), sZ(sZ_), gam(std::move(g)), K(N_, 0.0, L_) {
        VectorXd u, w;
        OBS.assign(N, MatrixXd::Zero(N, N)); VOL.assign(N, MatrixXd::Zero(N, N)); ADJ.assign(N, MatrixXd::Zero(N, N));
        for (int j = 0; j < N; ++j) {
            const double aj = K.x[j];
            if (L - aj > 0) {
                gauss_legendre(m, 0.0, L - aj, u, w);
                const MatrixXd Pk = K.interp(u), Ps = K.interp((u.array() + aj).matrix());
                OBS[j].noalias() = Pk.transpose() * w.asDiagonal() * Ps;
                ADJ[j].noalias() = Pk.transpose() * (w.array() * (-rho * u.array()).exp()).matrix().asDiagonal() * Ps;
            }
            if (aj > 0) {
                gauss_legendre(m, 0.0, aj, u, w);
                const MatrixXd Pk = K.interp((aj - u.array()).matrix()), Pc = K.interp(u);
                VOL[j].noalias() = Pk.transpose() * w.asDiagonal() * Pc;
            }
        }
        gauss_legendre(m, 0.0, L, u, w);
        {
            const MatrixXd Pu = K.interp(u);
            wq = (w.transpose() * Pu).transpose();
            Mass.noalias() = Pu.transpose() * w.asDiagonal() * Pu;
        }
        // Galerkin forms over the causal triangle a > j:
        //   Q_A(i,l)   = int_0^L da phi_i(a) int_0^a dP(a-j) phi_l(j) dj
        //   Q_A*(i,l)  = int_0^L dj phi_i(j) int_j^L e^{-rho(a-j)} dP(a-j) phi_l(a) da   (= transpose of the discounted form)
        QA.assign(N, MatrixXd::Zero(N, N)); QAT.assign(N, MatrixXd::Zero(N, N));
        for (int q = 0; q < m; ++q) {
            const double a = u[q];
            if (a <= 0) continue;
            VectorXd ui, wi; gauss_legendre(m, 0.0, a, ui, wi);
            const MatrixXd Pj = K.interp(ui), Pk = K.interp((a - ui.array()).matrix());
            const Eigen::RowVectorXd phia = K.interp(VectorXd::Constant(1, a)).row(0);
            const VectorXd disc = (-rho * (a - ui.array())).exp();
            for (int k = 0; k < N; ++k) {
                // weight of dP_k at each inner point: Pk(:, k)
                const VectorXd wk = w[q] * wi.array() * Pk.col(k).array();
                QA[k].noalias() += phia.transpose() * (wk.transpose() * Pj);              // rows: phi_i(a), cols: phi_l(j)
                QAT[k].noalias() += (Pj.transpose() * (wk.array() * disc.array()).matrix()) * phia;   // rows: phi_i(j), cols: phi_l(a)
            }
        }
        v = MatrixXd::Zero(N, NC); v.col(0).setConstant(sV);
    }

    // flat layout: channel-major, index ch*N + node
    static VectorXd flat(const MatrixXd& c) { VectorXd s(c.rows() * c.cols()); for (int ch = 0; ch < c.cols(); ++ch) s.segment(ch * c.rows(), c.rows()) = c.col(ch); return s; }
    MatrixXd unflat(const VectorXd& s) const { MatrixXd c(N, NC); for (int ch = 0; ch < NC; ++ch) c.col(ch) = s.segment(ch * N, N); return c; }

    // observation operator for a vector kernel k (N x NC): Ht (NC N x N) maps observation
    // coefficients y to shock kernels, (Ht y)(a) = y(a) e_id + scale * int_0^a k(a - j) y(j) dj;
    // H = Ht^T Mass is its L2 adjoint.
    void obs_ops(const MatrixXd& k, double scale, int identity_ch, MatrixXd& H, MatrixXd& Ht) const {
        Ht.setZero(NC * N, N);
        for (int a = 0; a < N; ++a)
            for (int ch = 0; ch < NC; ++ch)
                Ht.block(ch * N + a, 0, 1, N).noalias() = scale * (k.col(ch).transpose() * VOL[a]);
        for (int j = 0; j < N; ++j) Ht(identity_ch * N + j, j) += 1.0;
        H.setZero(N, NC * N);
        for (int ch = 0; ch < NC; ++ch) H.block(0, ch * N, N, N).noalias() = Ht.block(ch * N, 0, N, N).transpose() * Mass;
    }
    MatrixXd galerkin_A(const VectorXd& dP) const { MatrixXd Q = MatrixXd::Zero(N, N); for (int k = 0; k < N; ++k) Q += dP[k] * QA[k]; return Q; }
    MatrixXd galerkin_At(const VectorXd& dP) const { MatrixXd Q = MatrixXd::Zero(N, N); for (int k = 0; k < N; ++k) Q += dP[k] * QAT[k]; return Q; }
    MatrixXd volterra(const VectorXd& k) const { MatrixXd A(N, N); for (int a = 0; a < N; ++a) A.row(a) = k.transpose() * VOL[a]; return A; }
    MatrixXd volterra_adj(const VectorXd& k) const { MatrixXd A(N, N); for (int j = 0; j < N; ++j) A.row(j) = k.transpose() * ADJ[j]; return A; }

    struct Diag {
        VectorXd beta; MatrixXd p, g; double lam;
        std::vector<VectorXd> dP;
        std::vector<MatrixXd> A, K, G;              // per trader: impact operator (NC N square), FOC form and L2 metric (2N square)
        std::vector<VectorXd> a_lin;
    };

    // one joint best-response pass
    std::vector<MatrixXd> phi(const std::vector<MatrixXd>& cs, Diag* dg = nullptr) const {
        MatrixXd c_tot = MatrixXd::Zero(N, NC);
        for (const auto& c : cs) c_tot += c;
        MatrixXd Hf, Htf; obs_ops(c_tot, 1.0 / sZ, 1, Hf, Htf);
        const Eigen::PartialPivLU<MatrixXd> Gf(Hf * Htf);
        const VectorXd beta = Gf.solve(Hf * flat(v));
        const MatrixXd p = unflat(Htf * beta);
        const MatrixXd g = v - p;
        const double lam = beta[0] / sZ;
        const MatrixXd Vb = volterra(beta);
        std::vector<MatrixXd> out(NT);
        if (dg) { dg->beta = beta; dg->p = p; dg->g = g; dg->lam = lam; dg->dP.resize(NT); dg->A.resize(NT); dg->K.resize(NT); dg->G.resize(NT); dg->a_lin.resize(NT); }
        // opponents' policy rows in their own observation coordinates (flow excluding own trades + signal)
        std::vector<VectorXd> yf(NT), ys(NT);
        std::vector<MatrixXd> Hs(NT), Hts(NT), Hfo(NT), Htfo(NT);
        for (int j = 0; j < NT; ++j) {
            obs_ops(g, gam[j], 2 + j, Hs[j], Hts[j]);
            obs_ops(c_tot - cs[j], 1.0 / sZ, 1, Hfo[j], Htfo[j]);
            MatrixXd Hj(2 * N, NC * N); Hj << Hfo[j], Hs[j];
            MatrixXd Htj(NC * N, 2 * N); Htj << Htfo[j], Hts[j];
            const VectorXd y = (Hj * Htj).partialPivLu().solve(Hj * flat(cs[j]));
            yf[j] = y.head(N); ys[j] = y.tail(N);
        }
        const MatrixXd I = MatrixXd::Identity(N, N);
        for (int i = 0; i < NT; ++i) {
            // price-impact cascade: dP = beta/sZ + (1/sZ) V_beta x,  x = sum_{j != i} [ yf_j/sZ + (1/sZ) V_{yf_j} x - gam_j V_{ys_j} dP ]
            VectorXd dP;
            if (NT == 1) {
                dP = beta / sZ;
            } else {
                MatrixXd SF = MatrixXd::Zero(N, N), SS = MatrixXd::Zero(N, N);
                VectorXd f0 = VectorXd::Zero(N);
                for (int j = 0; j < NT; ++j) if (j != i) { SF += volterra(yf[j]); SS += gam[j] * volterra(ys[j]); f0 += yf[j] / sZ; }
                MatrixXd M(2 * N, 2 * N);
                M << I, -Vb / sZ, SS, I - SF / sZ;
                VectorXd rhs(2 * N); rhs << beta / sZ, f0;
                dP = M.partialPivLu().solve(rhs).head(N);
            }
            const MatrixXd A1 = volterra(dP);
            const MatrixXd Q1 = galerkin_A(dP) + galerkin_At(dP) + 2.0 * eps * Mass;
            MatrixXd A = MatrixXd::Zero(NC * N, NC * N), Mq = MatrixXd::Zero(NC * N, NC * N), MassB = MatrixXd::Zero(NC * N, NC * N);
            for (int ch = 0; ch < NC; ++ch) {
                A.block(ch * N, ch * N, N, N) = A1;
                Mq.block(ch * N, ch * N, N, N) = Q1;
                MassB.block(ch * N, ch * N, N, N) = Mass;
            }
            MatrixXd Ht(NC * N, 2 * N); Ht << Htfo[i], Hts[i];
            const VectorXd a_lin = flat(v - p) + A * flat(cs[i]);
            const MatrixXd Kmat = Ht.transpose() * Mq * Ht;                 // quadratic form in observation coordinates
            const VectorXd y = Kmat.partialPivLu().solve(Ht.transpose() * (MassB * a_lin));
            out[i] = unflat(Ht * y);
            if (dg) { dg->dP[i] = dP; dg->A[i] = A; dg->K[i] = Kmat; dg->G[i] = Ht.transpose() * MassB * Ht; dg->a_lin[i] = a_lin; }
        }
        return out;
    }

    VectorXd pack(const std::vector<MatrixXd>& cs) const { VectorXd z(NT * NC * N); for (int i = 0; i < NT; ++i) z.segment(i * NC * N, NC * N) = flat(cs[i]); return z; }
    std::vector<MatrixXd> unpack(const VectorXd& z) const { std::vector<MatrixXd> cs(NT); for (int i = 0; i < NT; ++i) cs[i] = unflat(z.segment(i * NC * N, NC * N)); return cs; }
    VectorXd residual(const VectorXd& z, Diag* dg = nullptr) const { return pack(phi(unpack(z), dg)) - z; }

    // per-trader profit flow and its channel decomposition: <c, a_lin> - <c, A c> - eps <c, c>
    std::vector<VectorXd> profit_by_channel(const std::vector<MatrixXd>& cs, const Diag& dg) const {
        std::vector<VectorXd> res(NT);
        for (int i = 0; i < NT; ++i) {
            const VectorXd c = flat(cs[i]);
            const VectorXd integrand = c.array() * (dg.a_lin[i] - dg.A[i] * c - eps * c).array();
            VectorXd by(NC);
            for (int ch = 0; ch < NC; ++ch) by[ch] = wq.dot(integrand.segment(ch * N, N));
            res[i] = by;
        }
        return res;
    }
    // lag by which half of trader i's profit has accrued (cumulative integrand over age)
    double half_profit_lag(const std::vector<MatrixXd>& cs, const Diag& dg, int i) const {
        const VectorXd c = flat(cs[i]);
        const VectorXd integrand = c.array() * (dg.a_lin[i] - dg.A[i] * c - eps * c).array();
        VectorXd tot = VectorXd::Zero(N);
        for (int ch = 0; ch < NC; ++ch) tot += integrand.segment(ch * N, N);
        const double total = wq.dot(tot);
        // cumulative integral via fine quadrature on [0, a]
        VectorXd u, w;
        for (int k = 1; k <= 400; ++k) {
            const double a = L * k / 400.0;
            gauss_legendre(m, 0.0, a, u, w);
            const double cum = (w.transpose() * K.interp(u) * tot)(0);
            if (cum >= 0.5 * total) return a;
        }
        return L;
    }
    std::vector<double> certificate(const Diag& dg) const {
        // smallest eigenvalue of the quadratic form relative to the L2 metric G = Ht^T Mass Ht
        std::vector<double> out(NT);
        for (int i = 0; i < NT; ++i) {
            const MatrixXd Ks = 0.5 * (dg.K[i] + dg.K[i].transpose());
            Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(Ks, dg.G[i], Eigen::EigenvaluesOnly);
            out[i] = es.eigenvalues().minCoeff();
        }
        return out;
    }
};

// ----------------------------------------------------------------- newton

struct Solver {
    Model& M;
    MatrixXd Jinv; bool have_J = false;
    long evals = 0;
    bool verbose = false;
    explicit Solver(Model& m) : M(m) {}

    struct Out { VectorXd z; double resid; int steps, jacobians; bool ok; };

    Out solve(VectorXd z, double tol, int pre = 0, double relax = 0.1, int maxsteps = 60) {
        Out o; o.jacobians = 0; o.ok = false;
        // damped pre-phase with monitoring
        {
            double lam = relax, best = std::numeric_limits<double>::infinity(); VectorXd zbest = z;
            for (int k = 0; k < pre; ++k) {
                const VectorXd r = M.residual(z); ++evals;
                const double rn = r.cwiseAbs().maxCoeff();
                if (!std::isfinite(rn) || rn > 2.0 * best) { z = zbest; lam *= 0.5; if (lam < 1e-3) break; continue; }
                if (rn < best) { best = rn; zbest = z; }
                if (rn < 1e-3) break;
                z += lam * r;
            }
            z = zbest;
        }
        VectorXd r = M.residual(z); ++evals;
        double rn = r.cwiseAbs().maxCoeff();
        const double rn0 = rn;
        const int n = static_cast<int>(z.size());
        double last_ratio = 0.0;
        int step = 0;
        for (; step < maxsteps && rn > tol; ++step) {
            if (!have_J || last_ratio > 0.5) {
                MatrixXd Jm(n, n);
                const double eps_fd = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
#pragma omp parallel for schedule(dynamic, 4)
                for (int i = 0; i < n; ++i) { VectorXd zp = z; zp[i] += eps_fd; Jm.col(i) = (M.residual(zp) - r) / eps_fd; }
                evals += n;
                Jinv = Jm.partialPivLu().inverse(); have_J = true; ++o.jacobians; last_ratio = 0.0;
            }
            const VectorXd dz = -(Jinv * r);
            double lam = 1.0; bool acc = false;
            for (int ls = 0; ls < 8; ++ls) {
                const VectorXd zt = z + lam * dz;
                const VectorXd rt = M.residual(zt); ++evals;
                const double rtn = rt.cwiseAbs().maxCoeff();
                if (std::isfinite(rtn) && rtn < (1.0 - 1e-4 * lam) * rn) {
                    const VectorXd sz = zt - z, yr = rt - r, Jy = Jinv * yr;
                    const double den = sz.dot(Jy);
                    if (std::abs(den) > 1e-14 * sz.norm() * Jy.norm()) Jinv.noalias() += ((sz - Jy) * (sz.transpose() * Jinv)) / den;
                    last_ratio = rtn / rn; z = zt; r = rt; rn = rtn; acc = true; break;
                }
                lam *= 0.5;
            }
            if (verbose) std::fprintf(stderr, "  newton %d: |r| %.3e step %.3g%s\n", step, rn, lam, acc ? "" : " (rejected)");
            if (!acc) { if (last_ratio <= 0.5) { last_ratio = 1.0; continue; } break; }
            if (!std::isfinite(rn) || rn > 1e6 * std::max(rn0, 1.0)) break;
        }
        o.z = z; o.resid = rn; o.steps = step; o.ok = rn <= tol;
        return o;
    }
};

// ------------------------------------------------------------------- json

void print_array(const VectorXd& v) {
    std::printf("[");
    for (int i = 0; i < v.size(); ++i) std::printf("%s%.15g", i ? "," : "", v[i]);
    std::printf("]");
}
void print_vec(const char* name, const VectorXd& v, bool last = false) {
    std::printf("\"%s\":", name); print_array(v); std::printf("%s", last ? "" : ",");
}
void print_karray(const MatrixXd& k) {
    std::printf("[");
    for (int ch = 0; ch < k.cols(); ++ch) { std::printf("%s", ch ? "," : ""); print_array(k.col(ch)); }
    std::printf("]");
}
void print_kernel(const char* name, const MatrixXd& k, bool last = false) {
    std::printf("\"%s\":", name); print_karray(k); std::printf("%s", last ? "" : ",");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s N L eps rho gamma1[,gamma2,...] [--sigma-z s] [--eps-path e1,e2,...] [--tol 1e-10] [--uniform n] [--threads t] [--verbose]\n", argv[0]);
        return 1;
    }
    const int N = std::atoi(argv[1]); const double L = std::atof(argv[2]), eps = std::atof(argv[3]), rho = std::atof(argv[4]);
    std::vector<double> gam; { std::string s = argv[5]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); gam.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; } }
    double sZ = 1.0, tol = 1e-10; int uniform = 0; bool verbose = false;
    std::vector<double> path;
    for (int i = 6; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sigma-z") && i + 1 < argc) sZ = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--uniform") && i + 1 < argc) uniform = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--eps-path") && i + 1 < argc) { std::string s = argv[++i]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); path.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; } }
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(std::atoi(argv[++i]));
#else
            ++i;
#endif
        }
    }
#ifdef _OPENMP
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_num_procs()));
#endif
    // default continuation path: from a well-posed cost down to the target
    if (path.empty()) { for (double e : {0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0}) if (e > eps) path.push_back(e); path.push_back(eps); }

    const auto t0 = std::chrono::steady_clock::now();
    Model M(N, L, path.front(), rho, gam, sZ);
    Solver S(M); S.verbose = verbose;
    VectorXd z = VectorXd::Zero(M.NT * M.NC * N);
    Solver::Out o;
    VectorXd zprev; double eprev = 0; bool have_prev = false;
    for (size_t k = 0; k < path.size(); ++k) {
        M.eps = path[k];
        VectorXd zstart = z;
        if (have_prev && eprev != path[k == 0 ? 0 : k - 1]) zstart = z + (z - zprev) * ((path[k] - path[k - 1]) / (path[k - 1] - eprev));
        o = S.solve(zstart, tol, k == 0 ? 200 : 0, 0.1);
        if (!o.ok) { S.have_J = false; o = S.solve(z, tol, k == 0 ? 0 : 30, 0.1); }
        if (verbose) std::fprintf(stderr, "eps=%g: %s |r| %.2e, %d steps, %d jacobians, %ld evals total\n", path[k], o.ok ? "ok" : "FAIL", o.resid, o.steps, o.jacobians, S.evals);
        if (!o.ok) break;
        if (k > 0) { zprev = z; eprev = path[k - 1]; have_prev = true; }
        z = o.z;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    Model::Diag dg;
    const std::vector<MatrixXd> cs = M.unpack(z);
    M.residual(z, &dg);
    const auto prof = M.profit_by_channel(cs, dg);
    const auto cert = M.certificate(dg);
    VectorXd q(4); q << 0.0, 1.0, 2.0, 4.0;
    const VectorXd gapV = M.K.interp(q) * dg.g.col(0);

    std::printf("{\"converged\":%s,\"residual\":%.3e,\"N\":%d,\"L\":%.15g,\"eps\":%.15g,\"rho\":%.15g,\"sigma_Z\":%.15g,\"NT\":%d,\"seconds\":%.4f,\"evaluations\":%ld,",
                o.ok ? "true" : "false", o.resid, N, L, M.eps, rho, sZ, M.NT, secs, S.evals);
    print_vec("gammas", Eigen::Map<const VectorXd>(gam.data(), gam.size()));
    std::printf("\"lambda\":%.15g,", dg.lam);
    print_vec("gapV_0_1_2_4", gapV);
    std::printf("\"traders\":[");
    for (int i = 0; i < M.NT; ++i) {
        std::printf("%s{\"flow\":%.15g,\"margin\":%.6g,\"half_profit_lag\":%.6g,", i ? "," : "", prof[i].sum(), cert[i], M.half_profit_lag(cs, dg, i));
        print_vec("flow_by_channel", prof[i]);
        print_kernel("c", cs[i]);
        print_vec("dP", dg.dP[i], true);
        std::printf("}");
    }
    std::printf("],");
    print_vec("lag", M.K.x);
    print_vec("beta", dg.beta);
    print_kernel("p", dg.p);
    print_kernel("g", dg.g, uniform == 0);
    if (uniform > 0) {
        VectorXd ul(uniform); for (int i = 0; i < uniform; ++i) ul[i] = L * i / (uniform - 1);
        const MatrixXd Iu = M.K.interp(ul);
        std::printf("\"uniform\":{"); print_vec("lag", ul); print_kernel("g", Iu * dg.g); print_vec("beta", Iu * dg.beta);
        std::printf("\"c\":[");
        for (int i = 0; i < M.NT; ++i) { std::printf("%s", i ? "," : ""); print_karray(Iu * cs[i]); }
        std::printf("]}");
    }
    std::printf("}\n");
    return o.ok ? 0 : 2;
}
