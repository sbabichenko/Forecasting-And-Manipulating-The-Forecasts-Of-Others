// Spectral solver for the q-asset stationary Kyle-Back system of Chapter 4
// (continuum form of numerics/kyleback/kb_multi.py).
//
//   dV   = Sigma_V^{1/2} dW^V              (q-dim random walk)
//   dZ   = sum_j D^j dt + Sigma_Z^{1/2} dW^Z
//   dY^j = diag(gamma_j) (V - P) dt + dW^j   (q-dim signal per trader; zero gain = no signal in that stock)
//   P    = E[V | flow history],  J^j = E int e^{-rho t} [D^j . (V - P) - eps |D^j|^2] dt.
//
// Primitive channels: q V-factors, q Z-factors, q signal-noise factors per
// trader, NC = q (2 + NT).  Trader i's kernel c^i(a) is (NC x q): channel
// by demand component.  Kernels are nodal values on N Chebyshev points of
// [0, L].  Observations carry q rows per age.  All inner products are L2
// with the exact mass matrix; the impact quadratic form is Galerkin.
//
// Usage: kb_spectral_q N L eps rho q "g11,..,g1q;g21,..,g2q;..." [--sigma-v v11,v12,...]
//        [--sigma-z z11,...] [--eps-path ...] [--tol 1e-10] [--uniform n] [--threads t] [--verbose]

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <malloc.h>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
static long g_pre_ok = 0, g_pre_fallback = 0, g_full = 0; static double g_t_full = 0, g_t_pre = 0;
static inline double now_s(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

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

// Chebyshev grid on [0, L]: one panel, or two panels [0, b] and [b, L] sharing the node at b
// (N1 nodes on the first, N - N1 + 1 on the second).  Kernels are piecewise smooth with a
// possible kink at b, so every integral is split at b and at the kink's image.
struct Grid {
    // Chebyshev nodes in a computational coordinate s in [0, L], mapped to a = phi(s) with
    // phi(s) = L (e^{alpha s/L} - 1) / (e^alpha - 1) (alpha = 0: identity).  The basis stays
    // globally smooth while nodes cluster near a = 0.  Optionally two panels split at b (kept
    // for experiments; it lets the discrete strategy space kink at b, which small-eps
    // equilibria exploit).
    int N, N1; double L, b, alpha; bool split;
    Panel p1, p2; VectorXd s_nodes, x;
    double phi(double s) const { return alpha == 0.0 ? s : L * (std::exp(alpha * s / L) - 1.0) / (std::exp(alpha) - 1.0); }
    double phi_inv(double a) const { return alpha == 0.0 ? a : (L / alpha) * std::log(1.0 + a * (std::exp(alpha) - 1.0) / L); }
    double dphi(double s) const { return alpha == 0.0 ? 1.0 : alpha * std::exp(alpha * s / L) / (std::exp(alpha) - 1.0); }
    Grid(int N_, double L_, int N1_ = 0, double b_ = 0.0, double alpha_ = 0.0)
        : N(N_), N1(N1_), L(L_), b(b_), alpha(alpha_), split(N1_ > 1 && b_ > 0 && b_ < L_),
          p1(split ? N1_ : N_, 0.0, split ? b_ : L_), p2(split ? N_ - N1_ + 1 : 2, split ? b_ : 0.0, L_) {
        if (split) { s_nodes.resize(N); s_nodes.head(N1) = p1.x; s_nodes.tail(N - N1) = p2.x.tail(N - N1); }
        else s_nodes = p1.x;
        x = s_nodes.unaryExpr([&](double s) { return phi(s); });
    }
    MatrixXd interp(const VectorXd& pts) const {
        const VectorXd sp = pts.unaryExpr([&](double a) { return phi_inv(std::max(0.0, std::min(L, a))); });
        if (!split) return p1.interp(sp);
        MatrixXd M = MatrixXd::Zero(pts.size(), N);
        for (int r = 0; r < pts.size(); ++r) {
            VectorXd one(1); one << sp[r];
            if (sp[r] < b) M.block(r, 0, 1, N1) = p1.interp(one);
            else           M.block(r, N1 - 1, 1, N - N1 + 1) = p2.interp(one);
        }
        return M;
    }
    // Gauss-Legendre on [lo, hi] (physical), split at the breakpoints given (and at b);
    // with a map each piece is integrated in s with the Jacobian phi'.
    void quad(double lo, double hi, std::vector<double> breaks, VectorXd& u, VectorXd& w, int m) const {
        if (split) breaks.push_back(phi(b));
        std::vector<double> pts = {lo};
        for (double c : breaks) if (c > lo + 1e-14 && c < hi - 1e-14) pts.push_back(c);
        pts.push_back(hi);
        std::sort(pts.begin(), pts.end());
        std::vector<double> U, W;
        for (size_t k = 0; k + 1 < pts.size(); ++k) {
            if (pts[k + 1] - pts[k] <= 1e-14) continue;
            VectorXd uu, ww;
            gauss_legendre(m, phi_inv(pts[k]), phi_inv(pts[k + 1]), uu, ww);
            for (int i = 0; i < m; ++i) { const double s = uu[i]; ww[i] *= dphi(s); uu[i] = phi(s); }
            for (int i = 0; i < m; ++i) { U.push_back(uu[i]); W.push_back(ww[i]); }
        }
        u = Eigen::Map<VectorXd>(U.data(), U.size()); w = Eigen::Map<VectorXd>(W.data(), W.size());
    }
};

std::vector<double> parse_list(const std::string& s, char sep = ',') {
    std::vector<double> v; size_t p = 0;
    while (p <= s.size()) { size_t q = s.find(sep, p); if (q == std::string::npos) q = s.size(); if (q > p) v.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; }
    return v;
}

// ------------------------------------------------------------------ model

struct Model {
    int N, q, NT, NC, m;
    double L, eps, rho;
    std::vector<VectorXd> gam;                     // per trader, per stock gain
    MatrixXd SV, SZ, SVh, SZh, SZhi;               // covariances, Cholesky factors, Sigma_Z^{-1/2}
    Grid K;
    VectorXd wq;
    std::vector<MatrixXd> VOL, QA, QAT;            // Volterra row maps and Galerkin maps (N x N each)
    MatrixXd Mass;
    MatrixXd v;                                    // value kernel (NC x q), constant in age: V-factor f row = SVh(:, f)^T

    Model(int N_, double L_, double eps_, double rho_, int q_, std::vector<VectorXd> g, MatrixXd SV_, MatrixXd SZ_, int N1_ = 0, double b_ = 0.0, double alpha_ = 0.0)
        : N(N_), q(q_), NT(static_cast<int>(g.size())), NC(q_ * (2 + static_cast<int>(g.size()))), m(std::max(N_ - std::max(N1_, 1) + 1, N1_) + (alpha_ > 0 ? N_ : 8)),
          L(L_), eps(eps_), rho(rho_), gam(std::move(g)), SV(std::move(SV_)), SZ(std::move(SZ_)), K(N_, L_, N1_, b_, alpha_) {
        SVh = Eigen::LLT<MatrixXd>(SV).matrixL();
        SZh = Eigen::LLT<MatrixXd>(SZ).matrixL();
        SZhi = SZh.inverse();
        VectorXd u, w;
        VOL.assign(N, MatrixXd::Zero(N, N));
        for (int j = 0; j < N; ++j) {
            const double aj = K.x[j];
            if (aj > 0) {
                // (A_k c)(a_j) = int_0^{a_j} k(a_j - u) c(u) du; k has its kink at u = a_j - b
                K.quad(0.0, aj, {aj - K.b}, u, w, m);
                const MatrixXd Pk = K.interp((aj - u.array()).matrix()), Pc = K.interp(u);
                VOL[j].noalias() = Pk.transpose() * w.asDiagonal() * Pc;
            }
        }
        K.quad(0.0, L, {}, u, w, m);
        { const MatrixXd Pu = K.interp(u); wq = (w.transpose() * Pu).transpose(); Mass.noalias() = Pu.transpose() * w.asDiagonal() * Pu; }
        QA.assign(N, MatrixXd::Zero(N, N)); QAT.assign(N, MatrixXd::Zero(N, N));
        for (int qq = 0; qq < u.size(); ++qq) {
            const double a = u[qq];
            if (a <= 0) continue;
            VectorXd ui, wi; K.quad(0.0, a, {a - K.b}, ui, wi, m);
            const MatrixXd Pj = K.interp(ui), Pk = K.interp((a - ui.array()).matrix());
            const Eigen::RowVectorXd phia = K.interp(VectorXd::Constant(1, a)).row(0);
            const VectorXd disc = (-rho * (a - ui.array())).exp();
            for (int k = 0; k < N; ++k) {
                const VectorXd wk = w[qq] * wi.array() * Pk.col(k).array();
                QA[k].noalias() += phia.transpose() * (wk.transpose() * Pj);
                QAT[k].noalias() += (Pj.transpose() * (wk.array() * disc.array()).matrix()) * phia;
            }
        }
        v = MatrixXd::Zero(NC, q);
        for (int f = 0; f < q; ++f) v.row(f) = SVh.col(f).transpose();   // V-factor f channel, component nu: SVh(nu, f)
    }

    // kernels: c (N x NC x q) stored as vector<MatrixXd> over ages? -> use flat layout: index (ch, comp, node) = (ch*q + comp)*N + node
    int dim() const { return NC * q * N; }
    MatrixXd volterra(const VectorXd& k) const { MatrixXd A(N, N); for (int a = 0; a < N; ++a) A.row(a) = k.transpose() * VOL[a]; return A; }

    // Observation operator Ht for drift kernel kw (per age: (q obs comps) x NC channels), stored as
    // kw[o][ch] = VectorXd over ages, plus identity picks: obs comp o reads channel id_ch0 + o.
    //   (Ht y)_ch(a) = sum_o [ch == id_ch0 + o] y_o(a) + sum_o int_0^a kw[o][ch](a - j) y_o(j) dj
    // Ht: (NC N) x (q N); H = Ht^T Mass_block.
    MatrixXd obs_Ht(const std::vector<std::vector<VectorXd>>& kw, int id_ch0) const {
        MatrixXd Ht = MatrixXd::Zero(NC * N, q * N);
        const int nb = q * NC;
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
        for (int b = 0; b < nb; ++b) {
            const int o = b / NC, ch = b % NC;
            if (kw[o][ch].size() && kw[o][ch].cwiseAbs().maxCoeff() != 0.0) Ht.block(ch * N, o * N, N, N) = volterra(kw[o][ch]);
        }
        for (int o = 0; o < q; ++o) Ht.block((id_ch0 + o) * N, o * N, N, N) += MatrixXd::Identity(N, N);
        return Ht;
    }
    MatrixXd Htmass(const MatrixXd& Ht) const {   // H = Ht^T blockdiag(Mass): (q N) x (NC N)
        MatrixXd H(Ht.cols(), Ht.rows());
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
        for (int ch = 0; ch < NC; ++ch) H.block(0, ch * N, Ht.cols(), N).noalias() = Ht.block(ch * N, 0, N, Ht.cols()).transpose() * Mass;
        return H;
    }
    // drift kernels of the flow observation for total demand kernel C (vector over ages of (NC x q)):
    //   kw[o][ch](a) = sum_u SZhi(o, u) C(a)[ch, u]
    std::vector<std::vector<VectorXd>> flow_kw(const std::vector<MatrixXd>& C) const {
        std::vector<std::vector<VectorXd>> kw(q, std::vector<VectorXd>(NC, VectorXd::Zero(N)));
        for (int o = 0; o < q; ++o) for (int ch = 0; ch < NC; ++ch) for (int a = 0; a < N; ++a) {
            double s = 0.0; for (int u = 0; u < q; ++u) s += SZhi(o, u) * C[a](ch, u);
            kw[o][ch][a] = s;
        }
        return kw;
    }
    // signal drift kernels for trader j: kw[o][ch](a) = gam_j[o] * g(a)[ch, o]
    std::vector<std::vector<VectorXd>> signal_kw(const std::vector<MatrixXd>& g, int j) const {
        std::vector<std::vector<VectorXd>> kw(q, std::vector<VectorXd>(NC, VectorXd::Zero(N)));
        for (int o = 0; o < q; ++o) { if (gam[j][o] == 0.0) { for (int ch = 0; ch < NC; ++ch) kw[o][ch].resize(0); continue; }
            for (int ch = 0; ch < NC; ++ch) for (int a = 0; a < N; ++a) kw[o][ch][a] = gam[j][o] * g[a](ch, o); }
        return kw;
    }
    // kernel <-> (q N)-column layout helpers: a kernel (ages x NC x q) as matrix (NC N) x q with row ch*N + a
    MatrixXd kmat(const std::vector<MatrixXd>& C) const { MatrixXd M(NC * N, q); for (int a = 0; a < N; ++a) for (int ch = 0; ch < NC; ++ch) M.row(ch * N + a) = C[a].row(ch); return M; }
    std::vector<MatrixXd> kvec(const MatrixXd& M) const { std::vector<MatrixXd> C(N, MatrixXd::Zero(NC, q)); for (int a = 0; a < N; ++a) for (int ch = 0; ch < NC; ++ch) C[a].row(ch) = M.row(ch * N + a); return C; }

    struct Diag { MatrixXd beta, p, g, lam; std::vector<MatrixXd> K, G, a_lin; std::vector<std::vector<MatrixXd>> dP; std::vector<double> margin; std::vector<Eigen::PartialPivLU<MatrixXd>> Klu, Glu; Eigen::PartialPivLU<MatrixXd> Gflu;
        // linearization state
        std::vector<MatrixXd> cs;                       // kernels at the base point
        MatrixXd Htf, Hf, vmat, ctot;                   // flow observation operator (total flow), its adjoint, value kernel, total kernel
        std::vector<MatrixXd> Htj, Hj, yj;              // per trader: observation operator (residual flow + signal), adjoint, policy rows
        std::vector<std::vector<std::vector<MatrixXd>>> Aq, Mqq;   // per trader: impact blocks A[nu][k], form blocks Mq[nu][k]
        std::vector<MatrixXd> Ymat, alin_raw;           // per trader: FOC solution Y (R x q), a_lin before profit accounting
        std::vector<Eigen::PartialPivLU<MatrixXd>> Mlu; std::vector<MatrixXd> Msol;   // per trader: cascade system LU and its solution [dP; X] (2qN x q); empty if NT == 1
        bool lin = false; };

    // one joint best-response pass; cs[i] is kmat layout (NC N x q)
    // Richardson iteration on A X = B with a nearby factorization as preconditioner (A given as a dense matrix here)
    static MatrixXd pre_solve(const MatrixXd& A, const MatrixXd& B, const Eigen::PartialPivLU<MatrixXd>& lu) {
        MatrixXd X = lu.solve(B);
        for (int it = 0; it < 6; ++it) {
            const MatrixXd res = B - A * X;
            if (res.cwiseAbs().maxCoeff() < 1e-14 * std::max(1.0, B.cwiseAbs().maxCoeff())) break;
            X += lu.solve(res);
        }
        return X;
    }
    // pre: a Diag from a nearby point whose FOC factorizations precondition a matrix-free solve (used for Jacobian columns)
    std::vector<MatrixXd> phi(const std::vector<MatrixXd>& cs, Diag* dg = nullptr, const Diag* pre = nullptr) const {
        MatrixXd ctot = MatrixXd::Zero(NC * N, q);
        for (const auto& c : cs) ctot += c;
        const std::vector<MatrixXd> Ctot = kvec(ctot);
        const MatrixXd Htf = obs_Ht(flow_kw(Ctot), q);
        const MatrixXd Hf = Htmass(Htf);
        const MatrixXd vmat = [&] { MatrixXd V = MatrixXd::Zero(NC * N, q); for (int a = 0; a < N; ++a) for (int ch = 0; ch < NC; ++ch) V.row(ch * N + a) = v.row(ch); return V; }();
        MatrixXd beta;                                                          // (q N) x q: obs (o, age) -> price comp
        {
            const MatrixXd Gf = Hf * Htf;
            if (pre && !dg) beta = pre_solve(Gf, Hf * vmat, pre->Gflu);
            else { Eigen::PartialPivLU<MatrixXd> lu(Gf); beta = lu.solve(Hf * vmat); if (dg) dg->Gflu = lu; }
        }
        const MatrixXd p = Htf * beta;
        const MatrixXd gmat = vmat - p;
        const std::vector<MatrixXd> g = kvec(gmat);
        MatrixXd lam(q, q);                                                    // instantaneous impact: lam(nu, k) = sum_o beta(o, age 0; nu) SZhi(o, k)
        for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) { double s = 0; for (int o = 0; o < q; ++o) s += beta(o * N + 0, nu) * SZhi(o, k); lam(nu, k) = s; }
        if (dg) { dg->beta = beta; dg->p = p; dg->g = gmat; dg->lam = lam; dg->K.resize(NT); dg->G.resize(NT); dg->a_lin.resize(NT); dg->dP.resize(NT); dg->margin.resize(NT); dg->Klu.resize(NT); dg->Glu.resize(NT);
                  dg->cs = cs; dg->Htf = Htf; dg->Hf = Hf; dg->vmat = vmat; dg->ctot = ctot; dg->Htj.resize(NT); dg->Hj.resize(NT); dg->yj.resize(NT); dg->Aq.resize(NT); dg->Mqq.resize(NT); dg->Ymat.resize(NT); dg->alin_raw.resize(NT); dg->Mlu.resize(NT); dg->Msol.resize(NT); dg->lin = true; }
        const bool use_pre = pre && static_cast<int>(pre->Klu.size()) == NT && !dg;
        // each trader's observation operator (residual flow + own signal) and policy rows y_j = G_j^{-1} H_j c^j
        std::vector<MatrixXd> Htj(NT), Hj(NT), yj(NT);
#pragma omp parallel for schedule(static) if (!omp_in_parallel() && NT > 1)
        for (int j = 0; j < NT; ++j) {
            const MatrixXd Htfo = obs_Ht(flow_kw(kvec(ctot - cs[j])), q);
            const MatrixXd Hts = obs_Ht(signal_kw(g, j), (2 + j) * q);
            Htj[j].resize(NC * N, 2 * q * N); Htj[j] << Htfo, Hts;
            Hj[j] = Htmass(Htj[j]);
            const MatrixXd Gj = Hj[j] * Htj[j];
            if (pre && !dg) yj[j] = pre_solve(Gj, Hj[j] * cs[j], pre->Glu[j]);
            else { Eigen::PartialPivLU<MatrixXd> lu(Gj); yj[j] = lu.solve(Hj[j] * cs[j]); if (dg) { dg->Glu[j] = lu; dg->Htj[j] = Htj[j]; dg->Hj[j] = Hj[j]; dg->yj[j] = yj[j]; } }
        }
        std::vector<MatrixXd> out(NT);
        const MatrixXd I = MatrixXd::Identity(N, N);
        for (int i = 0; i < NT; ++i) {
            // ---- cascade: dP[nu][k] (vector over ages), X[u][k]: unknown blocks per spike direction k
            // dP_nu = sum_o SZhi(o,k) beta_{o,nu} + sum_o V_{beta_{o,nu}} F_o,   F_o = sum_u SZhi(o,u) X_u
            // X_u = sum_{j!=i} sum_o [ SZhi(o,k) rf_{j,o,u} + V_{rf_{j,o,u}} F_o - gam_j[o] V_{rs_{j,o,u}} dP_o ]
            std::vector<std::vector<VectorXd>> dP(q, std::vector<VectorXd>(q, VectorXd::Zero(N)));   // dP[nu][k]
            std::vector<std::vector<MatrixXd>> Vb(q, std::vector<MatrixXd>(q));
            for (int o = 0; o < q; ++o) for (int nu = 0; nu < q; ++nu) Vb[o][nu] = volterra(beta.block(o * N, nu, N, 1));
            if (NT == 1) {
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) for (int o = 0; o < q; ++o) dP[nu][k] += SZhi(o, k) * beta.block(o * N, nu, N, 1);
            } else {
                // unknown vector per k: [dP_0..dP_{q-1}; X_0..X_{q-1}] each N -> size 2 q N
                MatrixXd Msys = MatrixXd::Zero(2 * q * N, 2 * q * N);
                MatrixXd rhs = MatrixXd::Zero(2 * q * N, q);
                for (int nu = 0; nu < q; ++nu) {
                    Msys.block(nu * N, nu * N, N, N) = I;
                    for (int o = 0; o < q; ++o) for (int u = 0; u < q; ++u) Msys.block(nu * N, (q + u) * N, N, N) -= SZhi(o, u) * Vb[o][nu];
                    for (int k = 0; k < q; ++k) for (int o = 0; o < q; ++o) rhs.block(nu * N, k, N, 1) += SZhi(o, k) * beta.block(o * N, nu, N, 1);
                }
                for (int u = 0; u < q; ++u) {
                    Msys.block((q + u) * N, (q + u) * N, N, N) = I;
                    for (int j = 0; j < NT; ++j) if (j != i) for (int o = 0; o < q; ++o) {
                        const VectorXd rf = yj[j].block(o * N, u, N, 1), rs = yj[j].block((q + o) * N, u, N, 1);
                        const MatrixXd Vrf = volterra(rf), Vrs = volterra(rs);
                        for (int uu = 0; uu < q; ++uu) Msys.block((q + u) * N, (q + uu) * N, N, N) -= SZhi(o, uu) * Vrf;
                        Msys.block((q + u) * N, o * N, N, N) += gam[j][o] * Vrs;
                        for (int k = 0; k < q; ++k) rhs.block((q + u) * N, k, N, 1) += SZhi(o, k) * rf;
                    }
                }
                Eigen::PartialPivLU<MatrixXd> mlu(Msys);
                const MatrixXd sol = mlu.solve(rhs);
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) dP[nu][k] = sol.block(nu * N, k, N, 1);
                if (dg) { dg->Mlu[i] = mlu; dg->Msol[i] = sol; }
            }
            // ---- impact operator blocks and Galerkin form blocks: A_{nu k} = V_{dP[nu][k]}, Mq_{nu k} = sum_m dP[nu][k]_m QA[m] + sum_m dP[k][nu]_m QAT[m] + 2 eps delta Mass
            std::vector<std::vector<MatrixXd>> A(q, std::vector<MatrixXd>(q)), Mq(q, std::vector<MatrixXd>(q));
            for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) {
                A[nu][k] = volterra(dP[nu][k]);
                MatrixXd Q1 = MatrixXd::Zero(N, N);
                for (int mm = 0; mm < N; ++mm) Q1 += dP[nu][k][mm] * QA[mm] + dP[k][nu][mm] * QAT[mm];
                if (nu == k) Q1 += 2.0 * eps * Mass;
                Mq[nu][k] = Q1;
            }
            // a_lin (NC N x q): v - p + A c_own  (per channel, components mix through A)
            MatrixXd a_lin = gmat;
            for (int ch = 0; ch < NC; ++ch) for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k)
                a_lin.block(ch * N, nu, N, 1) += A[nu][k] * cs[i].block(ch * N, k, N, 1);
            // FOC in observation coordinates: unknown Y (2 q N x q), flattened (o-block row, comp): index (r, nu) -> r*? use blocks
            const int R = 2 * q * N;                                           // observation rows (o-blocks x N)
            const MatrixXd& Ht = Htj[i];
            VectorXd rhs = VectorXd::Zero(R * q);
            for (int ch = 0; ch < NC; ++ch)
                for (int nu = 0; nu < q; ++nu)
                    rhs.segment(nu * R, R).noalias() += Ht.block(ch * N, 0, N, R).transpose() * (Mass * a_lin.block(ch * N, nu, N, 1));
            VectorXd yv;
            MatrixXd Kmat, Gm;
            bool solved = false;
            if (use_pre) {
                // matrix-free K apply, preconditioned Richardson with the base LU; fall back to assembly if it stalls
                auto applyK = [&](const VectorXd& y) {
                    VectorXd out = VectorXd::Zero(R * q);
                    for (int ch = 0; ch < NC; ++ch) {
                        const auto Hc = Ht.block(ch * N, 0, N, R);
                        MatrixXd C(N, q); for (int k = 0; k < q; ++k) C.col(k).noalias() = Hc * y.segment(k * R, R);
                        for (int nu = 0; nu < q; ++nu) { VectorXd w = VectorXd::Zero(N); for (int k = 0; k < q; ++k) w.noalias() += Mq[nu][k] * C.col(k); out.segment(nu * R, R).noalias() += Hc.transpose() * w; }
                    }
                    return out;
                };
                yv = pre->Klu[i].solve(rhs);
                const double rn0 = std::max(1.0, rhs.cwiseAbs().maxCoeff()); double prev = std::numeric_limits<double>::infinity();
                for (int it = 0; it < 12; ++it) {
                    const VectorXd res = rhs - applyK(yv);
                    const double rn = res.cwiseAbs().maxCoeff();
                    if (rn < 1e-14 * rn0) { solved = true; break; }
                    if (rn > 0.7 * prev) break;        // not contracting: stale preconditioner
                    prev = rn;
                    yv += pre->Klu[i].solve(res);
                }
            }
            if (solved) ++g_pre_ok; else if (use_pre) ++g_pre_fallback;
            if (!solved) {
                ++g_full;
                // assemble K = sum_ch Hc^T Mq Hc as blockdiag-apply plus one large product per (nu, k)
                Kmat = MatrixXd::Zero(R * q, R * q); Gm = MatrixXd::Zero(R * q, R * q);
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) {
                    MatrixXd W(NC * N, R);
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
                    for (int ch = 0; ch < NC; ++ch) W.block(ch * N, 0, N, R).noalias() = Mq[nu][k] * Ht.block(ch * N, 0, N, R);
                    Kmat.block(nu * R, k * R, R, R).noalias() = Ht.transpose() * W;
                }
                if (dg) for (int nu = 0; nu < q; ++nu) {
                    MatrixXd W(NC * N, R);
                    for (int ch = 0; ch < NC; ++ch) W.block(ch * N, 0, N, R).noalias() = Mass * Ht.block(ch * N, 0, N, R);
                    Gm.block(nu * R, nu * R, R, R).noalias() = Ht.transpose() * W;
                }
                Eigen::PartialPivLU<MatrixXd> lu(Kmat);
                yv = lu.solve(rhs);
                if (dg) dg->Klu[i] = lu;
            }
            MatrixXd Y(R, q); for (int nu = 0; nu < q; ++nu) Y.col(nu) = yv.segment(nu * R, R);
            out[i] = Ht * Y;
            if (dg) {
                dg->Aq[i] = A; dg->Mqq[i] = Mq; dg->Ymat[i] = Y; dg->alin_raw[i] = a_lin;
                dg->K[i] = Kmat; dg->G[i] = Gm; dg->a_lin[i] = a_lin; dg->dP[i].resize(q * q);
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) dg->dP[i][nu * q + k] = dP[nu][k];
                // profit flow needs A c: store in a_lin? compute here: flow = <c, a_lin - A c - eps c>
                MatrixXd Ac = MatrixXd::Zero(NC * N, q);
                for (int ch = 0; ch < NC; ++ch) for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) Ac.block(ch * N, nu, N, 1) += A[nu][k] * cs[i].block(ch * N, k, N, 1);
                dg->a_lin[i] = a_lin - Ac - eps * cs[i];                        // integrand partner: flow = <c, this>
                Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(0.5 * (Kmat + Kmat.transpose()), Gm, Eigen::EigenvaluesOnly);
                dg->margin[i] = es.eigenvalues().minCoeff();
            }
        }
        return out;
    }

    // Directional derivative d Phi(c)[dc] at the base point B (which must hold the linearization
    // state).  Every stage of phi is linear in the kernels that enter it, so the derivative is a
    // chain of products with base quantities and solves with base factorizations; nothing is
    // re-factored and no perturbed operator is assembled beyond the observation operators of dc.
    std::vector<MatrixXd> dphi(const Diag& B, const std::vector<MatrixXd>& dcs) const {
        MatrixXd dctot = MatrixXd::Zero(NC * N, q); for (const auto& d : dcs) dctot += d;
        auto Ht_of = [&](const MatrixXd& kmat_, bool flow, int j) {
            std::vector<std::vector<VectorXd>> kw = flow ? flow_kw(kvec(kmat_)) : signal_kw(kvec(kmat_), j);
            MatrixXd Ht = MatrixXd::Zero(NC * N, q * N);
            for (int o = 0; o < q; ++o) for (int ch = 0; ch < NC; ++ch)
                if (kw[o][ch].size() && kw[o][ch].cwiseAbs().maxCoeff() != 0.0) Ht.block(ch * N, o * N, N, N) = volterra(kw[o][ch]);
            return Ht;
        };
        auto massB = [&](const MatrixXd& X) { MatrixXd Y(X.rows(), X.cols()); for (int ch = 0; ch < NC; ++ch) Y.block(ch * N, 0, N, X.cols()).noalias() = Mass * X.block(ch * N, 0, N, X.cols()); return Y; };
        const MatrixXd dHtf = Ht_of(dctot, true, 0);
        // dHf X = dHtf^T (Mass X): apply, never form
        const MatrixXd Htfb = B.Htf * B.beta, dHtfb = dHtf * B.beta;
        const MatrixXd dbeta = B.Gflu.solve(dHtf.transpose() * massB(B.vmat - Htfb) - B.Hf * dHtfb);
        const MatrixXd dp = dHtfb + B.Htf * dbeta;
        const MatrixXd dg = -dp;
        std::vector<MatrixXd> dHtj(NT), dyj(NT);
        for (int j = 0; j < NT; ++j) {
            dHtj[j].resize(NC * N, 2 * q * N);
            dHtj[j] << Ht_of(dctot - dcs[j], true, 0), Ht_of(dg, false, j);
            // dHj X = dHtj^T (Mass X)
            dyj[j] = B.Glu[j].solve(dHtj[j].transpose() * massB(B.cs[j] - B.Htj[j] * B.yj[j]) + B.Hj[j] * (dcs[j] - dHtj[j] * B.yj[j]));
        }
        std::vector<MatrixXd> out(NT);
        for (int i = 0; i < NT; ++i) {
            std::vector<std::vector<VectorXd>> ddP(q, std::vector<VectorXd>(q, VectorXd::Zero(N)));
            if (NT == 1) {
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) for (int o = 0; o < q; ++o) ddP[nu][k] += SZhi(o, k) * dbeta.block(o * N, nu, N, 1);
            } else {
                MatrixXd dM = MatrixXd::Zero(2 * q * N, 2 * q * N), drhs = MatrixXd::Zero(2 * q * N, q);
                for (int nu = 0; nu < q; ++nu) {
                    for (int o = 0; o < q; ++o) { const MatrixXd dVb = volterra(dbeta.block(o * N, nu, N, 1)); for (int u = 0; u < q; ++u) dM.block(nu * N, (q + u) * N, N, N) -= SZhi(o, u) * dVb; }
                    for (int k = 0; k < q; ++k) for (int o = 0; o < q; ++o) drhs.block(nu * N, k, N, 1) += SZhi(o, k) * dbeta.block(o * N, nu, N, 1);
                }
                for (int u = 0; u < q; ++u) for (int j = 0; j < NT; ++j) if (j != i) for (int o = 0; o < q; ++o) {
                    const VectorXd drf = dyj[j].block(o * N, u, N, 1), drs = dyj[j].block((q + o) * N, u, N, 1);
                    const MatrixXd dVrf = volterra(drf), dVrs = volterra(drs);
                    for (int uu = 0; uu < q; ++uu) dM.block((q + u) * N, (q + uu) * N, N, N) -= SZhi(o, uu) * dVrf;
                    dM.block((q + u) * N, o * N, N, N) += gam[j][o] * dVrs;
                    for (int k = 0; k < q; ++k) drhs.block((q + u) * N, k, N, 1) += SZhi(o, k) * drf;
                }
                const MatrixXd dsol = B.Mlu[i].solve(drhs - dM * B.Msol[i]);
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) ddP[nu][k] = dsol.block(nu * N, k, N, 1);
            }
            std::vector<std::vector<MatrixXd>> dA(q, std::vector<MatrixXd>(q)), dMq(q, std::vector<MatrixXd>(q));
            for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) {
                dA[nu][k] = volterra(ddP[nu][k]);
                MatrixXd Q1 = MatrixXd::Zero(N, N);
                for (int mm = 0; mm < N; ++mm) Q1 += ddP[nu][k][mm] * QA[mm] + ddP[k][nu][mm] * QAT[mm];
                dMq[nu][k] = Q1;
            }
            MatrixXd dalin = -dp;
            for (int ch = 0; ch < NC; ++ch) for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k)
                dalin.block(ch * N, nu, N, 1) += dA[nu][k] * B.cs[i].block(ch * N, k, N, 1) + B.Aq[i][nu][k] * dcs[i].block(ch * N, k, N, 1);
            const int R = 2 * q * N;
            const MatrixXd& Ht = B.Htj[i]; const MatrixXd& dHt = dHtj[i]; const MatrixXd& Y = B.Ymat[i];
            VectorXd drhs_v = VectorXd::Zero(R * q), dKY = VectorXd::Zero(R * q);
            for (int ch = 0; ch < NC; ++ch) {
                const auto Hc = Ht.block(ch * N, 0, N, R); const auto dHc = dHt.block(ch * N, 0, N, R);
                MatrixXd C(N, q), dC(N, q); for (int k = 0; k < q; ++k) { C.col(k) = Hc * Y.col(k); dC.col(k) = dHc * Y.col(k); }
                for (int nu = 0; nu < q; ++nu) {
                    drhs_v.segment(nu * R, R) += dHc.transpose() * (Mass * B.alin_raw[i].block(ch * N, nu, N, 1)) + Hc.transpose() * (Mass * dalin.block(ch * N, nu, N, 1));
                    VectorXd w = VectorXd::Zero(N), wd = VectorXd::Zero(N);
                    for (int k = 0; k < q; ++k) { w += B.Mqq[i][nu][k] * C.col(k); wd += dMq[nu][k] * C.col(k) + B.Mqq[i][nu][k] * dC.col(k); }
                    dKY.segment(nu * R, R) += dHc.transpose() * w + Hc.transpose() * wd;
                }
            }
            const VectorXd dyv = B.Klu[i].solve(drhs_v - dKY);
            MatrixXd dY(R, q); for (int nu = 0; nu < q; ++nu) dY.col(nu) = dyv.segment(nu * R, R);
            out[i] = dHt * Y + Ht * dY;
        }
        return out;
    }

    // ---- fast linearization: base tensors so that every product with a perturbed kernel is N^2
    //   T_x (N x N): (A_k x)     = T_x k    with T_x(a, :) = (VOL[a] x)^T
    //   P_w (N x N): (A_k^T w)   = P_w^T k  with P_w = sum_a w_a VOL[a]
    //   S_x, S'_x  : (Q_A(dP) x) = S_x dP,  (Q_A*(dP) x) = S'_x dP  with columns QA[m] x, QAT[m] x
    MatrixXd Tmat(const VectorXd& x) const { MatrixXd T(N, N); for (int a = 0; a < N; ++a) T.row(a) = (VOL[a] * x).transpose(); return T; }
    MatrixXd Pmat(const VectorXd& w) const { MatrixXd P = MatrixXd::Zero(N, N); for (int a = 0; a < N; ++a) if (w[a] != 0.0) P += w[a] * VOL[a]; return P; }
    MatrixXd Smat(const VectorXd& x, bool adj) const { MatrixXd S(N, N); for (int m = 0; m < N; ++m) S.col(m) = (adj ? QAT[m] : QA[m]) * x; return S; }
    struct LinBase {
        // market maker
        std::vector<MatrixXd> Tbeta;                 // [o*q+nu]: T of beta block (o, nu)
        std::vector<MatrixXd> Pw1;                   // [ch*q+nu]: P of (Mass (v - Htf beta))[ch-block, nu]
        MatrixXd W1;                                 // Mass (vmat - Htf beta)
        // per trader j: policy rows
        std::vector<std::vector<MatrixXd>> Ty;       // [j][r*q+comp]: T of yj block r (r < 2q), comp
        std::vector<std::vector<MatrixXd>> Pw2;      // [j][ch*q+comp]: P of (Mass (c_j - Htj yj))[ch, comp]
        std::vector<MatrixXd> W2;
        // per trader i: cascade, impact, FOC
        std::vector<std::vector<MatrixXd>> TF, TdP;  // [i][o*q+k]: T of F_{o,k} = sum_u SZhi(o,u) X_{u,k}; T of dP_{o,k}
        std::vector<std::vector<MatrixXd>> Tc;       // [i][ch*q+k]: T of c_i[ch-block, k]
        std::vector<std::vector<MatrixXd>> SC, SCa;  // [i][ch*q+k]: S, S' of C_{ch,k} = Hc Y_k
        std::vector<std::vector<MatrixXd>> TY;       // [i][r*q+k]: T of Y block r, comp k
        std::vector<std::vector<MatrixXd>> Pma, Pwq; // [i][ch*q+nu]: P of Mass a_lin[ch,nu]; P of w_{ch,nu} = sum_k Mq_{nu k} C_{ch,k}
        std::vector<std::vector<VectorXd>> Cvec;     // [i][ch*q+k]: C_{ch,k}
    };
    LinBase linbase(const Diag& B) const {
        LinBase L;
        L.Tbeta.resize(q * q); for (int o = 0; o < q; ++o) for (int nu = 0; nu < q; ++nu) L.Tbeta[o * q + nu] = Tmat(B.beta.block(o * N, nu, N, 1));
        L.W1.resize(NC * N, q); { const MatrixXd D = B.vmat - B.Htf * B.beta; for (int ch = 0; ch < NC; ++ch) L.W1.block(ch * N, 0, N, q).noalias() = Mass * D.block(ch * N, 0, N, q); }
        L.Pw1.resize(NC * q); for (int ch = 0; ch < NC; ++ch) for (int nu = 0; nu < q; ++nu) L.Pw1[ch * q + nu] = Pmat(L.W1.block(ch * N, nu, N, 1));
        L.Ty.resize(NT); L.Pw2.resize(NT); L.W2.resize(NT);
        for (int j = 0; j < NT; ++j) {
            L.Ty[j].resize(2 * q * q); for (int r = 0; r < 2 * q; ++r) for (int c = 0; c < q; ++c) L.Ty[j][r * q + c] = Tmat(B.yj[j].block(r * N, c, N, 1));
            L.W2[j].resize(NC * N, q); { const MatrixXd D = B.cs[j] - B.Htj[j] * B.yj[j]; for (int ch = 0; ch < NC; ++ch) L.W2[j].block(ch * N, 0, N, q).noalias() = Mass * D.block(ch * N, 0, N, q); }
            L.Pw2[j].resize(NC * q); for (int ch = 0; ch < NC; ++ch) for (int c = 0; c < q; ++c) L.Pw2[j][ch * q + c] = Pmat(L.W2[j].block(ch * N, c, N, 1));
        }
        L.TF.resize(NT); L.TdP.resize(NT); L.Tc.resize(NT); L.SC.resize(NT); L.SCa.resize(NT); L.TY.resize(NT); L.Pma.resize(NT); L.Pwq.resize(NT); L.Cvec.resize(NT);
        const int R = 2 * q * N;
        for (int i = 0; i < NT; ++i) {
            if (NT > 1) {
                L.TF[i].resize(q * q); L.TdP[i].resize(q * q);
                for (int o = 0; o < q; ++o) for (int k = 0; k < q; ++k) {
                    VectorXd F = VectorXd::Zero(N); for (int u = 0; u < q; ++u) F += SZhi(o, u) * B.Msol[i].block((q + u) * N, k, N, 1);
                    L.TF[i][o * q + k] = Tmat(F);
                    L.TdP[i][o * q + k] = Tmat(B.Msol[i].block(o * N, k, N, 1));
                }
            }
            L.Tc[i].resize(NC * q); for (int ch = 0; ch < NC; ++ch) for (int k = 0; k < q; ++k) L.Tc[i][ch * q + k] = Tmat(B.cs[i].block(ch * N, k, N, 1));
            L.TY[i].resize(2 * q * q); for (int r = 0; r < 2 * q; ++r) for (int k = 0; k < q; ++k) L.TY[i][r * q + k] = Tmat(B.Ymat[i].block(r * N, k, N, 1));
            L.SC[i].resize(NC * q); L.SCa[i].resize(NC * q); L.Cvec[i].resize(NC * q); L.Pma[i].resize(NC * q); L.Pwq[i].resize(NC * q);
            for (int ch = 0; ch < NC; ++ch) {
                const auto Hc = B.Htj[i].block(ch * N, 0, N, R);
                std::vector<VectorXd> C(q); for (int k = 0; k < q; ++k) C[k] = Hc * B.Ymat[i].col(k);
                for (int k = 0; k < q; ++k) { L.Cvec[i][ch * q + k] = C[k]; L.SC[i][ch * q + k] = Smat(C[k], false); L.SCa[i][ch * q + k] = Smat(C[k], true); }
                for (int nu = 0; nu < q; ++nu) {
                    L.Pma[i][ch * q + nu] = Pmat(Mass * B.alin_raw[i].block(ch * N, nu, N, 1));
                    VectorXd w = VectorXd::Zero(N); for (int k = 0; k < q; ++k) w += B.Mqq[i][nu][k] * C[k];
                    L.Pwq[i][ch * q + nu] = Pmat(w);
                }
            }
        }
        return L;
    }
    // drift kernels of a perturbation, as vectors: flow kw[o][ch] = sum_u SZhi(o,u) dc[ch,u]; signal kw[o][ch] = gam_j[o] dg[ch,o]
    std::vector<MatrixXd> dphi_fast(const Diag& B, const LinBase& L, const std::vector<MatrixXd>& dcs) const {
        const int R = 2 * q * N;
        MatrixXd dctot = MatrixXd::Zero(NC * N, q); for (const auto& d : dcs) dctot += d;
        auto flowk = [&](const MatrixXd& dc, int o, int ch) { VectorXd k = VectorXd::Zero(N); for (int u = 0; u < q; ++u) if (SZhi(o, u) != 0.0) k += SZhi(o, u) * dc.block(ch * N, u, N, 1); return k; };
        // ---- market maker: dbeta = G^{-1} (dHtf^T W1 - Hf (dHtf beta))
        MatrixXd rhsb = MatrixXd::Zero(q * N, q), dHtfb = MatrixXd::Zero(NC * N, q);
        for (int o = 0; o < q; ++o) for (int ch = 0; ch < NC; ++ch) {
            const VectorXd k = flowk(dctot, o, ch); if (k.cwiseAbs().maxCoeff() == 0.0) continue;
            for (int nu = 0; nu < q; ++nu) { rhsb.block(o * N, nu, N, 1) += L.Pw1[ch * q + nu].transpose() * k; dHtfb.block(ch * N, nu, N, 1) += L.Tbeta[o * q + nu] * k; }
        }
        const MatrixXd dbeta = B.Gflu.solve(rhsb - B.Hf * dHtfb);
        const MatrixXd dp = dHtfb + B.Htf * dbeta;
        const MatrixXd dg = -dp;
        // ---- policy rows: dyj = G_j^{-1} (dHtj^T W2 + Hj (dc_j - dHtj yj))
        std::vector<MatrixXd> dyj(NT);
        // perturbed drift kernels of trader j's observation operator, kept for the FOC stage
        std::vector<std::vector<std::vector<VectorXd>>> kwj(NT, std::vector<std::vector<VectorXd>>(2 * q, std::vector<VectorXd>(NC)));
        for (int j = 0; j < NT; ++j) {
            MatrixXd rhs = MatrixXd::Zero(R, q), dHty = MatrixXd::Zero(NC * N, q);
            for (int r = 0; r < 2 * q; ++r) for (int ch = 0; ch < NC; ++ch) {
                VectorXd k;
                if (r < q) k = flowk(dctot - dcs[j], r, ch);
                else { if (gam[j][r - q] == 0.0) continue; k = gam[j][r - q] * dg.block(ch * N, r - q, N, 1); }
                if (k.cwiseAbs().maxCoeff() == 0.0) continue;
                kwj[j][r][ch] = k;
                for (int c = 0; c < q; ++c) { rhs.block(r * N, c, N, 1) += L.Pw2[j][ch * q + c].transpose() * k; dHty.block(ch * N, c, N, 1) += L.Ty[j][r * q + c] * k; }
            }
            dyj[j] = B.Glu[j].solve(rhs + B.Hj[j] * (dcs[j] - dHty));
        }
        std::vector<MatrixXd> out(NT);
        for (int i = 0; i < NT; ++i) {
            // ---- cascade
            std::vector<std::vector<VectorXd>> ddP(q, std::vector<VectorXd>(q, VectorXd::Zero(N)));
            if (NT == 1) {
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) for (int o = 0; o < q; ++o) ddP[nu][k] += SZhi(o, k) * dbeta.block(o * N, nu, N, 1);
            } else {
                MatrixXd drhs = MatrixXd::Zero(2 * q * N, q);   // d rhs - dM sol
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) {
                    VectorXd v = VectorXd::Zero(N);
                    for (int o = 0; o < q; ++o) { v += SZhi(o, k) * dbeta.block(o * N, nu, N, 1); v += L.TF[i][o * q + k] * dbeta.block(o * N, nu, N, 1); }   // -(-V_dbeta F)
                    drhs.block(nu * N, k, N, 1) = v;
                }
                for (int u = 0; u < q; ++u) for (int k = 0; k < q; ++k) {
                    VectorXd v = VectorXd::Zero(N);
                    for (int j = 0; j < NT; ++j) if (j != i) for (int o = 0; o < q; ++o) {
                        const VectorXd drf = dyj[j].block(o * N, u, N, 1), drs = dyj[j].block((q + o) * N, u, N, 1);
                        v += SZhi(o, k) * drf;                               // d rhs
                        v += L.TF[i][o * q + k] * drf;                       // -(-V_drf F)
                        v -= gam[j][o] * (L.TdP[i][o * q + k] * drs);        // -(+gam V_drs dP)
                    }
                    drhs.block((q + u) * N, k, N, 1) = v;
                }
                const MatrixXd dsol = B.Mlu[i].solve(drhs);
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) ddP[nu][k] = dsol.block(nu * N, k, N, 1);
            }
            // ---- a_lin: dalin = -dp + dA c + A dc
            MatrixXd dalin = -dp;
            for (int ch = 0; ch < NC; ++ch) for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k)
                dalin.block(ch * N, nu, N, 1) += L.Tc[i][ch * q + k] * ddP[nu][k] + B.Aq[i][nu][k] * dcs[i].block(ch * N, k, N, 1);
            // ---- FOC: dY = K^{-1} (d rhs - dK Y)
            const MatrixXd& Ht = B.Htj[i]; const MatrixXd& Y = B.Ymat[i];
            VectorXd drhs_v = VectorXd::Zero(R * q), dKY = VectorXd::Zero(R * q);
            MatrixXd dHtY = MatrixXd::Zero(NC * N, q);
            for (int ch = 0; ch < NC; ++ch) {
                const auto Hc = Ht.block(ch * N, 0, N, R);
                // dC_{ch,k} = (dHc Y)_k = sum_r T_{Y_{r,k}} kw_r,ch
                MatrixXd dC = MatrixXd::Zero(N, q);
                for (int r = 0; r < 2 * q; ++r) if (kwj[i][r][ch].size()) for (int k = 0; k < q; ++k) dC.col(k) += L.TY[i][r * q + k] * kwj[i][r][ch];
                dHtY.block(ch * N, 0, N, q) = dC;
                for (int nu = 0; nu < q; ++nu) {
                    // d rhs: dHc^T (M a) + Hc^T (M dalin)
                    for (int r = 0; r < 2 * q; ++r) if (kwj[i][r][ch].size()) drhs_v.segment(nu * R + r * N, N) += L.Pma[i][ch * q + nu].transpose() * kwj[i][r][ch];
                    drhs_v.segment(nu * R, R) += Hc.transpose() * (Mass * dalin.block(ch * N, nu, N, 1));
                    // dK Y: dHc^T w + Hc^T (dMq C + Mq dC)
                    for (int r = 0; r < 2 * q; ++r) if (kwj[i][r][ch].size()) dKY.segment(nu * R + r * N, N) += L.Pwq[i][ch * q + nu].transpose() * kwj[i][r][ch];
                    VectorXd wd = VectorXd::Zero(N);
                    for (int k = 0; k < q; ++k) {
                        wd += L.SC[i][ch * q + k] * ddP[nu][k] + L.SCa[i][ch * q + k] * ddP[k][nu];   // dMq_{nu k} C_k
                        wd += B.Mqq[i][nu][k] * dC.col(k);
                    }
                    dKY.segment(nu * R, R) += Hc.transpose() * wd;
                }
            }
            const VectorXd dyv = B.Klu[i].solve(drhs_v - dKY);
            MatrixXd dY(R, q); for (int nu = 0; nu < q; ++nu) dY.col(nu) = dyv.segment(nu * R, R);
            out[i] = dHtY + Ht * dY;
        }
        return out;
    }

    VectorXd pack(const std::vector<MatrixXd>& cs) const { VectorXd z(NT * NC * N * q); for (int i = 0; i < NT; ++i) for (int k = 0; k < q; ++k) z.segment((i * q + k) * NC * N, NC * N) = cs[i].col(k); return z; }
    std::vector<MatrixXd> unpack(const VectorXd& z) const { std::vector<MatrixXd> cs(NT, MatrixXd(NC * N, q)); for (int i = 0; i < NT; ++i) for (int k = 0; k < q; ++k) cs[i].col(k) = z.segment((i * q + k) * NC * N, NC * N); return cs; }
    VectorXd residual(const VectorXd& z, Diag* dg = nullptr, const Diag* pre = nullptr) const { const double t0 = now_s(); VectorXd r = pack(phi(unpack(z), dg, pre)) - z; (dg || !pre ? g_t_full : g_t_pre) += now_s() - t0; return r; }

    // profit flow per trader, by channel block (V, Z, trader noises) and by stock (demand component)
    MatrixXd profit(const std::vector<MatrixXd>& cs, const Diag& dg, int i) const {
        MatrixXd P = MatrixXd::Zero(NC, q);
        for (int ch = 0; ch < NC; ++ch) for (int nu = 0; nu < q; ++nu)
        {
            const VectorXd a = cs[i].block(ch * N, nu, N, 1), b = dg.a_lin[i].block(ch * N, nu, N, 1);
            P(ch, nu) = wq.dot(a.cwiseProduct(b));
        }
        return P;
    }
};

// ----------------------------------------------------------------- newton

struct Solver {
    Model& M; bool have_J = false; long evals = 0; bool verbose = false; double refresh_ratio = 0.8; int max_jacobians = 3;
    // inverse Jacobian as an LU plus Broyden rank-one corrections: Jinv r = LU^{-1} r + sum_k u_k (v_k . r)
    MatrixXd Jstore; Eigen::PartialPivLU<Eigen::Ref<MatrixXd>> Jlu{Jstore}; std::vector<VectorXd> bu, bv;
    VectorXd apply_Jinv(const VectorXd& r) const { VectorXd x = Jlu.solve(r); for (size_t k = 0; k < bu.size(); ++k) x += bu[k] * bv[k].dot(r); return x; }
    struct JinvOp { const Solver* S; VectorXd operator*(const VectorXd& r) const { return S->apply_Jinv(r); } };
    JinvOp Jinv{this};
    bool jfnk = true; int gmres_max = 12; double gmres_tol = 1e-3; bool analytic = true; bool krylov_stalled = false; bool exact_newton = false; bool range_newton = false;
    Model::Diag cur; bool have_cur = false;   // factorizations at the current iterate, preconditioning nearby evaluations
    long gmres_its = 0;
    explicit Solver(Model& m) : M(m) {}
    struct Out { VectorXd z; double resid; int steps, jacobians; bool ok; };
    Out solve(VectorXd z, double tol, int pre = 0, double relax = 0.1, int maxsteps = 30) {
        Out o; o.jacobians = 0; o.ok = false;
        { double lam = relax, best = std::numeric_limits<double>::infinity(); VectorXd zbest = z;
          for (int k = 0; k < pre; ++k) { const VectorXd r = have_cur ? M.residual(z, nullptr, &cur) : M.residual(z); ++evals; const double rn = r.cwiseAbs().maxCoeff();
              if (verbose && k % 20 == 0) std::fprintf(stderr, "  pre %d: |r| %.3e relax %.3g\n", k, rn, lam);
              if (!std::isfinite(rn) || rn > 2.0 * best) { z = zbest; lam *= 0.5; if (lam < 1e-3) break; continue; }
              if (rn < best) { best = rn; zbest = z; } if (rn < 1e-3) break; z += lam * r; }
          z = zbest; }
        VectorXd r = M.residual(z, &cur); have_cur = true; ++evals; double rn = r.cwiseAbs().maxCoeff(); const double rn0 = rn;
        const int n = static_cast<int>(z.size()); double last_ratio = 0.0; int step = 0; int fails = 0; bool fresh = false;
        double rn_at10 = std::numeric_limits<double>::infinity();
        bool use_krylov = jfnk && !exact_newton;     // hybrid: Newton-Krylov until GMRES stalls in this solve, then chord + Broyden for the rest of it
        for (; step < maxsteps && rn > tol; ++step) {
            if (step == 10) rn_at10 = rn;
            if (step == 20 && rn > 0.3 * rn_at10) break;      // not making progress: hand back for eps bisection
            if (!range_newton && (exact_newton || !have_J || (last_ratio > refresh_ratio && o.jacobians < (use_krylov ? 1 : max_jacobians)))) {   // under JFNK the Broyden-updated inverse is only a preconditioner: refresh at most once per solve
                Jstore.resize(n, n); MatrixXd& Jm = Jstore; const double eps_fd = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
                Model::Diag& base = cur; const VectorXd rb = r;   // factorizations at z precondition the columns
                if (analytic) {
                    const Model::LinBase LB = M.linbase(base);
#pragma omp parallel for schedule(dynamic, 8)
                    for (int i = 0; i < n; ++i) { VectorXd e = VectorXd::Zero(n); e[i] = 1.0; Jm.col(i) = M.pack(M.dphi_fast(base, LB, M.unpack(e))) - e; }
                } else {
#pragma omp parallel for schedule(dynamic, 4)
                    for (int i = 0; i < n; ++i) { VectorXd zp = z; zp[i] += eps_fd; Jm.col(i) = (M.residual(zp, nullptr, &base) - rb) / eps_fd; }
                    evals += n;
                } new (&Jlu) Eigen::PartialPivLU<Eigen::Ref<MatrixXd>>(Jstore); bu.clear(); bv.clear(); have_J = true; ++o.jacobians; last_ratio = 0.0; fresh = true;   // factors in place
            } else fresh = false;
            VectorXd dz;
            if (range_newton) {
                // EXPERIMENTAL (not consistent): Newton on the range manifold c_i = Ht_i Y_i with the basis Ht_i frozen.
                // The manifold's tangent also contains the basis change dHt_i[dc] Y_i, which runs through the market
                // maker's projection; omitting it makes this a quasi-Newton step (linear convergence, can diverge).
                // Unknowns dY (R q per trader), equations
                // Ht_i^T M (J [Ht dY]) = -Ht_i^T M r_i, columns from the tensor linearization.
                const int Rq = 2 * M.q * M.N * M.q, nred = M.NT * Rq;
                const Model::LinBase LB = M.linbase(cur);
                auto massB = [&](const MatrixXd& X) { MatrixXd Yv(X.rows(), X.cols()); for (int ch = 0; ch < M.NC; ++ch) Yv.block(ch * M.N, 0, M.N, X.cols()).noalias() = M.Mass * X.block(ch * M.N, 0, M.N, X.cols()); return Yv; };
                auto reduce = [&](const std::vector<MatrixXd>& v) {   // v_i (NC N x q) -> Ht_i^T M v_i flattened (r-major, comp)
                    VectorXd out(nred);
                    for (int i = 0; i < M.NT; ++i) { const MatrixXd t = cur.Htj[i].transpose() * massB(v[i]); for (int k = 0; k < M.q; ++k) out.segment(i * Rq + k * (Rq / M.q), Rq / M.q) = t.col(k); }
                    return out;
                };
                auto expand = [&](const VectorXd& w) {                 // dY -> c-space directions Ht_i dY_i
                    std::vector<MatrixXd> v(M.NT);
                    for (int i = 0; i < M.NT; ++i) { MatrixXd dY(Rq / M.q, M.q); for (int k = 0; k < M.q; ++k) dY.col(k) = w.segment(i * Rq + k * (Rq / M.q), Rq / M.q); v[i] = cur.Htj[i] * dY; }
                    return v;
                };
                const VectorXd rr = reduce(M.unpack(r));
                MatrixXd Jr(nred, nred);
#pragma omp parallel for schedule(dynamic, 8)
                for (int c = 0; c < nred; ++c) {
                    VectorXd e = VectorXd::Zero(nred); e[c] = 1.0;
                    const std::vector<MatrixXd> dir = expand(e);
                    std::vector<MatrixXd> dphi = M.dphi_fast(cur, LB, dir);
                    for (int i = 0; i < M.NT; ++i) dphi[i] -= dir[i];
                    Jr.col(c) = reduce(dphi);
                }
                ++o.jacobians;
                const VectorXd dw = Jr.partialPivLu().solve(-rr);
                dz = M.pack(expand(dw));
                fresh = true;
            } else if (use_krylov && !fresh) {
                // Newton-Krylov: solve J dz = -r by right-preconditioned GMRES, P = Jinv (stale),
                // J v by finite differences through the base-preconditioned residual.
                Model::Diag& base = cur; const VectorXd rb = r;
                const double hfd = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
                auto Jv = [&](const VectorXd& v) { const double vn = v.norm(); if (vn == 0) return VectorXd(VectorXd::Zero(n)); const VectorXd zp = z + (hfd / vn) * v; ++evals; return VectorXd((M.residual(zp, nullptr, &base) - rb) * (vn / hfd)); };
                const int m = gmres_max; const double bn = r.norm();
                std::vector<VectorXd> V; MatrixXd H = MatrixXd::Zero(m + 1, m);
                VectorXd cs = VectorXd::Zero(m), sn = VectorXd::Zero(m), g = VectorXd::Zero(m + 1);
                V.push_back(-r / bn); g[0] = bn;
                int k = 0; bool ok_g = false;
                for (; k < m; ++k) {
                    VectorXd w = Jv(apply_Jinv(V[k])); ++gmres_its;
                    for (int j = 0; j <= k; ++j) { H(j, k) = w.dot(V[j]); w -= H(j, k) * V[j]; }
                    H(k + 1, k) = w.norm(); const double hn = H(k + 1, k);
                    for (int j = 0; j < k; ++j) { const double t = cs[j] * H(j, k) + sn[j] * H(j + 1, k); H(j + 1, k) = -sn[j] * H(j, k) + cs[j] * H(j + 1, k); H(j, k) = t; }
                    const double d = std::hypot(H(k, k), H(k + 1, k)); cs[k] = H(k, k) / d; sn[k] = H(k + 1, k) / d; H(k, k) = d; H(k + 1, k) = 0.0;
                    g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
                    if (std::abs(g[k + 1]) < gmres_tol * bn || hn < 1e-14 * bn) { ++k; ok_g = true; break; }
                    V.push_back(w / hn);
                }
                const VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
                VectorXd wsum = VectorXd::Zero(n); for (int j = 0; j < k; ++j) wsum += y[j] * V[j];
                dz = apply_Jinv(wsum);
                if (verbose) std::fprintf(stderr, "    gmres %d its%s\n", k, ok_g ? "" : " (not converged)");
                if (!ok_g) { use_krylov = false; krylov_stalled = true; last_ratio = 1.0; if (verbose) std::fprintf(stderr, "    krylov stalled: switching to chord\n"); }   // preconditioner has drifted
            } else dz = -apply_Jinv(r);
            double lam = 1.0; bool acc = false;
            for (int ls = 0; ls < 8; ++ls) {
                const VectorXd zt = z + lam * dz; const VectorXd rt = M.residual(zt, nullptr, &cur); ++evals; const double rtn = rt.cwiseAbs().maxCoeff();
                if (std::isfinite(rtn) && rtn < (1.0 - 1e-4 * lam) * rn) {
                    const VectorXd sz = zt - z, yr = rt - r; const VectorXd Jy = (exact_newton || range_newton) ? VectorXd(VectorXd::Zero(sz.size())) : apply_Jinv(yr); const double den = sz.dot(Jy);
                    if (!exact_newton && !range_newton && std::abs(den) > 1e-14 * sz.norm() * Jy.norm()) {
                        // good Broyden update of the inverse: Jinv += (sz - Jy) (sz^T Jinv) / den  ->  u = (sz - Jy)/den, v = Jinv^T sz
                        VectorXd vT = Jlu.transpose().solve(sz); for (size_t k = 0; k < bu.size(); ++k) vT += bv[k] * bu[k].dot(sz);
                        bu.push_back((sz - Jy) / den); bv.push_back(vT);
                    }
                    last_ratio = rtn / rn; z = zt; r = rt; rn = rtn; acc = true;
                    Model::Diag nd; r = M.residual(z, &nd); ++evals; rn = r.cwiseAbs().maxCoeff(); cur = std::move(nd);   // refresh factorizations at the accepted point
                    if (range_newton) {
                        // project c_i onto range(Ht_i(c)):  c_i <- Ht_i G_i^{-1} Ht_i^T M c_i
                        std::vector<MatrixXd> cs = M.unpack(z);
                        for (int i = 0; i < M.NT; ++i) { MatrixXd mc(cs[i].rows(), cs[i].cols()); for (int ch = 0; ch < M.NC; ++ch) mc.block(ch * M.N, 0, M.N, M.q).noalias() = M.Mass * cs[i].block(ch * M.N, 0, M.N, M.q); cs[i] = cur.Htj[i] * cur.Glu[i].solve(cur.Htj[i].transpose() * mc); }
                        const VectorXd zp = M.pack(cs);
                        if ((zp - z).cwiseAbs().maxCoeff() > 1e-15) { z = zp; Model::Diag nd2; r = M.residual(z, &nd2); ++evals; rn = r.cwiseAbs().maxCoeff(); cur = std::move(nd2); }
                    }
                    break;
                }
                lam *= 0.5;
            }
            if (verbose) std::fprintf(stderr, "  newton %d: |r| %.3e step %.3g%s\n", step, rn, lam, acc ? "" : (fresh ? " (rejected, damped residual step)" : " (rejected, refresh)"));
            if (!acc) {
                if (!fresh) { last_ratio = 1.0; continue; }               // stale Jacobian: refresh once
                // fresh Jacobian and still no descent: damped residual step, then keep the chord
                if (++fails > 5) break;
                z += relax * r; { Model::Diag nd; r = M.residual(z, &nd); cur = std::move(nd); } ++evals; rn = r.cwiseAbs().maxCoeff(); last_ratio = 0.0; continue;
            }
            fails = 0;
            if (!std::isfinite(rn) || rn > 1e6 * std::max(rn0, 1.0)) break;
        }
        o.z = z; o.resid = rn; o.steps = step; o.ok = rn <= tol; return o;
    }
};

void print_array(const VectorXd& v) { std::printf("["); for (int i = 0; i < v.size(); ++i) std::printf("%s%.15g", i ? "," : "", v[i]); std::printf("]"); }
void print_mat(const MatrixXd& A) { std::printf("["); for (int r = 0; r < A.rows(); ++r) { std::printf("%s", r ? "," : ""); print_array(A.row(r).transpose()); } std::printf("]"); }

}  // namespace

static bool threads_set = false;

int main(int argc, char* argv[]) {
    mallopt(M_MMAP_THRESHOLD, 1 << 30); mallopt(M_TRIM_THRESHOLD, 1 << 30); mallopt(M_TOP_PAD, 256 << 20);
    if (argc < 7) { std::fprintf(stderr, "usage: %s N L eps rho q \"g11,..;g21,..\" [--sigma-v ...] [--sigma-z ...] [--eps-path ...] [--tol t] [--uniform n] [--threads t] [--verbose]\n", argv[0]); return 1; }
    const int N = std::atoi(argv[1]); const double L = std::atof(argv[2]), eps = std::atof(argv[3]), rho = std::atof(argv[4]); const int q = std::atoi(argv[5]);
    std::vector<VectorXd> gam;
    { std::string s = argv[6]; size_t p = 0; while (p <= s.size()) { size_t e = s.find(';', p); if (e == std::string::npos) e = s.size(); auto v = parse_list(s.substr(p, e - p)); VectorXd g(q); for (int k = 0; k < q; ++k) g[k] = v.size() == 1 ? v[0] : v[k]; gam.push_back(g); p = e + 1; } }
    MatrixXd SV = MatrixXd::Identity(q, q), SZ = MatrixXd::Identity(q, q);
    double tol = 1e-10, split_b = 0.0, map_alpha = 0.0; int uniform = 0, coarse = 0, n1 = 0; bool verbose = false, eval_only = false, adaptive = true, tangent = true, use_jfnk = true, use_analytic = true, check_jac = false, exact_nt = true, range_nt = false; int gm_max = 12; double gm_tol = 1e-3; std::vector<double> path; std::string init_file;
    for (int i = 7; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sigma-v") && i + 1 < argc) { auto v = parse_list(argv[++i]); for (int r = 0; r < q; ++r) for (int c = 0; c < q; ++c) SV(r, c) = v[r * q + c]; }
        else if (!std::strcmp(argv[i], "--sigma-z") && i + 1 < argc) { auto v = parse_list(argv[++i]); for (int r = 0; r < q; ++r) for (int c = 0; c < q; ++c) SZ(r, c) = v[r * q + c]; }
        else if (!std::strcmp(argv[i], "--eps-path") && i + 1 < argc) path = parse_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--uniform") && i + 1 < argc) uniform = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--eval-only")) eval_only = true;
        else if (!std::strcmp(argv[i], "--jfnk")) use_jfnk = true;
        else if (!std::strcmp(argv[i], "--chord")) use_jfnk = false;
        else if (!std::strcmp(argv[i], "--newton")) exact_nt = true;
        else if (!std::strcmp(argv[i], "--hybrid")) exact_nt = false;
        else if (!std::strcmp(argv[i], "--range")) range_nt = true;
        else if (!std::strcmp(argv[i], "--gmres") && i + 2 < argc) { gm_max = std::atoi(argv[++i]); gm_tol = std::atof(argv[++i]); }
        else if (!std::strcmp(argv[i], "--fd-jacobian")) use_analytic = false;
        else if (!std::strcmp(argv[i], "--check-jacobian")) check_jac = true;
        else if (!std::strcmp(argv[i], "--split") && i + 2 < argc) { split_b = std::atof(argv[++i]); n1 = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--map") && i + 1 < argc) map_alpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--adaptive")) adaptive = true;
        else if (!std::strcmp(argv[i], "--no-adaptive")) adaptive = false;
        else if (!std::strcmp(argv[i], "--no-tangent")) tangent = false;
        else if (!std::strcmp(argv[i], "--tangent")) tangent = true;
        else if (!std::strcmp(argv[i], "--coarse") && i + 1 < argc) coarse = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--init") && i + 1 < argc) init_file = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(std::atoi(argv[++i])); threads_set = true;
#else
            ++i;
#endif
        }
    }
#ifdef _OPENMP
    if (!threads_set && !std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_num_procs()));
#endif
    if (path.empty()) { for (double e : {0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002}) if (e > eps) path.push_back(e); path.push_back(eps); }   // correlated-value cases need the finer path
    const auto t0 = std::chrono::steady_clock::now();
    // continuation in eps on a given model from a start vector; returns the solution at the target (or the last good one)
    auto continuation = [&](Model& M, Solver& S, VectorXd z, std::vector<double> path, int pre0, bool& ok_out) {
        VectorXd zprev; double eprev = 0; bool have_prev = false;
        Solver::Out o; o.ok = false;
        int bisections = 0; double factor = 0.5;                   // adaptive: next eps = factor * current
        for (size_t k = 0; k < path.size(); ++k) {
            M.eps = path[k];
            VectorXd zstart = z;
            if (have_prev) {
                // candidates: plain warm start, secant extrapolation, tangent predictor (if a Jacobian exists)
                std::vector<std::pair<double, VectorXd>> cands;
                const VectorXd zx = z + (z - zprev) * ((path[k] - path[k - 1]) / (path[k - 1] - eprev));
                cands.emplace_back(M.residual(z).cwiseAbs().maxCoeff(), z);
                cands.emplace_back(M.residual(zx).cwiseAbs().maxCoeff(), zx); S.evals += 2;
                if (tangent && S.have_J) {
                    const double e1 = path[k - 1], d = 1e-3 * e1;
                    M.eps = e1;       const VectorXd r0 = M.residual(z);
                    M.eps = e1 + d;   const VectorXd r1 = M.residual(z);
                    M.eps = path[k];  S.evals += 2;
                    const VectorXd zt = z - S.Jinv * ((r1 - r0) / d) * (path[k] - e1);
                    cands.emplace_back(M.residual(zt).cwiseAbs().maxCoeff(), zt); ++S.evals;
                }
                size_t best = 0; for (size_t c = 1; c < cands.size(); ++c) if (std::isfinite(cands[c].first) && cands[c].first < cands[best].first) best = c;
                zstart = cands[best].second;
                if (verbose) { std::fprintf(stderr, "  start candidates |r|:"); for (auto& c : cands) std::fprintf(stderr, " %.2e", c.first); std::fprintf(stderr, " -> %zu\n", best); }
            }
            o = S.solve(zstart, tol, (k == 0 && pre0 > 0) ? pre0 : 0, 0.1);
            if (!o.ok) { S.have_J = false; o = S.solve(z, tol, k == 0 ? 0 : 30, 0.1); }
            if (verbose) std::fprintf(stderr, "[N=%d] eps=%g: %s |r| %.2e, %d steps, %d jacobians, %ld evals, %.1f s elapsed | FOC solves: preconditioned ok %ld, fallback %ld, full %ld; time full-evals %.2f s, pre-evals %.2f s\n", M.N, path[k], o.ok ? "ok" : "FAIL", o.resid, o.steps, o.jacobians, S.evals, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_pre_ok, g_pre_fallback, g_full, g_t_full, g_t_pre);
            if (!o.ok) {
                if (k > 0 && bisections < 6) { ++bisections; path.insert(path.begin() + k, 0.5 * (path[k - 1] + path[k])); have_prev = false; S.have_J = false; factor = std::sqrt(factor); --k; continue; }
                break;
            }
            if (k > 0) { zprev = z; eprev = path[k - 1]; have_prev = true; }
            z = o.z;
            if (adaptive && k + 1 < path.size()) {
                // replace the rest of the path by one geometric step, sized by how easy this step was:
                // quick solve -> allow a somewhat larger ratio, slow solve -> smaller; never below 0.25 per step
                if (o.steps <= 5) factor *= 0.8; else if (o.steps >= 12) factor = std::sqrt(factor);
                factor = std::max(0.25, std::min(0.7, factor));
                const double target = path.back();
                double next = path[k] * factor; if (next < target) next = target;
                path.erase(path.begin() + k + 1, path.end()); path.push_back(next); if (next > target) path.push_back(target);
            }
        }
        ok_out = o.ok;
        return o;
    };

    Model M(N, L, path.front(), rho, q, gam, SV, SZ, n1, split_b, map_alpha);
    Solver S(M); S.verbose = verbose; S.jfnk = use_jfnk; S.gmres_max = gm_max; S.gmres_tol = gm_tol; S.analytic = use_analytic; S.exact_newton = exact_nt; S.range_newton = range_nt;
    VectorXd z = VectorXd::Zero(M.dim() * M.NT);
    Solver::Out o; o.ok = false;
    if (!init_file.empty()) {
        std::FILE* f = std::fopen(init_file.c_str(), "r");
        if (!f) { std::fprintf(stderr, "cannot read %s\n", init_file.c_str()); return 1; }
        std::vector<MatrixXd> cs(M.NT, MatrixXd::Zero(M.NC * N, q));
        for (int i = 0; i < M.NT; ++i) for (int ch = 0; ch < M.NC; ++ch) for (int k = 0; k < q; ++k) for (int a = 0; a < N; ++a) { double x; if (std::fscanf(f, "%lf", &x) != 1) { std::fprintf(stderr, "short init file\n"); return 1; } cs[i](ch * N + a, k) = x; }
        std::fclose(f);
        z = M.pack(cs);
        path = {eps};
    }
    bool ok = false;
    if (check_jac) {
        M.eps = path.front();
        VectorXd zc = VectorXd::Random(z.size()) * 0.3;
        Model::Diag base; const VectorXd rb = M.residual(zc, &base);
        const double h = 1e-7 * std::max(1.0, zc.cwiseAbs().maxCoeff());
        double maxrel = 0, maxfast = 0; int worst = -1;
        const Model::LinBase LB = M.linbase(base);
        for (int i = 0; i < static_cast<int>(zc.size()); i += std::max(1, static_cast<int>(zc.size()) / 40)) {
            VectorXd e = VectorXd::Zero(zc.size()); e[i] = 1.0;
            const VectorXd ca = M.pack(M.dphi_fast(base, LB, M.unpack(e))) - e;
            { const VectorXd cr = M.pack(M.dphi(base, M.unpack(e))) - e; maxfast = std::max(maxfast, (ca - cr).norm() / std::max(1e-12, cr.norm())); }
            VectorXd zp = zc; zp[i] += h; const VectorXd cf = (M.residual(zp, nullptr, &base) - rb) / h;
            const double rel = (ca - cf).norm() / std::max(1e-12, cf.norm());
            if (rel > maxrel) { maxrel = rel; worst = i; }
        }
        std::fprintf(stderr, "check-jacobian: fast vs FD %.2e (column %d), fast vs reference linearization %.2e\n", maxrel, worst, maxfast);
        return maxrel < 1e-5 ? 0 : 3;
    }
    if (eval_only) {
        M.eps = eps;
        const VectorXd r = M.residual(z);
        std::fprintf(stderr, "eval-only: |r|_max %.3e  |r|_2 %.3e\n", r.cwiseAbs().maxCoeff(), r.norm());
        o.z = z; o.resid = r.cwiseAbs().maxCoeff(); o.steps = 0; o.jacobians = 0; o.ok = true;
    } else if (coarse > 0 && coarse < N && init_file.empty() && n1 == 0) {
        // coarse-to-fine: full continuation at N0 = coarse, interpolate, then solve the target directly at N
        Model Mc(coarse, L, path.front(), rho, q, gam, SV, SZ);
        Solver Sc(Mc); Sc.verbose = verbose; Sc.jfnk = use_jfnk; Sc.analytic = use_analytic; Sc.exact_newton = exact_nt; Sc.range_newton = range_nt;
        VectorXd zc = VectorXd::Zero(Mc.dim() * Mc.NT);
        bool okc = false;
        const Solver::Out oc = continuation(Mc, Sc, zc, path, 60, okc);
        S.evals += Sc.evals;
        if (!okc) { std::fprintf(stderr, "coarse continuation failed\n"); return 2; }
        const MatrixXd P = Grid(coarse, L).interp(M.K.x);            // (N x coarse)
        const std::vector<MatrixXd> csc = Mc.unpack(oc.z);
        std::vector<MatrixXd> cs(M.NT, MatrixXd::Zero(M.NC * N, q));
        for (int i = 0; i < M.NT; ++i) for (int ch = 0; ch < M.NC; ++ch) cs[i].block(ch * N, 0, N, q) = P * csc[i].block(ch * coarse, 0, coarse, q);
        z = M.pack(cs);
        M.eps = eps;
        o = S.solve(z, tol, 0, 0.1);
        if (!o.ok) { S.have_J = false; o = S.solve(z, tol, 30, 0.1); }
        if (verbose) std::fprintf(stderr, "[N=%d] eps=%g from coarse N=%d: %s |r| %.2e, %d steps, %d jacobians, %ld evals, %.1f s elapsed\n", N, eps, coarse, o.ok ? "ok" : "FAIL", o.resid, o.steps, o.jacobians, S.evals, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (o.ok) z = o.z;
    } else {
        o = continuation(M, S, z, path, init_file.empty() ? 60 : 0, ok);
        z = o.z;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    Model::Diag dg; const std::vector<MatrixXd> cs = M.unpack(z); M.residual(z, &dg);
    std::printf("{\"converged\":%s,\"residual\":%.3e,\"map_alpha\":%.15g,\"split_b\":%.15g,\"N1\":%d,\"N\":%d,\"L\":%.15g,\"eps\":%.15g,\"rho\":%.15g,\"q\":%d,\"NT\":%d,\"NC\":%d,\"seconds\":%.3f,\"evaluations\":%ld,\"gmres_iterations\":%ld,",
                o.ok ? "true" : "false", o.resid, map_alpha, split_b, n1, N, L, M.eps, rho, q, M.NT, M.NC, secs, S.evals, S.gmres_its);
    std::printf("\"lambda\":"); print_mat(dg.lam); std::printf(",");
    {   // unrevealed first value factor (channel 0, component 0) at lags 0, 1, 2, 4
        VectorXd qq(4); qq << 0.0, 1.0, 2.0, 4.0;
        const VectorXd gv = M.K.interp(qq) * dg.g.block(0, 0, N, 1);
        std::printf("\"gapV_0_1_2_4\":"); print_array(gv); std::printf(",");
    }
    std::printf("\"sigma_v\":"); print_mat(SV); std::printf(",\"sigma_z\":"); print_mat(SZ); std::printf(",");
    std::printf("\"lag\":"); print_array(M.K.x); std::printf(",");
    std::printf("\"traders\":[");
    for (int i = 0; i < M.NT; ++i) {
        const MatrixXd P = M.profit(cs, dg, i);
        std::printf("%s{\"gamma\":", i ? "," : ""); print_array(gam[i]);
        double half_lag = L;
        {   // lag by which half of the trader's profit has accrued
            VectorXd tot = VectorXd::Zero(N);
            for (int ch = 0; ch < M.NC; ++ch) for (int nu = 0; nu < q; ++nu) { const VectorXd a = cs[i].block(ch * N, nu, N, 1), b = dg.a_lin[i].block(ch * N, nu, N, 1); tot += a.cwiseProduct(b); }
            const double total = M.wq.dot(tot);
            for (int k = 1; k <= 400; ++k) { const double a = L * k / 400.0; VectorXd u, w; M.K.quad(0.0, a, {}, u, w, M.m); const double cum = (w.transpose() * M.K.interp(u) * tot)(0); if (cum >= 0.5 * total) { half_lag = a; break; } }
        }
        std::printf(",\"flow\":%.15g,\"margin\":%.6g,\"half_profit_lag\":%.6g,\"flow_by_channel_and_stock\":", P.sum(), dg.margin[i], half_lag); print_mat(P);
        // kernel c[ch][comp] = vector over nodes
        std::printf(",\"c\":[");
        for (int ch = 0; ch < M.NC; ++ch) { std::printf("%s[", ch ? "," : ""); for (int k = 0; k < q; ++k) { std::printf("%s", k ? "," : ""); print_array(cs[i].block(ch * N, k, N, 1)); } std::printf("]"); }
        std::printf("]}");
    }
    std::printf("],\"g\":[");
    for (int ch = 0; ch < M.NC; ++ch) { std::printf("%s[", ch ? "," : ""); for (int k = 0; k < q; ++k) { std::printf("%s", k ? "," : ""); print_array(dg.g.block(ch * N, k, N, 1)); } std::printf("]"); }
    std::printf("]");
    if (uniform > 0) {
        VectorXd ul(uniform); for (int i = 0; i < uniform; ++i) ul[i] = L * i / (uniform - 1);
        const MatrixXd Iu = M.K.interp(ul);
        std::printf(",\"uniform\":{\"lag\":"); print_array(ul); std::printf(",\"c\":[");
        for (int i = 0; i < M.NT; ++i) { std::printf("%s[", i ? "," : ""); for (int ch = 0; ch < M.NC; ++ch) { std::printf("%s[", ch ? "," : ""); for (int k = 0; k < q; ++k) { std::printf("%s", k ? "," : ""); print_array(Iu * cs[i].block(ch * N, k, N, 1)); } std::printf("]"); } std::printf("]"); }
        std::printf("],\"g\":[");
        for (int ch = 0; ch < M.NC; ++ch) { std::printf("%s[", ch ? "," : ""); for (int k = 0; k < q; ++k) { std::printf("%s", k ? "," : ""); print_array(Iu * dg.g.block(ch * N, k, N, 1)); } std::printf("]"); }
        std::printf("]}");
    }
    std::printf("}\n");
    return o.ok ? 0 : 2;
}
