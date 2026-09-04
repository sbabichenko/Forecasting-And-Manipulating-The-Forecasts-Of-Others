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
// kb_inventory: kb_spectral plus a market maker with inventory.  The quote is P = E[V | flow] + pq * Q with
// Q = -(flow over the window) the inventory, i.e. the flow->price kernel is beta(a) - pq.  pq < 0 tilts the quote
// down when the market maker is long.  Reports the inventory kernel, Var(Q), the truncation check q(L), the market
// maker's expected profit on informed flow, and the traders' reaction B_0 = -sum_i c_i^flow(0)/lambda_tot.
// Usage: kb_inventory N L eps rho gamma1[,gamma2,...] [--pq x] [--gq g] [--sigma-z s] [--eps-path e1,e2,...] [--progress file]
//        [--tol 1e-10] [--uniform n] [--threads t] [--verbose] [--progress file] [--nk] [--nk-m 60]
// Prints one JSON object.

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#ifdef EIGEN_USE_BLAS
extern "C" void openblas_set_num_threads(int);
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <malloc.h>
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

// ------------------------------------------------------------------ model

struct Model {
    int N, NC, NT, m;
    double L, eps, rho, sV, sZ; double pq = 0.0, gq = 0.0;   // inventory tilt of the quote and inventory aversion
    std::vector<double> gam;
    Grid K;
    VectorXd wq;                                   // integration weights on [0, L]
    // linear maps (kernel nodal values -> operator matrices), one per operator type:
    //   VOL[a]  (N x N):  row a of the Volterra operator:  (A_k c)(a) = k^T VOL[a] c
    std::vector<MatrixXd> VOL;
    MatrixXd Mass;                                 // exact L2 Gram of the nodal basis (N x N)
    std::vector<MatrixXd> QA, QAT;                 // Galerkin maps: Q_A = sum_k dP_k QA[k], Q_A*_rho = sum_k dP_k QAT[k]
    MatrixXd v;                                    // value kernel (N x NC)

    Model(int N_, double L_, double eps_, double rho_, std::vector<double> g, double sZ_, int N1_ = 0, double b_ = 0.0, double alpha_ = 0.0)
        : N(N_), NC(2 + static_cast<int>(g.size())), NT(static_cast<int>(g.size())), m(std::max(N_ - std::max(N1_, 1) + 1, N1_) + (alpha_ > 0 ? N_ : 8)),
          L(L_), eps(eps_), rho(rho_), sV(1.0), sZ(sZ_), gam(std::move(g)), K(N_, L_, N1_, b_, alpha_) {
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
        QA.assign(N, MatrixXd::Zero(N, N));
        if (rho != 0.0) QAT.assign(N, MatrixXd::Zero(N, N));   // with rho = 0 the adjoint map is the transpose
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
                if (rho != 0.0) QAT[k].noalias() += (Pj.transpose() * (wk.array() * disc.array()).matrix()) * phia;
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
        // with a zero kernel (the own-flow row of a single trader) only the identity part survives, and the
        // Volterra tensor need not be streamed at all
        const bool nonzero = scale != 0.0 && (k.array() != 0.0).any();
        if (nonzero)
            for (int a = 0; a < N; ++a)
                for (int ch = 0; ch < NC; ++ch)
                    Ht.block(ch * N + a, 0, 1, N).noalias() = scale * (k.col(ch).transpose() * VOL[a]);
        for (int j = 0; j < N; ++j) Ht(identity_ch * N + j, j) += 1.0;
        H.setZero(N, NC * N);
        if (nonzero) { for (int ch = 0; ch < NC; ++ch) H.block(0, ch * N, N, N).noalias() = Ht.block(ch * N, 0, N, N).transpose() * Mass; }
        else H.block(0, identity_ch * N, N, N) = Mass;                          // Ht is a single identity block
    }
    MatrixXd galerkin_A(const VectorXd& dP) const { MatrixXd Q = MatrixXd::Zero(N, N); for (int k = 0; k < N; ++k) Q += dP[k] * QA[k]; return Q; }
    MatrixXd galerkin_At(const VectorXd& dP) const { MatrixXd Q = MatrixXd::Zero(N, N); for (int k = 0; k < N; ++k) Q += dP[k] * QAT[k]; return Q; }
    MatrixXd volterra(const VectorXd& k) const { MatrixXd A(N, N); for (int a = 0; a < N; ++a) A.row(a) = k.transpose() * VOL[a]; return A; }

    struct Diag {
        std::vector<Eigen::PartialPivLU<MatrixXd>> Klu, Glu; Eigen::PartialPivLU<MatrixXd> Gflu;
        VectorXd beta; MatrixXd p, g, q; double lam;   // q: shock kernel of the inventory Q = -(flow over the window)
        std::vector<VectorXd> dP;
        std::vector<MatrixXd> A, K, G;              // per trader: impact operator (N square, per channel), FOC form and L2 metric (2N square)
        std::vector<VectorXd> a_lin;
    };

    // one joint best-response pass
    static VectorXd pre_solve(const MatrixXd& A, const VectorXd& b, const Eigen::PartialPivLU<MatrixXd>& lu) {
        VectorXd x = lu.solve(b);
        for (int it = 0; it < 6; ++it) { const VectorXd res = b - A * x; if (res.cwiseAbs().maxCoeff() < 1e-14 * std::max(1.0, b.cwiseAbs().maxCoeff())) break; x += lu.solve(res); }
        return x;
    }
    std::vector<MatrixXd> phi(const std::vector<MatrixXd>& cs, Diag* dg = nullptr, const Diag* pre = nullptr) const {
        MatrixXd c_tot = MatrixXd::Zero(N, NC);
        for (const auto& c : cs) c_tot += c;
        MatrixXd Hf, Htf; obs_ops(c_tot, 1.0 / sZ, 1, Hf, Htf);
        const bool use_pre = pre && static_cast<int>(pre->Klu.size()) == NT && !dg;
        VectorXd beta;
        { const MatrixXd G = Hf * Htf; if (use_pre) beta = pre_solve(G, Hf * flat(v), pre->Gflu); else { Eigen::PartialPivLU<MatrixXd> lu(G); beta = lu.solve(Hf * flat(v)); if (dg) dg->Gflu = lu; } }
        const VectorXd beta_tot = beta.array() - pq;                 // flow -> price kernel including the inventory tilt
        const MatrixXd p = unflat(Htf * beta_tot);
        const MatrixXd g = v - p;
        const double lam = beta[0] / sZ;
        const MatrixXd Vb = NT > 1 ? volterra(beta_tot) : MatrixXd();   // only the multi-trader cascade uses it
        std::vector<MatrixXd> out(NT);
        if (dg) { dg->beta = beta; dg->p = p; dg->g = g; dg->lam = lam; dg->q = -unflat(Htf * VectorXd::Ones(N)); dg->dP.resize(NT); dg->A.resize(NT); dg->K.resize(NT); dg->G.resize(NT); dg->a_lin.resize(NT); dg->Klu.resize(NT); dg->Glu.resize(NT); }
        // opponents' policy rows in their own observation coordinates (flow excluding own trades + signal)
        std::vector<VectorXd> yf(NT), ys(NT);
        std::vector<MatrixXd> Hs(NT), Hts(NT), Hfo(NT), Htfo(NT);
        for (int j = 0; j < NT; ++j) {
            obs_ops(g, gam[j], 2 + j, Hs[j], Hts[j]);
            obs_ops(c_tot - cs[j], 1.0 / sZ, 1, Hfo[j], Htfo[j]);
            MatrixXd Hj(2 * N, NC * N); Hj << Hfo[j], Hs[j];
            MatrixXd Htj(NC * N, 2 * N); Htj << Htfo[j], Hts[j];
            VectorXd y;
            { const MatrixXd G = Hj * Htj; if (use_pre) y = pre_solve(G, Hj * flat(cs[j]), pre->Glu[j]); else { Eigen::PartialPivLU<MatrixXd> lu(G); y = lu.solve(Hj * flat(cs[j])); if (dg) dg->Glu[j] = lu; } }
            yf[j] = y.head(N); ys[j] = y.tail(N);
        }
        const MatrixXd I = MatrixXd::Identity(N, N);
        for (int i = 0; i < NT; ++i) {
            // price-impact cascade: dP = beta/sZ + (1/sZ) V_beta x,  x = sum_{j != i} [ yf_j/sZ + (1/sZ) V_{yf_j} x - gam_j V_{ys_j} dP ]
            VectorXd dP;
            if (NT == 1) {
                dP = beta_tot / sZ;
            } else {
                MatrixXd SF = MatrixXd::Zero(N, N), SS = MatrixXd::Zero(N, N);
                VectorXd f0 = VectorXd::Zero(N);
                for (int j = 0; j < NT; ++j) if (j != i) { SF += volterra(yf[j]); SS += gam[j] * volterra(ys[j]); f0 += yf[j] / sZ; }
                MatrixXd M(2 * N, 2 * N);
                M << I, -Vb / sZ, SS, I - SF / sZ;
                VectorXd rhs(2 * N); rhs << beta_tot / sZ, f0;
                dP = M.partialPivLu().solve(rhs).head(N);
            }
            const MatrixXd A1 = volterra(dP);
            const MatrixXd QAf = galerkin_A(dP);
            const MatrixXd Q1 = QAf + (rho == 0.0 ? MatrixXd(QAf.transpose()) : galerkin_At(dP)) + 2.0 * eps * Mass;
            MatrixXd Ht(NC * N, 2 * N); Ht << Htfo[i], Hts[i];
            // a_lin = v - p + A c_own, channel by channel; FOC form and metric by channel blocks
            VectorXd a_lin(NC * N);
            MatrixXd Kmat = MatrixXd::Zero(2 * N, 2 * N), Gm = MatrixXd::Zero(2 * N, 2 * N);
            VectorXd rhs = VectorXd::Zero(2 * N);
            for (int ch = 0; ch < NC; ++ch) {
                const auto Hc = Ht.block(ch * N, 0, N, 2 * N);
                a_lin.segment(ch * N, N) = (v.col(ch) - p.col(ch)) + A1 * cs[i].col(ch);
                if (!use_pre) Kmat.noalias() += Hc.transpose() * Q1 * Hc;
                if (dg) Gm.noalias() += Hc.transpose() * Mass * Hc;
                rhs.noalias() += Hc.transpose() * (Mass * a_lin.segment(ch * N, N));
            }
            VectorXd y;
            if (use_pre) {
                auto applyK = [&](const VectorXd& yy) { VectorXd out = VectorXd::Zero(2 * N); for (int ch = 0; ch < NC; ++ch) { const auto Hc = Ht.block(ch * N, 0, N, 2 * N); out.noalias() += Hc.transpose() * (Q1 * (Hc * yy)); } return out; };
                y = pre->Klu[i].solve(rhs);
                for (int it = 0; it < 6; ++it) { const VectorXd res = rhs - applyK(y); if (res.cwiseAbs().maxCoeff() < 1e-14 * std::max(1.0, rhs.cwiseAbs().maxCoeff())) break; y += pre->Klu[i].solve(res); }
            } else {
                Eigen::PartialPivLU<MatrixXd> lu(Kmat); y = lu.solve(rhs); if (dg) dg->Klu[i] = lu;
            }
            out[i] = unflat(Ht * y);
            if (dg) { dg->dP[i] = dP; dg->A[i] = A1; dg->K[i] = Kmat; dg->G[i] = Gm; dg->a_lin[i] = a_lin; }
        }
        return out;
    }

    VectorXd pack(const std::vector<MatrixXd>& cs) const { VectorXd z(NT * NC * N); for (int i = 0; i < NT; ++i) z.segment(i * NC * N, NC * N) = flat(cs[i]); return z; }
    std::vector<MatrixXd> unpack(const VectorXd& z) const { std::vector<MatrixXd> cs(NT); for (int i = 0; i < NT; ++i) cs[i] = unflat(z.segment(i * NC * N, NC * N)); return cs; }
    VectorXd residual(const VectorXd& z, Diag* dg = nullptr, const Diag* pre = nullptr) const { return pack(phi(unpack(z), dg, pre)) - z; }

    // per-trader profit flow and its channel decomposition: <c, a_lin> - <c, A c> - eps <c, c>
    std::vector<VectorXd> profit_by_channel(const std::vector<MatrixXd>& cs, const Diag& dg) const {
        std::vector<VectorXd> res(NT);
        for (int i = 0; i < NT; ++i) {
            VectorXd by(NC);
            for (int ch = 0; ch < NC; ++ch) {
                const VectorXd c = cs[i].col(ch);
                const VectorXd integrand = c.array() * (dg.a_lin[i].segment(ch * N, N) - dg.A[i] * c - eps * c).array();
                by[ch] = wq.dot(integrand);
            }
            res[i] = by;
        }
        return res;
    }
    // lag by which half of trader i's profit has accrued (cumulative integrand over age)
    double half_profit_lag(const std::vector<MatrixXd>& cs, const Diag& dg, int i) const {
        VectorXd tot = VectorXd::Zero(N);
        for (int ch = 0; ch < NC; ++ch) {
            const VectorXd c = cs[i].col(ch);
            tot += (c.array() * (dg.a_lin[i].segment(ch * N, N) - dg.A[i] * c - eps * c).array()).matrix();
        }
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
    double refresh_ratio = 0.8;   // recompute the Jacobian when a step contracts by less than this
    bool nk = false; int nk_m = 60;   // matrix-free Newton-Krylov instead of a dense finite-difference Jacobian
    int blas_threads = 1;             // BLAS threads for the sequential phases (Krylov vectors, line searches)
    explicit Solver(Model& m) : M(m) {}

    struct Out { VectorXd z; double resid; int steps, jacobians; bool ok; };

    // matrix-free GMRES(m) on J dz = b, with finite-difference Jacobian-vector products that share the base
    // factorizations, so a Krylov vector costs one residual evaluation instead of the n needed for a dense Jacobian.
    VectorXd gmres(const VectorXd& z, const VectorXd& rb, const Model::Diag& base, const VectorXd& b, double rtol, int m, int& used, const MatrixXd* P = nullptr) {
        const int n = static_cast<int>(z.size());
        const double hbase = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
        auto Jv = [&](const VectorXd& v) {
            const double nv = v.norm();
            if (!(nv > 0)) return VectorXd::Zero(n).eval();
            const VectorXd pv = P ? (*P * v).eval() : v;   // right preconditioning: J P v
            const double npv = pv.norm(); if (!(npv > 0)) return VectorXd::Zero(n).eval();
            const double h = hbase / npv;
            ++evals; return ((M.residual(z + h * pv, nullptr, &base) - rb) / h).eval();
        };
        std::vector<VectorXd> V; MatrixXd H = MatrixXd::Zero(m + 1, m);
        VectorXd g = VectorXd::Zero(m + 1), cs = VectorXd::Zero(m), sn = VectorXd::Zero(m);
        const double bn = b.norm(); if (!(bn > 0)) { used = 0; return VectorXd::Zero(n); }
        V.push_back(b / bn); g[0] = bn;
        int k = 0;
        for (; k < m; ++k) {
            VectorXd w = Jv(V[k]);
            if (!w.allFinite()) break;
            for (int j = 0; j <= k; ++j) { H(j, k) = w.dot(V[j]); w -= H(j, k) * V[j]; }
            for (int j = 0; j <= k; ++j) { const double c2 = w.dot(V[j]); H(j, k) += c2; w -= c2 * V[j]; }   // reorthogonalize
            const double hn = w.norm(); H(k + 1, k) = hn;
            for (int j = 0; j < k; ++j) { const double t = cs[j] * H(j, k) + sn[j] * H(j + 1, k); H(j + 1, k) = -sn[j] * H(j, k) + cs[j] * H(j + 1, k); H(j, k) = t; }
            const double rr = std::hypot(H(k, k), H(k + 1, k));
            if (!(rr > 0)) { ++k; break; }
            cs[k] = H(k, k) / rr; sn[k] = H(k + 1, k) / rr; H(k, k) = rr; H(k + 1, k) = 0.0;
            g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
            if (std::abs(g[k + 1]) <= rtol * bn) { ++k; break; }
            if (hn <= 1e-14 * bn) { ++k; break; }
            V.push_back(w / hn);
        }
        used = k;
        if (k == 0) return VectorXd::Zero(n);
        const VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
        VectorXd u = VectorXd::Zero(n);
        for (int j = 0; j < k; ++j) u += y[j] * V[j];
        return P ? (*P * u).eval() : u;
    }
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
            if (nk) {   // Newton-Krylov: GMRES with finite-difference products, preconditioned by a dense Jacobian inverse
                bool fresh = false;
                if (!have_J) {   // the preconditioner is a dense Jacobian built once here and Broyden-updated afterwards
                    MatrixXd Jm(n, n);
                    const double eps_fd = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
                    Model::Diag b0; const VectorXd r0f = M.residual(z, &b0); ++evals;
#ifdef EIGEN_USE_BLAS
                    openblas_set_num_threads(1);
#endif
#pragma omp parallel for schedule(dynamic, 4)
                    for (int i = 0; i < n; ++i) { VectorXd zp = z; zp[i] += eps_fd; Jm.col(i) = (M.residual(zp, nullptr, &b0) - r0f) / eps_fd; }
                    evals += n;
#ifdef EIGEN_USE_BLAS
                    openblas_set_num_threads(blas_threads);
#endif
                    Jinv = Jm.partialPivLu().inverse(); have_J = true; ++o.jacobians; fresh = true;
                }
                Model::Diag base; const VectorXd rb = M.residual(z, &base); ++evals;
                const double rtol = std::min(0.1, std::max(1e-6, 1e-2 * rn / std::max(rn0, rn)));
                int used = 0; const VectorXd dz = gmres(z, rb, base, -rb, rtol, nk_m, used, &Jinv);
                double lam = 1.0; bool acc = false;
                for (int ls = 0; ls < 8; ++ls) {
                    const VectorXd zt = z + lam * dz; const VectorXd rt = M.residual(zt); ++evals;
                    const double rtn = rt.cwiseAbs().maxCoeff();
                    if (std::isfinite(rtn) && rtn < (1.0 - 1e-4 * lam) * rn) {
                        const VectorXd sz = zt - z, yr = rt - r, Jy = Jinv * yr; const double den = sz.dot(Jy);
                        if (std::abs(den) > 1e-14 * sz.norm() * Jy.norm()) Jinv.noalias() += ((sz - Jy) * (sz.transpose() * Jinv)) / den;
                        last_ratio = rtn / rn; z = zt; r = rt; rn = rtn; acc = true; break;
                    }
                    lam *= 0.5;
                }
                if (verbose) std::fprintf(stderr, "  nk %d: |r| %.3e krylov %d step %.3g%s\n", step, rn, used, lam, acc ? "" : " (rejected)");
                if (!acc) { if (!fresh) { have_J = false; continue; } break; }   // stale preconditioner: rebuild once
                if (!std::isfinite(rn) || rn > 1e6 * std::max(rn0, 1.0)) break;
                continue;
            }
            if (!have_J || last_ratio > refresh_ratio) {
                MatrixXd Jm(n, n);
                const double eps_fd = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
                Model::Diag base; const VectorXd rb = M.residual(z, &base); ++evals;
#ifdef EIGEN_USE_BLAS
                openblas_set_num_threads(1);
#endif
#pragma omp parallel for schedule(dynamic, 4)
                for (int i = 0; i < n; ++i) { VectorXd zp = z; zp[i] += eps_fd; Jm.col(i) = (M.residual(zp, nullptr, &base) - rb) / eps_fd; }
                evals += n;
#ifdef EIGEN_USE_BLAS
                openblas_set_num_threads(blas_threads);
#endif
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
            if (!acc) { if (last_ratio <= refresh_ratio) { last_ratio = 1.0; continue; } break; }
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

static bool threads_set = false;

int main(int argc, char* argv[]) {
    // Large Eigen temporaries would otherwise be mmap'ed and freed on every
    // evaluation, which serializes the parallel Jacobian in the kernel.
    mallopt(M_MMAP_THRESHOLD, 1 << 30); mallopt(M_TRIM_THRESHOLD, 1 << 30); mallopt(M_TOP_PAD, 256 << 20);
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s N L eps rho gamma1[,gamma2,...] [--sigma-z s] [--eps-path e1,e2,...] [--tol 1e-10] [--uniform n] [--threads t] [--verbose] [--progress file] [--nk] [--nk-m 60]\n", argv[0]);
        return 1;
    }
    const int N = std::atoi(argv[1]); const double L = std::atof(argv[2]), eps = std::atof(argv[3]), rho = std::atof(argv[4]);
    std::vector<double> gam; { std::string s = argv[5]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); gam.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; } }
    double sZ = 1.0, tol = 1e-10, refresh_ratio = 0.8, split_b = 0.0, map_alpha = 0.0; int uniform = 0, n1 = 0; bool verbose = false; const char* progress = nullptr; bool use_nk = false; int nk_m = 60; double pq_arg = 0.0, gq_arg = 0.0; const char* warm_file = nullptr; const char* dump_file = nullptr;
    std::vector<double> path;
    for (int i = 6; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sigma-z") && i + 1 < argc) sZ = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--pq") && i + 1 < argc) pq_arg = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--warm") && i + 1 < argc) warm_file = argv[++i];
        else if (!std::strcmp(argv[i], "--dump-z") && i + 1 < argc) dump_file = argv[++i];
        else if (!std::strcmp(argv[i], "--gq") && i + 1 < argc) gq_arg = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--uniform") && i + 1 < argc) uniform = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--split") && i + 2 < argc) { split_b = std::atof(argv[++i]); n1 = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--map") && i + 1 < argc) map_alpha = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--refresh-ratio") && i + 1 < argc) refresh_ratio = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--progress") && i + 1 < argc) progress = argv[++i];
        else if (!std::strcmp(argv[i], "--nk")) use_nk = true;
        else if (!std::strcmp(argv[i], "--nk-m") && i + 1 < argc) nk_m = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--eps-path") && i + 1 < argc) { std::string s = argv[++i]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); path.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; } }
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
    // default continuation path: from a well-posed cost down to the target
    if (path.empty()) { for (double e : {0.3, 0.05, 0.01, 0.002}) if (e > eps) path.push_back(e); path.push_back(eps); }

    const auto t0 = std::chrono::steady_clock::now();
    VectorXd q_probe(4); q_probe << 0.0, 1.0, 2.0, 4.0;
    Model M(N, L, path.front(), rho, gam, sZ, n1, split_b, map_alpha); M.pq = pq_arg; M.gq = gq_arg;
    Solver S(M); S.verbose = verbose; S.refresh_ratio = refresh_ratio; S.nk = use_nk; S.nk_m = nk_m;
#ifdef EIGEN_USE_BLAS
    { const char* bt = std::getenv("KB_BLAS_THREADS"); S.blas_threads = bt ? std::atoi(bt) : 8; openblas_set_num_threads(S.blas_threads); }
#endif
    VectorXd z = VectorXd::Zero(M.NT * M.NC * N);
    if (warm_file) {   // warm start from a previous solution on the same grid: skip the cost continuation
        FILE* wf = std::fopen(warm_file, "r");
        if (!wf) { std::fprintf(stderr, "cannot open warm file %s\n", warm_file); return 1; }
        int nz = 0; if (std::fscanf(wf, "%d", &nz) != 1 || nz != z.size()) { std::fprintf(stderr, "warm file size %d != %ld\n", nz, (long)z.size()); std::fclose(wf); return 1; }
        for (int k = 0; k < nz; ++k) if (std::fscanf(wf, "%lf", &z[k]) != 1) { std::fclose(wf); return 1; }
        std::fclose(wf);
        path.assign(1, path.back());
    }
    Solver::Out o;
    VectorXd zprev; double eprev = 0; bool have_prev = false;
    for (size_t k = 0; k < path.size(); ++k) {
        M.eps = path[k];
        VectorXd zstart = z;
        if (have_prev && eprev != path[k == 0 ? 0 : k - 1]) zstart = z + (z - zprev) * ((path[k] - path[k - 1]) / (path[k - 1] - eprev));
        o = S.solve(zstart, tol, k == 0 ? 200 : 0, 0.1);
        if (!o.ok) { S.have_J = false; o = S.solve(z, tol, k == 0 ? 0 : 30, 0.1); }
        if (verbose) std::fprintf(stderr, "eps=%g: %s |r| %.2e, %d steps, %d jacobians, %ld evals total\n", path[k], o.ok ? "ok" : "FAIL", o.resid, o.steps, o.jacobians, S.evals);
        if (progress) {   // snapshot after each continuation step, so a long run can be watched and stopped early
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            Model::Diag pdg; const VectorXd zp = o.ok ? o.z : z; M.residual(zp, &pdg);
            const VectorXd pg = M.K.interp(q_probe) * pdg.g.col(0);
            FILE* f = std::fopen(progress, "w");
            if (f) {
                std::fprintf(f, "{\"step\":%zu,\"steps_total\":%zu,\"eps\":%.15g,\"ok\":%s,\"residual\":%.3e,\"newton_steps\":%d,\"seconds\":%.1f,\"lambda\":%.6g,\"unrevealed_0_1_2_4\":[%.6g,%.6g,%.6g,%.6g]}\n",
                             k + 1, path.size(), path[k], o.ok ? "true" : "false", o.resid, o.steps, el, pdg.lam, pg[0], pg[1], pg[2], pg[3]);
                std::fclose(f);
            }
        }
        if (!o.ok) break;
        if (k > 0) { zprev = z; eprev = path[k - 1]; have_prev = true; }
        z = o.z;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (dump_file) { FILE* df = std::fopen(dump_file, "w"); if (df) { std::fprintf(df, "%ld\n", (long)z.size()); for (int k = 0; k < z.size(); ++k) std::fprintf(df, "%.17g\n", z[k]); std::fclose(df); } }

    Model::Diag dg;
    const std::vector<MatrixXd> cs = M.unpack(z);
    M.residual(z, &dg);
    const auto prof = M.profit_by_channel(cs, dg);
    const auto cert = M.certificate(dg);
    const VectorXd gapV = M.K.interp(q_probe) * dg.g.col(0);

    std::printf("{\"converged\":%s,\"residual\":%.3e,\"N\":%d,\"L\":%.15g,\"eps\":%.15g,\"rho\":%.15g,\"sigma_Z\":%.15g,\"NT\":%d,\"seconds\":%.4f,\"evaluations\":%ld,",
                o.ok ? "true" : "false", o.resid, N, L, M.eps, rho, sZ, M.NT, secs, S.evals);
    print_vec("gammas", Eigen::Map<const VectorXd>(gam.data(), gam.size()));
    std::printf("\"lambda\":%.15g,", dg.lam);
    {   // inventory diagnostics: Var(Q) = sum over shock channels of int q(a)^2 da (unit-variance shocks per unit age), q at the window end,
        // the market maker's expected profit rate on informed flow E[(P - V) D_tot] = sum_ch int (p - v) c_tot, the inventory cost, and B_0
        MatrixXd c_tot = MatrixXd::Zero(N, M.NC); for (const auto& c : cs) c_tot += c;
        double varQ = 0.0, qL = 0.0, profit = 0.0, b0 = 0.0;
        for (int ch = 0; ch < M.NC; ++ch) { varQ += dg.q.col(ch).dot(M.Mass * dg.q.col(ch)); qL += std::abs(dg.q(N - 1, ch)); profit += (dg.p.col(ch) - M.v.col(ch)).dot(M.Mass * c_tot.col(ch)); }
        const double lam_tot = (dg.beta[0] - M.pq) / M.sZ;
        for (int i = 0; i < M.NT; ++i) b0 -= cs[i](0, 1) / lam_tot;   // channel 1 = noise flow: each trader's lag-0 loading on the flow innovation
        std::printf("\"inventory\":{\"pq\":%.15g,\"gq\":%.15g,\"varQ\":%.15g,\"q_at_L\":%.6g,\"lam_tot\":%.15g,\"profit_informed\":%.15g,\"inventory_cost\":%.15g,\"B0\":%.15g,\"mm_loss\":%.15g},",
                    M.pq, M.gq, varQ, qL, lam_tot, profit, M.gq * varQ, b0, M.gq * varQ - profit);
        std::vector<double> ql; VectorXd qm(N); for (int a = 0; a < N; ++a) { double sq = 0.0; for (int ch = 0; ch < M.NC; ++ch) sq += dg.q(a, ch) * dg.q(a, ch); qm[a] = std::sqrt(sq); }
        print_vec("q_norm_by_lag", qm);
    }
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
