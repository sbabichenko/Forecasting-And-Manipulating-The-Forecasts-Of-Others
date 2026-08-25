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
    Panel K;
    VectorXd wq;
    std::vector<MatrixXd> VOL, QA, QAT;            // Volterra row maps and Galerkin maps (N x N each)
    MatrixXd Mass;
    MatrixXd v;                                    // value kernel (NC x q), constant in age: V-factor f row = SVh(:, f)^T

    Model(int N_, double L_, double eps_, double rho_, int q_, std::vector<VectorXd> g, MatrixXd SV_, MatrixXd SZ_)
        : N(N_), q(q_), NT(static_cast<int>(g.size())), NC(q_ * (2 + static_cast<int>(g.size()))), m(N_ + 8),
          L(L_), eps(eps_), rho(rho_), gam(std::move(g)), SV(std::move(SV_)), SZ(std::move(SZ_)), K(N_, 0.0, L_) {
        SVh = Eigen::LLT<MatrixXd>(SV).matrixL();
        SZh = Eigen::LLT<MatrixXd>(SZ).matrixL();
        SZhi = SZh.inverse();
        VectorXd u, w;
        VOL.assign(N, MatrixXd::Zero(N, N));
        for (int j = 0; j < N; ++j) {
            const double aj = K.x[j];
            if (aj > 0) {
                gauss_legendre(m, 0.0, aj, u, w);
                const MatrixXd Pk = K.interp((aj - u.array()).matrix()), Pc = K.interp(u);
                VOL[j].noalias() = Pk.transpose() * w.asDiagonal() * Pc;
            }
        }
        gauss_legendre(m, 0.0, L, u, w);
        { const MatrixXd Pu = K.interp(u); wq = (w.transpose() * Pu).transpose(); Mass.noalias() = Pu.transpose() * w.asDiagonal() * Pu; }
        QA.assign(N, MatrixXd::Zero(N, N)); QAT.assign(N, MatrixXd::Zero(N, N));
        for (int qq = 0; qq < m; ++qq) {
            const double a = u[qq];
            if (a <= 0) continue;
            VectorXd ui, wi; gauss_legendre(m, 0.0, a, ui, wi);
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
        for (int o = 0; o < q; ++o) {
            for (int ch = 0; ch < NC; ++ch)
                if (kw[o][ch].size()) Ht.block(ch * N, o * N, N, N) = volterra(kw[o][ch]);
            Ht.block((id_ch0 + o) * N, o * N, N, N) += MatrixXd::Identity(N, N);
        }
        return Ht;
    }
    MatrixXd Htmass(const MatrixXd& Ht) const {   // H = Ht^T blockdiag(Mass): (q N) x (NC N)
        MatrixXd H(Ht.cols(), Ht.rows());
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

    struct Diag { MatrixXd beta, p, g, lam; std::vector<MatrixXd> K, G, a_lin; std::vector<std::vector<MatrixXd>> dP; std::vector<double> margin; };

    // one joint best-response pass; cs[i] is kmat layout (NC N x q)
    std::vector<MatrixXd> phi(const std::vector<MatrixXd>& cs, Diag* dg = nullptr) const {
        MatrixXd ctot = MatrixXd::Zero(NC * N, q);
        for (const auto& c : cs) ctot += c;
        const std::vector<MatrixXd> Ctot = kvec(ctot);
        const MatrixXd Htf = obs_Ht(flow_kw(Ctot), q);
        const MatrixXd Hf = Htmass(Htf);
        const MatrixXd vmat = [&] { MatrixXd V = MatrixXd::Zero(NC * N, q); for (int a = 0; a < N; ++a) for (int ch = 0; ch < NC; ++ch) V.row(ch * N + a) = v.row(ch); return V; }();
        const MatrixXd beta = (Hf * Htf).partialPivLu().solve(Hf * vmat);     // (q N) x q: obs (o, age) -> price comp
        const MatrixXd p = Htf * beta;
        const MatrixXd gmat = vmat - p;
        const std::vector<MatrixXd> g = kvec(gmat);
        MatrixXd lam(q, q);                                                    // instantaneous impact: lam(nu, k) = sum_o beta(o, age 0; nu) SZhi(o, k)
        for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) { double s = 0; for (int o = 0; o < q; ++o) s += beta(o * N + 0, nu) * SZhi(o, k); lam(nu, k) = s; }
        // each trader's observation operator (residual flow + own signal) and policy rows y_j = G_j^{-1} H_j c^j
        std::vector<MatrixXd> Htj(NT), Hj(NT), yj(NT);
        for (int j = 0; j < NT; ++j) {
            const MatrixXd Htfo = obs_Ht(flow_kw(kvec(ctot - cs[j])), q);
            const MatrixXd Hts = obs_Ht(signal_kw(g, j), (2 + j) * q);
            Htj[j].resize(NC * N, 2 * q * N); Htj[j] << Htfo, Hts;
            Hj[j] = Htmass(Htj[j]);
            yj[j] = (Hj[j] * Htj[j]).partialPivLu().solve(Hj[j] * cs[j]);     // (2 q N) x q
        }
        std::vector<MatrixXd> out(NT);
        if (dg) { dg->beta = beta; dg->p = p; dg->g = gmat; dg->lam = lam; dg->K.resize(NT); dg->G.resize(NT); dg->a_lin.resize(NT); dg->dP.resize(NT); dg->margin.resize(NT); }
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
                const MatrixXd sol = Msys.partialPivLu().solve(rhs);
                for (int nu = 0; nu < q; ++nu) for (int k = 0; k < q; ++k) dP[nu][k] = sol.block(nu * N, k, N, 1);
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
            MatrixXd Kmat = MatrixXd::Zero(R * q, R * q), Gm = MatrixXd::Zero(R * q, R * q);
            VectorXd rhs = VectorXd::Zero(R * q);
            // K[(nu, r),(k, r')] = sum_ch Ht(ch-block, r)^T Mq_{nu k} Ht(ch-block, r')
            for (int ch = 0; ch < NC; ++ch) {
                const MatrixXd Hc = Ht.block(ch * N, 0, N, R);                 // N x R
                for (int nu = 0; nu < q; ++nu) {
                    for (int k = 0; k < q; ++k) Kmat.block(nu * R, k * R, R, R).noalias() += Hc.transpose() * Mq[nu][k] * Hc;
                    Gm.block(nu * R, nu * R, R, R).noalias() += Hc.transpose() * Mass * Hc;
                    rhs.segment(nu * R, R).noalias() += Hc.transpose() * (Mass * a_lin.block(ch * N, nu, N, 1));
                }
            }
            const VectorXd yv = Kmat.partialPivLu().solve(rhs);
            MatrixXd Y(R, q); for (int nu = 0; nu < q; ++nu) Y.col(nu) = yv.segment(nu * R, R);
            out[i] = Ht * Y;
            if (dg) {
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

    VectorXd pack(const std::vector<MatrixXd>& cs) const { VectorXd z(NT * NC * N * q); for (int i = 0; i < NT; ++i) for (int k = 0; k < q; ++k) z.segment((i * q + k) * NC * N, NC * N) = cs[i].col(k); return z; }
    std::vector<MatrixXd> unpack(const VectorXd& z) const { std::vector<MatrixXd> cs(NT, MatrixXd(NC * N, q)); for (int i = 0; i < NT; ++i) for (int k = 0; k < q; ++k) cs[i].col(k) = z.segment((i * q + k) * NC * N, NC * N); return cs; }
    VectorXd residual(const VectorXd& z, Diag* dg = nullptr) const { return pack(phi(unpack(z), dg)) - z; }

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
    Model& M; MatrixXd Jinv; bool have_J = false; long evals = 0; bool verbose = false; double refresh_ratio = 0.8; int max_jacobians = 3;
    explicit Solver(Model& m) : M(m) {}
    struct Out { VectorXd z; double resid; int steps, jacobians; bool ok; };
    Out solve(VectorXd z, double tol, int pre = 0, double relax = 0.1, int maxsteps = 30) {
        Out o; o.jacobians = 0; o.ok = false;
        { double lam = relax, best = std::numeric_limits<double>::infinity(); VectorXd zbest = z;
          for (int k = 0; k < pre; ++k) { const VectorXd r = M.residual(z); ++evals; const double rn = r.cwiseAbs().maxCoeff();
              if (verbose && k % 20 == 0) std::fprintf(stderr, "  pre %d: |r| %.3e relax %.3g\n", k, rn, lam);
              if (!std::isfinite(rn) || rn > 2.0 * best) { z = zbest; lam *= 0.5; if (lam < 1e-3) break; continue; }
              if (rn < best) { best = rn; zbest = z; } if (rn < 1e-3) break; z += lam * r; }
          z = zbest; }
        VectorXd r = M.residual(z); ++evals; double rn = r.cwiseAbs().maxCoeff(); const double rn0 = rn;
        const int n = static_cast<int>(z.size()); double last_ratio = 0.0; int step = 0; int fails = 0; bool fresh = false;
        for (; step < maxsteps && rn > tol; ++step) {
            if (!have_J || (last_ratio > refresh_ratio && o.jacobians < max_jacobians)) {
                MatrixXd Jm(n, n); const double eps_fd = 1e-7 * std::max(1.0, z.cwiseAbs().maxCoeff());
#pragma omp parallel for schedule(dynamic, 4)
                for (int i = 0; i < n; ++i) { VectorXd zp = z; zp[i] += eps_fd; Jm.col(i) = (M.residual(zp) - r) / eps_fd; }
                evals += n; Jinv = Jm.partialPivLu().inverse(); have_J = true; ++o.jacobians; last_ratio = 0.0; fresh = true;
            } else fresh = false;
            const VectorXd dz = -(Jinv * r); double lam = 1.0; bool acc = false;
            for (int ls = 0; ls < 8; ++ls) {
                const VectorXd zt = z + lam * dz; const VectorXd rt = M.residual(zt); ++evals; const double rtn = rt.cwiseAbs().maxCoeff();
                if (std::isfinite(rtn) && rtn < (1.0 - 1e-4 * lam) * rn) {
                    const VectorXd sz = zt - z, yr = rt - r, Jy = Jinv * yr; const double den = sz.dot(Jy);
                    if (std::abs(den) > 1e-14 * sz.norm() * Jy.norm()) Jinv.noalias() += ((sz - Jy) * (sz.transpose() * Jinv)) / den;
                    last_ratio = rtn / rn; z = zt; r = rt; rn = rtn; acc = true; break;
                }
                lam *= 0.5;
            }
            if (verbose) std::fprintf(stderr, "  newton %d: |r| %.3e step %.3g%s\n", step, rn, lam, acc ? "" : (fresh ? " (rejected, damped residual step)" : " (rejected, refresh)"));
            if (!acc) {
                if (!fresh) { last_ratio = 1.0; continue; }               // stale Jacobian: refresh once
                // fresh Jacobian and still no descent: damped residual step, then keep the chord
                if (++fails > 5) break;
                z += relax * r; r = M.residual(z); ++evals; rn = r.cwiseAbs().maxCoeff(); last_ratio = 0.0; continue;
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
    double tol = 1e-10; int uniform = 0, coarse = 0; bool verbose = false, eval_only = false, adaptive = true, tangent = true; std::vector<double> path; std::string init_file;
    for (int i = 7; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sigma-v") && i + 1 < argc) { auto v = parse_list(argv[++i]); for (int r = 0; r < q; ++r) for (int c = 0; c < q; ++c) SV(r, c) = v[r * q + c]; }
        else if (!std::strcmp(argv[i], "--sigma-z") && i + 1 < argc) { auto v = parse_list(argv[++i]); for (int r = 0; r < q; ++r) for (int c = 0; c < q; ++c) SZ(r, c) = v[r * q + c]; }
        else if (!std::strcmp(argv[i], "--eps-path") && i + 1 < argc) path = parse_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--uniform") && i + 1 < argc) uniform = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--eval-only")) eval_only = true;
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
        int bisections = 0; double factor = 0.4;                   // adaptive: next eps = factor * current
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
            if (verbose) std::fprintf(stderr, "[N=%d] eps=%g: %s |r| %.2e, %d steps, %d jacobians, %ld evals, %.1f s elapsed\n", M.N, path[k], o.ok ? "ok" : "FAIL", o.resid, o.steps, o.jacobians, S.evals, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            if (!o.ok) {
                if (k > 0 && bisections < 6) { ++bisections; path.insert(path.begin() + k, 0.5 * (path[k - 1] + path[k])); have_prev = false; S.have_J = false; factor = std::sqrt(factor); --k; continue; }
                break;
            }
            if (k > 0) { zprev = z; eprev = path[k - 1]; have_prev = true; }
            z = o.z;
            if (adaptive && k + 1 < path.size()) {
                // replace the rest of the path by one geometric step, sized by how easy this step was
                if (o.steps <= 8) factor *= 0.6; else if (o.steps >= 20) factor = std::sqrt(factor);
                factor = std::max(0.05, std::min(0.7, factor));
                const double target = path.back();
                double next = path[k] * factor; if (next < target) next = target;
                path.erase(path.begin() + k + 1, path.end()); path.push_back(next); if (next > target) path.push_back(target);
            }
        }
        ok_out = o.ok;
        return o;
    };

    Model M(N, L, path.front(), rho, q, gam, SV, SZ);
    Solver S(M); S.verbose = verbose;
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
    if (eval_only) {
        M.eps = eps;
        const VectorXd r = M.residual(z);
        std::fprintf(stderr, "eval-only: |r|_max %.3e  |r|_2 %.3e\n", r.cwiseAbs().maxCoeff(), r.norm());
        o.z = z; o.resid = r.cwiseAbs().maxCoeff(); o.steps = 0; o.jacobians = 0; o.ok = true;
    } else if (coarse > 0 && coarse < N && init_file.empty()) {
        // coarse-to-fine: full continuation at N0 = coarse, interpolate, then solve the target directly at N
        Model Mc(coarse, L, path.front(), rho, q, gam, SV, SZ);
        Solver Sc(Mc); Sc.verbose = verbose;
        VectorXd zc = VectorXd::Zero(Mc.dim() * Mc.NT);
        bool okc = false;
        const Solver::Out oc = continuation(Mc, Sc, zc, path, 60, okc);
        S.evals += Sc.evals;
        if (!okc) { std::fprintf(stderr, "coarse continuation failed\n"); return 2; }
        const MatrixXd P = Panel(coarse, 0.0, L).interp(M.K.x);            // (N x coarse)
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
    std::printf("{\"converged\":%s,\"residual\":%.3e,\"N\":%d,\"L\":%.15g,\"eps\":%.15g,\"rho\":%.15g,\"q\":%d,\"NT\":%d,\"NC\":%d,\"seconds\":%.3f,\"evaluations\":%ld,",
                o.ok ? "true" : "false", o.resid, N, L, M.eps, rho, q, M.NT, M.NC, secs, S.evals);
    std::printf("\"lambda\":"); print_mat(dg.lam); std::printf(",");
    std::printf("\"sigma_v\":"); print_mat(SV); std::printf(",\"sigma_z\":"); print_mat(SZ); std::printf(",");
    std::printf("\"lag\":"); print_array(M.K.x); std::printf(",");
    std::printf("\"traders\":[");
    for (int i = 0; i < M.NT; ++i) {
        const MatrixXd P = M.profit(cs, dg, i);
        std::printf("%s{\"gamma\":", i ? "," : ""); print_array(gam[i]);
        std::printf(",\"flow\":%.15g,\"margin\":%.6g,\"flow_by_channel_and_stock\":", P.sum(), dg.margin[i]); print_mat(P);
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
