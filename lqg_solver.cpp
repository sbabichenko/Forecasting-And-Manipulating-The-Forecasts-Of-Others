// Decentralized LQG noise-state game solver — F-free implementation
//
// The 36MB Kernel3D F is never materialized during solving. Products
// like sum_u F[j][u][s]^T * v[u] are computed from the rank-1
// decomposition F[j][u][s] = border + sum_k Xtilde[k][u] * A[k][s]^T, A[k][s] = dt gain^2 Xtilde[k][s] (s < k).

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_in_parallel() { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif
#include "lqg_solver.h"
#include <algorithm>
#include <limits>

// Runtime grid parameters
int g_n = 40;
double g_T = 1.0;
double g_dt = 1.0 / 39.0;

void set_grid(int n, double T) {
    g_n = std::max(4, std::min(n, N_MAX));
    g_T = T;
    g_dt = g_T / (g_n - 1);
}

double g_b1 = B1_DEFAULT;
double g_b2 = B2_DEFAULT;
double g_r1 = RHO;
double g_r2 = RHO;
double g_sigma = 1.0;
double g_x0 = 0.0;
double g_terminal_weight = DEFAULT_TERMINAL_STATE_WEIGHT;

static void enforce_initial_observation_noise_boundary(Kernel2D& calD, int obs_idx) {
    if (g_n <= 0) return;
    // Source index 0 is the left endpoint. There is no observation-noise
    // increment born at t=0, so primitive control cannot load on W^i_0.
    for (int j = 0; j < g_n; ++j)
        calD[j][0](obs_idx) = 0.0;
    calD[0][0].setZero();
}

SolverContext SolverContext::capture_current() {
    return SolverContext{g_n, g_T, g_b1, g_b2, g_r1, g_r2,
                         g_sigma, g_x0, g_terminal_weight};
}

void SolverContext::apply() const {
    set_grid(n, T);
    g_b1 = b1;
    g_b2 = b2;
    g_r1 = r1;
    g_r2 = r2;
    g_sigma = sigma;
    g_x0 = x0;
    g_terminal_weight = terminal_weight;
}

ScopedSolverContext::ScopedSolverContext(const SolverContext& next)
    : previous_(SolverContext::capture_current()) {
    next.apply();
}

ScopedSolverContext::~ScopedSolverContext() {
    previous_.apply();
}

// --- state_kernel_from_calD ---

void state_kernel_from_calD(const Kernel2D& calD1, const Kernel2D& calD2,
                            Kernel2D& X) {
    X.setZero();
    Vec3 sigE0 = g_sigma * E0();
    for (int s = 0; s < g_n; ++s) {
        X[s][s] = sigE0;
        Vec3 cumsum = Vec3::Zero();
        for (int t = s + 1; t < g_n; ++t) {
            cumsum += g_dt * (calD1[t - 1][s] + calD2[t - 1][s]);
            X[t][s] = sigE0 + cumsum;
        }
    }
}

// --- compute_filter_kernels (F-free) ---
//
// Computes Xtilde without materializing F.
// gamma_b (border_lt) is identically zero by causality:
// Xtilde[u][s] = 0 for s > u.

// Row j of the filter kernels from row j of X and rows k < j of Xtilde:
// closed-form Kalman gain, no sub-iteration.  Only rows <= j are read, so
// the rows can be produced in a single causal march (forward_environment).
static void filter_row(int j,
    const Kernel2D& X, const Mat3& Pi, int obs_index,
    double obs_gain_val, Kernel2D& Xtilde, double* scale_out = nullptr) {

    const Mat3 I_minus_Pi = Mat3::Identity() - Pi;
    const double g = obs_gain_val;
    const double prec = g * g;
    const double dt2_prec = g_dt * g_dt * prec;

    if (j == 0) {
        Xtilde[0][0] = I_minus_Pi * X[0][0];
        if (scale_out) *scale_out = 1.0;
        return;
    }
    using FlatC = Eigen::Map<const Eigen::VectorXd>;
    using Flat = Eigen::Map<Eigen::VectorXd>;
    const double* xj = X[j][0].data();               // row j of X, contiguous (j+1) Vec3

    // c_k = sum_{u <= k} Xtilde[k][u] . X[j][u]   (flat dot over the contiguous row k)
    std::array<double, N_MAX> c_k;
    for (int k = 0; k < j; ++k) {
        const double* xk = Xtilde[k][0].data();
        c_k[k] = (k > 0 ? FlatC(xk, 3 * k).dot(FlatC(xj, 3 * k)) : 0.0) + Xtilde[k][k].dot(X[j][k]);
    }

    // correction[s] = dt2_prec c_s Xtilde[s][s] + sum_{s < k < j} alpha_k Xtilde[k][s],
    // alpha_k = dt2_prec c_k + g_dt g X[j][k](obs): accumulated by rows (axpy over row k)
    std::array<Vec3, N_MAX> q;
    Flat qf(q[0].data(), 3 * j); qf.setZero();
    for (int k = 1; k < j; ++k) {
        const double alpha = dt2_prec * c_k[k] + g_dt * g * X[j][k](obs_index);
        qf.head(3 * k) += alpha * FlatC(Xtilde[k][0].data(), 3 * k);
    }
    for (int s = 0; s < j; ++s)
        Xtilde[j][s] = I_minus_Pi * X[j][s] - (dt2_prec * c_k[s] * Xtilde[s][s] + q[s]);
    Xtilde[j][j] = I_minus_Pi * X[j][j];

    // closed-form Kalman gain
    const double sigma_minus = g_dt * FlatC(Xtilde[j][0].data(), 3 * j).squaredNorm();
    const double scale = 1.0 / (1.0 + g_dt * prec * sigma_minus);
    Flat(Xtilde[j][0].data(), 3 * (j + 1)) *= scale;
    if (scale_out) *scale_out = scale;

}

// Worksharing versions of the row functions: called by every thread of the
// enclosing parallel team for the same (j, player).  The k-loops (dot products)
// are shared; the row accumulation q[s] = sum_k alpha_k Xtilde[k][s] is done as
// per-thread partials over k-chunks reduced under a critical section.  Same
// arithmetic as filter_row/control_row up to summation order.
struct RowScratch { std::array<double, N_MAX> c; std::array<Vec3, N_MAX> q; };
static void filter_row_ws(int j,
    const Kernel2D& X, const Mat3& Pi, int obs_index,
    double obs_gain_val, Kernel2D& Xtilde, RowScratch& sc) {
    const Mat3 I_minus_Pi = Mat3::Identity() - Pi;
    const double g = obs_gain_val, prec = g * g, dt2_prec = g_dt * g_dt * prec;
    if (j == 0) {
        #pragma omp single
        { Xtilde[0][0] = I_minus_Pi * X[0][0]; }
        return;
    }
    using FlatC = Eigen::Map<const Eigen::VectorXd>;
    using Flat = Eigen::Map<Eigen::VectorXd>;
    const double* xj = X[j][0].data();
    #pragma omp for schedule(static) nowait
    for (int k = 0; k < j; ++k) {
        const double* xk = Xtilde[k][0].data();
        sc.c[k] = (k > 0 ? FlatC(xk, 3 * k).dot(FlatC(xj, 3 * k)) : 0.0) + Xtilde[k][k].dot(X[j][k]);
    }
    #pragma omp single
    Flat(sc.q[0].data(), 3 * j).setZero();
    // implicit barrier: c complete, q zeroed
    {
        std::array<Vec3, N_MAX> ql; Flat qlf(ql[0].data(), 3 * j); qlf.setZero(); bool any = false;
        #pragma omp for schedule(static) nowait
        for (int k = 1; k < j; ++k) {
            const double alpha = dt2_prec * sc.c[k] + g_dt * g * X[j][k](obs_index);
            qlf.head(3 * k) += alpha * FlatC(Xtilde[k][0].data(), 3 * k); any = true;
        }
        if (any) {
            #pragma omp critical(row_reduce)
            Flat(sc.q[0].data(), 3 * j) += qlf;
        }
    }
    #pragma omp barrier
    #pragma omp for schedule(static)
    for (int s = 0; s < j; ++s)
        Xtilde[j][s] = I_minus_Pi * X[j][s] - (dt2_prec * sc.c[s] * Xtilde[s][s] + sc.q[s]);
    #pragma omp single
    {
        Xtilde[j][j] = I_minus_Pi * X[j][j];
        const double sigma_minus = g_dt * FlatC(Xtilde[j][0].data(), 3 * j).squaredNorm();
        const double scale = 1.0 / (1.0 + g_dt * prec * sigma_minus);
        Flat(Xtilde[j][0].data(), 3 * (j + 1)) *= scale;
    }
}
static void control_row_ws(int j,
    const Kernel2D& D, const Kernel2D& Xtilde,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD, RowScratch& sc) {
    Vec3 e_i = Vec3::Zero(); e_i(obs_index) = 1.0;
    const double g = obs_gain_val, DT_g = g_dt * g, dt_prec = g_dt * g * g;
    if (j == 0) {
        #pragma omp single
        calD[0][0].setZero();
        return;
    }
    using FlatC = Eigen::Map<const Eigen::VectorXd>;
    using Flat = Eigen::Map<Eigen::VectorXd>;
    const double* dj = D[j][0].data();
    #pragma omp for schedule(static) nowait
    for (int k = 1; k <= j; ++k) sc.c[k] = FlatC(Xtilde[k][0].data(), 3 * k).dot(FlatC(dj, 3 * k));
    #pragma omp single
    { sc.c[0] = 0.0; Flat(sc.q[0].data(), 3 * j).setZero(); }
    {
        std::array<Vec3, N_MAX> ql; Flat qlf(ql[0].data(), 3 * j); qlf.setZero(); bool any = false;
        #pragma omp for schedule(static) nowait
        for (int k = 1; k <= j; ++k) {
            const double beta = DT_g * D[j][k](obs_index) + g_dt * sc.c[k] * dt_prec;
            const int len = 3 * std::min(k, j);
            qlf.head(len) += beta * FlatC(Xtilde[k][0].data(), len); any = true;
        }
        if (any) {
            #pragma omp critical(row_reduce)
            Flat(sc.q[0].data(), 3 * j) += qlf;
        }
    }
    #pragma omp barrier
    #pragma omp for schedule(static)
    for (int s = 0; s < j; ++s)
        calD[j][s] = Pi * D[j][s] + sc.q[s] + DT_g * sc.c[s] * e_i;
    #pragma omp single
    { calD[j][j] = Pi * D[j][j] + DT_g * sc.c[j] * e_i; calD[j][0](obs_index) = 0.0; }
}

static void compute_filter_kernels(
    const Kernel2D& X, const Mat3& Pi, int obs_index,
    double obs_gain_val, int filter_iters, double relax,
    Kernel2D& Xtilde) {
    (void)filter_iters;
    (void)relax;
    for (int j = 0; j < g_n; ++j)
        filter_row(j, X, Pi, obs_index, obs_gain_val, Xtilde);
}

// --- primitive_control_kernel (F-free) ---
//
// calD[j][s] = Pi*D[j][s] + g_dt * sum_u F[j][u][s]^T * D[j][u]
// decomposed into border_gt, border_lt, and interior sums.

// Row j of the primitive control kernel from D[j] and rows k <= j of the
// filter kernels.
static void control_row(int j,
    const Kernel2D& D, const Kernel2D& Xtilde,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD, double* beta_out = nullptr) {

    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    const double g = obs_gain_val;
    const double DT_g = g_dt * g;

    if (j == 0) {
        // No observation increment has arrived at the initial grid point,
        // so there is no primitive-shock feedback on the first point.
        calD[0][0].setZero();
        if (beta_out) *beta_out = 0.0;
        return;
    }
    using FlatC = Eigen::Map<const Eigen::VectorXd>;
    using Flat = Eigen::Map<Eigen::VectorXd>;
    const double* dj = D[j][0].data();

    // partial_d_k = sum_{u < k} Xtilde[k][u] . D[j][u]
    std::array<double, N_MAX> partial_d_k;
    partial_d_k[0] = 0.0;
    for (int k = 1; k <= j; ++k)
        partial_d_k[k] = FlatC(Xtilde[k][0].data(), 3 * k).dot(FlatC(dj, 3 * k));

    // q[s] = sum_{s < k <= j} ( DT_g D[j][k](obs) Xtilde[k][s] + g_dt partial_d_k[k] A[k][s] ),
    // accumulated by rows; A[k][s] = dt prec Xtilde[k][s] for s < k
    std::array<Vec3, N_MAX> q;
    Flat qf(q[0].data(), 3 * j); qf.setZero();
    const double dt_prec = g_dt * g * g;
    for (int k = 1; k <= j; ++k) {
        const double beta = DT_g * D[j][k](obs_index) + g_dt * partial_d_k[k] * dt_prec;
        const int len = 3 * std::min(k, j);
        qf.head(len) += beta * FlatC(Xtilde[k][0].data(), len);
        if (k == j && beta_out) *beta_out = beta;      // same-time coefficient: calD[j][s] contains beta_j Xtilde[j][s]
    }
    for (int s = 0; s < j; ++s)
        calD[j][s] = Pi * D[j][s] + q[s] + DT_g * partial_d_k[s] * e_i;   // last term: border_lt

    // s = j: only border_lt contributes
    calD[j][j] = Pi * D[j][j] + DT_g * partial_d_k[j] * e_i;

    // No source-time-zero observation-noise shock is available to controls.
    calD[j][0](obs_index) = 0.0;
}

void primitive_control_kernel(
    const Kernel2D& D, const Kernel2D& Xtilde,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD) {
    for (int j = 0; j < g_n; ++j)
        control_row(j, D, Xtilde, obs_gain_val, obs_index, Pi, calD);
}

// --- CE-based filter: incremental rank-1 projection ---
//
// Builds M_j incrementally for j = 0..n-1 via rank-1 updates.
// At each j, extracts Xtilde[j] = (I - M_j) X[j] and calD[j] = M_j D[j].

// Thread-local buffers for compute_ce_filter_and_calD.
// Avoids repeated heap allocation of the V matrix (~600KB at N=160)
// and scratch vectors across Picard iterations.
struct CEFilterBufs {
    int dim_alloc = 0;
    Eigen::MatrixXd V;
    Eigen::VectorXd h, v, Xvec, Dvec, coeff;
    void ensure(int dim, int n_obs) {
        if (dim_alloc == dim) return;
        dim_alloc = dim;
        V.resize(dim, n_obs);
        h.resize(dim); v.resize(dim); Xvec.resize(dim); Dvec.resize(dim);
        coeff.resize(n_obs);
    }
};
static thread_local CEFilterBufs s_ce_bufs;

static void compute_ce_filter_and_calD(
    const Kernel2D& X, const Kernel2D& D,
    int obs_idx, double obs_gain,
    Kernel2D& Xtilde, Kernel2D& calD) {

    int n = g_n;
    int dim = 3 * n;
    int n_obs = n - 1;
    double g = obs_gain;

    // Thin orthonormal basis: M_j = V_j V_j^T where V_j has j columns.
    // Mat-vec M*x = V*(V^T*x) costs O(active*j) instead of O(dim*j).
    //
    // Key: at step j, all vectors (h, Xvec, Dvec) and all V columns are
    // zero below row 3(j+1). So mat-vecs are restricted to the top
    // "active" = 3(j+1) rows, cutting average cost by ~2x.
    s_ce_bufs.ensure(dim, n_obs);
    auto& V = s_ce_bufs.V;
    auto& h = s_ce_bufs.h;
    auto& v = s_ce_bufs.v;
    auto& Xvec = s_ce_bufs.Xvec;
    auto& Dvec = s_ce_bufs.Dvec;
    auto& coeff = s_ce_bufs.coeff;

    V.setZero();
    int rank = 0;
    h.setZero();
    v.setZero();
    Xvec.setZero();
    Dvec.setZero();

    // j = 0: no observations yet, M_0 = 0
    Xtilde[0][0] = X[0][0];
    calD[0][0].setZero();

    for (int j = 1; j < n; ++j) {
        int active = 3 * (j + 1);  // nonzero rows at step j
        auto Vact = V.topRows(active).leftCols(rank);

        // Build h_j (only active entries)
        h.head(active).setZero();
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                h(3*z + k) = g * g_dt * X[j][z](k);
        h(3*j + obs_idx) += 1.0;

        // Innovation: v = (I - V V^T) h, restricted to active rows
        v.head(active) = h.head(active);
        if (rank > 0) {
            auto c = coeff.head(rank);
            c.noalias() = Vact.transpose() * h.head(active);
            v.head(active).noalias() -= Vact * c;
        }
        double vnorm = v.head(active).norm();
        if (vnorm > 1e-15) {
            V.col(rank).head(active) = v.head(active) / vnorm;
            rank++;
        }

        // Refresh Vact after potential rank increase
        auto Vr = V.topRows(active).leftCols(rank);
        auto c = coeff.head(rank);

        // Xtilde[j] = (I - V V^T) X_vec_j
        for (int z = 0; z <= j; ++z)
            Xvec.segment<3>(3*z) = X[j][z];
        if (rank > 0) {
            c.noalias() = Vr.transpose() * Xvec.head(active);
            Xvec.head(active).noalias() -= Vr * c;
        }
        for (int z = 0; z <= j; ++z)
            Xtilde[j][z] = Xvec.segment<3>(3*z);

        // calD[j] = V V^T D_vec_j
        for (int z = 0; z <= j; ++z)
            Dvec.segment<3>(3*z) = D[j][z];
        if (rank > 0) {
            c.noalias() = Vr.transpose() * Dvec.head(active);
            Dvec.head(active).noalias() = Vr * c;
        } else {
            Dvec.head(active).setZero();
        }
        for (int z = 0; z <= j; ++z)
            calD[j][z] = Dvec.segment<3>(3*z);
        calD[j][0](obs_idx) = 0.0;
    }
}

// Per-row form of the exact projection, for the causal march: the orthonormal
// basis V of the observation space grows by one column per row (Gram-Schmidt
// on h_j = g dt X[j][.] + e_obs at z = j), then Xtilde[j] = (I - V V^T) X[j] and
// calD[j] = V V^T D[j].  Same arithmetic as compute_ce_filter_and_calD.
// Controls are predictable (default): the control over step j is the projection of D[j] onto the
// observations up to j-1, as for an Euler-Maruyama discretization of an adapted control.  With the
// step-j observation included (LQG_PREDICTABLE=0, the former game) the control could react to the
// same-step increments, a one-step anticipation that acts as a cheap noise-injection channel at
// small effort cost.  Under predictability the diagonal coordinate D(j,j) is inert in every channel.
static bool predictable_control() { static const bool v = [] { const char* e = std::getenv("LQG_PREDICTABLE"); return !(e && std::atoi(e) == 0); }(); return v; }
static void enforce_predictable(Kernel2D& D1, Kernel2D& D2) { if (!predictable_control()) return; for (int t = 0; t < g_n; ++t) { D1[t][t].setZero(); D2[t][t].setZero(); } }
struct CERowFilter {
    int n = 0, rank = 0, obs = 0; double g = 0.0;
    Eigen::MatrixXd V; Eigen::VectorXd h, v, coeff, Xvec, Dvec, Pd, cH, cD, cX;
    std::vector<int> rank_after;     // rank after row j (basis of observations up to j)
    // Per-row coefficients kept for the exact adjoint: column j of cHs / cDs holds V_{j-1}^T h_j and
    // V_{j-1}^T D[j] (first rank_after[j-1] entries), vns[j] the norm of the new column's residual.
    Eigen::MatrixXd cHs, cDs; std::vector<double> vns;
    void reset(int n_, int obs_, double g_) {
        n = n_; obs = obs_; g = g_; rank = 0;
        const int dim = 3 * n;
        if (cHs.rows() != n) { cHs.resize(n, n); cDs.resize(n, n); }
        vns.assign(n, 0.0);
        if (V.rows() != dim || V.cols() != n) { V.resize(dim, n); h.resize(dim); v.resize(dim); Xvec.resize(dim); Dvec.resize(dim); Pd.resize(dim); coeff.resize(n); cH.resize(n); cD.resize(n); cX.resize(n); }
        V.setZero(); h.setZero(); v.setZero(); Xvec.setZero(); Dvec.setZero();
        rank_after.assign(n, 0);
    }
    void row(int j, const Kernel2D& X, const Kernel2D& D, Kernel2D& Xtilde, Kernel2D& calD) {
        if (j == 0) { Xtilde[0][0] = X[0][0]; calD[0][0].setZero(); return; }
        const int active = 3 * (j + 1);
        auto Vact = V.topRows(active).leftCols(rank);          // basis of the observations up to j-1
        // Two passes over the basis instead of six: (1) V^T [X_j, D_j] in one product; V^T h_j
        // follows from it since h_j = g dt X_j + e_(j,obs); (2) V [c_h, c_D, c_X] in one product.
        for (int z = 0; z <= j; ++z) { Xvec.segment<3>(3 * z) = X[j][z]; Dvec.segment<3>(3 * z) = D[j][z]; }
        h.head(active) = g * g_dt * Xvec.head(active); h(3 * j + obs) += 1.0;
        if (rank > 0) {
            // pass 1: per basis column, two dots (X_j and D_j); c_h from c_X and the (j,obs) row
            if (static_cast<int>(cH.size()) < n) { cH.resize(n); cD.resize(n); cX.resize(n); }
            const double* xp = Xvec.data(); const double* dp = Dvec.data(); const int row_obs = 3 * j + obs;
            using FlatC = Eigen::Map<const Eigen::VectorXd>;
            const FlatC xm(xp, active), dm(dp, active);
            for (int k = 0; k < rank; ++k) {            // vectorized reductions (a plain loop does not vectorize without -ffast-math)
                const double* vk = V.col(k).data(); const FlatC vm(vk, active);
                const double sx = vm.dot(xm), sd = vm.dot(dm);
                cX[k] = sx; cD[k] = sd; cH[k] = g * g_dt * sx + vk[row_obs];
            }
            for (int k = 0; k < rank; ++k) { cHs(k, j) = cH[k]; cDs(k, j) = cD[k]; }
            // pass 2: per basis column, three axpy's into v (= h - V c_h), the control (V c_D) and Xtilde (X - V c_X)
            v.head(active) = h.head(active); Pd.head(active).setZero();
            double* vp = v.data(); double* pd = Pd.data(); double* xw = Xvec.data();
            using Flat = Eigen::Map<Eigen::VectorXd>;
            Flat vv(vp, active), pv(pd, active), xv(xw, active);
            for (int k = 0; k < rank; ++k) {
                const double* vk = V.col(k).data(); const FlatC vm(vk, active); const double a = cH[k], b = cD[k], c = cX[k];
                vv -= a * vm; pv += b * vm; xv -= c * vm;
            }
            if (predictable_control()) { for (int z = 0; z <= j; ++z) calD[j][z] = Pd.segment<3>(3 * z); }
        } else {
            v.head(active) = h.head(active);
            if (predictable_control()) for (int z = 0; z <= j; ++z) calD[j][z].setZero();
        }
        // new basis column from the step-j observation
        const double vnorm = v.head(active).norm(); vns[j] = vnorm;
        bool added = false;
        if (vnorm > 1e-15) { V.col(rank).head(active) = v.head(active) / vnorm; ++rank; added = true; }
        if (added) {   // rank-1 corrections with the new column
            auto vn = V.col(rank - 1).head(active);
            Xvec.head(active) -= vn * vn.dot(Xvec.head(active));
            if (!predictable_control()) {
                // adapted (former) game: control projected with the step-j observation included
                for (int z = 0; z <= j; ++z) Dvec.segment<3>(3 * z) = D[j][z];
                Eigen::VectorXd cD = V.topRows(active).leftCols(rank).transpose() * Dvec.head(active);
                Dvec.head(active) = V.topRows(active).leftCols(rank) * cD;
                for (int z = 0; z <= j; ++z) calD[j][z] = Dvec.segment<3>(3 * z);
            }
        } else if (!predictable_control()) {
            for (int z = 0; z <= j; ++z) calD[j][z] = (rank > 0) ? Vect3(Dvec, z) : Vec3::Zero();
        }
        for (int z = 0; z <= j; ++z) Xtilde[j][z] = Xvec.segment<3>(3 * z);
        calD[j][0](obs) = 0.0;
        rank_after[j] = rank;
    }
    static Vec3 Vect3(const Eigen::VectorXd& w, int z) { return w.segment<3>(3 * z); }

    // Workshared row for j >= 1 (predictable control): called by all `S` threads of this player's
    // subgroup (sub-thread index `t`) inside one parallel region.  Pass 1 splits the basis columns,
    // pass 2 splits the rows (Vec3 units).  One team-wide barrier
    // per row; both players' subgroups must call this with the same j so the barriers match.
    void row_ws(int j, const Kernel2D& X, const Kernel2D& D, Kernel2D& Xtilde, Kernel2D& calD, int t, int S) {
        // One barrier per row: after the column (dot) pass every thread has c_h, c_D, c_X, and the
        // two reductions the row pass needs follow algebraically,
        //   v^T X = h^T X - c_h . c_X,   |v|^2 = |h|^2 - |c_h|^2,
        // the latter without cancellation because v keeps the unit entry on the new observation
        // coordinate (row (j, obs) is zero in every earlier basis column), so |v| >= 1.
        const int active = 3 * (j + 1); const int r0 = rank;
        using FlatC = Eigen::Map<const Eigen::VectorXd>; using Flat = Eigen::Map<Eigen::VectorXd>;
        static thread_local Eigen::VectorXd xl, dl, hl;
        if (xl.size() < 3 * n) { xl.resize(3 * n); dl.resize(3 * n); hl.resize(3 * n); }
        for (int z = 0; z <= j; ++z) { xl.segment<3>(3 * z) = X[j][z]; dl.segment<3>(3 * z) = D[j][z]; }
        hl.head(active) = g * g_dt * xl.head(active); hl(3 * j + obs) += 1.0;
        {   // pass 1: columns [k0, k1)
            const int k0 = static_cast<int>(static_cast<long>(r0) * t / S), k1 = static_cast<int>(static_cast<long>(r0) * (t + 1) / S);
            const FlatC xm(xl.data(), active), dm(dl.data(), active); const int row_obs = 3 * j + obs;
            for (int k = k0; k < k1; ++k) { const double* vk = V.col(k).data(); const FlatC vm(vk, active); const double sx = vm.dot(xm), sd = vm.dot(dm); cX[k] = sx; cD[k] = sd; cH[k] = g * g_dt * sx + vk[row_obs]; cHs(k, j) = cH[k]; cDs(k, j) = sd; }
        }
        #pragma omp barrier
        const FlatC chv(cH.data(), r0), cxv(cX.data(), r0);
        const double hX = FlatC(hl.data(), active).dot(FlatC(xl.data(), active)), hh = FlatC(hl.data(), active).squaredNorm();
        const double vX = hX - chv.dot(cxv), vn2 = hh - chv.squaredNorm();
        const double vnorm = std::sqrt(std::max(vn2, 0.0)); const bool added = vnorm > 1e-15;
        const int z0 = static_cast<int>(static_cast<long>(j + 1) * t / S), z1 = static_cast<int>(static_cast<long>(j + 1) * (t + 1) / S);
        const int i0 = 3 * z0, L = 3 * (z1 - z0);
        {   // row pass on rows [i0, i0+L): v, the control and Xtilde, then the new basis column
            Flat vv(v.data() + i0, L), pv(Pd.data() + i0, L), xv(Xvec.data() + i0, L);
            vv = FlatC(hl.data() + i0, L); pv.setZero(); xv = FlatC(xl.data() + i0, L);
            for (int k = 0; k < r0; ++k) { const FlatC vm(V.col(k).data() + i0, L); vv -= cH[k] * vm; pv += cD[k] * vm; xv -= cX[k] * vm; }
            if (added) { Flat vc(V.col(r0).data() + i0, L); vc = vv / vnorm; xv -= (vX / vnorm) * vc; }
            for (int z = z0; z < z1; ++z) { calD[j][z] = Pd.segment<3>(3 * z); Xtilde[j][z] = Xvec.segment<3>(3 * z); }
        }
        if (t == 0) { calD[j][0](obs) = 0.0; vns[j] = vnorm; if (added) ++rank; rank_after[j] = rank; }
        #pragma omp barrier            // the next X row needs every thread's calD[j]; `single` has no entry barrier
    }
};

static bool ce_filter_enabled() { static const bool v = [] { const char* e = std::getenv("LQG_FILTER"); return !(e && std::strcmp(e, "pi") == 0); }(); return v; }

// Forward environment using discrete CE projection.
// Replaces compute_filter_kernels + primitive_control_kernel with
// the exact discrete conditional expectation at each time step.

static void forward_environment_ce(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    struct ForwardCEBufs {
        Kernel2D calD1, calD2, X, calD1_new, calD2_new, X_new;
        void resize() {
            calD1.resize(); calD2.resize(); X.resize();
            calD1_new.resize(); calD2_new.resize(); X_new.resize();
        }
    };
    static thread_local ForwardCEBufs bufs;
    bufs.resize();

    // Initial calD from Pi*D (same seed as standard forward_environment)
    // Reused thread-local temporaries avoid repeated Kernel2D allocation
    // across Picard iterations.
    Kernel2D& calD1 = bufs.calD1;
    Kernel2D& calD2 = bufs.calD2;
    Kernel2D& X = bufs.X;
    Kernel2D& calD1_new = bufs.calD1_new;
    Kernel2D& calD2_new = bufs.calD2_new;
    Kernel2D& X_new = bufs.X_new;
    for (int j = 0; j < g_n; ++j)
        for (int s = 0; s <= j; ++s) {
            if (j == 0) {
                calD1[j][s].setZero();
                calD2[j][s].setZero();
            } else {
                calD1[j][s] = Pi_1 * D1[j][s];
                calD2[j][s] = Pi_2 * D2[j][s];
            }
        }
    enforce_initial_observation_noise_boundary(calD1, obs_idx_1);
    enforce_initial_observation_noise_boundary(calD2, obs_idx_2);

    state_kernel_from_calD(calD1, calD2, X);

    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_ce_filter_and_calD(X, D1, obs_idx_1, obs_gain1,
                                        env.Xtilde1, calD1_new);
            #pragma omp section
            compute_ce_filter_and_calD(X, D2, obs_idx_2, obs_gain2,
                                        env.Xtilde2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);
        for (int t = 0; t < g_n; ++t)
            for (int s = 0; s <= t; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];
    }

    env.X = X;
    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
}

// --- forward_environment ---

void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env) {

    static const int relaxed_sweeps = [] { const char* e = std::getenv("LQG_FORWARD_RELAXED"); return e ? std::atoi(e) : 0; }();
    if (ce_filter_enabled()) {
        // Exact causal march with the exact projection filter (the production filter):
        // row j of X from the primitive controls at earlier rows; row j of each player's
        // filter and controls from the projection onto its observations up to j.
        Kernel2D& X = env.X; X.resize(); X.setZero();
        env.calD1.resize(); env.calD2.resize(); env.calD1.setZero(); env.calD2.setZero(); env.has_calD = true;
        Kernel2D& calD1 = env.calD1; Kernel2D& calD2 = env.calD2;
        const Vec3 sigE0 = g_sigma * E0();
        // thread-local so that concurrent solves (the figure driver's parallel pre-solve) do not
        // share buffers; inside the two-thread region below the second thread must use the
        // calling thread's objects, hence the shared pointers
        static thread_local CERowFilter F1, F2;
        F1.reset(g_n, obs_idx_1, obs_gain1); F2.reset(g_n, obs_idx_2, obs_gain2);
        CERowFilter* pF1 = &F1; CERowFilter* pF2 = &F2;
        env.basis1 = &F1; env.basis2 = &F2;
        static const bool ws_ce = [] { const char* e = std::getenv("LQG_FORWARD_WS"); return !(e && std::atoi(e) == 0); }();
        const int team = std::min(8, omp_get_max_threads());
        if (ws_ce && predictable_control() && g_n >= 64 && !omp_in_parallel() && team >= 4) {
            // Split-pass march: half the team per player, the basis passes shared within each half.
            const int half = team / 2;
            #pragma omp parallel num_threads(2 * half)
            {
                const int tid = omp_get_thread_num(); const int player = tid < half ? 0 : 1; const int st = tid - player * half;
                for (int j = 0; j < g_n; ++j) {
                    #pragma omp single
                    {
                        X[j][j] = sigE0;
                        for (int s = 0; s < j; ++s) X[j][s] = X[j - 1][s] + g_dt * (calD1[j - 1][s] + calD2[j - 1][s]);
                        if (j == 0) { pF1->row(0, X, D1, env.Xtilde1, calD1); pF2->row(0, X, D2, env.Xtilde2, calD2); }
                    }   // implicit barrier
                    if (j == 0) continue;
                    if (player == 0) pF1->row_ws(j, X, D1, env.Xtilde1, calD1, st, half);
                    else             pF2->row_ws(j, X, D2, env.Xtilde2, calD2, st, half);
                }
            }
            env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
            env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
            return;
        }
        #pragma omp parallel num_threads(2) if (g_n >= 64 && !omp_in_parallel())
        {
            const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
            for (int j = 0; j < g_n; ++j) {
                #pragma omp single
                {
                    X[j][j] = sigE0;
                    for (int s = 0; s < j; ++s)
                        X[j][s] = X[j - 1][s] + g_dt * (calD1[j - 1][s] + calD2[j - 1][s]);
                }   // implicit barrier
                if (tid == 0) pF1->row(j, X, D1, env.Xtilde1, calD1);
                if (tid == nth - 1) pF2->row(j, X, D2, env.Xtilde2, calD2);
                #pragma omp barrier
            }
        }
        (void)Pi_1; (void)Pi_2; (void)inner_iters;
        env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
        env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
        return;
    }
    env.has_calD = false;
    if (relaxed_sweeps <= 0) {
        // Exact causal march.  Row j of the state kernel depends on the primitive
        // controls at earlier times only, row j of the filter kernels on row j of X
        // and earlier filter rows, and row j of the controls on filter rows <= j.
        // Marching j = 0..n-1 therefore produces the exact state--filter--control
        // fixed point in one sweep, with no relaxation.
        Kernel2D& X = env.X; Kernel2D calD1, calD2;
        X.resize(); X.setZero(); calD1.setZero(); calD2.setZero();
        const Vec3 sigE0 = g_sigma * E0();
        static const bool ws_march = [] { const char* e = std::getenv("LQG_FORWARD_PAIR"); return !(e && std::atoi(e)); }();
        static const bool implicit_req = [] { const char* e = std::getenv("LQG_FORWARD_IMPLICIT"); return e && std::atoi(e); }();
        if (ws_march && !implicit_req && g_n >= 400 && !omp_in_parallel() && omp_get_max_threads() > 2) {   // per-row barriers only pay off for large N
            // Worksharing march: all threads work on the same row j of both players.
            static RowScratch sc1, sc2;     // shared scratch per player (one march at a time at top level)
            #pragma omp parallel
            {
                for (int j = 0; j < g_n; ++j) {
                    #pragma omp single
                    {
                        X[j][j] = sigE0;
                        for (int s = 0; s < j; ++s)
                            X[j][s] = X[j - 1][s] + g_dt * (calD1[j - 1][s] + calD2[j - 1][s]);
                    }
                    filter_row_ws(j, X, Pi_1, obs_idx_1, obs_gain1, env.Xtilde1, sc1);
                    filter_row_ws(j, X, Pi_2, obs_idx_2, obs_gain2, env.Xtilde2, sc2);
                    control_row_ws(j, D1, env.Xtilde1, obs_gain1, obs_idx_1, Pi_1, calD1, sc1);
                    control_row_ws(j, D2, env.Xtilde2, obs_gain2, obs_idx_2, Pi_2, calD2, sc2);
                }
            }
            env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
            env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
            return;
        }
        // One two-thread region for the whole march (a region per row would be
        // spawned thousands of times inside the parallel pre-solve); the two
        // players' rows are independent given row j of X.
        static const bool implicit_march = [] { const char* e = std::getenv("LQG_FORWARD_IMPLICIT"); return e && std::atoi(e); }();
        static const int implicit_iters = [] { const char* e = std::getenv("LQG_FORWARD_IMPLICIT"); return e ? std::max(1, std::atoi(e)) : 1; }();   // corrector passes
        double sc_scale[2] = {1.0, 1.0}, sc_beta[2] = {0.0, 0.0};
        #pragma omp parallel num_threads(2) if (g_n >= 64 && !omp_in_parallel())
        {
            const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
            for (int j = 0; j < g_n; ++j) {
                #pragma omp single
                {
                    X[j][j] = sigE0;
                    for (int s = 0; s < j; ++s)
                        X[j][s] = X[j - 1][s] + g_dt * (calD1[j - 1][s] + calD2[j - 1][s]);
                }   // implicit barrier
                if (tid == 0) {
                    filter_row(j, X, Pi_1, obs_idx_1, obs_gain1, env.Xtilde1, &sc_scale[0]);
                    control_row(j, D1, env.Xtilde1, obs_gain1, obs_idx_1, Pi_1, calD1, &sc_beta[0]);
                }
                if (tid == nth - 1) {
                    filter_row(j, X, Pi_2, obs_idx_2, obs_gain2, env.Xtilde2, &sc_scale[1]);
                    control_row(j, D2, env.Xtilde2, obs_gain2, obs_idx_2, Pi_2, calD2, &sc_beta[1]);
                }
                #pragma omp barrier
                for (int corr = 0; corr < implicit_iters && implicit_march && j > 0; ++corr) {
                    // Semi-implicit step: the same-time part of calD[j][s] is beta_j Xtilde[j][s] with
                    // Xtilde[j][s] ~ scale_j (I - Pi) X[j][s]; take that part implicitly (3x3 solve per s,
                    // both players summed), the cross-s corrections explicitly from the predictor, then
                    // re-evaluate the row at the corrected X[j].
                    #pragma omp single
                    {
                        const Mat3 A = g_dt * (sc_beta[0] * sc_scale[0] * (Mat3::Identity() - Pi_1) + sc_beta[1] * sc_scale[1] * (Mat3::Identity() - Pi_2));
                        const Mat3 Minv = (Mat3::Identity() - A).inverse();
                        for (int s = 0; s < j; ++s) {
                            const Vec3 rhs = X[j - 1][s] + g_dt * (calD1[j][s] + calD2[j][s]) - A * X[j][s];
                            X[j][s] = Minv * rhs;
                        }
                    }   // implicit barrier
                    if (tid == 0) {
                        filter_row(j, X, Pi_1, obs_idx_1, obs_gain1, env.Xtilde1, &sc_scale[0]);
                        control_row(j, D1, env.Xtilde1, obs_gain1, obs_idx_1, Pi_1, calD1, &sc_beta[0]);
                    }
                    if (tid == nth - 1) {
                        filter_row(j, X, Pi_2, obs_idx_2, obs_gain2, env.Xtilde2, &sc_scale[1]);
                        control_row(j, D2, env.Xtilde2, obs_gain2, obs_idx_2, Pi_2, calD2, &sc_beta[1]);
                    }
                    #pragma omp barrier
                }
            }
        }
        env.X = X;
        env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
        env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
        return;
    }
    inner_iters = relaxed_sweeps;

    Kernel2D calD1, calD2;
    for (int j = 0; j < g_n; ++j)
        for (int s = 0; s <= j; ++s) {
            if (j == 0) {
                calD1[j][s].setZero();
                calD2[j][s].setZero();
            } else {
                calD1[j][s] = Pi_1 * D1[j][s];
                calD2[j][s] = Pi_2 * D2[j][s];
            }
        }
    enforce_initial_observation_noise_boundary(calD1, obs_idx_1);
    enforce_initial_observation_noise_boundary(calD2, obs_idx_2);

    Kernel2D X;
    state_kernel_from_calD(calD1, calD2, X);

    Kernel2D calD1_new, calD2_new, X_new;
    for (int it = 0; it < inner_iters; ++it) {
        #pragma omp parallel sections
        {
            #pragma omp section
            compute_filter_kernels(X, Pi_1, obs_idx_1, obs_gain1,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.Xtilde1);
            #pragma omp section
            compute_filter_kernels(X, Pi_2, obs_idx_2, obs_gain2,
                                   FILTER_INNER_ITERS, FILTER_RELAX,
                                   env.Xtilde2);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            primitive_control_kernel(D1, env.Xtilde1,
                                     obs_gain1, obs_idx_1, Pi_1, calD1_new);
            #pragma omp section
            primitive_control_kernel(D2, env.Xtilde2,
                                     obs_gain2, obs_idx_2, Pi_2, calD2_new);
        }

        state_kernel_from_calD(calD1_new, calD2_new, X_new);
        for (int t = 0; t < g_n; ++t)
            for (int s = 0; s <= t; ++s)
                X[t][s] = 0.6 * X_new[t][s] + 0.4 * X[t][s];
    }

    env.X = X;
    env.obs_gain1 = obs_gain1; env.obs_gain2 = obs_gain2;
    env.obs_idx1 = obs_idx_1; env.obs_idx2 = obs_idx_2;
}

// --- backward_kernels (ping-pong, no Kernel3D) ---
//
// Two N_MAX×N_MAX Mat3 slices replace the full Kernel3D.

// Dynamically sized HkSlice: g_n × g_n Mat3 entries.
struct HkSlice {
    int n;
    std::vector<Mat3> data;
    HkSlice() : n(0) {}
    void resize(int nn) {
        if (n != nn) { n = nn; data.resize(nn * nn); }
    }
    Mat3& operator()(int z, int r) { return data[z * n + r]; }
    const Mat3& operator()(int z, int r) const { return data[z * n + r]; }
};

// Thread-local HkSlice buffers — reused across backward_kernels calls within a
// thread to avoid heap allocation churn.
// thread_local is needed because OMP parallel sections call backward_kernels
// concurrently from different threads.
static thread_local std::unique_ptr<HkSlice> s_hk_buf0, s_hk_buf1;

static void ensure_hk_buffers() {
    if (!s_hk_buf0) s_hk_buf0 = std::make_unique<HkSlice>();
    if (!s_hk_buf1) s_hk_buf1 = std::make_unique<HkSlice>();
    s_hk_buf0->resize(g_n);
    s_hk_buf1->resize(g_n);
}

void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double obs_gain_k, int obs_idx_k,
                      double terminal_state_weight,
    Kernel2D& Hx) {
    Hx.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = terminal_state_weight * X[g_n - 1][s];

    Vec3 e_obs = Vec3::Zero();
    e_obs(obs_idx_k) = 1.0;

    // Hk-free formulation.  With Hk[n-1] = 0 and
    //   Hk[t](z, r) = Hk[t+1](z, r) + dt (Dk[t+1][r] Hx[t+1][z]^T - X[t+1][z] W[t+1][r]^T),
    // Hk[t] is the sum over t' > t of rank-one (in z, r) terms, so the two
    // contractions the recursion needs are formed from 3x3 moments instead of
    // an N^2 array of 3x3 blocks:
    //   sum_z Hk[tp](z, r)^T Xtilde[tp][z]
    //     = sum_{t' > tp} dt ( M_{t'} Dk[t'][r] - p_{t'} W[t'][r] ),
    //   M_{t'} = sum_{z <= tp} Hx[t'][z] Xtilde[tp][z]^T,   p_{t'} = sum_{z <= tp} X[t'][z] . Xtilde[tp][z],
    //   Hk[tp](tp, r)^T e = sum_{t' > tp} dt ( Hx[t'][tp] (Dk[t'][r] . e) - W[t'][r] (X[t'][tp] . e) ).
    // Cost O(N^3) flops with O(N^2) vectors of memory, no block array.
    const int n = g_n;
    static thread_local std::vector<Vec3> Whist;      // W[t][r], r <= t-1, row-major t * n + r
    if (static_cast<int>(Whist.size()) < n * (n + 1) / 2) Whist.resize(n * (n + 1) / 2);   // triangular
    static thread_local std::vector<Mat3> Mbuf;       // M_{t'} for the current level
    static thread_local std::vector<double> pbuf;     // p_{t'}
    if (static_cast<int>(Mbuf.size()) < n) { Mbuf.resize(n); pbuf.resize(n); }

    std::array<Vec3, N_MAX> W;
    std::array<Vec3, N_MAX> acc, diag;
    using FlatC = Eigen::Map<const Eigen::VectorXd>;
    using Flat = Eigen::Map<Eigen::VectorXd>;
    for (int j = n - 2; j >= 0; --j) {
        const int tp = j + 1;  // t+1
        const double Pk = prec_k[tp];
        const int len = tp + 1;                       // z = 0..tp

        // moments of the later levels against Xtilde[tp][.]:
        //   M_{t2} = sum_{z <= tp} Hx[t2][z] Xtilde[tp][z]^T  (3x3),  p_{t2} = sum_z X[t2][z] . Xtilde[tp][z]
        for (int t2 = tp + 1; t2 < n; ++t2) {
            Mat3 M = Mat3::Zero();
            const Vec3* hxrow = &Hx[t2][0]; const Vec3* xtrow = &Xtildek[tp][0];
            for (int z = 0; z < len; ++z) M.noalias() += hxrow[z] * xtrow[z].transpose();
            Mbuf[t2] = M;
            pbuf[t2] = FlatC(X[t2][0].data(), 3 * len).dot(FlatC(Xtildek[tp][0].data(), 3 * len));
        }
        // acc[r] = sum_{t2 > tp} ( M_{t2} Dk[t2][r] - p_{t2} W[t2][r] ),  r <= j
        // diag[r] = sum_{t2 > tp} ( Hx[t2][tp] Dk[t2][r](obs) - W[t2][r] X[t2][tp](obs) )
        Flat(acc[0].data(), 3 * len).setZero(); Flat(diag[0].data(), 3 * len).setZero();
        for (int t2 = tp + 1; t2 < n; ++t2) {
            const Mat3& M = Mbuf[t2]; const double pv = pbuf[t2];
            const Vec3& hxd = Hx[t2][tp]; const double xo = X[t2][tp](obs_idx_k);
            const Vec3* drow = &Dk[t2][0]; const Vec3* wrow = &Whist[t2 * (t2 + 1) / 2];
            for (int r = 0; r <= j; ++r) {
                acc[r].noalias() += M * drow[r]; acc[r] -= pv * wrow[r];
                diag[r] += hxd * drow[r](obs_idx_k) - wrow[r] * xo;
            }
        }
        // W[r] = Gamma^T E Hk[tp][tp][r] + g_dt * Pk * sum_z Hk[tp][z][r]^T * Xtilde[tp][z]
        // Diagonal innovation-birth term: Gamma^T E H^k_t(t, .), from delta w^k_t(t) = E^{k,T} Gamma^k(t) e^k_t.
        for (int r = 0; r <= j; ++r) {
            W[r] = obs_gain_k * g_dt * diag[r] + g_dt * Pk * g_dt * acc[r];
            Whist[tp * (tp + 1) / 2 + r] = W[r];
        }
        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + W[r]);
    }
}

// Both players' backward adjoints in lockstep inside one parallel region:
// the two recursions run over the same levels, so each level is two
// worksharing loops (moments over (player, t2); then accumulation, W and the
// new Hx row over (player, r)) with an implicit barrier each.  Same arithmetic
// as backward_kernels; results identical.
struct BackwardPlayer {
    const Kernel2D* Xt; const Kernel2D* Dk; const std::array<double, N_MAX>* prec; double gain; int obs; Kernel2D* Hx;
    std::vector<Vec3> W;            // W[t][r], row-major t * n + r
    std::vector<Mat3> M; std::vector<double> pv;
};
void backward_kernels_pair(const Kernel2D& X, BackwardPlayer& P1, BackwardPlayer& P2, double terminal_state_weight) {
    const int n = g_n;
    BackwardPlayer* P[2] = {&P1, &P2};
    for (int q = 0; q < 2; ++q) {
        BackwardPlayer& p = *P[q];
        p.Hx->setZero();
        for (int s = 0; s < n; ++s) (*p.Hx)[n - 1][s] = terminal_state_weight * X[n - 1][s];
        if (static_cast<int>(p.W.size()) < n * (n + 1) / 2) p.W.resize(n * (n + 1) / 2);   // W[t][r], r <= t: triangular
        if (static_cast<int>(p.M.size()) < n) { p.M.resize(n); p.pv.resize(n); }
    }
    using FlatC = Eigen::Map<const Eigen::VectorXd>;
    #pragma omp parallel if (!omp_in_parallel())
    {
        for (int j = n - 2; j >= 0; --j) {
            const int tp = j + 1, len = tp + 1, nt2 = n - tp - 1;
            // moments of the later levels against Xtilde[tp][.], both players
            #pragma omp for schedule(static)
            for (int q = 0; q < 2 * nt2; ++q) {
                BackwardPlayer& p = *P[q / nt2]; const int t2 = tp + 1 + q % nt2;
                const Kernel2D& Hx = *p.Hx; const Kernel2D& Xt = *p.Xt;
                Mat3 Mm = Mat3::Zero();
                const Vec3* hxrow = &Hx[t2][0]; const Vec3* xtrow = &Xt[tp][0];
                for (int z = 0; z < len; ++z) Mm.noalias() += hxrow[z] * xtrow[z].transpose();
                p.M[t2] = Mm;
                p.pv[t2] = FlatC(X[t2][0].data(), 3 * len).dot(FlatC(Xt[tp][0].data(), 3 * len));
            }
            // accumulation, W[tp][r] and the new row Hx[j][r], both players
            #pragma omp for schedule(static)
            for (int q = 0; q < 2 * (j + 1); ++q) {
                BackwardPlayer& p = *P[q / (j + 1)]; const int r = q % (j + 1);
                Kernel2D& Hx = *p.Hx; const Kernel2D& Dk = *p.Dk;
                Vec3 acc = Vec3::Zero(), diag = Vec3::Zero();
                for (int t2 = tp + 1; t2 < n; ++t2) {
                    const Vec3& d = Dk[t2][r]; const Vec3& w = p.W[t2 * (t2 + 1) / 2 + r];
                    acc.noalias() += p.M[t2] * d; acc -= p.pv[t2] * w;
                    diag += Hx[t2][tp] * d(p.obs) - w * X[t2][tp](p.obs);
                }
                const double Pk = (*p.prec)[tp];
                const Vec3 Wr = p.gain * g_dt * diag + g_dt * Pk * g_dt * acc;
                p.W[tp * (tp + 1) / 2 + r] = Wr;
                Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + Wr);
            }
        }
    }
}

// Exact discrete adjoint: gradient of player a's fluctuation cost
//   J_a = sum_j dt ( dt sum_{s<=j} |X[j][s]|^2 + r_a dt sum_{s<=j} |calD_a[j][s]|^2 ) + tw dt sum_s |X[n-1][s]|^2
// with respect to D_a (the opponent's kernel and the mean frozen), by reverse-mode differentiation of
// the march: X-row update, h_j = g dt X[j] + e_(j,obs), Gram-Schmidt column u_j, calD_i[j] = V_{j-1} V_{j-1}^T D_i[j].
// Returned as Hx_a := grad / (2 dt^2) - r_a D_a, so that the fixed-point map G = -Hx / r reads
// D - grad / (2 r dt^2) and vanishes exactly at a best response.
static bool exact_adjoint_enabled() { static const bool v = [] { const char* e = std::getenv("LQG_EXACT_ADJOINT"); return !(e && std::atoi(e) == 0); }(); return v; }
static void exact_adjoint_player_ref(int a, const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2, double r_a, int obs1, int obs2, double g1, double g2, Kernel2D& Hx_out) {
    const int n = g_n, dim = 3 * n;
    const CERowFilter* B[2] = {static_cast<const CERowFilter*>(env.basis1), static_cast<const CERowFilter*>(env.basis2)};
    const Kernel2D* D[2] = {&D1, &D2}; const Kernel2D* C[2] = {&env.calD1, &env.calD2}; const int obs[2] = {obs1, obs2}; const double gg[2] = {g1, g2};
    const Kernel2D& X = env.X;
    static thread_local Kernel2D Xbar, cbar0, cbar1, grad; Xbar.resize(); cbar0.resize(); cbar1.resize(); grad.resize();
    Kernel2D* cbar[2] = {&cbar0, &cbar1};
    static thread_local Eigen::MatrixXd Vbar[2];
    for (int q = 0; q < 2; ++q) { if (Vbar[q].rows() != dim || Vbar[q].cols() != n) Vbar[q].resize(dim, n); Vbar[q].setZero(); }
    // cost adjoints
    for (int j = 0; j < n; ++j) for (int s = 0; s <= j; ++s) { Xbar[j][s] = 2.0 * g_dt * g_dt * X[j][s]; (*cbar[a])[j][s] = 2.0 * r_a * g_dt * g_dt * (*C[a])[j][s]; (*cbar[1 - a])[j][s].setZero(); }
    if (g_terminal_weight != 0.0) for (int s = 0; s < n; ++s) Xbar[n - 1][s] += 2.0 * g_terminal_weight * g_dt * X[n - 1][s];   // tw dt sum_s |X[n-1][s]|^2
    grad.setZero();
    Eigen::VectorXd cb, dv, Vtc, Vtd, h, c, u, ubar, vbar, hbar, Vtvb;
    for (int j = n - 1; j >= 1; --j) {
        const int active = 3 * (j + 1);
        // (b) calD_i[j] = P_{j-1} D_i[j]
        for (int q = 0; q < 2; ++q) {
            const CERowFilter& b = *B[q]; const int r0 = b.rank_after[j - 1];
            cb.resize(active); dv.resize(active);
            for (int z = 0; z <= j; ++z) { cb.segment<3>(3 * z) = (*cbar[q])[j][z]; dv.segment<3>(3 * z) = (*D[q])[j][z]; }
            cb(obs[q]) = 0.0;                                   // calD[j][0](obs) is forced to zero
            if (r0 > 0) {
                auto Vact = b.V.topRows(active).leftCols(r0);
                Vtc.noalias() = Vact.transpose() * cb; Vtd.noalias() = Vact.transpose() * dv;
                if (q == a) { Eigen::VectorXd pdb = Vact * Vtc; for (int z = 0; z <= j; ++z) grad[j][z] += pdb.segment<3>(3 * z); }
                Vbar[q].topRows(active).leftCols(r0).noalias() += cb * Vtd.transpose() + dv * Vtc.transpose();
            }
        }
        // (c) Gram-Schmidt adjoint of the column added at row j (both players' filters depend on X)
        for (int q = 0; q < 2; ++q) {
            const CERowFilter& b = *B[q]; const int r0 = b.rank_after[j - 1]; if (b.rank_after[j] <= r0) continue;
            auto Vact = b.V.topRows(active).leftCols(r0);
            h.resize(active); for (int z = 0; z <= j; ++z) h.segment<3>(3 * z) = gg[q] * g_dt * X[j][z]; h(3 * j + obs[q]) += 1.0;
            c.noalias() = Vact.transpose() * h; const double vn = std::sqrt(std::max(h.squaredNorm() - c.squaredNorm(), 1e-300));
            u = b.V.col(r0).head(active); ubar = Vbar[q].col(r0).head(active);
            vbar = (ubar - u * u.dot(ubar)) / vn;
            hbar = vbar;
            if (r0 > 0) { Vtvb.noalias() = Vact.transpose() * vbar; hbar.noalias() -= Vact * Vtvb; Vbar[q].topRows(active).leftCols(r0).noalias() += -vbar * c.transpose() - h * Vtvb.transpose(); }
            for (int z = 0; z <= j; ++z) Xbar[j][z] += gg[q] * g_dt * hbar.segment<3>(3 * z);
        }
        // (d) X[j][s] = X[j-1][s] + dt (calD_1[j-1][s] + calD_2[j-1][s])
        for (int s = 0; s < j; ++s) { Xbar[j - 1][s] += Xbar[j][s]; (*cbar[0])[j - 1][s] += g_dt * Xbar[j][s]; (*cbar[1])[j - 1][s] += g_dt * Xbar[j][s]; }
    }
    const double sc = 1.0 / (2.0 * g_dt * g_dt);
    Hx_out.resize();
    for (int j = 0; j < n; ++j) for (int s = 0; s <= j; ++s) Hx_out[j][s] = sc * grad[j][s] - r_a * (*D[a])[j][s];
}
static int adjoint_batch() { static const int v = [] { const char* e = std::getenv("LQG_ADJ_BATCH"); return e ? std::max(1, std::atoi(e)) : 8; }(); return v; }
// Four-role sweep: role (a, q) carries player a's adjoint through player q's basis (its own
// Vbar, W, K).  The roles of one player meet once per row to merge their Gram-Schmidt
// contributions to Xbar[j] before the state recursion; the team of up to 4 threads runs the
// roles in lockstep (one barrier per row), any smaller team takes several roles per thread.
struct AdjRole {
    Eigen::MatrixXd Vbar, W, K, Rm, M2;      // Rm = [cb, ubar] (dim x 2), M2 = V^T Rm (n x 2): shared by the role's sub-threads
    Eigen::VectorXd hpart[2];
};
static void exact_adjoint_pair(const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2, double r1, double r2, int obs1, int obs2, double g1, double g2, Kernel2D& Hx1, Kernel2D& Hx2) {
    if (std::getenv("LQG_ADJ_REF")) {
        #pragma omp parallel sections num_threads(2) if (!omp_in_parallel())
        {
            #pragma omp section
            exact_adjoint_player_ref(0, env, D1, D2, r1, obs1, obs2, g1, g2, Hx1);
            #pragma omp section
            exact_adjoint_player_ref(1, env, D1, D2, r2, obs1, obs2, g1, g2, Hx2);
        }
        return;
    }
    const int n = g_n, dim = 3 * n, m = adjoint_batch();
    const CERowFilter* B[2] = {static_cast<const CERowFilter*>(env.basis1), static_cast<const CERowFilter*>(env.basis2)};
    const Kernel2D* D[2] = {&D1, &D2}; const Kernel2D* C[2] = {&env.calD1, &env.calD2}; const int obs[2] = {obs1, obs2}; const double gg[2] = {g1, g2}, rr[2] = {r1, r2};
    const Kernel2D& X = env.X;
    // Scratch is per calling thread: the figure pipeline solves several equilibria concurrently,
    // each solve then runs its sweep on one thread (omp_in_parallel) with its own buffers.
    static thread_local Kernel2D Xbar_tl[2], cbar_tl[2][2], grad_tl[2];
    static thread_local AdjRole R_tl[4];
    // pointers to the calling thread's buffers: inside the parallel region a thread_local name would
    // denote each worker's own instance
    Kernel2D* Xbar = Xbar_tl; Kernel2D (*cbar)[2] = cbar_tl; Kernel2D* grad = grad_tl; AdjRole* R = R_tl;
    for (int a = 0; a < 2; ++a) {
        Xbar[a].resize(); grad[a].resize(); grad[a].setZero(); cbar[a][0].resize(); cbar[a][1].resize();
        for (int j = 0; j < n; ++j) for (int s = 0; s <= j; ++s) { Xbar[a][j][s] = 2.0 * g_dt * g_dt * X[j][s]; cbar[a][a][j][s] = 2.0 * rr[a] * g_dt * g_dt * (*C[a])[j][s]; cbar[a][1 - a][j][s].setZero(); }
        if (g_terminal_weight != 0.0) for (int s = 0; s < n; ++s) Xbar[a][n - 1][s] += 2.0 * g_terminal_weight * g_dt * X[n - 1][s];   // tw dt sum_s |X[n-1][s]|^2
    }
    for (int k = 0; k < 4; ++k) {
        AdjRole& r = R[k];
        if (r.Vbar.rows() != dim || r.Vbar.cols() != n) { r.Vbar.resize(dim, n); r.Rm.resize(dim, 2); r.M2.resize(n, 2); }
        r.Vbar.setZero();
        if (r.W.rows() != dim || r.W.cols() != 4 * m) { r.W.resize(dim, 4 * m); r.K.resize(n, 4 * m); }
        for (int e = 0; e < 2; ++e) if (r.hpart[e].size() != dim) r.hpart[e].resize(dim);
    }
    static const bool prof = std::getenv("LQG_ADJ_PROF") != nullptr;
    // Team: 4 roles x S sub-threads when at least 4 threads are available (S = T / 4; the products of a
    // role are split among its sub-threads: V^T Rm by basis column, V M3 and the row-local updates by
    // row block, the flush by column of Vbar; five team barriers per row).  With fewer threads each
    // thread takes every fourth role alone.
    // Sub-threads pay five barriers per row, which only amortizes once the row work is large enough
    // (N=640 gains 30%, N=320 nothing measurable on an 8-core desktop); LQG_ADJ_SUB fixes the number per role.
    static const int sub_env = [] { const char* e = std::getenv("LQG_ADJ_SUB"); return e ? std::max(1, std::atoi(e)) : 0; }();
    const int avail = omp_in_parallel() ? 1 : omp_get_max_threads();
    const int S = sub_env > 0 ? std::min(sub_env, std::max(1, avail / 4)) : std::max(1, std::min(avail / 4, n / 320)), nthr = avail >= 4 ? 4 * S : avail;
    #pragma omp parallel num_threads(nthr)
    {
        const int T = omp_get_num_threads(), t = omp_get_thread_num();
        const int Sx = T >= 4 ? T / 4 : 1;           // sub-threads per role (actual team)
        const int sub = T >= 4 ? t % Sx : 0;
        const bool have_role = T >= 4 ? t < 4 * Sx : true;
        auto roles_of = [&](std::vector<int>& v) { v.clear(); if (!have_role) return; if (T >= 4) v.push_back(t / Sx); else for (int r = t; r < 4; r += T) v.push_back(r); };
        std::vector<int> my; roles_of(my);
        int nbl[4] = {0, 0, 0, 0}, bAl[4] = {0, 0, 0, 0}, bRl[4] = {0, 0, 0, 0};   // batch state, private (identical on every thread)
        Eigen::MatrixXd M3, P; Eigen::VectorXd vbar, hbar;
        double tacc[6] = {0, 0, 0, 0, 0, 0}; double t0 = prof ? omp_get_wtime() : 0.0;
        auto tick = [&](int k) { if (prof) { const double t1 = omp_get_wtime(); tacc[k] += t1 - t0; t0 = t1; } };
        using FlatC = Eigen::Map<const Eigen::VectorXd>; using Flat = Eigen::Map<Eigen::VectorXd>;
        for (int j = n - 1; j >= 1; --j) {
            const int active = 3 * (j + 1), par = j & 1;
            // block of Vec3 rows [z0, z1) of this sub-thread
            const int z0 = static_cast<int>(static_cast<long>(j + 1) * sub / Sx), z1 = static_cast<int>(static_cast<long>(j + 1) * (sub + 1) / Sx);
            const int i0 = 3 * z0, L = 3 * (z1 - z0);
            // phase A: Rm = [cb, ubar] rows, W_j columns 0,1 rows, K_j column 0 (sub 0)
            for (int role : my) {
                const int a = role >> 1, q = role & 1; AdjRole& r = R[role];
                const CERowFilter& b = *B[q]; const int r0 = b.rank_after[j - 1];
                const bool gs = r0 > 0 && b.rank_after[j] > r0;
                Flat(r.hpart[par].data() + i0, L).setZero();
                if (r0 == 0) continue;
                if (nbl[role] == 0) { bAl[role] = active; bRl[role] = r0; }
                const int c0 = 4 * nbl[role];
                auto Wj = r.W.middleCols(c0, 4); auto Kj = r.K.middleCols(c0, 4);
                Wj.middleRows(i0, L).setZero();
                if (sub == 0) { Kj.setZero(); Kj.topRows(r0).col(0) = b.cDs.col(j).head(r0); if (gs) Kj.topRows(r0).col(2) = b.cHs.col(j).head(r0); }
                for (int z = z0; z < z1; ++z) { r.Rm.block<3, 1>(3 * z, 0) = cbar[a][q][j][z]; Wj.block<3, 1>(3 * z, 1) = (*D[q])[j][z]; }
                if (z0 == 0) r.Rm(obs[q], 0) = 0.0;                 // calD[j][0](obs) is forced to zero
                Wj.col(0).segment(i0, L) = r.Rm.col(0).segment(i0, L);
                if (gs) {
                    Flat ub(r.Rm.col(1).data() + i0, L);
                    ub = r.Vbar.col(r0).segment(i0, L);
                    if (c0 > 0) ub.noalias() += r.W.middleRows(i0, L).leftCols(c0) * r.K.row(r0).head(c0).transpose();
                }
            }
            tick(4);
            #pragma omp barrier
            // phase B: M2 = V^T Rm over the basis columns [k0, k1) of this sub-thread
            for (int role : my) {
                const int q = role & 1; AdjRole& r = R[role];
                const CERowFilter& b = *B[q]; const int r0 = b.rank_after[j - 1]; if (r0 == 0) continue;
                const bool gs = b.rank_after[j] > r0;
                const int k0 = static_cast<int>(static_cast<long>(r0) * sub / Sx), k1 = static_cast<int>(static_cast<long>(r0) * (sub + 1) / Sx);
                const FlatC x0(r.Rm.col(0).data(), active), x1(r.Rm.col(1).data(), active);
                for (int k = k0; k < k1; ++k) {
                    const FlatC vm(b.V.col(k).data(), active);
                    r.M2(k, 0) = vm.dot(x0); if (gs) r.M2(k, 1) = vm.dot(x1);
                }
            }
            tick(0);
            #pragma omp barrier
            // phase C: K_j columns 1,3; P = V M3 on the row block; gradient, vbar, hbar, W_j columns 2,3, hpart rows
            for (int role : my) {
                const int a = role >> 1, q = role & 1; AdjRole& r = R[role];
                const CERowFilter& b = *B[q]; const int r0 = b.rank_after[j - 1]; if (r0 == 0) continue;
                const bool gs = b.rank_after[j] > r0;
                const int c0 = 4 * nbl[role]; auto Wj = r.W.middleCols(c0, 4); auto Kj = r.K.middleCols(c0, 4);
                const double vn = gs ? b.vns[j] : 1.0;
                if (sub == 0) { Kj.topRows(r0).col(1) = r.M2.col(0).head(r0); if (gs) Kj.topRows(r0).col(3) = r.M2.col(1).head(r0) / vn; }   // V^T vbar = V^T ubar / |v|
                const int nm = (q == a ? 1 : 0) + (gs ? 1 : 0);
                if (nm == 0) continue;
                M3.resize(r0, nm); int k = 0;
                if (q == a) M3.col(k++) = r.M2.col(0).head(r0);
                if (gs) M3.col(k++) = r.M2.col(1).head(r0) / vn;
                P.resize(L, nm); P.setZero();
                {
                    Flat p0(P.col(0).data(), L), p1(P.col(nm - 1).data(), L);
                    for (int kc = 0; kc < r0; ++kc) { const FlatC vm(b.V.col(kc).data() + i0, L); p0 += M3(kc, 0) * vm; if (nm == 2) p1 += M3(kc, 1) * vm; }
                }
                int kk = 0;
                if (q == a) { for (int z = z0; z < z1; ++z) grad[a][j][z] += P.block<3, 1>(3 * (z - z0), kk); ++kk; }
                if (gs) {
                    const FlatC u(b.V.col(r0).data(), active), ub(r.Rm.col(1).data(), active);
                    const double uu = u.dot(ub);
                    vbar = (ub.segment(i0, L) - uu * u.segment(i0, L)) / vn;
                    hbar = vbar - P.col(kk);
                    Wj.col(2).segment(i0, L) = -vbar;
                    for (int z = z0; z < z1; ++z) Wj.block<3, 1>(3 * z, 3) = -gg[q] * g_dt * X[j][z];
                    if (z1 == j + 1) Wj(3 * j + obs[q], 3) -= 1.0;
                    Flat(r.hpart[par].data() + i0, L) = gg[q] * g_dt * hbar;
                }
            }
            tick(1);
            #pragma omp barrier
            // flush (every m rows): Vbar += W K^T on this sub-thread's block of columns
            for (int role : my) {
                const int q = role & 1; AdjRole& r = R[role];
                const int r0 = B[q]->rank_after[j - 1]; if (r0 == 0) continue;
                if (++nbl[role] == m) {
                    nbl[role] = 0;
                    const int bA = bAl[role], bR = bRl[role];
                    const int cA = static_cast<int>(static_cast<long>(bR) * sub / Sx), cB = static_cast<int>(static_cast<long>(bR) * (sub + 1) / Sx);
                    if (cB > cA) r.Vbar.topRows(bA).middleCols(cA, cB - cA).noalias() += r.W.topRows(bA).leftCols(4 * m) * r.K.middleRows(cA, cB - cA).leftCols(4 * m).transpose();
                }
            }
            tick(3);
            #pragma omp barrier
            // state recursion X[j][s] = X[j-1][s] + dt (calD_1[j-1][s] + calD_2[j-1][s]): role (a, q) updates
            // its own cbar[a][q][j-1] on its row block; role (a, 0) also carries Xbar[a][j-1]
            for (int role : my) {
                const int a = role >> 1, q = role & 1;
                const Eigen::VectorXd& h0 = R[2 * a].hpart[par]; const Eigen::VectorXd& h1 = R[2 * a + 1].hpart[par];
                const int s0 = z0, s1 = std::min(z1, j);
                for (int s = s0; s < s1; ++s) {
                    const Vec3 tot = Xbar[a][j][s] + h0.segment<3>(3 * s) + h1.segment<3>(3 * s);
                    cbar[a][q][j - 1][s] += g_dt * tot;
                    if (q == 0) Xbar[a][j - 1][s] += tot;
                }
            }
            tick(2);
            #pragma omp barrier      // the row blocks of the recursion and of the next row's phase A differ
        }
        if (prof) {
            #pragma omp critical
            std::fprintf(stderr, "adj thread %d: VtR %.2f  VM3 %.2f  recur %.2f  flush %.2f  phaseA %.2f ms\n", t, 1e3 * tacc[0], 1e3 * tacc[1], 1e3 * tacc[2], 1e3 * tacc[3], 1e3 * tacc[4]);
        }
    }
    const double sc = 1.0 / (2.0 * g_dt * g_dt);
    Hx1.resize(); Hx2.resize();
    for (int j = 0; j < n; ++j) for (int s = 0; s <= j; ++s) { Hx1[j][s] = sc * grad[0][j][s] - r1 * D1[j][s]; Hx2[j][s] = sc * grad[1][j][s] - r2 * D2[j][s]; }
}

// Version that also outputs the kernel information wedge V^i(t,r).
// V[t][r] = g_dt * Pk * sum_z Hk[t][z][r]^T * Xtilde[t][z]
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double obs_gain_k, int obs_idx_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel2D& Vkernel) {
    Hx.setZero();
    Vkernel.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = terminal_state_weight * X[g_n - 1][s];

    Vec3 e_obs = Vec3::Zero();
    e_obs(obs_idx_k) = 1.0;

    ensure_hk_buffers();
    for (int z = 0; z < g_n; ++z)
        for (int r = 0; r < g_n; ++r)
            (*s_hk_buf0)(z, r).setZero();
    HkSlice* cur = s_hk_buf0.get();
    HkSlice* nxt = s_hk_buf1.get();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += (*cur)(z, r).transpose() * Xtildek[tp][z];
            // Diagonal innovation-birth term:
            // Gamma^T E H^k_t(t, .), from delta w^k_t(t)
            // = E^{k,T} Gamma^k(t) e^k_t.
            W[r] = obs_gain_k * ((*cur)(tp, r).transpose() * e_obs)
                 + g_dt * Pk * acc;
        }

        // Store corrected W[r] = V^i(t_{j+1}, r) into Vkernel
        for (int r = 0; r <= j; ++r)
            Vkernel[tp][r] = W[r];

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                (*nxt)(z, r) = (*cur)(z, r)
                    + g_dt * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());

        std::swap(cur, nxt);
    }
}

// Legacy version that also fills Hk (for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk,
                      const std::array<double, N_MAX>& prec_k,
                      double obs_gain_k, int obs_idx_k,
                      double terminal_state_weight,
    Kernel2D& Hx, Kernel3D& Hk) {
    Hx.setZero();
    for (int s = 0; s < g_n; ++s)
        Hx[g_n - 1][s] = terminal_state_weight * X[g_n - 1][s];

    Vec3 e_obs = Vec3::Zero();
    e_obs(obs_idx_k) = 1.0;

    for (int z = 0; z < g_n; ++z)
        for (int r = 0; r < g_n; ++r)
            Hk[g_n - 1][z][r].setZero();

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        std::array<Vec3, N_MAX> W;
        for (int r = 0; r <= j; ++r) {
            Vec3 acc = Vec3::Zero();
            for (int z = 0; z <= tp; ++z)
                acc += Hk[tp][z][r].transpose() * Xtildek[tp][z];
            // Diagonal innovation-birth term:
            // Gamma^T E H^k_t(t, .), from delta w^k_t(t)
            // = E^{k,T} Gamma^k(t) e^k_t.
            W[r] = obs_gain_k * (Hk[tp][tp][r].transpose() * e_obs)
                 + g_dt * Pk * acc;
        }

        for (int r = 0; r <= j; ++r)
            Hx[j][r] = Hx[tp][r] + g_dt * (X[tp][r] + W[r]);

        for (int z = 0; z <= j; ++z)
            for (int r = 0; r <= j; ++r)
                Hk[j][z][r] = Hk[tp][z][r]
                    + g_dt * (Dk[tp][r] * Hx[tp][z].transpose()
                          - X[tp][z] * W[r].transpose());
    }
}

// --- backward_bar_adjoints ---

BackwardBarResult backward_bar_adjoints(
    const Kernel2D& X, const Kernel2D& Xtildek, const Kernel2D& Dk,
    const std::array<double, N_MAX>& barX, double b,
    const std::array<double, N_MAX>& prec_k,
    double obs_gain_k, int obs_idx_k,
    double terminal_weight) {

    BackwardBarResult res;
    res.barHx.fill(0.0);
    res.barHk.setZero();
    res.barHx[g_n - 1] = terminal_weight * (barX[g_n - 1] - b);

    for (int j = g_n - 2; j >= 0; --j) {
        int tp = j + 1;
        double Pk = prec_k[tp];

        double I_val = 0.0;
        for (int z = 0; z <= tp; ++z)
            I_val += Xtildek[tp][z].dot(res.barHk[tp][z]);
        // Diagonal innovation-birth term:
        // Gamma^T E H^k_t(t, .), from delta w^k_t(t)
        // = E^{k,T} Gamma^k(t) e^k_t.
        I_val = obs_gain_k * res.barHk[tp][tp](obs_idx_k)
              + g_dt * Pk * I_val;

        res.barHx[j] = res.barHx[tp] + g_dt * ((barX[tp] - b) + I_val);

        for (int s = 0; s <= j; ++s)
            res.barHk[j][s] = res.barHk[tp][s]
                + g_dt * (Dk[tp][s] * res.barHx[tp] - X[tp][s] * I_val);
    }
    return res;
}

double mean_information_wedge_at(
    const Kernel2D& Xtildek, const Kernel2D& barHk,
    const std::array<double, N_MAX>& prec_k,
    double obs_gain_k, int obs_idx_k, int t_idx) {
    // No observation increment has arrived at the initial grid point. The
    // backward solver uses right-endpoint wedges on intervals, so the first
    // displayed/exported point should remain zero.
    if (t_idx <= 0)
        return 0.0;

    double acc = 0.0;
    for (int z = 0; z <= t_idx; ++z)
        acc += Xtildek[t_idx][z].dot(barHk[t_idx][z]);
    // Diagonal innovation-birth term:
    // Gamma^T E H^k_t(t, .), from delta w^k_t(t)
    // = E^{k,T} Gamma^k(t) e^k_t.
    return obs_gain_k * barHk[t_idx][t_idx](obs_idx_k)
         + g_dt * prec_k[t_idx] * acc;
}

// --- solve_bar_equilibrium ---

BarSolution solve_bar_equilibrium(
    const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2,
    double prec1, double prec2,
    int max_iters, double relax, double tol, bool /*verbose*/) {

    // The mean-field system is affine in d = (barD1, barD2): barX is a cumulative
    // sum of d, the bar adjoints are affine in barX, and the best response is
    // -(1/r) barHx.  Write the map as F(d) = A d + c and solve (I - A) d = c by
    // GMRES with F applied matrix-free (one pair of backward bar adjoints per
    // application).  The relaxed iteration d <- d + relax (F(d) - d) used before is
    // only stable when the spectral radius of I - relax (I - A) is below one,
    // which fails at some N; it is kept as a fallback if GMRES does not reach tol.
    const int n = g_n, dim = 2 * n;
    auto prec1_arr = make_constant_prec(prec1);
    auto prec2_arr = make_constant_prec(prec2);

    auto apply_F = [&](const Eigen::VectorXd& d, Eigen::VectorXd& out, std::array<double, N_MAX>* barX_out) {
        std::array<double, N_MAX> barX;
        barX[0] = g_x0;
        for (int j = 0; j < n - 1; ++j) barX[j + 1] = barX[j] + g_dt * (d[j] + d[n + j]);
        BackwardBarResult bba1, bba2;
        #pragma omp parallel sections num_threads(2) if (!omp_in_parallel())
        {
            #pragma omp section
            bba1 = backward_bar_adjoints(env.X, env.Xtilde2, D2, barX, g_b1, prec2_arr, env.obs_gain2, env.obs_idx2, g_terminal_weight);
            #pragma omp section
            bba2 = backward_bar_adjoints(env.X, env.Xtilde1, D1, barX, g_b2, prec1_arr, env.obs_gain1, env.obs_idx1, g_terminal_weight);
        }
        out.resize(dim);
        for (int j = 0; j < n; ++j) { out[j] = -(1.0 / g_r1) * bba1.barHx[j]; out[n + j] = -(1.0 / g_r2) * bba2.barHx[j]; }
        if (barX_out) *barX_out = barX;
    };

    // affine part c = F(0); linear operator L d = F(d) - c; solve (I - L) d = c
    Eigen::VectorXd c; apply_F(Eigen::VectorXd::Zero(dim), c, nullptr);
    auto apply_M = [&](const Eigen::VectorXd& d, Eigen::VectorXd& out) { apply_F(d, out, nullptr); out = d - (out - c); };

    Eigen::VectorXd d = Eigen::VectorXd::Zero(dim);
    double last_err = 1e30;
    const double bnorm = std::max(1e-300, c.norm());
    static const int dense_max = [] { const char* e = std::getenv("LQG_BAR_DENSE_MAX"); return e ? std::atoi(e) : 1600; }();
    if (dim <= dense_max) {
        // Direct solve: build I - L column by column (2N applications of the affine map, each a
        // pair of backward bar adjoints) and factor.  Robust when the effort cost is small and
        // the map is far from a contraction, where restarted GMRES can stagnate.
        Eigen::MatrixXd Mtx(dim, dim);
        #pragma omp parallel for schedule(dynamic, 4) if (!omp_in_parallel())
        for (int k = 0; k < dim; ++k) { Eigen::VectorXd e = Eigen::VectorXd::Unit(dim, k), col; apply_M(e, col); Mtx.col(k) = col; }   // columns are independent
        d = Mtx.partialPivLu().solve(c);
    } else
    // GMRES(m) with modified Gram-Schmidt for large N
    for (int restart = 0, m = std::min(dim, 60); restart < 20; ++restart) {
        Eigen::VectorXd r; apply_M(d, r); r = c - r;
        double beta = r.norm();
        if (beta / bnorm < 1e-14) break;
        Eigen::MatrixXd V(dim, m + 1), H = Eigen::MatrixXd::Zero(m + 1, m);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(m + 1); g[0] = beta;
        std::vector<double> cs(m), sn(m);
        V.col(0) = r / beta;
        int k = 0;
        for (; k < m; ++k) {
            Eigen::VectorXd w; apply_M(V.col(k), w);
            for (int i = 0; i <= k; ++i) { H(i, k) = V.col(i).dot(w); w -= H(i, k) * V.col(i); }
            H(k + 1, k) = w.norm();
            if (H(k + 1, k) > 1e-300) V.col(k + 1) = w / H(k + 1, k);
            for (int i = 0; i < k; ++i) { const double t = cs[i] * H(i, k) + sn[i] * H(i + 1, k); H(i + 1, k) = -sn[i] * H(i, k) + cs[i] * H(i + 1, k); H(i, k) = t; }
            const double den = std::hypot(H(k, k), H(k + 1, k));
            cs[k] = H(k, k) / den; sn[k] = H(k + 1, k) / den;
            H(k, k) = den; H(k + 1, k) = 0.0;
            g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
            if (std::fabs(g[k + 1]) / bnorm < 1e-13 || H(k + 1, k) <= 1e-300) { ++k; break; }
        }
        const Eigen::VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
        d += V.leftCols(k) * y;
        if (std::fabs(g[k]) / bnorm < 1e-13) break;
    }

    // residual in the fixed-point sense, as before: |F(d) - d| / max(1, |F(d)|), per player
    Eigen::VectorXd Fd; std::array<double, N_MAX> barX; apply_F(d, Fd, &barX);
    {
        const double e1 = (Fd.head(n) - d.head(n)).norm() / std::max(1.0, Fd.head(n).norm());
        const double e2 = (Fd.tail(n) - d.tail(n)).norm() / std::max(1.0, Fd.tail(n).norm());
        last_err = std::max(e1, e2);
    }
    if (!(last_err < std::max(tol, 1e-9))) {
        // fallback: the relaxed iteration from the GMRES iterate
        for (int it = 1; it <= max_iters; ++it) {
            apply_F(d, Fd, &barX);
            const double e1 = (Fd.head(n) - d.head(n)).norm() / std::max(1.0, Fd.head(n).norm());
            const double e2 = (Fd.tail(n) - d.tail(n)).norm() / std::max(1.0, Fd.tail(n).norm());
            last_err = std::max(e1, e2);
            if (!std::isfinite(last_err)) break;
            d += relax * (Fd - d);
            if (last_err < tol) break;
        }
        apply_F(d, Fd, &barX);
    }

    BarSolution sol;
    sol.barX[0] = g_x0;
    for (int j = 0; j < n - 1; ++j) sol.barX[j + 1] = sol.barX[j] + g_dt * (d[j] + d[n + j]);
    for (int j = 0; j < n; ++j) { sol.barD1[j] = d[j]; sol.barD2[j] = d[n + j]; }
    sol.bar_residual = last_err;
    return sol;
}

// --- solve_equilibrium: relaxed Picard iteration ---

static double best_response_residual(const Kernel2D& D1, const Kernel2D& D2,
                                     const Kernel2D& Hx1, const Kernel2D& Hx2,
                                     int tri_size,
                                     double neg_inv_r1, double neg_inv_r2,
                                     Eigen::VectorXd* f_out = nullptr) {
    // residual |G - D| / max(1, |G|); optionally also writes f = G - D (stacked D1, D2) for Anderson
    double norm_f = 0.0, norm_g = 0.0;
    if (f_out) f_out->resize(6 * tri_size);
    double* fo = f_out ? f_out->data() : nullptr;
    #pragma omp parallel for reduction(+:norm_f,norm_g) schedule(static) if (tri_size > 4000 && !omp_in_parallel())
    for (int i = 0; i < tri_size; ++i) {
        const Vec3 g1 = neg_inv_r1 * Hx1.data[i];
        const Vec3 g2 = neg_inv_r2 * Hx2.data[i];
        const Vec3 f1 = g1 - D1.data[i];
        const Vec3 f2 = g2 - D2.data[i];
        norm_f += f1.squaredNorm() + f2.squaredNorm();
        norm_g += g1.squaredNorm() + g2.squaredNorm();
        if (fo) { Eigen::Map<Vec3>(fo + 3 * i) = f1; Eigen::Map<Vec3>(fo + 3 * (tri_size + i)) = f2; }
    }
    return std::sqrt(norm_f) / std::max(1.0, std::sqrt(norm_g));
}

static void apply_picard_update(Kernel2D& D1, Kernel2D& D2,
                                const Kernel2D& Hx1, const Kernel2D& Hx2,
                                int tri_size,
                                double neg_inv_r1, double neg_inv_r2,
                                double relax) {
    for (int i = 0; i < tri_size; ++i) {
        D1.data[i] += relax * (neg_inv_r1 * Hx1.data[i] - D1.data[i]);
        D2.data[i] += relax * (neg_inv_r2 * Hx2.data[i] - D2.data[i]);
    }
}

static bool adapt_picard_damping(double err, double& prev_err,
                                 double& relax,
                                 Kernel2D& D1, Kernel2D& D2,
                                 Kernel2D& prev_D1, Kernel2D& prev_D2,
                                 bool& have_prev,
                                 bool verbose, const char* prefix) {
    if (have_prev && err > prev_err * PICARD_RESIDUAL_GROWTH_LIMIT
        && relax > PICARD_RELAX_MIN) {
        D1 = prev_D1;
        D2 = prev_D2;
        relax = std::max(PICARD_RELAX_MIN, relax * PICARD_RELAX_BACKOFF);
        if (verbose)
            std::cout << "  " << prefix << "backtracking Picard step; relax="
                      << relax << std::endl;
        return false;
    }

    if (have_prev && err < prev_err * PICARD_RESIDUAL_DECAY_FOR_GROWTH)
        relax = std::min(PICARD_RELAX_MAX, relax * PICARD_RELAX_GROWTH);

    prev_D1 = D1;
    prev_D2 = D2;
    prev_err = err;
    have_prev = true;
    return true;
}

// Anderson acceleration (type II) on the stacked kernel vector x = (D1, D2).
// Given the history of iterates x_k and their images g_k = G(x_k), the next
// iterate is x = sum_i a_i ((1 - beta) x_i + beta g_i) with sum a_i = 1
// minimising |sum_i a_i f_i|, f_i = g_i - x_i.  Depth m keeps the last m + 1
// pairs.
long g_anderson_fallbacks = 0;   // solves in which acceleration was abandoned (diagnostic)
struct AndersonState {
    int depth = 5; double beta = 0.6; double first_step = 0.15; double growth = 3.0;   // growth: residual increase tolerated in one accelerated step
    // ring of the last `depth` differences dX_i = x_{i+1} - x_i, dF_i = f_{i+1} - f_i (columns), their Gram matrix,
    // and the previous pair (x, f)
    // differences stored in single precision: they only feed the least-squares coefficients and the
    // extrapolation correction, whose error is at the 1e-7 level against a 1e-5 stopping tolerance
    Eigen::MatrixXf dX, dF; Eigen::MatrixXd G; int m = 0, head = 0; bool have_prev = false; Eigen::VectorXd x_prev, f_prev;
    void reset() { m = 0; head = 0; have_prev = false; }
    // x, f (= g - x) of the current iterate; on return x holds the next iterate
    bool step(Eigen::VectorXd& x, const Eigen::VectorXd& f) {
        const int n = static_cast<int>(f.size());
        if (dX.rows() != n || dX.cols() != depth) { dX.resize(n, depth); dF.resize(n, depth); G.resize(depth, depth); reset(); }
        if (have_prev) {
            // append the newest difference (overwriting the oldest when full) and update the Gram matrix column
            const int c = head; head = (head + 1) % depth; if (m < depth) ++m;
            {
                float* dxc = dX.col(c).data(); float* dfc = dF.col(c).data();
                const double* xp = x.data(); const double* xq = x_prev.data(); const double* fp = f.data(); const double* fq = f_prev.data();
                #pragma omp parallel for schedule(static) if (n > 20000 && !omp_in_parallel())
                for (int i = 0; i < n; ++i) { dxc[i] = static_cast<float>(xp[i] - xq[i]); dfc[i] = static_cast<float>(fp[i] - fq[i]); }
            }
            for (int i = 0; i < m; ++i) {
                const float* a = dF.col(i).data(); const float* b = dF.col(c).data(); double acc = 0.0;
                #pragma omp parallel for reduction(+:acc) schedule(static) if (n > 20000 && !omp_in_parallel())
                for (int k = 0; k < n; ++k) acc += static_cast<double>(a[k]) * b[k];
                G(i, c) = acc; G(c, i) = acc;
            }
        }
        x_prev = x; f_prev = f; have_prev = true;
        if (m == 0) { x = x + first_step * f; return true; }   // first move: the safe relaxation, not beta
        // gamma = argmin |f - dF gamma| over the m stored columns (any order: the columns are a set)
        Eigen::MatrixXd Gm = G.topLeftCorner(m, m);
        Eigen::VectorXd rhs(m);
        for (int i = 0; i < m; ++i) {
            const float* a = dF.col(i).data(); const double* fp = f.data(); double acc = 0.0;
            #pragma omp parallel for reduction(+:acc) schedule(static) if (n > 20000 && !omp_in_parallel())
            for (int k = 0; k < n; ++k) acc += a[k] * fp[k];
            rhs[i] = acc;
        }
        Eigen::LDLT<Eigen::MatrixXd> ldlt(Gm + 1e-12 * Gm.trace() * Eigen::MatrixXd::Identity(m, m));
        const Eigen::VectorXd gamma = ldlt.solve(rhs);
        if (!gamma.allFinite()) { reset(); return false; }
        {
            double* xp = x.data(); const double* fp = f.data();
            #pragma omp parallel for schedule(static) if (n > 20000 && !omp_in_parallel())
            for (int k = 0; k < n; ++k) {
                double cx = 0.0, cf = 0.0;
                for (int i = 0; i < m; ++i) { cx += dX(k, i) * gamma[i]; cf += dF(k, i) * gamma[i]; }
                xp[k] = xp[k] + beta * fp[k] - cx - beta * cf;
            }
        }
        return true;
    }
};

static void pack_kernels(const Kernel2D& D1, const Kernel2D& D2, int tri, Eigen::VectorXd& x) {
    x.resize(6 * tri);
    std::memcpy(x.data(), D1.data[0].data(), sizeof(double) * 3 * tri);            // Vec3 rows are contiguous
    std::memcpy(x.data() + 3 * tri, D2.data[0].data(), sizeof(double) * 3 * tri);
}
static void unpack_kernels(const Eigen::VectorXd& x, int tri, Kernel2D& D1, Kernel2D& D2) {
    std::memcpy(D1.data[0].data(), x.data(), sizeof(double) * 3 * tri);
    std::memcpy(D2.data[0].data(), x.data() + 3 * tri, sizeof(double) * 3 * tri);
}

// Core solver: takes initial D1, D2 (may be zero or warm-started)
static EquilibriumResult solve_equilibrium_core(
    double p1_val, double p2_val,
    Kernel2D D1, Kernel2D D2, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {

    std::vector<double> residuals;
    auto prec1 = make_constant_prec(p1_val * p1_val);
    auto prec2 = make_constant_prec(p2_val * p2_val);
    EnvironmentResult env;

    const int TRI = g_n * (g_n + 1) / 2;

    // Hoist adjoint buffers out of the Picard loop to avoid repeated
    // Kernel2D allocation at large N.
    Kernel2D Hx1, Hx2;
    Kernel2D prev_D1, prev_D2;
    double prev_err = std::numeric_limits<double>::infinity();
    double relax = PICARD_RELAX;
    bool have_prev = false;

    // Anderson acceleration of the outer fixed point (LQG_ANDERSON="depth,beta";
    // depth 0 restores plain relaxed Picard).  Engaged once the relaxed Picard
    // iteration has brought the residual below ANDERSON_START; a step that
    // increases the residual by more than ANDERSON_GROWTH is rejected, the
    // history cleared, and a Picard step taken from the previous iterate.
    static const AndersonState and_cfg = [] { AndersonState a; if (const char* e = std::getenv("LQG_ANDERSON")) { double d = 5, b = 0.6, g = 3.0; if (std::sscanf(e, "%lf,%lf,%lf", &d, &b, &g) >= 1) { a.depth = static_cast<int>(d); a.beta = b; a.growth = g; } } return a; }();
    AndersonState anderson = and_cfg;
    static const double ANDERSON_START = [] { const char* e = std::getenv("LQG_ANDERSON_START"); return e ? std::atof(e) : 0.5; }(); const double ANDERSON_GROWTH = anderson.growth;
    Eigen::VectorXd xa, fa;
    bool anderson_active = false;
    static const bool lockstep = [] { const char* e = std::getenv("LQG_BACKWARD_SECTIONS"); return !(e && std::atoi(e)); }();   // default: lockstep pair
    BackwardPlayer bp1, bp2;
    double best_err = std::numeric_limits<double>::infinity(); int since_best = 0;   // stagnation guard
    int nonfinite_recoveries = 0;
    enforce_predictable(D1, D2);
    static const bool newton_enabled_flag = [] { const char* e = std::getenv("LQG_NEWTON"); return !(e && std::atoi(e) == 0); }();
    static const int newton_after = [] { const char* e = std::getenv("LQG_NEWTON_AFTER"); return e ? std::atoi(e) : 40; }();   // switch to the Newton-Krylov engine after this many outer iterations (0 = never)
    constexpr int ANDERSON_STALL = 8;

    static const bool prof = std::getenv("LQG_PROF") != nullptr; double tprof[3] = {0, 0, 0}; double tp0 = prof ? omp_get_wtime() : 0.0;
    auto ptick = [&](int k) { if (prof) { const double t1 = omp_get_wtime(); tprof[k] += t1 - tp0; tp0 = t1; } };
    for (int it = 1; it <= MAX_PICARD_ITERS; ++it) {
        ptick(2);
        forward_environment(D1, D2, p1_val, p2_val, FORWARD_INNER_ITERS,
                            Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);
        ptick(0);
        if (exact_adjoint_enabled() && env.has_calD && env.basis1) {
            exact_adjoint_pair(env, D1, D2, g_r1, g_r2, obs_idx_1, obs_idx_2, p1_val, p2_val, Hx1, Hx2);
            ptick(1);
        } else if (lockstep) {
            bp1.Xt = &env.Xtilde2; bp1.Dk = &D2; bp1.prec = &prec2; bp1.gain = p2_val; bp1.obs = obs_idx_2; bp1.Hx = &Hx1;
            bp2.Xt = &env.Xtilde1; bp2.Dk = &D1; bp2.prec = &prec1; bp2.gain = p1_val; bp2.obs = obs_idx_1; bp2.Hx = &Hx2;
            backward_kernels_pair(env.X, bp1, bp2, g_terminal_weight);
            if (predictable_control()) for (int t = 0; t < g_n; ++t) { Hx1[t][t].setZero(); Hx2[t][t].setZero(); }   // inert coordinate
        } else {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            backward_kernels(env.X, env.Xtilde2, D2, prec2,
                             p2_val, obs_idx_2, g_terminal_weight, Hx1);
            #pragma omp section
            backward_kernels(env.X, env.Xtilde1, D1, prec1,
                             p1_val, obs_idx_1, g_terminal_weight, Hx2);
        }
        }

        ptick(1);
        if (prof && it % 5 == 0) std::fprintf(stderr, "prof it %d: forward %.1f  adjoint %.1f  rest %.1f ms\n", it, 1e3 * tprof[0], 1e3 * tprof[1], 1e3 * tprof[2]);
        // Compute G = -(1/r_k)*Hx and residual F = G - D
        const double neg_inv_r1 = -(1.0 / g_r1);
        const double neg_inv_r2 = -(1.0 / g_r2);
        double err = best_response_residual(D1, D2, Hx1, Hx2, TRI,
                                            neg_inv_r1, neg_inv_r2, anderson.depth > 0 ? &fa : nullptr);
        residuals.push_back(err);
        if (!std::isfinite(err)) {
            // the relaxed step overshot into overflow (small effort cost makes the best-response
            // map -(1/r) Hx large): restore the last finite iterate and cut the relaxation
            if (!have_prev && !anderson_active) { if (verbose) std::cout << "  non-finite residual at the start; giving up" << std::endl; break; }
            if (anderson_active) unpack_kernels(anderson.x_prev, TRI, D1, D2); else { D1 = prev_D1; D2 = prev_D2; }
            anderson.reset(); anderson.depth = 0; anderson_active = false; have_prev = false;
            relax = std::max(PICARD_RELAX_MIN, 0.25 * relax); ++nonfinite_recoveries;
            if (verbose) std::cout << "  non-finite residual; restored previous iterate, relax=" << relax << std::endl;
            if (nonfinite_recoveries > 12) break;
            // re-evaluate at the restored iterate on the next pass (its Hx is recomputed)
            continue;
        }

        if (verbose && (it <= 5 || it % 50 == 0))
            std::cout << "  it=" << it << "  resid=" << err
                      << "  relax=" << relax << std::endl;
        if (err < PICARD_TOL) {
            if (verbose)
                std::cout << "  Converged at iteration " << it << std::endl;
            break;
        }
        if (it == MAX_PICARD_ITERS)
            break;
        if (newton_after > 0 && it >= newton_after) break;   // hand over to the Newton-Krylov engine

        if (err < 0.98 * best_err) { best_err = err; since_best = 0; } else ++since_best;
        if (anderson.depth > 0 && (err < ANDERSON_START || anderson_active)) {   // once active, the rejection logic applies whatever the residual
            const bool stalled = anderson_active && since_best >= ANDERSON_STALL;
            const bool blew_up = anderson_active && have_prev && err > prev_err * ANDERSON_GROWTH;
            if (stalled || blew_up) {
                // Give up on acceleration for this solve: restore the best Picard-safe
                // iterate seen and continue with the relaxed Picard iteration, which is
                // known to converge (44 iterations on the benchmark).
                unpack_kernels(anderson.x_prev, TRI, D1, D2);      // the previous iterate lives in the history
                anderson.reset(); anderson.depth = 0; anderson_active = false; have_prev = false;
                relax = PICARD_RELAX; ++g_anderson_fallbacks;
                if (verbose || std::getenv("LQG_ANDERSON_TRACE")) std::fprintf(stderr, "  [anderson %s at it=%d err=%.2e prev=%.2e] p=(%.3g,%.3g) r=(%g,%g) sigma=%g b=(%g,%g)\n", stalled ? "stalled" : "diverging", it, err, prev_err, p1_val, p2_val, g_r1, g_r2, g_sigma, g_b1, g_b2);
                // Hand over to the Newton-Krylov engine from the restored iterate.  The relaxed
                // Picard iteration is useless here: dG/dD has eigenvalues of order -1/r (-200 at
                // r = 0.015 on the low-precision path), so any relaxation above 2/(1 + |lambda|)
                // diverges and would wreck the iterate before Newton starts.
                if (newton_enabled_flag) break;
            } else {
                pack_kernels(D1, D2, TRI, xa);
                prev_err = err; have_prev = true; anderson.first_step = relax;
                if (anderson.step(xa, fa)) { unpack_kernels(xa, TRI, D1, D2); enforce_predictable(D1, D2); anderson_active = true; continue; }
            }
        }

        if (!adapt_picard_damping(err, prev_err, relax, D1, D2,
                                  prev_D1, prev_D2, have_prev,
                                  verbose, ""))
            continue;

        apply_picard_update(D1, D2, Hx1, Hx2, TRI,
                            neg_inv_r1, neg_inv_r2, relax);
        enforce_predictable(D1, D2);
    }

    // ---- Newton-Krylov fallback ----
    // The relaxed iteration (and Anderson on top of it) needs the best-response map
    // to be a contraction; for small effort cost r the map -(1/r) Hx is large and the
    // fixed point, though it exists, is unstable under Picard.  Solve F(x) = G(x) - x = 0
    // by inexact Newton: GMRES on finite-difference Jacobian actions (one forward march
    // plus one backward pass each), with a backtracking line search on |F|.
    static const bool newton_enabled = [] { const char* e = std::getenv("LQG_NEWTON"); return !(e && std::atoi(e) == 0); }();
    const bool converged_picard = !residuals.empty() && std::isfinite(residuals.back()) && residuals.back() < PICARD_TOL;
    if (newton_enabled && !converged_picard) {
        // start from the last finite iterate
        if (!residuals.empty() && !std::isfinite(residuals.back())) { if (anderson_active) unpack_kernels(anderson.x_prev, TRI, D1, D2); else if (have_prev) { D1 = prev_D1; D2 = prev_D2; } }
        const int dim = 6 * TRI;
        const double neg_inv_r1 = -(1.0 / g_r1), neg_inv_r2 = -(1.0 / g_r2);
        Eigen::VectorXd x(dim), F(dim), Fx(dim);
        pack_kernels(D1, D2, TRI, x);
        Kernel2D Dn1, Dn2;
        // F(x) = G(x) - x with G = -(1/r) Hx(x); returns the relative residual, or NaN if the march overflows
        auto evalF = [&](const Eigen::VectorXd& xv, Eigen::VectorXd& Fv) -> double {
            unpack_kernels(xv, TRI, Dn1, Dn2);
            forward_environment(Dn1, Dn2, p1_val, p2_val, FORWARD_INNER_ITERS, Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);
            if (exact_adjoint_enabled() && env.has_calD && env.basis1) {
                exact_adjoint_pair(env, Dn1, Dn2, g_r1, g_r2, obs_idx_1, obs_idx_2, p1_val, p2_val, Hx1, Hx2);
            } else {
            bp1.Xt = &env.Xtilde2; bp1.Dk = &Dn2; bp1.prec = &prec2; bp1.gain = p2_val; bp1.obs = obs_idx_2; bp1.Hx = &Hx1;
            bp2.Xt = &env.Xtilde1; bp2.Dk = &Dn1; bp2.prec = &prec1; bp2.gain = p1_val; bp2.obs = obs_idx_1; bp2.Hx = &Hx2;
            backward_kernels_pair(env.X, bp1, bp2, g_terminal_weight);
            if (predictable_control()) for (int t = 0; t < g_n; ++t) { Hx1[t][t].setZero(); Hx2[t][t].setZero(); }
            }
            return best_response_residual(Dn1, Dn2, Hx1, Hx2, TRI, neg_inv_r1, neg_inv_r2, &Fv);
        };
        double res = evalF(x, F);
        if (verbose) std::cout << "  Newton-Krylov fallback from resid " << res << std::endl;
        static const int GM = [] { const char* e = std::getenv("LQG_NEWTON_GMRES"); return e ? std::atoi(e) : 120; }();   // Krylov size per cycle
        const int NEWTON_MAX = 40, GM_RESTARTS = 3;
        Eigen::VectorXd xt(dim), Ft(dim), v(dim), w(dim);
        for (int nit = 0; nit < NEWTON_MAX && std::isfinite(res) && res >= PICARD_TOL; ++nit) {
            // GMRES on J d = -F, J v ~ (F(x + eps v) - F(x)) / eps
            const double xnorm = x.norm();
            auto applyJ = [&](const Eigen::VectorXd& vv, Eigen::VectorXd& out) {
                const double vn = vv.norm(); if (vn == 0.0) { out.setZero(); return; }
                const double eps = 1e-7 * std::max(1.0, xnorm) / vn;
                xt = x + eps * vv; evalF(xt, Ft); out = (Ft - F) / eps;
            };
            // restarted GMRES(GM) on J d = -F, inexact tolerance 1e-4 relative
            Eigen::VectorXd d = Eigen::VectorXd::Zero(dim); const double gm_tol = 1e-4 * F.norm(); int k = 0; bool gm_ok = false;
            const int m = std::min(dim, GM);
            for (int cycle = 0; cycle < GM_RESTARTS && !gm_ok; ++cycle) {
                Eigen::VectorXd r0; if (cycle == 0) r0 = -F; else { applyJ(d, w); r0 = -F - w; }
                const double beta = r0.norm(); if (beta < gm_tol) { gm_ok = true; break; }
                Eigen::MatrixXd V(dim, m + 1), H = Eigen::MatrixXd::Zero(m + 1, m);
                Eigen::VectorXd g = Eigen::VectorXd::Zero(m + 1); g[0] = beta; V.col(0) = r0 / beta;
                std::vector<double> cs(m), sn(m); k = 0;
                for (; k < m; ++k) {
                    applyJ(V.col(k), w);
                    if (!w.allFinite()) break;
                    for (int i2 = 0; i2 <= k; ++i2) { H(i2, k) = V.col(i2).dot(w); w -= H(i2, k) * V.col(i2); }
                    H(k + 1, k) = w.norm(); if (H(k + 1, k) > 1e-300) V.col(k + 1) = w / H(k + 1, k);
                    for (int i2 = 0; i2 < k; ++i2) { const double t = cs[i2] * H(i2, k) + sn[i2] * H(i2 + 1, k); H(i2 + 1, k) = -sn[i2] * H(i2, k) + cs[i2] * H(i2 + 1, k); H(i2, k) = t; }
                    const double den = std::hypot(H(k, k), H(k + 1, k)); cs[k] = H(k, k) / den; sn[k] = H(k + 1, k) / den; H(k, k) = den; H(k + 1, k) = 0.0;
                    g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k];
                    if (std::fabs(g[k + 1]) < gm_tol) { ++k; gm_ok = true; break; }
                }
                if (k == 0) break;
                const Eigen::VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
                d += V.leftCols(k) * y;
            }
            if (k == 0 && d.norm() == 0.0) break;
            // backtracking line search on the residual (finite evaluations only)
            double step = 1.0, res_new = std::numeric_limits<double>::quiet_NaN();
            for (int ls = 0; ls < 12; ++ls) {
                xt = x + step * d; res_new = evalF(xt, Ft);
                if (std::isfinite(res_new) && res_new < (1.0 - 1e-4 * step) * res) break;
                step *= 0.5;
            }
            if (!(std::isfinite(res_new) && res_new < res)) { if (verbose) std::cout << "  Newton: no descent; stopping" << std::endl; break; }
            x = xt; F = Ft; res = res_new; residuals.push_back(res);
            if (predictable_control()) { unpack_kernels(x, TRI, Dn1, Dn2); enforce_predictable(Dn1, Dn2); pack_kernels(Dn1, Dn2, TRI, x); }
            if (verbose) std::cout << "  Newton it=" << nit + 1 << " gmres " << k << (gm_ok ? "" : " (not converged)") << " step " << step << " resid " << res << std::endl;
            if (step < 1e-3) { if (verbose) std::cout << "  Newton: line search collapsed; stopping" << std::endl; break; }
        }
        unpack_kernels(x, TRI, D1, D2);
    }

    // Ensure the returned environment matches the final Picard iterate, including
    // the non-converged case where the loop exits immediately after an update.
    forward_environment(D1, D2, p1_val, p2_val, FORWARD_INNER_ITERS,
                        Pi_1, obs_idx_1, Pi_2, obs_idx_2, env);

    Kernel2D calD1, calD2;
    if (env.has_calD) { calD1 = env.calD1; calD2 = env.calD2; }
    else {
        #pragma omp parallel sections
        {
            #pragma omp section
            primitive_control_kernel(D1, env.Xtilde1,
                                     p1_val, obs_idx_1, Pi_1, calD1);
            #pragma omp section
            primitive_control_kernel(D2, env.Xtilde2,
                                     p2_val, obs_idx_2, Pi_2, calD2);
        }
    }

    return {D1, D2, std::move(env), std::move(calD1), std::move(calD2), residuals};
}

static bool eq_converged(const EquilibriumResult& eq) {
    return !eq.residuals.empty() && std::isfinite(eq.residuals.back()) && eq.residuals.back() < PICARD_TOL;
}

EquilibriumResult solve_equilibrium(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {
    Kernel2D D1, D2;
    D1.setZero();
    D2.setZero();
    EquilibriumResult eq = solve_equilibrium_core(p1_val, p2_val, D1, D2, verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);
    if (eq_converged(eq)) return eq;
    // Diagnostic: the stiff regime is small effort cost over a long horizon (the map's
    // sensitivity grows like exp(gain T)); dt/r summarises the grid side of it.
    std::fprintf(stderr, "lqg_solver: cold start did not converge (N = %d, T = %g, r = (%g, %g), dt/r = %.2f); trying continuation in r\n",
                 g_n, g_T, g_r1, g_r2, g_dt / std::min(g_r1, g_r2));
    // Cold start failed (small effort cost: the best-response map is far from a contraction
    // at D = 0).  Continuation in r from an easy problem, warm-starting each step.
    static const bool cont_enabled = [] { const char* e = std::getenv("LQG_R_CONTINUATION"); return !(e && std::atoi(e) == 0); }();
    const double r1 = g_r1, r2 = g_r2, rmin = std::min(r1, r2);
    if (!cont_enabled || rmin >= 0.5) return eq;
    if (verbose) std::cout << "  cold start did not converge; continuation in r from 0.5" << std::endl;
    // Adaptive path: multiply r by `ratio` per stage; a failed stage is bisected in log r
    // (up to 10 bisections), since the basin of the stiff fixed point shrinks with r and T.
    double ratio = 0.7, f = 0.5 / rmin; int bisections = 0, stages = 0;
    Kernel2D W1, W2; W1.setZero(); W2.setZero(); bool have = false;
    double f_ok = std::numeric_limits<double>::infinity();     // last factor that converged
    EquilibriumResult last = eq;
    while (stages < 200) {
        SolverContext ctx = SolverContext::capture_current(); ctx.r1 = r1 * f; ctx.r2 = r2 * f;
        EquilibriumResult step;
        { ScopedSolverContext guard(ctx); step = solve_equilibrium_core(p1_val, p2_val, W1, W2, verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2); }
        ++stages;
        if (verbose) std::cout << "  continuation r x " << f << ": " << (eq_converged(step) ? "ok" : "FAIL") << " in " << step.residuals.size() << std::endl;
        if (!eq_converged(step)) {
            if (!have || bisections >= 10) return step;          // even the easy problem failed, or the path is lost
            ++bisections; f = std::sqrt(f * f_ok); continue;   // bisect in log r between the last success and this failure
        }
        W1 = step.D1; W2 = step.D2; have = true; f_ok = f; last = std::move(step);
        if (f <= 1.0 + 1e-12) return last;
        f = std::max(1.0, f * ratio);
    }
    return last;
}

EquilibriumResult solve_equilibrium_warm(
    double p1_val, double p2_val,
    const Kernel2D& D1_init, const Kernel2D& D2_init,
    bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {
    Kernel2D D1(D1_init), D2(D2_init);
    return solve_equilibrium_core(p1_val, p2_val, std::move(D1), std::move(D2),
                                  verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);
}

// --- solve_equilibrium_ce: Picard iteration with discrete CE projection ---

EquilibriumResult solve_equilibrium_ce(
    double p1_val, double p2_val, bool verbose,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2) {
    // The exact projection is the production filter (LQG_FILTER=pi selects the former
    // Pi-based closed-form filter); this entry point is kept for API compatibility.
    return solve_equilibrium(p1_val, p2_val, verbose, Pi_1, obs_idx_1, Pi_2, obs_idx_2);
}

// --- compute_costs_general ---

CostPair compute_costs_general(const EnvironmentResult& env,
                               const Kernel2D& calD1, const Kernel2D& calD2,
                               const BarSolution& bar_sol,
                               double r1_val, double r2_val,
                               double b1_val, double b2_val) {
    // Variance sums run over s <= j: the coordinate born at j (the state's own increment and
    // the control's loading on the current innovation) is part of X_j and D_j at step j.  The
    // first-order condition charges r D(j,j); a cost that omitted the diagonal was not
    // stationary at the solver's fixed point there (finite-difference check, Aug 2026).
    double J1 = 0.0, J2 = 0.0;
    for (int j = 0; j < g_n; ++j) {
        double var_X = 0.0, var_D1 = 0.0, var_D2 = 0.0;
        for (int s = 0; s <= j; ++s) {
            var_X += g_dt * env.X[j][s].squaredNorm();
            var_D1 += g_dt * calD1[j][s].squaredNorm();
            var_D2 += g_dt * calD2[j][s].squaredNorm();
        }
        double dx1 = bar_sol.barX[j] - b1_val;
        double dx2 = bar_sol.barX[j] - b2_val;
        J1 += g_dt * (dx1*dx1 + var_X + r1_val * (bar_sol.barD1[j]*bar_sol.barD1[j] + var_D1));
        J2 += g_dt * (dx2*dx2 + var_X + r2_val * (bar_sol.barD2[j]*bar_sol.barD2[j] + var_D2));
    }
    if (g_terminal_weight != 0.0) {
        const int j = g_n - 1;
        double var_X_T = 0.0;
        for (int s = 0; s <= j; ++s)
            var_X_T += g_dt * env.X[j][s].squaredNorm();
        double dx1_T = bar_sol.barX[j] - b1_val;
        double dx2_T = bar_sol.barX[j] - b2_val;
        J1 += g_terminal_weight * (dx1_T * dx1_T + var_X_T);
        J2 += g_terminal_weight * (dx2_T * dx2_T + var_X_T);
    }
    return {J1, J2};
}

// --- materialize_F (for figure output only) ---
//
// Builds F[j][u][s] incrementally from Xtilde.
// F[j][u][s] = F[j-1][u][s] + Xtilde[j][u] * A[j][s]^T  (interior), A[j][s] = dt g^2 Xtilde[j][s]
// Borders: F[j][u][j] = g*Xtilde[j][u]*e_i^T, F[j][j][s] = g*e_i*Xtilde[j][s]^T

void materialize_F(const Kernel2D& Xtilde,
                   double obs_gain, int obs_index, Kernel3D& F) {
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain;
    const double dt_prec = g_dt * g * g;

    F[0][0][0].setZero();
    for (int j = 1; j < g_n; ++j) {
        for (int u = 0; u < j; ++u) {
            F[j][u][j] = g * Xtilde[j][u] * e_i.transpose();
            F[j][j][u] = g * e_i * Xtilde[j][u].transpose();
        }
        F[j][j][j].setZero();

        for (int u = 0; u < j; ++u)
            for (int s = 0; s < j; ++s)
                F[j][u][s] = F[j - 1][u][s] + Xtilde[j][u] * (dt_prec * Xtilde[j][s]).transpose();
    }
}

// --- compute_F_slice_at_T (memory-efficient) ---
//
// Computes F[g_n-1][u][s] using ping-pong of two N_MAX×N_MAX slices
// instead of the full 3D kernel. Memory: ~2 * N_MAX^2 * 72 bytes ≈ 0.9MB.

std::unique_ptr<FSlice> compute_F_slice_at(const Kernel2D& Xtilde,
                                            double obs_gain, int obs_index, int t_idx) {
    Vec3 e_i = Vec3::Zero();
    e_i(obs_index) = 1.0;
    double g = obs_gain;
    const double dt_prec = g_dt * g * g;

    // Two slices for ping-pong: prev = F[j-1], cur = F[j]
    // Heap-allocated; pointer swap avoids deep copy.
    auto prev = std::make_unique<FSlice>();
    auto cur  = std::make_unique<FSlice>();

    // j = 0: F[0][0][0] = 0
    (*prev)(0, 0).setZero();

    for (int j = 1; j <= t_idx; ++j) {
        // Borders
        for (int u = 0; u < j; ++u) {
            (*cur)(u, j) = g * Xtilde[j][u] * e_i.transpose();
            (*cur)(j, u) = g * e_i * Xtilde[j][u].transpose();
        }
        (*cur)(j, j).setZero();

        // Interior: F[j][u][s] = F[j-1][u][s] + Xtilde[j][u] * A[j][s]^T, A[j][s] = dt g^2 Xtilde[j][s]
        for (int u = 0; u < j; ++u)
            for (int s = 0; s < j; ++s)
                (*cur)(u, s) = (*prev)(u, s) + Xtilde[j][u] * (dt_prec * Xtilde[j][s]).transpose();

        std::swap(prev, cur);
    }

    // After the loop, prev holds F[t_idx]
    return prev;
}

std::unique_ptr<FSlice> compute_F_slice_at_T(const Kernel2D& Xtilde,
                                              double obs_gain, int obs_index) {
    return compute_F_slice_at(Xtilde, obs_gain, obs_index, g_n - 1);
}

// --- Exact discrete conditional expectation ---

DiscreteProjection discrete_conditional_expectation(
    const Kernel2D& X, int obs_idx, double obs_gain, int t_idx) {

    int n = t_idx + 1;
    int dim = 3 * n;
    int n_obs = t_idx;  // observations at j = 1, ..., t_idx
    double g = obs_gain;

    // 1. Build measurement matrix H: n_obs × dim
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_obs, dim);
    for (int j = 1; j <= t_idx; ++j) {
        int row = j - 1;
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                H(row, 3*z + k) = g * g_dt * X[j][z](k);
        H(row, 3*j + obs_idx) += 1.0;
    }

    // 2. Factorize HH^T (n_obs × n_obs, much smaller than dim × dim)
    Eigen::MatrixXd HHt = H * H.transpose();
    auto ldlt = HHt.ldlt();

    // 3. Extract Xtilde without forming the full dim × dim M matrix.
    //    (I - M) Xmat = Xmat - H^T (HH^T)^{-1} (H Xmat)
    //    Peak memory: H (n_obs × dim) + HXmat (n_obs × n) instead of M (dim × dim).
    Kernel2D Xt_out;
    Eigen::MatrixXd Xmat = Eigen::MatrixXd::Zero(dim, n);
    for (int j = 0; j <= t_idx; ++j)
        for (int z = 0; z <= j; ++z)
            Xmat.block<3,1>(3*z, j) = X[j][z];

    Eigen::MatrixXd HXmat = H * Xmat;              // n_obs × n
    Eigen::MatrixXd solved = ldlt.solve(HXmat);     // n_obs × n
    Eigen::MatrixXd Xt_mat = Xmat - H.transpose() * solved;  // dim × n

    for (int j = 0; j <= t_idx; ++j)
        for (int z = 0; z <= j; ++z)
            Xt_out[j][z] = Xt_mat.block<3,1>(3*z, j);

    // 4. Diagnostics from the LDLT factorization.
    //    Rank = number of positive pivots.
    //    Idempotency: compute trace(M) = trace(H^T (HH^T)^{-1} H)
    //    = trace((HH^T)^{-1} HH^T) = trace(I_{n_obs}) = n_obs for full rank.
    //    Check via ||H (HH^T)^{-1} H^T H - H||_F / ||H||_F.
    int rank = 0;
    for (int i = 0; i < n_obs; ++i)
        if (ldlt.vectorD()(i) > 1e-12) rank++;

    // Cheap idempotency check: ||(HH^T)^{-1}(HH^T) - I||_F
    Eigen::MatrixXd I_check = ldlt.solve(HHt);
    double idem = (I_check - Eigen::MatrixXd::Identity(n_obs, n_obs)).norm()
                  / std::max(1e-15, std::sqrt(static_cast<double>(n_obs)));

    return {Eigen::MatrixXd(), std::move(Xt_out), rank, idem};
}

IncrementalProjection build_projections_incremental(
    const Kernel2D& X, int obs_idx, double obs_gain) {

    int n = g_n;
    int dim = 3 * n;
    int n_obs = n - 1;
    double g = obs_gain;

    // Thin V basis instead of dense M (saves dim×dim = 1.8MB at N=160)
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(dim, n_obs);
    int rank = 0;
    Eigen::VectorXd h(dim), v(dim), coeff(n_obs);
    h.setZero();
    v.setZero();

    for (int j = 1; j < n; ++j) {
        int active = 3 * (j + 1);
        h.head(active).setZero();
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                h(3*z + k) = g * g_dt * X[j][z](k);
        h(3*j + obs_idx) += 1.0;

        v.head(active) = h.head(active);
        if (rank > 0) {
            auto Vact = V.topRows(active).leftCols(rank);
            auto c = coeff.head(rank);
            c.noalias() = Vact.transpose() * h.head(active);
            v.head(active).noalias() -= Vact * c;
        }
        double vnorm = v.head(active).norm();
        if (vnorm > 1e-15) {
            V.col(rank).head(active) = v.head(active) / vnorm;
            rank++;
        }
    }

    // Extract Xtilde using V: (I - VV^T) Xmat
    Kernel2D Xt_out;
    Eigen::MatrixXd Xmat = Eigen::MatrixXd::Zero(dim, n);
    for (int j = 0; j < n; ++j)
        for (int z = 0; z <= j; ++z)
            Xmat.block<3,1>(3*z, j) = X[j][z];

    // VtX = V^T * Xmat (rank × n), then Xt = Xmat - V * VtX
    Eigen::MatrixXd VtX = V.leftCols(rank).transpose() * Xmat;
    Eigen::MatrixXd Xt_mat = Xmat - V.leftCols(rank) * VtX;

    for (int j = 0; j < n; ++j)
        for (int z = 0; z <= j; ++z)
            Xt_out[j][z] = Xt_mat.block<3,1>(3*z, j);

    // Idempotency: VV^T is exact projection by construction, check V^T V = I
    Eigen::MatrixXd VtV = V.leftCols(rank).transpose() * V.leftCols(rank);
    double idem = (VtV - Eigen::MatrixXd::Identity(rank, rank)).norm()
                  / std::max(1e-15, std::sqrt(static_cast<double>(rank)));

    return {Eigen::MatrixXd(), std::move(Xt_out), idem};
}

std::unique_ptr<FSlice> compute_ce_F_slice_at_T(
    const Kernel2D& X, int obs_idx, double obs_gain, const Mat3& Pi) {

    int n = g_n;
    int dim = 3 * n;
    int n_obs = n - 1;
    double g = obs_gain;

    // Build thin V basis incrementally
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(dim, n_obs);
    int rank = 0;
    Eigen::VectorXd h(dim), v(dim), coeff(n_obs);
    h.setZero();
    v.setZero();

    for (int j = 1; j < n; ++j) {
        int active = 3 * (j + 1);
        h.head(active).setZero();
        for (int z = 0; z <= j; ++z)
            for (int k = 0; k < 3; ++k)
                h(3*z + k) = g * g_dt * X[j][z](k);
        h(3*j + obs_idx) += 1.0;

        v.head(active) = h.head(active);
        if (rank > 0) {
            auto Vact = V.topRows(active).leftCols(rank);
            auto c = coeff.head(rank);
            c.noalias() = Vact.transpose() * h.head(active);
            v.head(active).noalias() -= Vact * c;
        }
        double vnorm = v.head(active).norm();
        if (vnorm > 1e-15) {
            V.col(rank).head(active) = v.head(active) / vnorm;
            rank++;
        }
    }

    // Extract F: F(u,s) = [VV^T(3u:3u+3, 3s:3s+3) - Pi*delta(u,s)] / dt
    auto F = std::make_unique<FSlice>();
    double inv_dt = 1.0 / g_dt;
    auto Vr = V.leftCols(rank);
    for (int u = 0; u < n; ++u) {
        auto Vu = Vr.middleRows(3*u, 3);
        for (int s = 0; s < n; ++s) {
            auto Vs = Vr.middleRows(3*s, 3);
            Mat3 Mus = Vu * Vs.transpose();
            if (u == s) Mus -= Pi;
            (*F)(u, s) = Mus * inv_dt;
        }
    }

    return F;
}

ProjectionPair exact_discrete_CE(const EquilibriumResult& eq) {
    int t_idx = g_n - 1;
    DiscreteProjection p1, p2;
    #pragma omp parallel sections
    {
        #pragma omp section
        p1 = discrete_conditional_expectation(eq.env.X, eq.env.obs_idx1, eq.env.obs_gain1, t_idx);
        #pragma omp section
        p2 = discrete_conditional_expectation(eq.env.X, eq.env.obs_idx2, eq.env.obs_gain2, t_idx);
    }
    return {std::move(p1), std::move(p2)};
}
