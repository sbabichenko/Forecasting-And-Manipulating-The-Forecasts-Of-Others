#pragma once
// ============================================================
// Decentralized LQG noise-state game solver
// F-free optimized solver with triangular Kernel2D storage
// ============================================================

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------- compile-time maximum ----------
constexpr int N_MAX = 640;
constexpr int D_W = 3;

// Legacy 3D kernel max — kept small to avoid 160^3 * 72 B memory usage.
// Only used by generate_figures; the interactive solver never touches Kernel3D.
constexpr int N_MAX_3D = 80;

// ---------- runtime grid parameters ----------
extern int g_n;        // current grid size (default 40, max N_MAX)
extern double g_T;     // time horizon (default 1.0)
extern double g_dt;    // = g_T / (g_n - 1)

// Call this before any solver calls when changing N or T
void set_grid(int n, double T);
extern long g_anderson_fallbacks;   // diagnostic: solves that fell back from Anderson to Picard

// ---------- default constants ----------
constexpr double RHO = 0.1;
extern double g_x0;  // initial state (default 0.0)
constexpr double DEFAULT_TERMINAL_STATE_WEIGHT = 0.0;
constexpr double B1_DEFAULT = 1.0;
constexpr double B2_DEFAULT = -1.0;

constexpr int FORWARD_INNER_ITERS = 2;
constexpr int FILTER_INNER_ITERS = 6;
constexpr double FILTER_RELAX = 0.55;
constexpr int MAX_PICARD_ITERS = 10000;
constexpr double PICARD_RELAX = 0.15;
constexpr double PICARD_RELAX_MIN = 0.01;
constexpr double PICARD_RELAX_MAX = 0.25;
constexpr double PICARD_RELAX_BACKOFF = 0.5;
constexpr double PICARD_RELAX_GROWTH = 1.03;
constexpr double PICARD_RESIDUAL_GROWTH_LIMIT = 1.02;
constexpr double PICARD_RESIDUAL_DECAY_FOR_GROWTH = 0.90;
constexpr double PICARD_TOL = 1e-5;

// Mutable game parameters (modified by interactive app)
extern double g_b1;
extern double g_b2;
extern double g_r1;  // player 1 control cost weight
extern double g_r2;  // player 2 control cost weight
extern double g_sigma;  // state diffusion coefficient (default 1.0)
extern double g_terminal_weight;  // terminal tracking weight q_T

// Snapshot of all mutable runtime solver parameters.
struct SolverContext {
    int n;
    double T;
    double b1;
    double b2;
    double r1;
    double r2;
    double sigma;
    double x0;
    double terminal_weight;

    static SolverContext capture_current();
    void apply() const;
};

// RAII guard: applies a context on construction and restores the previous
// context on destruction.
class ScopedSolverContext {
public:
    explicit ScopedSolverContext(const SolverContext& next);
    ~ScopedSolverContext();

    ScopedSolverContext(const ScopedSolverContext&) = delete;
    ScopedSolverContext& operator=(const ScopedSolverContext&) = delete;

    ScopedSolverContext(ScopedSolverContext&&) = delete;
    ScopedSolverContext& operator=(ScopedSolverContext&&) = delete;

private:
    SolverContext previous_;
};

// ---------- type aliases ----------
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using VecN = Eigen::VectorXd;

// 2D kernel: lower-triangular packed storage (s <= t only)
// Dynamically sized to g_n*(g_n+1)/2 entries — safe on the stack at any N.
struct Kernel2D {
    std::vector<Vec3> data;

    Kernel2D() : data(g_n * (g_n + 1) / 2, Vec3::Zero()) {}

    struct RowProxy {
        Vec3* base;
        Vec3& operator[](int s) { return base[s]; }
        const Vec3& operator[](int s) const { return base[s]; }
    };
    struct ConstRowProxy {
        const Vec3* base;
        const Vec3& operator[](int s) const { return base[s]; }
    };

    RowProxy operator[](int t) { return {&data[t * (t + 1) / 2]}; }
    ConstRowProxy operator[](int t) const { return {&data[t * (t + 1) / 2]}; }

    void setZero() {
        for (auto& v : data) v.setZero();
    }

    // Resize to current g_n (used after set_grid changes).
    // Skips reallocation when size already matches.
    void resize() {
        const int need = g_n * (g_n + 1) / 2;
        if (static_cast<int>(data.size()) == need) return;
        data.assign(need, Vec3::Zero());
    }

    int tri_size() const { return static_cast<int>(data.size()); }
};

// 3D kernel: shape (N_MAX_3D, N_MAX_3D, N_MAX_3D, D_W, D_W) → F[t][u][s] is a 3x3 matrix
// Only used by legacy code paths (generate_figures). NOT used by interactive app.
using Kernel3DRaw = std::array<std::array<std::array<Mat3, N_MAX_3D>, N_MAX_3D>, N_MAX_3D>;

struct Kernel3D {
    std::unique_ptr<Kernel3DRaw> data;

    Kernel3D() : data(std::make_unique<Kernel3DRaw>()) {}
    Kernel3D(const Kernel3D& o) : data(std::make_unique<Kernel3DRaw>(*o.data)) {}
    Kernel3D(Kernel3D&&) noexcept = default;
    Kernel3D& operator=(const Kernel3D& o) {
        if (this != &o) data = std::make_unique<Kernel3DRaw>(*o.data);
        return *this;
    }
    Kernel3D& operator=(Kernel3D&&) noexcept = default;

    auto& operator[](int i) { return (*data)[i]; }
    const auto& operator[](int i) const { return (*data)[i]; }
};

// ---------- t_grid ----------
// Returns the time grid for the current g_n and g_T.
inline const std::array<double, N_MAX>& t_grid() {
    static std::array<double, N_MAX> g;
    for (int i = 0; i < g_n; ++i)
        g[i] = static_cast<double>(i) * g_T / (g_n - 1);
    return g;
}

// ---------- fixed matrices ----------
inline Vec3 E0() {
    return Vec3(1.0, 0.0, 0.0);
}

inline Mat3 Pi1() {
    Mat3 m = Mat3::Zero();
    m(1, 1) = 1.0;
    return m;
}

inline Mat3 Pi2() {
    Mat3 m = Mat3::Zero();
    m(2, 2) = 1.0;
    return m;
}

// ---------- environment result ----------
struct EnvironmentResult {
    Kernel2D X;
    Kernel2D Xtilde1, Xtilde2;
    // The rank-1 filter factor A[k][s] = dt * gain^2 * Xtilde[k][s] (s < k), A[k][k] = 0,
    // is not stored; it is formed from Xtilde where F is materialized.
    // Observation parameters (needed for F materialization)
    double obs_gain1, obs_gain2;
    int obs_idx1, obs_idx2;
};

// ---------- bar solution ----------
struct BarSolution {
    std::array<double, N_MAX> barX;
    std::array<double, N_MAX> barD1;
    std::array<double, N_MAX> barD2;
    double bar_residual;
};

// ---------- backward adjoint result ----------
struct BackwardBarResult {
    std::array<double, N_MAX> barHx;
    Kernel2D barHk; // (g_n, g_n, D_W)
};

// ---------- functions ----------
void state_kernel_from_calD(const Kernel2D& calD1, const Kernel2D& calD2,
                            Kernel2D& X);

void primitive_control_kernel(
    const Kernel2D& D, const Kernel2D& Xtilde,
    double obs_gain_val, int obs_index, const Mat3& Pi, Kernel2D& calD);

void forward_environment(
    const Kernel2D& D1, const Kernel2D& D2,
    double obs_gain1, double obs_gain2,
    int inner_iters,
    const Mat3& Pi_1, int obs_idx_1,
    const Mat3& Pi_2, int obs_idx_2,
    EnvironmentResult& env);

// Hk-free version: uses internal ping-pong buffers instead of 36MB Kernel3D
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk, const std::array<double, N_MAX>& prec_k,
                      double obs_gain_k, int obs_idx_k,
                      double terminal_state_weight,
                      Kernel2D& Hx);

// Version that also outputs the kernel information wedge V^i(t,r)
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk, const std::array<double, N_MAX>& prec_k,
                      double obs_gain_k, int obs_idx_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel2D& Vkernel);

// Legacy version that also fills Hk (only needed for figure output)
void backward_kernels(const Kernel2D& X, const Kernel2D& Xtildek,
                      const Kernel2D& Dk, const std::array<double, N_MAX>& prec_k,
                      double obs_gain_k, int obs_idx_k,
                      double terminal_state_weight,
                      Kernel2D& Hx, Kernel3D& Hk);

BackwardBarResult backward_bar_adjoints(
    const Kernel2D& X, const Kernel2D& Xtildek, const Kernel2D& Dk,
    const std::array<double, N_MAX>& barX, double b,
    const std::array<double, N_MAX>& prec_k,
    double obs_gain_k, int obs_idx_k,
    double terminal_weight);

double mean_information_wedge_at(
    const Kernel2D& Xtildek, const Kernel2D& barHk,
    const std::array<double, N_MAX>& prec_k,
    double obs_gain_k, int obs_idx_k, int t_idx);

BarSolution solve_bar_equilibrium(
    const EnvironmentResult& env, const Kernel2D& D1, const Kernel2D& D2,
    // Precision inputs P_i = gamma_i^2.
    double prec1, double prec2,
    int max_iters = 2000, double relax = 0.08, double tol = 1e-10,
    bool verbose = false);

struct EquilibriumResult {
    Kernel2D D1, D2;
    EnvironmentResult env;
    Kernel2D calD1, calD2;  // primitive control kernels (computed once after convergence)
    std::vector<double> residuals;
};

// Solver-level convention: these arguments are observation gains gamma_i.
// Paper-level precision p_i should be passed as std::sqrt(p_i).
EquilibriumResult solve_equilibrium(
    double p1_val, double p2_val, bool verbose = true,
    const Mat3& Pi_1 = Pi1(), int obs_idx_1 = 1,
    const Mat3& Pi_2 = Pi2(), int obs_idx_2 = 2);

// CE-based version: uses discrete conditional expectation in the Picard loop.
// The first two arguments are observation gains gamma_i.
EquilibriumResult solve_equilibrium_ce(
    double p1_val, double p2_val, bool verbose = true,
    const Mat3& Pi_1 = Pi1(), int obs_idx_1 = 1,
    const Mat3& Pi_2 = Pi2(), int obs_idx_2 = 2);

// Warm-started version: uses D1_init, D2_init as initial guesses.
// The first two arguments are observation gains gamma_i.
EquilibriumResult solve_equilibrium_warm(
    double p1_val, double p2_val,
    const Kernel2D& D1_init, const Kernel2D& D2_init,
    bool verbose = true,
    const Mat3& Pi_1 = Pi1(), int obs_idx_1 = 1,
    const Mat3& Pi_2 = Pi2(), int obs_idx_2 = 2);

// ---------- cost computation ----------
struct CostPair {
    double J1, J2;
};

CostPair compute_costs_general(const EnvironmentResult& env,
                               const Kernel2D& calD1, const Kernel2D& calD2,
                               const BarSolution& bar_sol,
                               double r1_val, double r2_val,
                               double b1_val, double b2_val);

// ---------- F materialization (for figure output only) ----------
// Builds F[j][u][s] for ALL j. Writes into pre-allocated Kernel3D.
void materialize_F(const Kernel2D& Xtilde,
                   double obs_gain, int obs_index, Kernel3D& F);

// ---------- F slice computation (memory-efficient) ----------
// Computes F[g_n-1][u][s] for all (u,s), returning a g_n x g_n array of Mat3.
// Only allocates two g_n x g_n slices (ping-pong), NOT the full 3D kernel.
struct FSlice {
    int n;  // grid size at construction time
    std::vector<Mat3> data;
    FSlice() : n(g_n), data(g_n * g_n, Mat3::Zero()) {}
    Mat3& operator()(int u, int s) { return data[u * n + s]; }
    const Mat3& operator()(int u, int s) const { return data[u * n + s]; }
};

std::unique_ptr<FSlice> compute_F_slice_at_T(const Kernel2D& Xtilde,
                                              double obs_gain, int obs_index);

// Compute F kernel at arbitrary time index t_idx (0-based grid index).
// Same algorithm as compute_F_slice_at_T but stops iteration at t_idx.
std::unique_ptr<FSlice> compute_F_slice_at(const Kernel2D& Xtilde,
                                            double obs_gain, int obs_index, int t_idx);

// ---------- exact discrete conditional expectation (post-processing) ----------

struct DiscreteProjection {
    Eigen::MatrixXd M;       // 3n x 3n projection matrix, n = t_idx + 1
    Kernel2D Xtilde;         // extracted Xtilde(j,z) for j = 0..t_idx
    int rank;                // rank of M (should equal t_idx)
    double idem_residual;    // ||M^2 - M|| / ||M||
};

// Build exact discrete CE projection M = H^T (H H^T)^{-1} H at a given time index.
DiscreteProjection discrete_conditional_expectation(
    const Kernel2D& X, int obs_idx, double obs_gain, int t_idx);

// Incremental version: builds M via rank-1 updates at each observation step.
struct IncrementalProjection {
    Eigen::MatrixXd M;       // terminal projection (3*g_n x 3*g_n)
    Kernel2D Xtilde;         // extracted from terminal M
    double idem_residual;
};

IncrementalProjection build_projections_incremental(
    const Kernel2D& X, int obs_idx, double obs_gain);

// Convenience: exact discrete CE for both players at terminal time.
struct ProjectionPair {
    DiscreteProjection player1, player2;
};

ProjectionPair exact_discrete_CE(const EquilibriumResult& eq);

// Compute F kernel from the exact discrete CE projection at terminal time.
// F_ce(u,s) = [M(u,s) - Pi*delta(u,s)] / dt  where M = VV^T is the CE projection.
std::unique_ptr<FSlice> compute_ce_F_slice_at_T(
    const Kernel2D& X, int obs_idx, double obs_gain, const Mat3& Pi);

// ---------- utility ----------
inline std::array<double, N_MAX> make_constant_prec(double val) {
    std::array<double, N_MAX> a;
    a.fill(val);
    return a;
}
