// WASM entry point for the interactive LQG solver.
// Compiled with Emscripten, called from JavaScript via ccall/cwrap.
//
// Architecture: a C++-side cache holds the last EquilibriumResult, BarSolution,
// and derived quantities. Multiple WASM endpoints (solve_light, solve_full,
// solve_f_kernel, solve_sweep) all use ensure_solve() to avoid redundant
// recomputation when the same parameters are queried from different tabs.

#include "lqg_solver.h"
#include <emscripten/emscripten.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

// Shared output buffer (grows as needed)
static std::string g_output;

static void out(const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    g_output += buf;
}

static void out_val(double v) {
    if (std::isfinite(v)) out("%.6g", v);
    else out("null");
}

static void out_array(const char* name, const double* arr, int n) {
    out("\"%s\":[", name);
    for (int i = 0; i < n; ++i) {
        if (i) out(",");
        out_val(arr[i]);
    }
    out("]");
}

// (out_kernel2d removed — kernel2d data now uses binary transfer via solve_full_bin)

static void compute_perfect_info(double b1, double b2,
                                  std::array<double, N_MAX>& barD1_pi,
                                  std::array<double, N_MAX>& barD2_pi,
                                  std::array<double, N_MAX>& barX_pi) {
    std::array<double, N_MAX> S_pi;
    S_pi.fill(0.0);
    S_pi[g_n - 1] = g_terminal_weight;
    for (int j = g_n - 2; j >= 0; --j)
        S_pi[j] = S_pi[j + 1] + g_dt * (1.0 - (1.0 / g_r1 + 1.0 / g_r2) * S_pi[j + 1] * S_pi[j + 1]);
    barX_pi[0] = g_x0;
    for (int j = 0; j < g_n; ++j) {
        barD1_pi[j] = -(1.0 / g_r1) * S_pi[j] * (barX_pi[j] - b1);
        barD2_pi[j] = -(1.0 / g_r2) * S_pi[j] * (barX_pi[j] - b2);
        if (j < g_n - 1)
            barX_pi[j + 1] = barX_pi[j] + g_dt * (barD1_pi[j] + barD2_pi[j]);
    }
}

// (out_fslice removed — F kernel now uses binary transfer via solve_f_kernel_bin)

// ============================================================
// Solver result cache
// ============================================================
// Single-level cache: recompute everything if any parameter changes.
// F kernel is still lazy (only computed when requested).

static bool g_use_ce = false;

static struct SolveCache {
    double p1, p2, b1, b2, r1, r2;
    double sigma, x0, terminal_weight;
    int n; double T;
    bool valid;
    bool ce_mode;

    EquilibriumResult eq;
    Kernel2D Vkernel1, Vkernel2;  // kernel information wedges V^i(t,r)
    BarSolution bar;
    CostPair costs;
    std::array<double, N_MAX> V1, V2;
    Kernel2D barHk1, barHk2;  // mean-field adjoint kernels
    std::array<double, N_MAX> barD1_pi, barD2_pi, barX_pi;

    // F kernel (computed lazily on first F kernel tab visit)
    std::unique_ptr<FSlice> F1, F2;
    bool f_valid;

    // Error tracking for debugging
    std::string last_error;       // empty if last solve succeeded
    std::string failed_stage;     // which pipeline stage failed
} g_cache = {};

static std::vector<double> g_full_buf;
static std::vector<double> g_fkernel_buf;

static bool cache_matches(double p1, double p2, double b1, double b2, double r1, double r2) {
    return g_cache.valid &&
           g_cache.p1 == p1 && g_cache.p2 == p2 &&
           g_cache.b1 == b1 && g_cache.b2 == b2 &&
           g_cache.r1 == r1 && g_cache.r2 == r2 &&
           g_cache.n == g_n && g_cache.T == g_T &&
           g_cache.ce_mode == g_use_ce &&
           g_cache.sigma == g_sigma && g_cache.x0 == g_x0 &&
           g_cache.terminal_weight == g_terminal_weight;
}

// Run the full solve pipeline if not cached. After this, g_cache holds all results.
static void ensure_solve(double p1, double p2, double b1, double b2, double r1, double r2) {
    if (cache_matches(p1, p2, b1, b2, r1, r2)) return;

    // Invalidate cache FIRST — if anything below fails, we won't serve stale data
    g_cache.valid = false;
    g_cache.f_valid = false;
    g_cache.F1.reset();
    g_cache.F2.reset();
    g_cache.last_error.clear();
    g_cache.failed_stage.clear();

    // Resize cached Kernel2D members to current g_n.
    // After set_grid changes g_n, these may still be sized for the old grid.
    g_cache.Vkernel1.resize();
    g_cache.Vkernel2.resize();
    g_cache.barHk1.resize();
    g_cache.barHk2.resize();

    SolverContext solve_ctx = SolverContext::capture_current();
    solve_ctx.b1 = b1;
    solve_ctx.b2 = b2;
    solve_ctx.r1 = r1;
    solve_ctx.r2 = r2;
    ScopedSolverContext guard(solve_ctx);

    try {

    const double obs_gain1 = std::sqrt(p1);
    const double obs_gain2 = std::sqrt(p2);

    // Solve equilibrium (standard or CE mode)
    g_cache.failed_stage = g_use_ce ? "solve_equilibrium_ce" : "solve_equilibrium";
    if (g_use_ce)
        g_cache.eq = solve_equilibrium_ce(obs_gain1, obs_gain2, false);
    else
        g_cache.eq = solve_equilibrium(obs_gain1, obs_gain2, false);

    // Compute kernel information wedges V^i(t,r)
    g_cache.failed_stage = "backward_kernels (V wedges)";
    auto prec1_eq = make_constant_prec(p1);
    auto prec2_eq = make_constant_prec(p2);
    Kernel2D Hx1_tmp, Hx2_tmp;
    backward_kernels(g_cache.eq.env.X, g_cache.eq.env.Xtilde2, g_cache.eq.D2,
                     prec2_eq, g_cache.eq.env.obs_gain2, g_cache.eq.env.obs_idx2,
                     g_terminal_weight, Hx1_tmp, g_cache.Vkernel1);
    backward_kernels(g_cache.eq.env.X, g_cache.eq.env.Xtilde1, g_cache.eq.D1,
                     prec1_eq, g_cache.eq.env.obs_gain1, g_cache.eq.env.obs_idx1,
                     g_terminal_weight, Hx2_tmp, g_cache.Vkernel2);

    // Bar solution, costs, wedges
    g_cache.failed_stage = "solve_bar_equilibrium";
    g_cache.bar = solve_bar_equilibrium(g_cache.eq.env, g_cache.eq.D1, g_cache.eq.D2,
                                         p1, p2, 2000, 0.08, 1e-10);

    g_cache.failed_stage = "compute_costs_general";
    g_cache.costs = compute_costs_general(g_cache.eq.env, g_cache.eq.calD1, g_cache.eq.calD2,
                                           g_cache.bar, r1, r2, b1, b2);

    g_cache.failed_stage = "backward_bar_adjoints";
    auto prec1_arr = make_constant_prec(p1);
    auto prec2_arr = make_constant_prec(p2);
    auto bba1 = backward_bar_adjoints(g_cache.eq.env.X, g_cache.eq.env.Xtilde2, g_cache.eq.D2,
                                       g_cache.bar.barX, b1, prec2_arr,
                                       g_cache.eq.env.obs_gain2, g_cache.eq.env.obs_idx2, g_terminal_weight);
    auto bba2 = backward_bar_adjoints(g_cache.eq.env.X, g_cache.eq.env.Xtilde1, g_cache.eq.D1,
                                       g_cache.bar.barX, b2, prec1_arr,
                                       g_cache.eq.env.obs_gain1, g_cache.eq.env.obs_idx1, g_terminal_weight);
    g_cache.barHk1 = std::move(bba1.barHk);
    g_cache.barHk2 = std::move(bba2.barHk);

    g_cache.failed_stage = "V wedge accumulation";
    for (int j = 0; j < g_n; ++j) {
        g_cache.V1[j] = mean_information_wedge_at(
            g_cache.eq.env.Xtilde2, g_cache.barHk1, prec2_arr,
            g_cache.eq.env.obs_gain2, g_cache.eq.env.obs_idx2, j);
        g_cache.V2[j] = mean_information_wedge_at(
            g_cache.eq.env.Xtilde1, g_cache.barHk2, prec1_arr,
            g_cache.eq.env.obs_gain1, g_cache.eq.env.obs_idx1, j);
    }

    g_cache.failed_stage = "compute_perfect_info";
    compute_perfect_info(b1, b2, g_cache.barD1_pi, g_cache.barD2_pi, g_cache.barX_pi);

    g_cache.p1 = p1; g_cache.p2 = p2;
    g_cache.b1 = b1; g_cache.b2 = b2;
    g_cache.r1 = r1; g_cache.r2 = r2;
    g_cache.n = g_n; g_cache.T = g_T;
    g_cache.ce_mode = g_use_ce;
    g_cache.sigma = g_sigma; g_cache.x0 = g_x0;
    g_cache.terminal_weight = g_terminal_weight;
    g_cache.valid = true;
    g_cache.failed_stage.clear();
    // f_valid already false from top of function

    } catch (const std::exception& e) {
        g_cache.valid = false;
        g_cache.f_valid = false;
        g_cache.last_error = e.what();
    } catch (...) {
        // Solver failed — leave cache invalid
        g_cache.valid = false;
        g_cache.f_valid = false;
        g_cache.last_error = "unknown C++ exception";
    }
}

// Compute F kernel slices if not cached. Requires ensure_solve() first.
static void ensure_f_kernel(double p1, double p2, double b1, double b2, double r1, double r2) {
    ensure_solve(p1, p2, b1, b2, r1, r2);
    if (g_cache.f_valid) return;

    const auto& env = g_cache.eq.env;
    if (g_use_ce) {
        g_cache.F1 = compute_ce_F_slice_at_T(env.X, env.obs_idx1, env.obs_gain1, Pi1());
        g_cache.F2 = compute_ce_F_slice_at_T(env.X, env.obs_idx2, env.obs_gain2, Pi2());
    } else {
        g_cache.F1 = compute_F_slice_at_T(env.Xtilde1,
                                           env.obs_gain1, env.obs_idx1);
        g_cache.F2 = compute_F_slice_at_T(env.Xtilde2,
                                           env.obs_gain2, env.obs_idx2);
    }
    g_cache.f_valid = true;
}

// Serialize only the "light" data (bar solution, costs, wedges, residuals).
// ~5KB JSON. Used by Equilibrium and Diagnostics tabs.
static void serialize_light() {
    const auto& tg = t_grid();
    const auto& eq = g_cache.eq;
    const auto& bar = g_cache.bar;

    out("{");
    out_array("t", tg.data(), g_n);
    out(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i) {
        if (i) out(",");
        out_val(eq.residuals[i]);
    }
    out("],\"n_iters\":%zu", eq.residuals.size());

    out(","); out_array("barD1", bar.barD1.data(), g_n);
    out(","); out_array("barD2", bar.barD2.data(), g_n);
    out(","); out_array("barX", bar.barX.data(), g_n);
    out(",\"bar_residual\":"); out_val(bar.bar_residual);
    out(","); out_array("V1", g_cache.V1.data(), g_n);
    out(","); out_array("V2", g_cache.V2.data(), g_n);
    out(","); out_array("barD1_pi", g_cache.barD1_pi.data(), g_n);
    out(","); out_array("barD2_pi", g_cache.barD2_pi.data(), g_n);
    out(","); out_array("barX_pi", g_cache.barX_pi.data(), g_n);
    out(",\"J1\":"); out_val(g_cache.costs.J1);
    out(",\"J2\":"); out_val(g_cache.costs.J2);
    // Convergence flag: true if final residual < tolerance
    bool converged = !eq.residuals.empty() &&
        std::isfinite(eq.residuals.back()) && eq.residuals.back() < PICARD_TOL;
    out(",\"converged\":%s", converged ? "true" : "false");
    out(",\"p1\":%.6g,\"p2\":%.6g,\"b1\":%.6g,\"b2\":%.6g",
        g_cache.p1, g_cache.p2, g_cache.b1, g_cache.b2);
    out(",\"obs_gain1\":%.6g,\"obs_idx1\":%d,\"obs_gain2\":%.6g,\"obs_idx2\":%d",
        g_cache.eq.env.obs_gain1, g_cache.eq.env.obs_idx1,
        g_cache.eq.env.obs_gain2, g_cache.eq.env.obs_idx2);
    out(",\"terminal_weight\":%.6g", g_cache.terminal_weight);
    out(",\"ce_mode\":%s", g_use_ce ? "true" : "false");
    out(",\"bar_converged\":%s", bar.bar_residual < 1e-8 ? "true" : "false");
    out(",\"max_picard_iters\":%d", MAX_PICARD_ITERS);
    out(",\"picard_tol\":%.1e", PICARD_TOL);
    out(",\"picard_relax\":%.6g", PICARD_RELAX);
    out(",\"picard_relax_min\":%.6g", PICARD_RELAX_MIN);
    out(",\"picard_relax_max\":%.6g", PICARD_RELAX_MAX);
    // Check for NaN/Inf in costs
    bool costs_finite = std::isfinite(g_cache.costs.J1) && std::isfinite(g_cache.costs.J2);
    out(",\"costs_finite\":%s", costs_finite ? "true" : "false");
    out("}");
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void wasm_set_grid(int n, double T) {
    set_grid(n, T);
    g_cache.valid = false;
    g_cache.f_valid = false;
    g_cache.F1.reset();
    g_cache.F2.reset();
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_sigma(double sigma) {
    if (sigma != g_sigma) {
        g_sigma = sigma;
        g_cache.valid = false;
        g_cache.f_valid = false;
        g_cache.F1.reset();
        g_cache.F2.reset();
    }
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_x0(double x0) {
    if (x0 != g_x0) {
        g_x0 = x0;
        g_cache.valid = false;
        g_cache.f_valid = false;
        g_cache.F1.reset();
        g_cache.F2.reset();
    }
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_terminal_weight(double terminal_weight) {
    if (terminal_weight != g_terminal_weight) {
        g_terminal_weight = terminal_weight;
        g_cache.valid = false;
        g_cache.f_valid = false;
        g_cache.F1.reset();
        g_cache.F2.reset();
    }
}

EMSCRIPTEN_KEEPALIVE
void wasm_set_ce_mode(int use_ce) {
    bool new_mode = (use_ce != 0);
    if (new_mode != g_use_ce) {
        g_use_ce = new_mode;
        g_cache.valid = false;
        g_cache.f_valid = false;
        g_cache.F1.reset();
        g_cache.F2.reset();
    }
}

EMSCRIPTEN_KEEPALIVE
void wasm_release_transfer_buffers() {
    std::vector<double>().swap(g_full_buf);
    std::vector<double>().swap(g_fkernel_buf);
}

// Light solve: bar solution + costs + wedges + residuals. ~5KB JSON.
// Used by Equilibrium tab and Diagnostics tab.
EMSCRIPTEN_KEEPALIVE
const char* solve_light(double p1, double p2, double b1, double b2, double r1, double r2) {
    g_output.clear();
    g_output.reserve(8 * 1024);
    ensure_solve(p1, p2, b1, b2, r1, r2);
    if (!g_cache.valid) {
        // Return a JSON error object so JS can display details
        out("{\"solver_error\":true");
        out(",\"failed_stage\":\"%s\"", g_cache.failed_stage.c_str());
        out(",\"error_message\":\"%s\"", g_cache.last_error.c_str());
        out(",\"p1\":%.6g,\"p2\":%.6g,\"b1\":%.6g,\"b2\":%.6g,\"r1\":%.6g,\"r2\":%.6g,\"terminal_weight\":%.6g",
            p1, p2, b1, b2, r1, r2, g_terminal_weight);
        out(",\"N\":%d,\"T\":%.6g,\"ce_mode\":%s", g_n, g_T, g_use_ce ? "true" : "false");
        out("}");
        return g_output.c_str();
    }
    serialize_light();
    return g_output.c_str();
}

// Binary full solve: writes 5 kernels × 3 channels × N×N doubles to buffer.
// Layout: [X_ch0[N×N], X_ch1[N×N], X_ch2[N×N], D1_ch0, ..., calD1_ch2, calD2_ch2]
// Each channel is row-major lower-triangular (upper entries are 0).
// JS reads directly from HEAPF64 — no JSON serialize/parse.

EMSCRIPTEN_KEEPALIVE
int full_kernel_count() { return 11; }

EMSCRIPTEN_KEEPALIVE
double* solve_full_bin(double p1, double p2, double b1, double b2, double r1, double r2) {
    ensure_solve(p1, p2, b1, b2, r1, r2);

    const int n = g_n;
    const int block = n * n;
    // 11 kernels × 3 channels × N×N
    g_full_buf.resize(11 * 3 * block);

    // rank-1 filter factors A[k][s] = dt gain^2 Xtilde[k][s] (s < k), formed here for export
    Kernel2D A1, A2;
    for (int k = 0; k < n; ++k) {
        for (int s = 0; s < k; ++s) { A1[k][s] = g_dt * g_cache.eq.env.obs_gain1 * g_cache.eq.env.obs_gain1 * g_cache.eq.env.Xtilde1[k][s]; A2[k][s] = g_dt * g_cache.eq.env.obs_gain2 * g_cache.eq.env.obs_gain2 * g_cache.eq.env.Xtilde2[k][s]; }
        A1[k][k].setZero(); A2[k][k].setZero();
    }
    const Kernel2D* kernels[11] = {
        &g_cache.eq.env.X, &g_cache.eq.D1, &g_cache.eq.D2, &g_cache.eq.calD1, &g_cache.eq.calD2,
        &g_cache.eq.env.Xtilde1, &g_cache.eq.env.Xtilde2,
        &g_cache.Vkernel1, &g_cache.Vkernel2,
        &A1, &A2
    };

    for (int k = 0; k < 11; ++k) {
        const Kernel2D& K = *kernels[k];
        for (int ch = 0; ch < 3; ++ch) {
            double* dst = g_full_buf.data() + (k * 3 + ch) * block;
            for (int ti = 0; ti < n; ++ti) {
                for (int si = 0; si <= ti; ++si)
                    dst[ti * n + si] = K[ti][si](ch);
                for (int si = ti + 1; si < n; ++si)
                    dst[ti * n + si] = 0.0;
            }
        }
    }

    return g_full_buf.data();
}

EMSCRIPTEN_KEEPALIVE
const char* solve_sweep(double p1, double b1, double b2, double r1, double r2,
                         const double* p2_vals, int n_p2) {
    g_output.clear();
    g_output.reserve(32 * 1024);

    SolverContext sweep_ctx = SolverContext::capture_current();
    sweep_ctx.b1 = b1;
    sweep_ctx.b2 = b2;
    sweep_ctx.r1 = r1;
    sweep_ctx.r2 = r2;
    ScopedSolverContext guard(sweep_ctx);

    std::array<double, N_MAX> barD1_pi, barD2_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barD2_pi, barX_pi);

    const auto& tg = t_grid();
    out("{");
    out_array("t", tg.data(), g_n);
    out(","); out_array("barD1_pi", barD1_pi.data(), g_n);
    out(","); out_array("barD2_pi", barD2_pi.data(), g_n);
    out(",\"p1\":%.6g,\"b1\":%.6g,\"b2\":%.6g", p1, b1, b2);

    out(",\"sweeps\":[");
    for (int i = 0; i < n_p2; ++i) {
        double p2 = p2_vals[i];
        if (i > 0) out(",");
        out("{\"p2\":%.6g", p2);

        double obs_gain1 = std::sqrt(p1);
        double obs_gain2 = std::sqrt(p2);
        auto eq = g_use_ce ? solve_equilibrium_ce(obs_gain1, obs_gain2, false)
                           : solve_equilibrium(obs_gain1, obs_gain2, false);
        auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                          p1, p2, 2000, 0.08, 1e-10);
        auto costs_priv = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                                 bar, r1, r2, b1, b2);
        out(","); out_array("barD1", bar.barD1.data(), g_n);
        out(","); out_array("barD2", bar.barD2.data(), g_n);
        out(",\"J1_priv\":%.6g,\"J2_priv\":%.6g", costs_priv.J1, costs_priv.J2);

        double p_common = p1 + p2;
        double obs_gain_common = std::sqrt(p_common);
        auto eq_pool = g_use_ce ? solve_equilibrium_ce(obs_gain_common, obs_gain_common, false,
                                                       Pi1(), 1, Pi1(), 1)
                                : solve_equilibrium(obs_gain_common, obs_gain_common, false,
                                                    Pi1(), 1, Pi1(), 1);
        auto bar_pool = solve_bar_equilibrium(eq_pool.env, eq_pool.D1, eq_pool.D2,
                                               p_common, p_common,
                                               2000, 0.08, 1e-10);
        auto costs_pool = compute_costs_general(eq_pool.env, eq_pool.calD1, eq_pool.calD2,
                                                 bar_pool, r1, r2, b1, b2);
        out(",\"J1_pool\":%.6g,\"J2_pool\":%.6g", costs_pool.J1, costs_pool.J2);

        auto prec2_arr = make_constant_prec(p2);
        auto prec1_arr = make_constant_prec(p1);
        auto bba1 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde2, eq.D2,
                                           bar.barX, b1, prec2_arr,
                                           eq.env.obs_gain2, eq.env.obs_idx2, g_terminal_weight);
        auto bba2 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde1, eq.D1,
                                           bar.barX, b2, prec1_arr,
                                           eq.env.obs_gain1, eq.env.obs_idx1, g_terminal_weight);
        std::array<double, N_MAX> V1_arr, V2_arr;
        for (int j = 0; j < g_n; ++j) {
            V1_arr[j] = mean_information_wedge_at(
                eq.env.Xtilde2, bba1.barHk, prec2_arr,
                eq.env.obs_gain2, eq.env.obs_idx2, j);
            V2_arr[j] = mean_information_wedge_at(
                eq.env.Xtilde1, bba2.barHk, prec1_arr,
                eq.env.obs_gain1, eq.env.obs_idx1, j);
        }
        out(","); out_array("V1", V1_arr.data(), g_n);
        out(","); out_array("V2", V2_arr.data(), g_n);
        out("}");
    }
    out("]}");

    return g_output.c_str();
}

// Single-point sweep: solve for one p2 value and return JSON for that point.
// Called repeatedly from JS with yields between calls to allow UI repaint.
EMSCRIPTEN_KEEPALIVE
const char* solve_sweep_point(double p1, double p2, double b1, double b2, double r1, double r2) {
    g_output.clear();
    g_output.reserve(8 * 1024);

    SolverContext point_ctx = SolverContext::capture_current();
    point_ctx.b1 = b1;
    point_ctx.b2 = b2;
    point_ctx.r1 = r1;
    point_ctx.r2 = r2;
    ScopedSolverContext guard(point_ctx);

    out("{\"p2\":%.6g", p2);

    double obs_gain1 = std::sqrt(p1);
    double obs_gain2 = std::sqrt(p2);
    auto eq = g_use_ce ? solve_equilibrium_ce(obs_gain1, obs_gain2, false)
                       : solve_equilibrium(obs_gain1, obs_gain2, false);
    auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                      p1, p2, 2000, 0.08, 1e-10);
    auto costs_priv = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                             bar, r1, r2, b1, b2);
    out(","); out_array("barD1", bar.barD1.data(), g_n);
    out(","); out_array("barD2", bar.barD2.data(), g_n);
    out(",\"J1_priv\":%.6g,\"J2_priv\":%.6g", costs_priv.J1, costs_priv.J2);

    double p_common = p1 + p2;
    double obs_gain_common = std::sqrt(p_common);
    auto eq_pool = g_use_ce ? solve_equilibrium_ce(obs_gain_common, obs_gain_common, false,
                                                   Pi1(), 1, Pi1(), 1)
                            : solve_equilibrium(obs_gain_common, obs_gain_common, false,
                                                Pi1(), 1, Pi1(), 1);
    auto bar_pool = solve_bar_equilibrium(eq_pool.env, eq_pool.D1, eq_pool.D2,
                                           p_common, p_common,
                                           2000, 0.08, 1e-10);
    auto costs_pool = compute_costs_general(eq_pool.env, eq_pool.calD1, eq_pool.calD2,
                                             bar_pool, r1, r2, b1, b2);
    out(",\"J1_pool\":%.6g,\"J2_pool\":%.6g", costs_pool.J1, costs_pool.J2);

    auto prec2_arr = make_constant_prec(p2);
    auto prec1_arr = make_constant_prec(p1);
    auto bba1 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde2, eq.D2,
                                       bar.barX, b1, prec2_arr,
                                       eq.env.obs_gain2, eq.env.obs_idx2, g_terminal_weight);
    auto bba2 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde1, eq.D1,
                                       bar.barX, b2, prec1_arr,
                                       eq.env.obs_gain1, eq.env.obs_idx1, g_terminal_weight);
    std::array<double, N_MAX> V1_arr, V2_arr;
    for (int j = 0; j < g_n; ++j) {
        V1_arr[j] = mean_information_wedge_at(
            eq.env.Xtilde2, bba1.barHk, prec2_arr,
            eq.env.obs_gain2, eq.env.obs_idx2, j);
        V2_arr[j] = mean_information_wedge_at(
            eq.env.Xtilde1, bba2.barHk, prec1_arr,
            eq.env.obs_gain1, eq.env.obs_idx1, j);
    }
    out(","); out_array("V1", V1_arr.data(), g_n);
    out(","); out_array("V2", V2_arr.data(), g_n);
    out("}");

    return g_output.c_str();
}

// Binary F kernel: writes 2 * 9 * g_n * g_n doubles to a static buffer.
// Layout: [F1_r0c0[0..n*n], F1_r0c1[..], ..., F1_r2c2[..], F2_r0c0[..], ..., F2_r2c2[..]]
// Each block is row-major: F[u*n + s] for u=0..n-1, s=0..n-1.
// JS reads directly from HEAPF64 — no JSON serialization or parsing.

EMSCRIPTEN_KEEPALIVE
double* solve_f_kernel_bin(double p1, double p2, double b1, double b2, double r1, double r2) {
    ensure_f_kernel(p1, p2, b1, b2, r1, r2);  // cached — no re-solve if params match

    const int n = g_n;
    const int block = n * n;
    g_fkernel_buf.resize(2 * 9 * block);

    // Write F1 then F2
    const FSlice* slices[2] = { g_cache.F1.get(), g_cache.F2.get() };
    for (int p = 0; p < 2; ++p) {
        const FSlice& F = *slices[p];
        double* base = g_fkernel_buf.data() + p * 9 * block;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                double* dst = base + (r * 3 + c) * block;
                for (int u = 0; u < n; ++u)
                    for (int s = 0; s < n; ++s)
                        dst[u * n + s] = F(u, s)(r, c);
            }
        }
    }

    return g_fkernel_buf.data();
}


} // extern "C"
