// CLI tool for the interactive Dash app.
//
// Modes:
//   solve_interactive single <p1> <p2> <b1> <b2> <r1> <r2> [qT] [--ce] [--N <N>] [--T <T>]
//     → JSON with kernels, bar solution, wedges, costs for one (p1, p2)
//
//   solve_interactive sweep <p1> <b1> <b2> <r1> <r2> [--qT <qT>] [--ce] [--N <N>] [--T <T>] <p2_0> ...
//     → JSON array: for each p2, private and pooled costs + barD1 curves
//
//   solve_interactive check <p1> <p2> <r1> <r2> <qT> [--ce] [--N <N>] [--T <T>]
//     → compact JSON convergence diagnostic for parameter sweeps

#include "lqg_solver.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void print_array(const char* name, const double* arr, int n) {
    printf("\"%s\":[", name);
    for (int i = 0; i < n; ++i)
        printf("%s%.12g", i ? "," : "", arr[i]);
    printf("]");
}

static void print_val(double v) {
    if (std::isfinite(v))
        printf("%.12g", v);
    else
        printf("null");
}

static void print_kernel2d(const char* name, const Kernel2D& K) {
    printf("\"%s\":{", name);
    bool first = true;
    for (int ch = 0; ch < 3; ++ch) {
        printf("%s\"ch%d\":[", first ? "" : ",", ch);
        first = false;
        bool first_row = true;
        for (int ti = 0; ti < g_n; ++ti) {
            printf("%s[", first_row ? "" : ",");
            first_row = false;
            for (int si = 0; si <= ti; ++si)
                printf("%s%.12g", si ? "," : "", K[ti][si](ch));
            for (int si = ti + 1; si < g_n; ++si)
                printf(",0");
            printf("]");
        }
        printf("]");
    }
    printf("}");
}

static double final_residual(const std::vector<double>& residuals) {
    return residuals.empty() ? std::numeric_limits<double>::quiet_NaN() : residuals.back();
}

static bool residual_converged(const std::vector<double>& residuals) {
    double r = final_residual(residuals);
    return std::isfinite(r) && r < PICARD_TOL;
}

static int parse_optional_flags(int argc, char* argv[], int start,
                                bool& use_ce, int& n, double& T) {
    int consumed = 0;
    for (int i = start; i < argc; ++i) {
        if (strcmp(argv[i], "--ce") == 0) {
            use_ce = true;
            ++consumed;
        } else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
            consumed += 2;
        } else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
            T = atof(argv[++i]);
            consumed += 2;
        }
    }
    return consumed;
}

// Perfect-info Riccati solution (depends on b1, b2 for the bar dynamics)
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

static int run_single(int argc, char* argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s single <p1> <p2> <b1> <b2> <r1> <r2> [qT] [--ce] [--N <N>] [--T <T>]\n", argv[0]);
        return 1;
    }
    double p1_prec = atof(argv[2]), p2_prec = atof(argv[3]);
    double obs_gain1 = std::sqrt(p1_prec), obs_gain2 = std::sqrt(p2_prec);
    double b1 = atof(argv[4]), b2 = atof(argv[5]);
    double r1 = atof(argv[6]), r2 = atof(argv[7]);
    double qT = 0.0;
    bool use_ce = false;
    int n = g_n;
    double T = g_T;
    for (int i = 8; i < argc; ++i) {
        if (strcmp(argv[i], "--ce") == 0)
            use_ce = true;
        else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc)
            T = atof(argv[++i]);
        else
            qT = atof(argv[i]);
    }
    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = n;
    run_ctx.T = T;
    run_ctx.b1 = b1;
    run_ctx.b2 = b2;
    run_ctx.r1 = r1;
    run_ctx.r2 = r2;
    run_ctx.terminal_weight = qT;
    ScopedSolverContext guard(run_ctx);

    auto eq = use_ce ? solve_equilibrium_ce(obs_gain1, obs_gain2, false)
                     : solve_equilibrium(obs_gain1, obs_gain2, false);
    auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                      p1_prec, p2_prec, 2000, 0.08, 1e-10);
    auto costs = compute_costs_general(eq.env, eq.calD1, eq.calD2, bar, r1, r2, b1, b2);

    // Wedges
    auto prec1_arr = make_constant_prec(p1_prec);
    auto prec2_arr = make_constant_prec(p2_prec);
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

    std::array<double, N_MAX> barD1_pi, barD2_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barD2_pi, barX_pi);

    const auto& tg = t_grid();
    printf("{");
    print_array("t", tg.data(), g_n);

    printf(",\"residuals\":[");
    for (size_t i = 0; i < eq.residuals.size(); ++i)
        printf("%s%.12g", i ? "," : "", eq.residuals[i]);
    printf("],\"n_iters\":%zu", eq.residuals.size());

    printf(","); print_kernel2d("X", eq.env.X);
    printf(","); print_kernel2d("D1", eq.D1);
    printf(","); print_kernel2d("D2", eq.D2);
    printf(","); print_kernel2d("calD1", eq.calD1);
    printf(","); print_kernel2d("calD2", eq.calD2);
    printf(","); print_kernel2d("Xtilde1", eq.env.Xtilde1);
    printf(","); print_kernel2d("Xtilde2", eq.env.Xtilde2);

    printf(","); print_array("barD1", bar.barD1.data(), g_n);
    printf(","); print_array("barD2", bar.barD2.data(), g_n);
    printf(","); print_array("barX", bar.barX.data(), g_n);
    printf(",\"bar_residual\":%.12g", bar.bar_residual);

    printf(","); print_array("V1", V1_arr.data(), g_n);
    printf(","); print_array("V2", V2_arr.data(), g_n);

    printf(","); print_array("barD1_pi", barD1_pi.data(), g_n);
    printf(","); print_array("barD2_pi", barD2_pi.data(), g_n);
    printf(","); print_array("barX_pi", barX_pi.data(), g_n);

    printf(",\"J1\":%.12g,\"J2\":%.12g", costs.J1, costs.J2);
    printf(",\"p1\":%.12g,\"p2\":%.12g,\"b1\":%.12g,\"b2\":%.12g",
           p1_prec, p2_prec, b1, b2);
    printf(",\"r1\":%.12g,\"r2\":%.12g,\"N\":%d,\"T\":%.12g",
           r1, r2, g_n, g_T);
    printf(",\"obs_gain1\":%.12g,\"obs_gain2\":%.12g", obs_gain1, obs_gain2);
    printf(",\"terminal_weight\":%.12g", g_terminal_weight);
    printf(",\"ce_mode\":%s", use_ce ? "true" : "false");
    printf("}\n");
    return 0;
}

static int run_check(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s check <p1> <p2> <r1> <r2> <qT> [--ce] [--N <N>] [--T <T>]\n", argv[0]);
        return 1;
    }
    double p1_prec = atof(argv[2]), p2_prec = atof(argv[3]);
    double r1 = atof(argv[4]), r2 = atof(argv[5]);
    double qT = atof(argv[6]);
    bool use_ce = false;
    int n = g_n;
    double T = g_T;
    parse_optional_flags(argc, argv, 7, use_ce, n, T);

    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = n;
    run_ctx.T = T;
    run_ctx.b1 = B1_DEFAULT;
    run_ctx.b2 = B2_DEFAULT;
    run_ctx.r1 = r1;
    run_ctx.r2 = r2;
    run_ctx.terminal_weight = qT;
    ScopedSolverContext guard(run_ctx);

    double obs_gain1 = std::sqrt(p1_prec), obs_gain2 = std::sqrt(p2_prec);
    auto eq = use_ce ? solve_equilibrium_ce(obs_gain1, obs_gain2, false)
                     : solve_equilibrium(obs_gain1, obs_gain2, false);
    const double final = final_residual(eq.residuals);
    const bool eq_ok = residual_converged(eq.residuals);

    double bar_res = std::numeric_limits<double>::quiet_NaN();
    double J1 = std::numeric_limits<double>::quiet_NaN();
    double J2 = std::numeric_limits<double>::quiet_NaN();
    bool bar_ok = false;
    bool costs_ok = false;
    if (eq_ok) {
        auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                          p1_prec, p2_prec, 2000, 0.08, 1e-10);
        bar_res = bar.bar_residual;
        bar_ok = std::isfinite(bar_res) && bar_res < 1e-8;
        auto costs = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                           bar, r1, r2, B1_DEFAULT, B2_DEFAULT);
        J1 = costs.J1;
        J2 = costs.J2;
        costs_ok = std::isfinite(J1) && std::isfinite(J2);
    }

    printf("{\"p1\":%.12g,\"p2\":%.12g,\"r1\":%.12g,\"r2\":%.12g,\"qT\":%.12g,\"N\":%d,\"T\":%.12g",
           p1_prec, p2_prec, r1, r2, qT, g_n, g_T);
    printf(",\"ce_mode\":%s,\"n_iters\":%zu,\"final_residual\":", use_ce ? "true" : "false", eq.residuals.size());
    print_val(final);
    printf(",\"eq_converged\":%s,\"bar_residual\":", eq_ok ? "true" : "false");
    print_val(bar_res);
    printf(",\"bar_converged\":%s,\"J1\":", bar_ok ? "true" : "false");
    print_val(J1);
    printf(",\"J2\":");
    print_val(J2);
    printf(",\"costs_finite\":%s,\"ok\":%s}\n",
           costs_ok ? "true" : "false",
           (eq_ok && bar_ok && costs_ok) ? "true" : "false");
    return 0;
}

static int run_sweep(int argc, char* argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s sweep <p1> <b1> <b2> <r1> <r2> [--qT <qT>] [--ce] [--N <N>] [--T <T>] <p2_0> [p2_1 ...]\n", argv[0]);
        return 1;
    }
    double p1_prec = atof(argv[2]);
    double obs_gain1 = std::sqrt(p1_prec);
    double b1 = atof(argv[3]), b2 = atof(argv[4]);
    double r1 = atof(argv[5]), r2 = atof(argv[6]);
    double qT = 0.0;
    bool use_ce = false;
    int n = g_n;
    double T = g_T;
    std::vector<double> p2_vals;
    for (int i = 7; i < argc; ++i) {
        if (strcmp(argv[i], "--qT") == 0 && i + 1 < argc) {
            qT = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ce") == 0) {
            use_ce = true;
        } else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
            T = atof(argv[++i]);
        } else {
            p2_vals.push_back(atof(argv[i]));
        }
    }
    if (p2_vals.empty()) {
        fprintf(stderr, "Usage: %s sweep <p1> <b1> <b2> <r1> <r2> [--qT <qT>] [--ce] [--N <N>] [--T <T>] <p2_0> [p2_1 ...]\n", argv[0]);
        return 1;
    }
    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = n;
    run_ctx.T = T;
    run_ctx.b1 = b1;
    run_ctx.b2 = b2;
    run_ctx.r1 = r1;
    run_ctx.r2 = r2;
    run_ctx.terminal_weight = qT;
    ScopedSolverContext guard(run_ctx);

    int n_p2 = static_cast<int>(p2_vals.size());

    std::array<double, N_MAX> barD1_pi, barD2_pi, barX_pi;
    compute_perfect_info(b1, b2, barD1_pi, barD2_pi, barX_pi);

    const auto& tg = t_grid();
    printf("{");
    print_array("t", tg.data(), g_n);
    printf(","); print_array("barD1_pi", barD1_pi.data(), g_n);
    printf(","); print_array("barD2_pi", barD2_pi.data(), g_n);
    printf(",\"p1\":%.12g,\"b1\":%.12g,\"b2\":%.12g", p1_prec, b1, b2);
    printf(",\"r1\":%.12g,\"r2\":%.12g,\"N\":%d,\"T\":%.12g",
           r1, r2, g_n, g_T);
    printf(",\"terminal_weight\":%.12g", g_terminal_weight);
    printf(",\"ce_mode\":%s", use_ce ? "true" : "false");

    printf(",\"sweeps\":[");
    for (int i = 0; i < n_p2; ++i) {
        double p2_prec = p2_vals[i];
        double obs_gain2 = std::sqrt(p2_prec);
        if (i > 0) printf(",");
        printf("{\"p2\":%.12g", p2_prec);

        // Private equilibrium
        auto eq = use_ce ? solve_equilibrium_ce(obs_gain1, obs_gain2, false)
                         : solve_equilibrium(obs_gain1, obs_gain2, false);
        auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2,
                                          p1_prec, p2_prec, 2000, 0.08, 1e-10);
        auto costs_priv = compute_costs_general(eq.env, eq.calD1, eq.calD2,
                                                 bar, r1, r2, b1, b2);

        printf(","); print_array("barD1", bar.barD1.data(), g_n);
        printf(","); print_array("barD2", bar.barD2.data(), g_n);
        printf(","); print_array("barX", bar.barX.data(), g_n);
        printf(",\"J1_priv\":%.12g,\"J2_priv\":%.12g", costs_priv.J1, costs_priv.J2);
        printf(",\"n_iters\":%zu", eq.residuals.size());

        // Pooled equilibrium: both players see same signal through Pi1
        double p_common = p1_prec + p2_prec;
        double obs_gain_common = std::sqrt(p_common);
        auto eq_pool = use_ce ? solve_equilibrium_ce(obs_gain_common, obs_gain_common, false,
                                                     Pi1(), 1, Pi1(), 1)
                              : solve_equilibrium(obs_gain_common, obs_gain_common, false,
                                                  Pi1(), 1, Pi1(), 1);
        auto bar_pool = solve_bar_equilibrium(eq_pool.env, eq_pool.D1, eq_pool.D2,
                                               p_common, p_common,
                                               2000, 0.08, 1e-10);
        auto costs_pool = compute_costs_general(eq_pool.env, eq_pool.calD1, eq_pool.calD2,
                                                 bar_pool, r1, r2, b1, b2);

        printf(",\"J1_pool\":%.12g,\"J2_pool\":%.12g", costs_pool.J1, costs_pool.J2);
        printf(",\"p_common\":%.12g", p_common);

        // Wedges (private)
        auto prec2_arr = make_constant_prec(p2_prec);
        auto prec1_arr = make_constant_prec(p1_prec);
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
        printf(","); print_array("V1", V1_arr.data(), g_n);
        printf(","); print_array("V2", V2_arr.data(), g_n);

        printf("}");
    }
    printf("]}\n");
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize grid to default (N=40, T=1.0) — matches previous compile-time constants
    SolverContext init_ctx = SolverContext::capture_current();
    init_ctx.n = 40;
    init_ctx.T = 1.0;
    init_ctx.apply();

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <single|sweep|check> ...\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "sweep") == 0)
        return run_sweep(argc, argv);
    if (strcmp(argv[1], "single") == 0)
        return run_single(argc, argv);
    if (strcmp(argv[1], "check") == 0)
        return run_check(argc, argv);

    // Legacy: positional args without subcommand
    if (argc >= 5) {
        // Shift args to match single mode
        char* new_argv[] = {argv[0], (char*)"single", argv[1], argv[2], argv[3], argv[4]};
        return run_single(6, new_argv);
    }
    fprintf(stderr, "Usage: %s <single|sweep|check> ...\n", argv[0]);
    return 1;
}
