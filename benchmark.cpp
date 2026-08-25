// Quick benchmark to profile solver phases at different N values.
// Build: cd build && cmake .. && make benchmark
// Run:   ./build/benchmark

#include "lqg_solver.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static void bench(int n, double T, double p1, double p2, double b1, double b2, double r1, double r2) {
    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = n;
    run_ctx.T = T;
    run_ctx.b1 = b1;
    run_ctx.b2 = b2;
    run_ctx.r1 = r1;
    run_ctx.r2 = r2;
    ScopedSolverContext guard(run_ctx);
    printf("\n=== N=%d, T=%.1f, p1=%.1f, p2=%.1f ===\n", n, T, p1, p2);

    // 1. solve_equilibrium
    auto t0 = Clock::now();
    double obs_gain1 = std::sqrt(p1);
    double obs_gain2 = std::sqrt(p2);
    auto eq = solve_equilibrium(obs_gain1, obs_gain2, false);
    double eq_ms = ms_since(t0);
    printf("  solve_equilibrium:    %7.1f ms  (%zu iters)\n", eq_ms, eq.residuals.size());

    // 2. solve_bar_equilibrium
    t0 = Clock::now();
    auto bar = solve_bar_equilibrium(eq.env, eq.D1, eq.D2, p1, p2, 2000, 0.08, 1e-10);
    double bar_ms = ms_since(t0);
    printf("  solve_bar_equilibrium:%7.1f ms\n", bar_ms);

    // 3. compute_costs_general
    t0 = Clock::now();
    auto costs = compute_costs_general(eq.env, eq.calD1, eq.calD2, bar, r1, r2, b1, b2);
    double cost_ms = ms_since(t0);
    printf("  compute_costs:        %7.1f ms\n", cost_ms);

    // 4. wedges (backward_bar_adjoints)
    auto prec1 = make_constant_prec(p1);
    auto prec2 = make_constant_prec(p2);
    t0 = Clock::now();
    auto bba1 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde2, eq.D2,
                                      bar.barX, b1, prec2,
                                      eq.env.obs_gain2, eq.env.obs_idx2, g_terminal_weight);
    auto bba2 = backward_bar_adjoints(eq.env.X, eq.env.Xtilde1, eq.D1,
                                      bar.barX, b2, prec1,
                                      eq.env.obs_gain1, eq.env.obs_idx1, g_terminal_weight);
    double wedge_ms = ms_since(t0);
    printf("  wedges (bba x2):      %7.1f ms\n", wedge_ms);

    // 5. compute_F_slice_at_T (both players)
    t0 = Clock::now();
    auto F1 = compute_F_slice_at_T(eq.env.Xtilde1, eq.env.obs_gain1, eq.env.obs_idx1);
    double f1_ms = ms_since(t0);
    t0 = Clock::now();
    auto F2 = compute_F_slice_at_T(eq.env.Xtilde2, eq.env.obs_gain2, eq.env.obs_idx2);
    double f2_ms = ms_since(t0);
    printf("  F_slice player 1:     %7.1f ms\n", f1_ms);
    printf("  F_slice player 2:     %7.1f ms\n", f2_ms);

    // 6. Serialization cost: simulate JSON output for kernel2d
    std::string out;
    out.reserve(1024 * 1024);
    char buf[32];
    t0 = Clock::now();
    for (const Kernel2D* K : {&eq.env.X, &eq.D1, &eq.D2, &eq.calD1}) {
        for (int ch = 0; ch < 3; ++ch) {
            for (int ti = 0; ti < g_n; ++ti) {
                for (int si = 0; si < g_n; ++si) {
                    double v = (si <= ti) ? (*K)[ti][si](ch) : 0.0;
                    int len = snprintf(buf, sizeof(buf), "%.6g", v);
                    out.append(buf, len);
                    out += ',';
                }
            }
        }
    }
    double serial_ms = ms_since(t0);
    printf("  kernel2d JSON serial: %7.1f ms  (%zu KB)\n", serial_ms, out.size() / 1024);

    // 7. Serialization cost: F kernel binary write
    out.clear();
    int block = g_n * g_n;
    std::vector<double> fbuf(2 * 9 * block);
    t0 = Clock::now();
    const FSlice* slices[2] = {F1.get(), F2.get()};
    for (int p = 0; p < 2; ++p) {
        const FSlice& F = *slices[p];
        double* base = fbuf.data() + p * 9 * block;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                double* dst = base + (row * 3 + col) * block;
                for (int u = 0; u < g_n; ++u)
                    for (int s = 0; s < g_n; ++s)
                        dst[u * g_n + s] = F(u, s)(row, col);
            }
    }
    double fbin_ms = ms_since(t0);
    printf("  F kernel binary write:%7.1f ms  (%zu KB)\n", fbin_ms, fbuf.size() * 8 / 1024);

    // 8. Simulate F kernel JSON serialization for comparison
    out.clear();
    out.reserve(1024 * 1024);
    t0 = Clock::now();
    for (int p = 0; p < 2; ++p) {
        const FSlice& F = *slices[p];
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                for (int u = 0; u < g_n; ++u)
                    for (int s = 0; s < g_n; ++s) {
                        int len = snprintf(buf, sizeof(buf), "%.6g", F(u, s)(row, col));
                        out.append(buf, len);
                        out += ',';
                    }
    }
    double fjson_ms = ms_since(t0);
    printf("  F kernel JSON serial: %7.1f ms  (%zu KB)\n", fjson_ms, out.size() / 1024);

    double total = eq_ms + bar_ms + cost_ms + wedge_ms + f1_ms + f2_ms;
    printf("  TOTAL computation:    %7.1f ms\n", total);
    printf("  J1=%.4f J2=%.4f\n", costs.J1, costs.J2);

    // Report peak memory from /proc
    std::ifstream proc("/proc/self/status");
    std::string line;
    while (std::getline(proc, line)) {
        if (line.find("VmPeak") != std::string::npos ||
            line.find("VmRSS") != std::string::npos) {
            printf("  %s\n", line.c_str());
        }
    }
}

int main() {
    // Quick single test matching solve_interactive
    bench(40, 1.0, 3.0, 3.0, 1.0, -1.0, 0.1, 0.1);

    // Test larger N to verify dynamic sizing
    bench(160, 1.0, 3.0, 3.0, 1.0, -1.0, 0.1, 0.1);

    return 0;
}
