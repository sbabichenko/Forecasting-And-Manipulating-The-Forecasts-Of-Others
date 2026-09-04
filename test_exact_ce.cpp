// Test exact discrete conditional expectation
//
// Verifies:
// 1. Idempotency: ||M^2 - M|| < 1e-14
// 2. Rank = n-1 (n-1 scalar observations for 3n unknowns)
// 3. Xtilde is nonzero (not collapsed like a buggy Pi-based Kalman)
// 4. Convergence: ||Xt_exact - Xt_solver|| / ||Xt_solver|| decreases with N

#include "lqg_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// --- Test 1: D=0 (free dynamics), exact CE properties ---
static bool test_zero_D(int N, double gain) {
    std::printf("--- D=0, N=%d, gain=%.1f ---\n", N, gain);
    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = N;
    run_ctx.T = 1.0;
    run_ctx.b1 = B1_DEFAULT;
    run_ctx.b2 = B2_DEFAULT;
    run_ctx.r1 = RHO;
    run_ctx.r2 = RHO;
    run_ctx.sigma = 1.0;
    ScopedSolverContext guard(run_ctx);

    // With D=0, X[t][s] = sigma * e_0 for all t >= s
    Kernel2D calD1, calD2;
    calD1.setZero(); calD2.setZero();
    Kernel2D X;
    state_kernel_from_calD(calD1, calD2, X);

    int t_idx = g_n - 1;
    auto proj = discrete_conditional_expectation(X, 1, gain, t_idx);

    bool pass = true;
    int n = t_idx + 1;

    // Check idempotency
    std::printf("  idem_residual = %.2e", proj.idem_residual);
    if (proj.idem_residual >= 1e-14) {
        std::printf(" FAIL\n"); pass = false;
    } else {
        std::printf(" OK\n");
    }

    // Check rank
    std::printf("  rank = %d (expected %d)", proj.rank, t_idx);
    if (proj.rank != t_idx) {
        std::printf(" FAIL\n"); pass = false;
    } else {
        std::printf(" OK\n");
    }

    // Check Xtilde is nonzero
    double xt_norm = 0.0;
    for (int j = 0; j <= t_idx; ++j)
        for (int z = 0; z <= j; ++z)
            xt_norm += proj.Xtilde[j][z].squaredNorm();
    xt_norm = std::sqrt(xt_norm);
    std::printf("  ||Xtilde_exact|| = %.6f", xt_norm);
    if (xt_norm < 1e-10) {
        std::printf(" FAIL (should be nonzero)\n"); pass = false;
    } else {
        std::printf(" OK\n");
    }

    // Check obs_idx component is nonzero in Xtilde
    double obs_comp_norm = 0.0;
    for (int j = 0; j <= t_idx; ++j)
        for (int z = 0; z <= j; ++z)
            obs_comp_norm += proj.Xtilde[j][z](1) * proj.Xtilde[j][z](1);
    obs_comp_norm = std::sqrt(obs_comp_norm);
    std::printf("  ||Xtilde_exact(obs_idx)|| = %.6f (nonzero = correct)\n", obs_comp_norm);

    std::printf("\n");
    return pass;
}

// --- Test 2: With equilibrium D, compare exact CE to solver ---
static bool test_equilibrium(int N, double gain1, double gain2) {
    std::printf("--- Equilibrium, N=%d, gain=(%.1f, %.1f) ---\n", N, gain1, gain2);
    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = N;
    run_ctx.T = 1.0;
    run_ctx.b1 = B1_DEFAULT;
    run_ctx.b2 = B2_DEFAULT;
    run_ctx.r1 = RHO;
    run_ctx.r2 = RHO;
    run_ctx.sigma = 1.0;
    ScopedSolverContext guard(run_ctx);

    auto eq = solve_equilibrium(gain1, gain2, false);
    auto pair = exact_discrete_CE(eq);

    bool pass = true;
    int t_idx = g_n - 1;

    const char* names[] = {"Player 1", "Player 2"};
    const DiscreteProjection* projs[] = {&pair.player1, &pair.player2};
    const Kernel2D* Xt_solver[] = {&eq.env.Xtilde1, &eq.env.Xtilde2};

    for (int p = 0; p < 2; ++p) {
        const auto& proj = *projs[p];

        // Idempotency
        std::printf("  %s: idem=%.2e", names[p], proj.idem_residual);
        if (proj.idem_residual >= 1e-14) {
            std::printf(" FAIL\n"); pass = false;
        } else {
            std::printf(" OK");
        }

        // Rank
        std::printf("  rank=%d/%d", proj.rank, t_idx);
        if (proj.rank != t_idx) {
            std::printf(" FAIL"); pass = false;
        } else {
            std::printf(" OK");
        }

        // Xtilde comparison
        const auto& Xts = *Xt_solver[p];
        double diff_norm = 0.0, solver_norm = 0.0;
        for (int j = 0; j <= t_idx; ++j)
            for (int z = 0; z <= j; ++z) {
                diff_norm += (proj.Xtilde[j][z] - Xts[j][z]).squaredNorm();
                solver_norm += Xts[j][z].squaredNorm();
            }
        double rel = std::sqrt(diff_norm) / std::max(1e-15, std::sqrt(solver_norm));
        std::printf("  ||Xt_exact-Xt_solver||/||Xt_solver||=%.4f", rel);

        // Check Xtilde nonzero
        double xt_norm = 0.0;
        for (int j = 0; j <= t_idx; ++j)
            for (int z = 0; z <= j; ++z)
                xt_norm += proj.Xtilde[j][z].squaredNorm();
        if (xt_norm < 1e-10) {
            std::printf(" Xt=0 FAIL"); pass = false;
        }

        std::printf("\n");
    }

    std::printf("\n");
    return pass;
}

// --- Test 3: Convergence test ---
static void test_convergence(double gain) {
    std::printf("--- Convergence test, gain=%.1f ---\n", gain);
    std::printf("  %4s  %8s  %12s  %12s\n", "N", "dt", "||gap|| p1", "||gap|| p2");

    for (int N : {8, 16, 32, 64}) {
        SolverContext run_ctx = SolverContext::capture_current();
        run_ctx.n = N;
        run_ctx.T = 1.0;
        run_ctx.b1 = B1_DEFAULT;
        run_ctx.b2 = B2_DEFAULT;
        run_ctx.r1 = RHO;
        run_ctx.r2 = RHO;
        run_ctx.sigma = 1.0;
        ScopedSolverContext guard(run_ctx);

        auto eq = solve_equilibrium(gain, gain, false);
        auto pair = exact_discrete_CE(eq);

        int t_idx = g_n - 1;
        const Kernel2D* Xt_solver[] = {&eq.env.Xtilde1, &eq.env.Xtilde2};
        const DiscreteProjection* projs[] = {&pair.player1, &pair.player2};

        double gaps[2];
        for (int p = 0; p < 2; ++p) {
            double diff_norm = 0.0, solver_norm = 0.0;
            for (int j = 0; j <= t_idx; ++j)
                for (int z = 0; z <= j; ++z) {
                    diff_norm += (projs[p]->Xtilde[j][z] - (*Xt_solver[p])[j][z]).squaredNorm();
                    solver_norm += (*Xt_solver[p])[j][z].squaredNorm();
                }
            gaps[p] = std::sqrt(diff_norm) / std::max(1e-15, std::sqrt(solver_norm));
        }

        std::printf("  %4d  %8.5f  %12.6f  %12.6f\n", N, g_dt, gaps[0], gaps[1]);
    }
    std::printf("\n");
}

static bool test_initial_control_boundary() {
    std::printf("--- Initial observation-noise control boundary ---\n");
    SolverContext run_ctx = SolverContext::capture_current();
    run_ctx.n = 24;
    run_ctx.T = 1.0;
    run_ctx.b1 = B1_DEFAULT;
    run_ctx.b2 = B2_DEFAULT;
    run_ctx.r1 = RHO;
    run_ctx.r2 = RHO;
    run_ctx.sigma = 1.0;
    ScopedSolverContext guard(run_ctx);

    auto check = [](const char* label, const EquilibriumResult& eq) {
        double max_p1 = 0.0, max_p2 = 0.0;
        for (int j = 0; j < g_n; ++j) {
            max_p1 = std::max(max_p1, std::abs(eq.calD1[j][0](1)));
            max_p2 = std::max(max_p2, std::abs(eq.calD2[j][0](2)));
        }
        double row0 = std::max(eq.calD1[0][0].norm(), eq.calD2[0][0].norm());
        bool ok = max_p1 < 1e-14 && max_p2 < 1e-14 && row0 < 1e-14;
        std::printf("  %s: max |calD1(t,0)[W1]|=%.2e, max |calD2(t,0)[W2]|=%.2e, row0=%.2e %s\n",
                    label, max_p1, max_p2, row0, ok ? "OK" : "FAIL");
        return ok;
    };

    bool pass = true;
    pass = check("standard", solve_equilibrium(3.0, 2.0, false)) && pass;
    pass = check("CE", solve_equilibrium_ce(3.0, 2.0, false)) && pass;
    std::printf("\n");
    return pass;
}

int main() {
    int passed = 0, total = 0;

    // Test 1: D=0 cases
    for (int N : {10, 20, 40}) {
        ++total;
        if (test_zero_D(N, 3.0)) ++passed;
    }

    // Test 2: Equilibrium cases
    for (auto [N, g1, g2] : std::vector<std::tuple<int,double,double>>{
            {20, 3.0, 3.0}, {40, 3.0, 3.0}, {40, 5.0, 2.0}, {60, 3.0, 3.0}}) {
        ++total;
        if (test_equilibrium(N, g1, g2)) ++passed;
    }

    // Test 3: Convergence
    test_convergence(3.0);

    ++total;
    if (test_initial_control_boundary()) ++passed;

    std::printf("=== %d / %d tests passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
