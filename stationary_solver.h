#pragma once

#include "lqg_solver.h"
#include <vector>

struct StationaryParams {
    // Paper-level signal precisions p_i. Observation gains are sqrt(p_i).
    double p1 = 3.0;
    double p2 = 3.0;
    double r1 = RHO;
    double r2 = RHO;
    double sigma = 1.0;
    double A = 0.0;

    int n_lag = 25;          // one-sided lag grid points, including 0 and lag_max
    double lag_max = 3.0;    // truncation L for a in [0,L] and b in [-L,L]
    bool use_simpson_quadrature = true;

    int max_iters = 500;
    double relax = 0.003;
    double tol = 1e-5;
    double abs_tol = 1e-5;

    int forward_iters = 500;
    double forward_relax = 0.05;
    double forward_tol = 1e-8;
    bool inexact_forward = false;
    bool newton_krylov = false;     // JFNK outer iteration instead of Anderson/relaxation
    int newton_gmres = 25;          // max GMRES iterations per Newton step   // loosen forward_tol while the outer residual is large

    int backward_iters = 500;
    double backward_relax = 0.05;
    double backward_tol = 1e-8;

    // Forward block solver: false = relaxed Picard on the filter kernels
    // (historical default), true = exact per-observer projection solve
    // (dense least squares; unconditionally stable, no forward_relax needed).
    bool exact_forward = false;

    // Anderson acceleration for the outer policy iteration. depth 0 keeps the
    // plain relaxed update; depth m mixes the last m residual differences with
    // the given mixing parameter once the relative residual is below 0.5.
    int anderson_depth = 0;
    double anderson_mixing = 0.6;

    // Forward Gram solves: false = dense Cholesky with refinement cache,
    // true = FFT-based PCG with a Strang circulant preconditioner.
    bool circulant_cg = false;

    // Optional warm start: policy kernels sampled on init_lag (one-sided lags,
    // ascending). When non-empty, linearly interpolated onto the solver grid in
    // place of the closed-form certainty-equivalent initialization.
    std::vector<double> init_lag;
    std::vector<Vec3> init_d1;
    std::vector<Vec3> init_d2;
};

struct StationarySolution {
    StationaryParams params;
    double h = 0.0;
    bool converged = false;
    double residual = 0.0;
    double relative_residual = 0.0;
    double absolute_residual = 0.0;
    double forward_residual = 0.0;
    double backward_residual1 = 0.0;
    double backward_residual2 = 0.0;
    std::vector<double> residuals;
    std::vector<double> relative_residuals;
    std::vector<double> absolute_residuals;

    // One-sided lag grid a >= 0.
    std::vector<double> lag;
    std::vector<Vec3> x;
    std::vector<Vec3> xhat1;
    std::vector<Vec3> xhat2;
    std::vector<Vec3> xtilde1;
    std::vector<Vec3> xtilde2;
    std::vector<Vec3> d1;
    std::vector<Vec3> d2;
    std::vector<Vec3> calD1;
    std::vector<Vec3> calD2;

    // Two-sided primitive-shock lag b in [-L,L].
    std::vector<double> b_lag;
    std::vector<Vec3> hx1;
    std::vector<Vec3> hx2;
    std::vector<Vec3> wedge1;
    std::vector<Vec3> wedge2;
};

StationarySolution solve_stationary(const StationaryParams& params,
                                    bool verbose = false);
