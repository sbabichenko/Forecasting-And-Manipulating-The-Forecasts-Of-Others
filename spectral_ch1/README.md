# Spectral-in-time solver for the Chapter 1 finite-horizon game (prototype, JAX)

`spec_ch1.cpp` (C++/Eigen, `g++ -O3 -march=native -fopenmp spec_ch1.cpp`; usage in the header) and
`spec_ch1.py` (JAX prototype) — kernels on the triangle 0 <= s <= t <= T in Duffy coordinates (t, theta = s/t),
Chebyshev-Lobatto nodes in both, barycentric interpolation, Clenshaw-Curtis / Gauss-Legendre
quadrature.  Player i's control is parametrized by its coefficient on the player's own
observation increments, g^i_t(u), so the primitive-noise control is
calD^i_t(s) = sqrt(p_i) int_s^t g^i_t(u) X_u(s) du + g^i_t(s) e_i with no projection, and the
state kernel solves a LINEAR Volterra system X_t(s) = sigma e_0 + int_s^t (calD^1 + calD^2)(u, s) du.
Costs (variance part) by triangle quadrature; the first-order conditions grad_{g^i} J_i = 0 by
reverse-mode AD; Newton with a forward-mode Jacobian built in chunks (a single jacfwd exhausts
the GPU above ~600 unknowns).  Run: `.venv/bin/python spec_ch1.py Nt Nth m`.

Findings (2026-08-25, benchmark p = (3, 3), r = 0.1, T = 1, sigma = 1, variance part only):

* Forward map: spectrally exact.  For a prescribed smooth control the cost is 0.406841 at every
  resolution from 12x12 up (six identical digits); the FD model (`fdeval.cpp`, the exact-projection
  march run on the same control) gives 0.42386 / 0.41517 at N = 80 / 160, Richardson 0.40648, i.e.
  0.09% from the spectral value -- the Richardson residual of a first-order scheme.  The FD grid
  would need N ~ 2000 for what 12x12 delivers.
* Equilibrium: the raw collocation root is polluted by weakly determined modes -- the degenerate
  t = 0 slice (N_theta unknowns at one point, cost-blind, spread into every slice by the
  t-interpolation) and the low-weight corner t < 0.15 / theta endpoints.  Chebyshev coefficients
  of the root plateau at 1e-2 instead of decaying, and the near-diagonal own-noise loading swings
  by O(1) between resolutions while J_1 and the state channel are fine to 3 digits.  The same
  happens for a single player's best response (convex), so it is not the Nash coupling.
  Tying the t = 0 slice to the first interior slice (`tie0_test.py`) removes most of it
  (J_1 0.39617 / 0.39624 / 0.39625 at 16x16 / 16x24 / 24x32); a second-derivative Tikhonov
  penalty lambda = 1e-7 (`reg_study.py`) removes the rest: resolution-independent profile,
  theta-coefficients 6e-1, 4e-3, 8e-4, 4e-5, 7e-6, J_1 = 0.39689 / 0.39690.  The penalty biases
  J_1 by +0.0006 at 1e-7 (+0.001 at 1e-5); the lambda -> 0 limit ~0.3966 agrees with the FD
  Richardson value 0.39657.  Own-noise loading of calD_1 at t = 0.5, lags 0.013 .. 0.4:
  spectral -1.59 -1.49 -1.30 -0.96 -0.49 -0.20 -0.06, FD N = 160 -1.64 -1.54 -1.34 -0.99 -0.50
  -0.21 -0.06 (FD 2-3% off in the band).
* Cost: 16x24 (768 unknowns) 14 s, 24x32 99 s on the RTX 5070, almost all of it the chunked
  Jacobian (60 batched jvp's of the map); Newton converges quadratically in 4-5 steps.

Not done: asymmetric parameters, a penalty acting on high modes only
(the smooth part would then be unbiased), Newton-Krylov instead of the dense Jacobian, and the
relation to the discrete game's predictable control (the spectral solver solves the continuous
game; the FD approximates its dt -> 0 limit).

C++ port (`spec_ch1.cpp`): same grids and conventions; the gradient of J_i is a hand-written
reverse sweep through the three stages (operator assembly, linear solve with the transposed LU,
control quadrature), checked against central differences (max relative difference 2e-6, the FD
accuracy); `SPEC_GRADCHECK=1` runs the check.  Newton with a finite-difference Jacobian, columns
in parallel.  At 16x24, m = 16, lambda = 1e-7 it reproduces the JAX result to every printed digit
(J1 = 0.3968879, own-noise profile at t = 0.5: -1.5889 -1.4868 -1.3069 -0.9610 -0.4878 -0.2034
-0.0585) in 90 s on 8 threads (the JAX/GPU version: 14 s); the cost is the FD Jacobian (720
gradient evaluations per Newton step, ~24 s), so Newton-Krylov or a semi-analytic Hessian is the
next step if the C++ path is the one to develop.

Mean part (2026-08-27, C++ only).  Given the equilibrium kernels, the mean paths solve a deterministic
LQ game with a Volterra feedback: player i chooses its mean control freely, the opponent reacts to the
mean state through its kernel on raw observations, delta Dbar^j_t = sqrt(p_j) int_0^t g^j_t(u) Xbar_u du
(the naive response, i.e. the information wedge acting on the mean).  Pontryagin gives
Xbar' = Dbar^1 + Dbar^2, Dbar^i = lambda^i/(2 r_i),
lambda^i'(u) = 2 (Xbar_u - b_i) - sqrt(p_j) int_u^T lambda^i_t g^j_t(u) dt, lambda^i(T) = 0,
solved as one 3 Nt dense system on the Lobatto t-nodes (integration operators from the differentiation
matrix with a boundary row; the Volterra kernel by Gauss quadrature in t with barycentric interpolation
of g).  `--b1 --b2 --x0` set the targets and initial state, `--out-mean file` writes t, Xbar, Dbar1,
Dbar2 and the closed-loop perfect-information Dbar1 (coupled Riccati + target ODE, RK4) on a uniform
grid.  Benchmark p = (3, 3): Dbar1(0) = 8.8306 (12x16) against 8.805 from the first-order FD solver at
N = 79; the perfect-information (feedback Nash) value is 4.647.  See mean_sweep results below.

Precision sweep of the mean control (12x16, m = 12; 16x24 changes Dbar1(0) by 0.03%), r = 0.1, T = 1,
b = (1, -1), x0 = 0, both players at precision p:

    p        0.1    0.3    1      2      3      5      10     30     100    300    1000   3000
    Dbar1(0) 9.937  9.819  9.477  9.110  8.831  8.419  7.798  6.826  5.981  5.459  5.116  4.983

The limits are the two Nash concepts of the deterministic game.  p -> 0: the kernels scale like sqrt(p),
the wedge vanishes, and Dbar1(0) -> 10.000, the OPEN-LOOP Nash path Dbar1_t = (T - t)/r (Xbar = 0 by
symmetry, lambda' = -2 b_1), reached to 4 digits at p = 1e-3.  p -> infinity: Dbar1(0) -> 4.647, the
CLOSED-LOOP (feedback) Nash of the perfect-information game (coupled Riccati + target ODE), approached
like p^{-0.3}.  The partial-information mean control moves monotonically from the open-loop to the
closed-loop value as the opponent's signal improves; this is the mean-path form of the separation
failure of Chapter 1 (Figure 1.3).  Data and plot: mean_sweep_2026-08-27/.

Pooled signals and the cost figure (2026-08-27).  `--pooled` puts both players on one common
observation (noise channel 1 for both, precisions p1 = p2 = the pooled precision); the gradient check
passes (rel. 1e-8) and the pooled (6,6) total cost 3.936 is within 0.5% of the FD Richardson value
3.917.  `fig12_sweep.py` produces dissertation Figure 1.4 (private p = (3, p2) vs pooled p1 + p2,
competitive (1,-1) and common (0,0) targets, total cost = variance part + mean part) on a log grid
p2 = 0.1 .. 1000, ten per decade, 12x16 m = 12, ~10 s per p2; output data/fig12_costs_spectral.csv,
which plot_figures.py prefers over the FD table when present.  All curves are monotone decreasing in
p2 up to 1000; under the common target player 1's private cost meets the pooled cost at large p2
(0.185 vs 0.183), under opposing targets it does not (3.17 vs 2.44).
Note on the FD pipeline: generate_figures crashes for N >= ~65 with more than one thread (heap
corruption in exact_adjoint_pair when the per-equilibrium parallel loop nests with the sweep's own
parallel region; OMP_NUM_THREADS=1 works).  Not fixed.
Figure 1.4 is now three panels: competitive (1,-1), hypercompetitive (5,-5), cooperative (0,0), all from the
same sweep (data/fig12_costs_spectral_parts.csv holds the variance and mean parts; total at (k,-k) is
Jvar + k^2 Jbar).  At (5,-5) pooling stops helping player 2 above p2 ~ 250 (-0.9 at p2 = 1000, player 1 +18);
at (3,-3) the crossing is at p2 ~ 320; at (1,-1) there is none up to 1000.
