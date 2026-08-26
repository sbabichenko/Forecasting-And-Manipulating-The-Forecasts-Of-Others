# Spectral-in-time solver for the Chapter 1 finite-horizon game (prototype, JAX)

`spec_ch1.py` — kernels on the triangle 0 <= s <= t <= T in Duffy coordinates (t, theta = s/t),
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

Not done: mean part (bar system), asymmetric parameters, a penalty acting on high modes only
(the smooth part would then be unbiased), Newton-Krylov instead of the dense Jacobian, and the
relation to the discrete game's predictable control (the spectral solver solves the continuous
game; the FD approximates its dt -> 0 limit).
