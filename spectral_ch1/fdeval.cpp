// Evaluate a given pair of control kernels in the FD model: D_i := calD_i (read from file, rows
// "j l c0 c1 c2 d0 d1 d2"), run the exact projection march, print the variance costs
// J_i = sum_j dt (sum_s dt |X|^2 + r_i sum_s dt |calD_i|^2) and a few kernel values.
#include "lqg_solver.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    const int n = std::atoi(argv[1]); const char* file = argv[2];
    SolverContext ctx = SolverContext::capture_current(); ctx.n = n; ctx.T = 1.0; ctx.b1 = 1.0; ctx.b2 = -1.0; ctx.r1 = 0.1; ctx.r2 = 0.1; ctx.sigma = 1.0;
    ScopedSolverContext guard(ctx);
    Kernel2D D1, D2; D1.setZero(); D2.setZero();
    FILE* f = std::fopen(file, "r"); int j, l; double v[6];
    while (std::fscanf(f, "%d %d %lf %lf %lf %lf %lf %lf", &j, &l, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 8) {
        D1[j][l] = Vec3(v[0], v[1], v[2]); D2[j][l] = Vec3(v[3], v[4], v[5]);
    }
    std::fclose(f);
    EnvironmentResult env;
    forward_environment(D1, D2, std::sqrt(3.0), std::sqrt(3.0), 1, Pi1(), 1, Pi2(), 2, env);
    const double dt = g_dt; double JX = 0, JD1 = 0, JD2 = 0;
    for (int t = 0; t < n; ++t) for (int s = 0; s <= t; ++s) { JX += dt * dt * env.X[t][s].squaredNorm(); JD1 += dt * dt * env.calD1[t][s].squaredNorm(); JD2 += dt * dt * env.calD2[t][s].squaredNorm(); }
    std::printf("N=%d  J_X %.6f  r J_calD1 %.6f  J1_var %.6f  J2_var %.6f\n", n, JX, 0.1 * JD1, JX + 0.1 * JD1, JX + 0.1 * JD2);
    const int tj = static_cast<int>(std::lround(0.5 / dt));
    for (double s : {0.1, 0.25, 0.4, 0.49}) { const int sl = static_cast<int>(std::lround(s / dt)); const Vec3& c = env.calD1[tj][sl]; const Vec3& d = D1[tj][sl]; const Vec3& x = env.X[tj][sl];
        std::printf("t=%.3f s=%.3f  input D1=(%.4f %.4f %.4f)  projected calD1=(%.4f %.4f %.4f)  X=(%.4f %.4f %.4f)\n", tj * dt, sl * dt, d(0), d(1), d(2), c(0), c(1), c(2), x(0), x(1), x(2)); }
}
