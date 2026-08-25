#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdlib>
#ifdef EIGEN_FFTW_DEFAULT
#include <fftw3.h>
#endif
#include "stationary_solver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

namespace {

static void print_json_num(double x) {
    if (std::isfinite(x))
        std::printf("%.12g", x);
    else
        std::printf("null");
}

static void print_vec3_array(const char* name, const std::vector<Vec3>& v) {
    std::printf("\"%s\":{\"ch0\":[", name);
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%s", i ? "," : "");
        print_json_num(v[i](0));
    }
    std::printf("],\"ch1\":[");
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%s", i ? "," : "");
        print_json_num(v[i](1));
    }
    std::printf("],\"ch2\":[");
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%s", i ? "," : "");
        print_json_num(v[i](2));
    }
    std::printf("]}");
}

static void print_double_array(const char* name, const std::vector<double>& v) {
    std::printf("\"%s\":[", name);
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%s", i ? "," : "");
        print_json_num(v[i]);
    }
    std::printf("]");
}

static void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [p1 p2 r1 r2] [--N <n_lag>] [--L <lag_max>] [--iters <n>] "
        "[--relax <x>] [--tol <x>] [--abs-tol <x>] "
        "[--quadrature simpson|trapezoid] "
        "[--forward-iters <n>] [--forward-relax <x>] "
        "[--backward-iters <n>] [--backward-relax <x>] "
        "[--init <warm-start csv>] [--inexact-forward] [--verbose]\n",
        argv0);
}

// Reads a warm-start CSV with rows "lag,d1_ch0,d1_ch1,d1_ch2,d2_ch0,d2_ch1,d2_ch2";
// non-matching lines (e.g. a header) are skipped.
static bool load_init_csv(const char* path, StationaryParams& p) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        double v[7];
        if (std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                        &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) == 7) {
            p.init_lag.push_back(v[0]);
            p.init_d1.push_back(Vec3(v[1], v[2], v[3]));
            p.init_d2.push_back(Vec3(v[4], v[5], v[6]));
        }
    }
    std::fclose(f);
    return !p.init_lag.empty();
}

} // namespace

int main(int argc, char* argv[]) {
#ifdef EIGEN_FFTW_DEFAULT
    fftw_make_planner_thread_safe();
#endif
#ifdef _OPENMP
    // The solver has at most four concurrent tasks; more threads only spin
    // and compete with them for cores.
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(4);
#endif

    StationaryParams p;
    bool verbose = false;

    int pos = 1;
    if (argc >= 5 && argv[1][0] != '-') {
        p.p1 = std::atof(argv[1]);
        p.p2 = std::atof(argv[2]);
        p.r1 = std::atof(argv[3]);
        p.r2 = std::atof(argv[4]);
        pos = 5;
    }

    for (int i = pos; i < argc; ++i) {
        if (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc)
            p.n_lag = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--L") == 0 && i + 1 < argc)
            p.lag_max = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            p.max_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--relax") == 0 && i + 1 < argc)
            p.relax = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc)
            p.tol = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--abs-tol") == 0 && i + 1 < argc)
            p.abs_tol = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--quadrature") == 0 && i + 1 < argc) {
            const char* q = argv[++i];
            if (std::strcmp(q, "simpson") == 0)
                p.use_simpson_quadrature = true;
            else if (std::strcmp(q, "trapezoid") == 0 || std::strcmp(q, "trap") == 0)
                p.use_simpson_quadrature = false;
            else {
                usage(argv[0]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--pcg") == 0)
            p.circulant_cg = true;
        else if (std::strcmp(argv[i], "--anderson") == 0 && i + 1 < argc)
            p.anderson_depth = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--anderson-mixing") == 0 && i + 1 < argc)
            p.anderson_mixing = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--forward") == 0 && i + 1 < argc) {
            const char* m = argv[++i];
            if (std::strcmp(m, "exact") == 0)
                p.exact_forward = true;
            else if (std::strcmp(m, "picard") == 0)
                p.exact_forward = false;
            else {
                usage(argv[0]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--inexact-forward") == 0) p.inexact_forward = true;
        else if (std::strcmp(argv[i], "--init") == 0 && i + 1 < argc) {
            if (!load_init_csv(argv[++i], p)) {
                std::fprintf(stderr, "failed to read --init warm-start CSV: %s\n", argv[i]);
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "--forward-iters") == 0 && i + 1 < argc)
            p.forward_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--forward-relax") == 0 && i + 1 < argc)
            p.forward_relax = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--backward-iters") == 0 && i + 1 < argc)
            p.backward_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--backward-relax") == 0 && i + 1 < argc)
            p.backward_relax = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--sigma") == 0 && i + 1 < argc)
            p.sigma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--A") == 0 && i + 1 < argc)
            p.A = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--verbose") == 0)
            verbose = true;
        else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    StationarySolution sol = solve_stationary(p, verbose);

    const double x_tail = sol.x.empty() ? 0.0 : sol.x.back().norm();
    const double c_tail = sol.calD1.empty() ? 0.0
        : std::max(sol.calD1.back().norm(), sol.calD2.back().norm());
    const int b0 = static_cast<int>(sol.lag.size()) - 1;
    const double hx_left = sol.hx1.empty() ? 0.0
        : std::max(sol.hx1.front().norm(), sol.hx2.front().norm());
    const double hx_right = sol.hx1.empty() ? 0.0
        : std::max(sol.hx1.back().norm(), sol.hx2.back().norm());

    std::printf("{");
    std::printf("\"converged\":%s", sol.converged ? "true" : "false");
    std::printf(",\"residual\":"); print_json_num(sol.residual);
    std::printf(",\"relative_residual\":"); print_json_num(sol.relative_residual);
    std::printf(",\"absolute_residual\":"); print_json_num(sol.absolute_residual);
    std::printf(",\"forward_residual\":"); print_json_num(sol.forward_residual);
    std::printf(",\"backward_residual1\":"); print_json_num(sol.backward_residual1);
    std::printf(",\"backward_residual2\":"); print_json_num(sol.backward_residual2);
    std::printf(",\"p1\":%.12g,\"p2\":%.12g,\"r1\":%.12g,\"r2\":%.12g",
                p.p1, p.p2, p.r1, p.r2);
    std::printf(",\"tol\":%.12g,\"abs_tol\":%.12g", p.tol, p.abs_tol);
    std::printf(",\"quadrature\":\"%s\"", p.use_simpson_quadrature ? "simpson" : "trapezoid");
    std::printf(",\"obs_gain1\":%.12g,\"obs_gain2\":%.12g",
                std::sqrt(std::max(0.0, p.p1)), std::sqrt(std::max(0.0, p.p2)));
    std::printf(",\"N\":%d,\"L\":%.12g,\"h\":%.12g",
                p.n_lag, p.lag_max, sol.h);
    std::printf(",\"n_iters\":%zu", sol.residuals.size());
    std::printf(",\"tail\":{\"x\":"); print_json_num(x_tail);
    std::printf(",\"calD\":"); print_json_num(c_tail);
    std::printf(",\"hx_left\":"); print_json_num(hx_left);
    std::printf(",\"hx_right\":"); print_json_num(hx_right);
    std::printf("}");

    std::printf(",");
    print_double_array("lag", sol.lag);
    std::printf(",");
    print_double_array("b_lag", sol.b_lag);
    std::printf(",");
    print_double_array("residuals", sol.residuals);
    std::printf(",");
    print_double_array("relative_residuals", sol.relative_residuals);
    std::printf(",");
    print_double_array("absolute_residuals", sol.absolute_residuals);
    std::printf(",");
    print_vec3_array("x", sol.x);
    std::printf(",");
    print_vec3_array("xhat1", sol.xhat1);
    std::printf(",");
    print_vec3_array("xhat2", sol.xhat2);
    std::printf(",");
    print_vec3_array("xtilde1", sol.xtilde1);
    std::printf(",");
    print_vec3_array("xtilde2", sol.xtilde2);
    std::printf(",");
    print_vec3_array("d1", sol.d1);
    std::printf(",");
    print_vec3_array("d2", sol.d2);
    std::printf(",");
    print_vec3_array("calD1", sol.calD1);
    std::printf(",");
    print_vec3_array("calD2", sol.calD2);
    std::printf(",");
    print_vec3_array("hx1", sol.hx1);
    std::printf(",");
    print_vec3_array("hx2", sol.hx2);
    std::printf(",");
    print_vec3_array("wedge1", sol.wedge1);
    std::printf(",");
    print_vec3_array("wedge2", sol.wedge2);
    std::printf(",\"zero_lag\":{\"calD1_obs\":");
    print_json_num(sol.calD1.empty() ? 0.0 : sol.calD1[0](1));
    std::printf(",\"calD2_obs\":");
    print_json_num(sol.calD2.empty() ? 0.0 : sol.calD2[0](2));
    std::printf(",\"hx1\":");
    print_json_num(sol.hx1.empty() ? 0.0 : sol.hx1[b0].norm());
    std::printf(",\"hx2\":");
    print_json_num(sol.hx2.empty() ? 0.0 : sol.hx2[b0].norm());
    std::printf("}");
    std::printf("}\n");
    return sol.converged ? 0 : 2;
}
