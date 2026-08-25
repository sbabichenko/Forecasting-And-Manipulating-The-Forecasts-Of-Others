cmake_minimum_required(VERSION 3.14)
project(NoiseStateGames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 dependency
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP (optional — graceful fallback if unavailable)
find_package(OpenMP)

add_executable(generate_figures
    generate_figures.cpp
    lqg_solver.cpp
)

target_link_libraries(generate_figures PRIVATE Eigen3::Eigen)
target_compile_options(generate_figures PRIVATE -O3 -march=native -Wall -Wextra)

if(OpenMP_CXX_FOUND)
    target_link_libraries(generate_figures PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(solve_interactive
    solve_interactive.cpp
    lqg_solver.cpp
)
target_link_libraries(solve_interactive PRIVATE Eigen3::Eigen)
target_compile_options(solve_interactive PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(solve_interactive PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(solve_stationary
    solve_stationary.cpp
    stationary_solver.cpp
    lqg_solver.cpp
)
target_link_libraries(solve_stationary PRIVATE Eigen3::Eigen)
target_compile_options(solve_stationary PRIVATE -O3 -march=native -DNDEBUG -Wall -Wextra)
# FFTW backend for Eigen's FFT (optional; kissfft fallback if absent)
include(CheckIncludeFileCXX)
check_include_file_cxx(fftw3.h HAVE_FFTW3_H)
find_library(FFTW3_LIB fftw3)
find_library(FFTW3_THREADS_LIB fftw3_threads)
if(HAVE_FFTW3_H AND FFTW3_LIB AND FFTW3_THREADS_LIB)
    target_compile_definitions(solve_stationary PRIVATE EIGEN_FFTW_DEFAULT)
    target_link_libraries(solve_stationary PRIVATE ${FFTW3_THREADS_LIB} ${FFTW3_LIB} pthread)
    message(STATUS "solve_stationary: using FFTW")
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(solve_stationary PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark
    benchmark.cpp
    lqg_solver.cpp
)
target_link_libraries(benchmark PRIVATE Eigen3::Eigen)
target_compile_options(benchmark PRIVATE -O3 -march=native -Wall -Wextra)

add_executable(test_exact_ce
    test_exact_ce.cpp
    lqg_solver.cpp
)
target_link_libraries(test_exact_ce PRIVATE Eigen3::Eigen)
target_compile_options(test_exact_ce PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_exact_ce PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_ce_picard
    test_ce_picard.cpp
    lqg_solver.cpp
)
target_link_libraries(test_ce_picard PRIVATE Eigen3::Eigen)
target_compile_options(test_ce_picard PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_ce_picard PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(solve_spectral spectral_solver.cpp)
target_link_libraries(solve_spectral PRIVATE Eigen3::Eigen)
target_compile_options(solve_spectral PRIVATE -O3 -march=native -DNDEBUG -std=c++17 -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(solve_spectral PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(kb_spectral kb_spectral.cpp)
target_link_libraries(kb_spectral PRIVATE Eigen3::Eigen)
target_compile_options(kb_spectral PRIVATE -O3 -march=native -DNDEBUG -std=c++17 -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kb_spectral PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(kb_spectral_q kb_spectral_q.cpp)
target_link_libraries(kb_spectral_q PRIVATE Eigen3::Eigen)
target_compile_options(kb_spectral_q PRIVATE -O3 -march=native -DNDEBUG -std=c++17 -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kb_spectral_q PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_stationarity test_stationarity.cpp lqg_solver.cpp)
target_include_directories(test_stationarity PRIVATE ${EIGEN3_INCLUDE_DIR})
target_compile_options(test_stationarity PRIVATE -O3 -march=native)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_stationarity PRIVATE OpenMP::OpenMP_CXX)
endif()
