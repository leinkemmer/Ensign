/**
 * \defgroup common_utilities Common utilities
 * Common utility functions
 */

/**
 * \defgroup linear_algebra Linear algebra
 * Dense matrix and vector operations used throughout the library.
 */

/**
 * \defgroup dynamical_low_rank_matrix Dynamical low-rank (matrix)
 * Two-factor (i.e.\ matrix) low-rank decompositions and their corresponding operations.
 */

/**
 * \defgroup dynamical_low_rank_tensor Dynamical low-rank (tensor)
 * Tensor low-rank representations and supporting tree and index operations.
 */

/**
 * \defgroup interpolatory_low_rank Interpolatory low-rank
 * Interpolation-based construction of low-rank representations.
 */

/**
 * \defgroup fast_fourier_transform Fast Fourier transform
 * Batched Fourier transforms, commonly applying one transform per rank index.
 */

/**
 * \file common.hpp
 * \brief Provides shared types, platform headers, and basic definitions.
 * \ingroup common_utilities
 */
#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <iterator>
#include <array>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <complex>
#include <string>
#include <memory>

#ifdef __MKL__
  #include <mkl.h>
#else
  #include <cblas.h>
#endif
#include <fftw3.h>
#include <functional>
#include <random>

#ifdef __CUDA__
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cufft.h>
#include <curand.h>
#include <cusolverDn.h>
#endif

#ifdef __OPENMP__
#include <omp.h>
#endif

using std::cout;
using std::endl;
using std::ofstream;
using std::max;
using std::array;
using std::vector;
using std::fill;
using std::abs;
using std::complex;
using std::string;
using std::to_string;

typedef ptrdiff_t Index;
enum class stloc { host, device };

#ifdef __CUDA__
const int n_threads = 128;
#endif

#ifdef __OPENMP__
const int n_threads_omp = 32;
#endif


namespace Ensign {

// make_unique implementation for compiler that do not support it yet
template<typename T, typename... Args>
std::unique_ptr<T> make_unique_ptr(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace Ensign