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
#ifdef __MKL__
  #include <mkl.h>
#else
  #include <cblas.h>
#endif
#include <fftw3.h>
#include <functional>
#include <random>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cufft.h>
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

typedef ptrdiff_t Index;
enum class stloc { host, device };

#ifdef __CUDACC__
  const int n_threads = 128;
  extern cublasHandle_t  handle;
  extern cublasHandle_t handle_dot;
#endif

#ifdef __OPENMP__
  const int n_threads_omp = 6;
#endif

extern double tot_gpu_mem;
