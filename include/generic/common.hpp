#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <iterator>
#include <array>
#include <vector>
#include <algorithm>
#include <assert.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

using std::cout;
using std::endl;
using std::ofstream;
using std::max;
using std::array;
using std::vector;
using std::fill;
using std::abs;

typedef ptrdiff_t Index;
