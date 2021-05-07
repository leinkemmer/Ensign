cmake_minimum_required(VERSION 3.9)

project(dynlr)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_STANDARD 14)

option("CUDA_ENABLED" OFF)
if(${CUDA_ENABLED})
  find_package(CUDA QUIET)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;-O3;-g;--compiler-options;-DNDEBUG;--compiler-options;-fopenmp)
endif()

include_directories(include)

add_subdirectory(extern/Catch2)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(fft_cuda)
