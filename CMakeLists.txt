cmake_minimum_required(VERSION 3.9)

project(dynlr)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_STANDARD 11)

option("CUDA_ENABLED" OFF)
if(${CUDA_ENABLED})
  find_package(CUDA QUIET)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-O3;-g;--compiler-options;-DNDEBUG;--compiler-options;-fopenmp)
endif()

option("FFTW_SERVER" OFF)
if(${FFTW_SERVER})
  # FFTW
  set(FFTW_INCLUDE_DIRS "$ENV{FFTW_DIR}/include")
  set(FFTW_LIBRARIES "$ENV{FFTW_DIR}/lib/libfftw3.so")

  add_definitions("-D__FFTW__")
  include_directories(${FFTW_INCLUDE_DIRS})
endif()

# MKL
option("MKL_ENABLED" OFF)
# MKL link options might be different on each platform...
set(MKL_DEFINES    "-D__MKL__" "-DMKL_ILP64" "-m64")
set(MKL_INCLUDEDIR "$ENV{MKLROOT}/include")
set(MKL_LIBRARIES  "-L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
if(${MKL_ENABLED})
   add_definitions(${MKL_DEFINES})
   include_directories(${MKL_INCLUDEDIR})
endif()

include_directories(include)

add_subdirectory(extern/Catch2)
add_subdirectory(tests)
add_subdirectory(examples)
