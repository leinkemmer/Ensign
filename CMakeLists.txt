cmake_minimum_required(VERSION 3.9)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-modules")

project(dynlr)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 11)

option("OPENMP" "Enable OpenMP" ON)
if(${OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(OPENMP_DEFINES "-D__OPENMP__")
    add_definitions(${OPENMP_DEFINES})
  endif()
endif()

option("CUDA_ENABLED" "Enable CUDA" OFF)
if(${CUDA_ENABLED})
  find_package(CUDA QUIET)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-O3;-g;--compiler-options;-DNDEBUG;--compiler-options;-fopenmp)
else()
endif()

find_package(NetCDF)
if(${NETCDF_FOUND})
  include_directories(${NETCDF_INCLUDE_DIR})
  add_definitions("-D__NETCDF__")
else()
  message("NetCDF has _not_ been found. If you want to write snapshots to disk please set the path in CMake appropriately.")
  set(NETCDF_LIBRARIES "")
endif()

#
# FFTW, cxxopts, and OpenBLAS (if MKL_ENABLED is off), and Catch2 are automatically downloaded and build.
#
ExternalProject_Add(fftw
  PREFIX "${PROJECT_BINARY_DIR}/extern/build/fftw"
  URL "http://fftw.org/fftw-3.3.10.tar.gz"
  CONFIGURE_COMMAND ./configure --enable-threads --enable-openmp --enable-static --disable-fortran --prefix=${PROJECT_BINARY_DIR}/extern/fftw
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND make install
  UPDATE_COMMAND "" # do not rebuild external projects as part of the normal workflow
)
include_directories("${PROJECT_BINARY_DIR}/extern/fftw/include")
set(FFTW_LIBRARIES "${PROJECT_BINARY_DIR}/extern/fftw/lib/libfftw3.a" "${PROJECT_BINARY_DIR}/extern/fftw/lib/libfftw3_omp.a")


ExternalProject_Add(cxxopts
  PREFIX "${PROJECT_BINARY_DIR}/extern/cxxopts"
  GIT_REPOSITORY "https://github.com/jarro2783/cxxopts.git"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
include_directories("${PROJECT_BINARY_DIR}/extern/cxxopts/src/cxxopts/include")


# BLAS and LAPACK
option("MKL_ENABLED" OFF)
if(${MKL_ENABLED})
  set(MKL_DEFINES    "-D__MKL__" "-DMKL_ILP64" "-m64")
  set(MKL_INCLUDEDIR "$ENV{MKLROOT}/include")
  set(MKL_LIBRARIES  "-L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
  add_definitions(${MKL_DEFINES})
  include_directories(${MKL_INCLUDEDIR})
else()
  message("OpenBLAS is used by default. MKL support is available by setting MKL_ENABLED to ON (set MKLROOT environment variable appropriately).")
  ExternalProject_Add(OpenBLAS
    GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
    GIT_TAG "develop"  # appearently master is years out of date
    PREFIX "${PROJECT_BINARY_DIR}/extern/build/OpenBLAS"
    SOURCE_DIR "${PROJECT_BINARY_DIR}/extern/OpenBLAS" 
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND DESTDIR=${PROJECT_BINARY_DIR}/extern/build/OpenBLAS make install
    UPDATE_COMMAND ""
  )
  set(OPENBLAS_LIBRARIES "${PROJECT_BINARY_DIR}/extern/build/OpenBLAS/opt/OpenBLAS/lib/libopenblas.a")
endif()
find_package(Threads REQUIRED)


ExternalProject_Add(Catch2
  PREFIX "${PROJECT_BINARY_DIR}/extern/Catch2"
  GIT_REPOSITORY "https://github.com/catchorg/Catch2"
  GIT_TAG "v2.x"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
include_directories("${PROJECT_BINARY_DIR}/extern/Catch2/src/Catch2/single_include")



include_directories(include)
add_subdirectory(tests)
add_subdirectory(examples)
