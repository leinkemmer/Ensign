cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(Ensign VERSION 1.0.1)
enable_testing()

include(CMakePackageConfigHelpers)
include(ExternalProject)
include(FetchContent)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-modules")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
    "Release" CACHE STRING 
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE
    )
endif(NOT CMAKE_BUILD_TYPE)

add_compile_options(-Wall -Wno-maybe-uninitialized)


################################################################################
# Configuration options
################################################################################

option("Sanitizer" "Enable address sanitizer" OFF)
option("OPENMP" "Enable OpenMP" ON)
option("CUDA_ENABLED" "Enable CUDA" OFF)
option("MKL_ENABLED" "Enable Intel MKL" OFF)


################################################################################
# Address sanitizer

if(${Sanitizer})
  add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
  add_link_options(-fsanitize=address)
endif()


################################################################################
# OpenMP

if(${OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(OPENMP_DEFINES "-D__OPENMP__")
  endif()
endif()


################################################################################
# CUDA

if(${CUDA_ENABLED})
  find_package(CUDA QUIET)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-O3;-g;--compiler-options;-DNDEBUG;--compiler-options;-fopenmp)
else()
endif()


################################################################################
# External libraries
################################################################################

set_directory_properties(PROPERTIES EP_BASE ${FETCHCONTENT_BASE_DIR})

# Retrieve a copy of the `COMPILE_OPTIONS` of the current directory
get_directory_property(old_dir_compile_options COMPILE_OPTIONS)

# Disable warnings for external libraries
add_compile_options(-w)


################################################################################
# NetCDF

find_package(NetCDF)
if(${NETCDF_FOUND})
  include_directories(${NETCDF_INCLUDE_DIR})
  add_definitions("-D__NETCDF__")
else()
  message("NetCDF has _not_ been found. If you want to write snapshots to disk please set the path in CMake appropriately.")
  set(NETCDF_LIBRARIES "")
endif()


################################################################################
# Threads

find_package(Threads REQUIRED)


################################################################################
# catch2

find_package(Catch2 QUIET)
if(NOT catch2_FOUND)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY "https://github.com/catchorg/Catch2"
    GIT_TAG "v2.13.10"
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(Catch2)
  find_package(Catch2 REQUIRED)
endif()


################################################################################
# cxxopts

find_package(cxxopts QUIET)
if(NOT cxxopts_FOUND)
  FetchContent_Declare(cxxopts
    GIT_REPOSITORY "https://github.com/jarro2783/cxxopts.git"
    GIT_TAG "v3.2.1"
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(cxxopts)
  find_package(cxxopts REQUIRED)
endif()


################################################################################
# fftw3

find_package(FFTW3 QUIET)
if(NOT fftw3_FOUND)
  ExternalProject_Add(fftw3
    URL "http://fftw.org/fftw-3.3.10.tar.gz"
    URL_HASH MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c
    CMAKE_ARGS
      -Wno-dev
      -DBUILD_TESTS=OFF
      -DENABLE_THREADS=ON
      -DENABLE_OPENMP=${OPENMP}
      -DBUILD_SHARED_LIBS=OFF
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
  )
  ExternalProject_Get_Property(fftw3 SOURCE_DIR)
  ExternalProject_Get_Property(fftw3 BINARY_DIR)
  set(fftw3_HDR_DIR ${SOURCE_DIR}/api)
  if(${OPENMP})
    set(fftw3_LIBRARIES "${BINARY_DIR}/libfftw3.a" "${BINARY_DIR}/libfftw3_omp.a")
  else()
    set(fftw3_LIBRARIES "${BINARY_DIR}/libfftw3.a")
  endif()
endif()


################################################################################
# BLAS and LAPACK

if(${MKL_ENABLED})
  find_package(MKL QUIET)
  if(NOT mkl_FOUND)
    set(MKL_INCLUDEDIR "$ENV{MKLROOT}/include" CACHE PATH "Intel MKL include directory")
    set(MKL_LIBDIR "$ENV{MKLROOT}/lib/intel64" CACHE PATH "Intel MKL lib directory")
    # TODO: Why are the following set commands shadowing the cache variables?
    set(MKL_INCLUDEDIR "$ENV{MKLINCLUDEDIR}")
    set(MKL_LIBDIR "$ENV{MKLLIBDIR}")
    set(MKL_DEFINES    "-D__MKL__" "-DMKL_ILP64" "-m64")
    set(MKL_LIBRARIES  "-L{MKL_LIBDIR} -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
  endif()
else()
  message("OpenBLAS is used by default. MKL support is available by setting MKL_ENABLED to ON (set MKLROOT environment variable appropriately).")
  find_package(OpenBLAS QUIET)
  if(NOT openblas_FOUND)
    ExternalProject_Add(openblas
      GIT_REPOSITORY "https://github.com/OpenMathLib/OpenBLAS.git"
      GIT_TAG "v0.3.28"
      GIT_SHALLOW ON
      GIT_PROGRESS ON
      CMAKE_ARGS
        -Wno-dev
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_LAPACK_DEPRECATED=OFF
      INSTALL_COMMAND ""
      UPDATE_COMMAND ""
    )
    ExternalProject_Get_Property(openblas SOURCE_DIR)
    ExternalProject_Get_Property(openblas BINARY_DIR)
    set(openblas_HDR_DIRS ${SOURCE_DIR};${BINARY_DIR})
    set(openblas_LIB_DIRS ${BINARY_DIR}/lib)
    set(openblas_LIBRARY ${BINARY_DIR}/lib/libopenblas.a)
  endif()
endif()


################################################################################
# Ensign library
################################################################################

# Restore the old `COMPILE_OPTIONS` of the current directory
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${old_dir_compile_options}")

add_subdirectory(tests)
add_subdirectory(examples)

add_library(Ensign
  STATIC
  "src/generic/fft.cpp"
  "src/generic/matrix.cpp"
  "src/generic/netcdf.cpp"
  "src/generic/tensor.cpp"
  "src/generic/timer.cpp"
  "src/generic/tree.cpp"
  "src/generic/utility.cpp"
  "src/lr/coefficients.cpp"
  "src/lr/lr.cpp"
)

add_library(Ensign::Ensign ALIAS Ensign)

# Include directories for the library
target_include_directories(Ensign
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(Ensign PUBLIC ${OPENMP_DEFINES})


################################################################################
# Link and include directories for external libraries

# BLAS and LAPACK
if (${MKL_ENABLED})
  if(mkl_FOUND)
    target_link_libraries(Ensign PUBLIC MKL::MKL)
  else()
    target_compile_definitions(Ensign ${MKL_DEFINES})
    target_include_directories(Ensign
      PUBLIC
      $<BUILD_INTERFACE:${MKL_INCLUDEDIR}>
    )
  endif()
else()
  if(openblas_FOUND)
    target_link_libraries(Ensign PUBLIC OpenBLAS::OpenBLAS)
  else()
    target_include_directories(Ensign
      PUBLIC 
      "$<BUILD_INTERFACE:${openblas_HDR_DIRS}>"
    )
    target_link_directories(Ensign
      PUBLIC
      "$<BUILD_INTERFACE:${openblas_LIB_DIRS}>"
    )
    target_link_libraries(Ensign PUBLIC ${openblas_LIBRARY})
  endif()
endif()

# NetCDF
target_link_libraries(Ensign PRIVATE ${NETCDF_LIBRARIES})

# Threads
target_link_libraries(Ensign PRIVATE Threads::Threads)

# fftw3
if(fftw3_FOUND)
  target_link_libraries(Ensign PRIVATE FFTW3:fftw3)
else()
  target_include_directories(Ensign PRIVATE ${fftw3_HDR_DIR})
  target_link_libraries(Ensign PRIVATE ${fftw3_LIBRARIES})
endif()

# gfortran
target_link_options(Ensign PUBLIC -lgfortran)


################################################################################
# Install tree configuration

# Install targets and export configuration
install(TARGETS Ensign
  EXPORT EnsignTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the headers
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT EnsignTargets
  FILE EnsignTargets.cmake
  NAMESPACE Ensign::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ensign
)

# Include config helpers
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/EnsignConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ensign
)

# Package versioning support
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/EnsignConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/EnsignConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/EnsignConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ensign
)