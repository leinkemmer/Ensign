cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(Ensign 
  VERSION 1.0.1
  DESCRIPTION "Framework for dynamical low-rank simulations"
  HOMEPAGE_URL "https://github.com/leinkemmer/Ensign"
  LANGUAGES CXX C
)
enable_testing()

include(CheckCXXCompilerFlag)
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
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Configuration options
################################################################################

option("Sanitizer" "Enable address sanitizer" OFF)
option("OPENMP" "Enable OpenMP" ON)
option("CUDA_ENABLED" "Enable CUDA" OFF)
option("MKL_ENABLED" "Enable Intel MKL" OFF)
set("MKL_INCLUDEDIR" "" CACHE PATH "Path to include directory for MKL")
set("MKL_LIBDIR" "" CACHE PATH "Path to lib directory for MKL")


################################################################################
# OpenMP

if(${OPENMP})
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message("OpenMP has been enabled and found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(OPENMP_DEFINES "-D__OPENMP__ -DUSE_OPENMP")
    add_compile_definitions(${OPENMP_DEFINES})
  else()
    message("Could not find support for OpenMP")
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


################################################################################
# Eigen

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT eigen3_FOUND)
  ExternalProject_Add(eigen
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
    GIT_TAG "3.4.0"
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    CMAKE_ARGS
      -DBUILD_TESTING=OFF
      -DEIGEN_BUILD_PKGCONFIG=OFF
      -DEIGEN_BUILD_DOC=OFF
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(eigen SOURCE_DIR)
  set(eigen_HDR_DIR ${SOURCE_DIR})
endif()


################################################################################
# Fortran
if(NOT ${MKL_ENABLED})
  enable_language(Fortran)
endif()


################################################################################
# NetCDF

find_package(NetCDF QUIET)
if(${NETCDF_FOUND})
  message("NetCDF was found.")
else()
  message("NetCDF was NOT found, we will build it from scratch.")
  set("NETCDF_ADD_DEPENDENCY" "TRUE")
  ExternalProject_Add(zlib
    GIT_REPOSITORY "https://github.com/madler/zlib"
    GIT_TAG "v1.3.1"
    CMAKE_ARGS
      -Wno-dev
      -DZLIB_BUILD_EXAMPLES=OFF
      -DBUILD_SHARED_LIBS=OFF
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    UPDATE_COMMAND ""
  )
  ExternalProject_Get_Property(zlib INSTALL_DIR)
  set(zlib_HDR_DIR "${INSTALL_DIR}/include")
  set(zlib_LIBRARIES "${INSTALL_DIR}/lib/libz.a")

  ExternalProject_Add(hdf5
    GIT_REPOSITORY "https://github.com/HDFGroup/hdf5"
    GIT_TAG "hdf5_1.14.6"
    CMAKE_ARGS
      -Wno-dev
      -DBUILD_TESTING=OFF
      -DBUILD_SHARED_LIBS=OFF
      -DHDF5_ENABLE_Z_LIB_SUPPORT=ON
      -DHDF5_ENABLE_SZIP_ENCODING=OFF
      -DHDF5_ENABLE_SZIP_SUPPORT=OFF
      -DHDF5_USE_ZLIB_STATIC=ON
      -DZLIB_USE_EXTERNAL=OFF
      -DZLIB_INCLUDE_DIR=${zlib_HDR_DIR}
      -DZLIB_LIBRARY=${zlib_LIBRARIES}
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    UPDATE_COMMAND ""
    DEPENDS zlib
  )
  ExternalProject_Get_Property(hdf5 INSTALL_DIR)
  set(hdf5_INSTALL_DIR "${INSTALL_DIR}")
  set(hdf5_HDR_DIR "${INSTALL_DIR}/include")
  set(hdf5_LIBRARY "${INSTALL_DIR}/lib/libhdf5.a")
  set(hdf5_HL_LIBRARY "${INSTALL_DIR}/lib/libhdf5_hl.a")

  ExternalProject_Add(netcdf
    GIT_REPOSITORY "https://github.com/Unidata/netcdf-c"
    GIT_TAG "v4.9.3"
    CMAKE_ARGS
      -Wno-dev
      -DCMAKE_BUILD_TYPE=Release
      -DNETCDF_ENABLE_TESTS=OFF
      -DNETCDF_ENABLE_DAP=OFF
      -DNETCDF_ENABLE_SZIP=OFF
      -DNETCDF_ENABLE_ZSTD=OFF
      -DNETCDF_BUILD_UTILITIES=OFF
      -DBUILD_SHARED_LIBS=OFF
      -DHDF5_ROOT=${hdf5_INSTALL_DIR}
      -DHDF5_INCLUDE_DIR=${hdf5_HDR_DIR}
      -DHDF5_C_LIBRARIES=${hdf5_LIBRARY}
      -DHDF5_HL_LIBRARIES=${hdf5_HL_LIBRARY}
      -DZLIB_INCLUDE_DIR=${zlib_HDR_DIR}
      -DZLIB_LIBRARY=${zlib_LIBRARIES}
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DCMAKE_INSTALL_LIBDIR=lib
    UPDATE_COMMAND ""
    DEPENDS hdf5
    DEPENDS zlib
  )

  ExternalProject_Get_Property(netcdf INSTALL_DIR)
  set(NETCDF_INCLUDE_DIR "${INSTALL_DIR}/include")
  set(NETCDF_LIBRARIES "${INSTALL_DIR}/lib/libnetcdf.a")
endif()


################################################################################
# Threads

find_package(Threads REQUIRED)


################################################################################
# fftw3

find_package(FFTW3 QUIET)
if(NOT fftw3_FOUND)
  ExternalProject_Add(fftw
    URL "http://fftw.org/fftw-3.3.10.tar.gz"
    URL_HASH MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c
    CMAKE_ARGS
      -Wno-dev
      -DBUILD_TESTS=OFF
      -DENABLE_THREADS=ON
      -DENABLE_OPENMP=${OPENMP}
      -DBUILD_SHARED_LIBS=OFF
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    UPDATE_COMMAND ""
  )
  ExternalProject_Get_Property(fftw SOURCE_DIR)
  ExternalProject_Get_Property(fftw BINARY_DIR)
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
    set(MKL_DEFINES    "-D__MKL__" "-DMKL_ILP64" "-m64")
    set(MKL_LIBRARIES  "-L${MKL_LIBDIR} -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
    if(NOT (MKL_INCLUDEDIR AND MKL_LIBDIR))
      message("MKL not found automatically. Please set MKL_INCLUDEDIR and MKL_LIBDIR")
    endif()
  else()
    message("MKL found automatically.")
  endif()
else()
  message("OpenBLAS is used by default. MKL support is available by setting MKL_ENABLED to ON (set MKL_INCLUDEDIR and MKL_LIBDIR environment variables appropriately).")
  find_package(OpenBLAS QUIET)
  if(NOT openblas_FOUND)
    ExternalProject_Add(openblas
      GIT_REPOSITORY "https://github.com/OpenMathLib/OpenBLAS.git"
      GIT_TAG "v0.3.29"
      GIT_SHALLOW ON
      GIT_PROGRESS ON
      CMAKE_ARGS
        -Wno-dev
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_LAPACK_DEPRECATED=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DUSE_OPENMP=${OPENMP}
        -DCMAKE_INSTALL_LIBDIR=lib
      UPDATE_COMMAND ""
    )
    ExternalProject_Get_Property(openblas INSTALL_DIR)
    set(openblas_HDR_DIRS ${INSTALL_DIR}/include/openblas)
    set(openblas_LIB_DIRS ${INSTALL_DIR}/lib)
    set(openblas_LIBRARY ${INSTALL_DIR}/lib/libopenblas.a)
  endif()

endif()


################################################################################
# Compiler options
################################################################################

if(${Sanitizer})
  add_compile_options("-fno-omit-frame-pointer;-fsanitize=address")
  add_link_options("-fsanitize=address")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  add_compile_options("-Warn all")
  add_compile_options(
    "$<$<CONFIG:RELEASE>:-O3;-ip;-xHOST>"
    "$<$<CONFIG:DEBUG>:-g;-check all;-traceback>"
  )

elseif(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  add_compile_options(-Wall -Wno-maybe-uninitialized)
  add_compile_options(
    "$<$<CONFIG:RELEASE>:-O3;-DNDEBUG>"
    "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb3;-Wall;-D_GLIBCXX_DEBUG;-fno-omit-frame-pointer;-ftrapv>"
  )

  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options("$<$<CONFIG:RELEASE>:-march=native>")
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  add_compile_options("-Wall;-Wno-uninitialized")
endif()


################################################################################
# Ensign library
################################################################################

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
add_dependencies(Ensign fftw eigen)
if(${NETCDF_ADD_DEPENDENCY})
    add_dependencies(Ensign netcdf)
endif()
if(NOT ${MKL_ENABLED})
  add_dependencies(Ensign openblas)
endif()

# Include directories for the library
target_include_directories(Ensign
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# If Ensign is being bundled, disable tests and examples
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  add_subdirectory(tests)
  add_subdirectory(examples)
endif()


################################################################################
# Link and include directories for external libraries

if(eigen_FOUND)
  target_link_libraries(Ensign PUBLIC Eigen3::Eigen)
else()
  target_include_directories(Ensign
    PUBLIC
    "$<BUILD_INTERFACE:${eigen_HDR_DIR}>"
  )
endif()

if(${MKL_ENABLED})
  if(mkl_FOUND)
    target_link_libraries(Ensign PUBLIC MKL::MKL)
  else()
    target_compile_options(Ensign PUBLIC ${MKL_DEFINES})
    target_include_directories(Ensign
      PUBLIC
      "$<BUILD_INTERFACE:${MKL_INCLUDEDIR}>"
    )
    target_link_directories(Ensign
      PUBLIC
      "$<BUILD_INTERFACE:${MKL_LIBDIR}>"
      "$<INSTALL_INTERFACE:lib>"
    )
    target_link_libraries(Ensign
      PUBLIC 
      "$<BUILD_INTERFACE:${MKL_LIBDIR}/libblas.so>"
      "$<INSTALL_INTERFACE:lib>"
    )
  endif()
else()
  if(openblas_FOUND)
    target_link_libraries(Ensign PUBLIC OpenBLAS::OpenBLAS)
  else()
    target_include_directories(Ensign
      PUBLIC 
      "$<BUILD_INTERFACE:${openblas_HDR_DIRS}>"
      "$<INSTALL_INTERFACE:include/openblas>"
    )
    target_link_directories(Ensign
      PUBLIC
      "$<BUILD_INTERFACE:${openblas_LIB_DIRS}>"
      "$<INSTALL_INTERFACE:lib>"
    )
    target_link_libraries(Ensign
      PUBLIC 
      "$<BUILD_INTERFACE:${openblas_LIBRARY}>"
      "$<INSTALL_INTERFACE:libopenblas.a>"
    )
  endif()
endif()

if(${NETCDF_FOUND})
  target_link_libraries(Ensign PUBLIC ${NETCDF_LIBRARIES})
else()
  target_include_directories(Ensign 
    PUBLIC 
    "$<BUILD_INTERFACE:${hdf5_HDR_DIR}>"
  )
  target_link_libraries(Ensign PUBLIC ${NETCDF_LIBRARIES})
  target_link_libraries(Ensign PRIVATE ${CMAKE_DL_LIBS} ${hdf5_HL_LIBRARY} ${hdf5_LIBRARY} ${zlib_LIBRARIES})
endif()
target_include_directories(Ensign 
  PUBLIC
  "$<BUILD_INTERFACE:${NETCDF_INCLUDE_DIR}>"
)

target_link_libraries(Ensign PRIVATE Threads::Threads)

if(fftw3_FOUND)
  target_link_libraries(Ensign PUBLIC FFTW3::fftw3)
else()
  target_include_directories(Ensign
    PUBLIC
    "$<BUILD_INTERFACE:${fftw3_HDR_DIR}>"
  )
  target_link_libraries(Ensign PUBLIC 
  "$<BUILD_INTERFACE:${fftw3_LIBRARIES}>"
  "$<INSTALL_INTERFACE:libfftw3.a>"
  )
endif()

if(NOT ${MKL_ENABLED})
  target_link_directories(Ensign PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})

  if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    target_link_libraries(Ensign PUBLIC "-lgfortran")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES Flang)
    target_link_libraries(Ensign PUBLIC "-lFortranRuntime")
    target_link_libraries(Ensign PUBLIC "-lFortranDecimal")
  endif()
endif()


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
  FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT EnsignTargets
  FILE EnsignTargets.cmake
  NAMESPACE Ensign::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ensign
)

# Include config helpers
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/EnsignConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/EnsignConfig.cmake
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
