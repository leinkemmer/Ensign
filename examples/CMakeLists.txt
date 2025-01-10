################################################################################
# cxxopts

find_package(cxxopts QUIET)
if(NOT cxxopts_FOUND)
  FetchContent_Declare(cxxopts
    GIT_REPOSITORY "https://github.com/jarro2783/cxxopts.git"
    GIT_TAG "v3.2.1"
    GIT_SHALLOW ON
    GIT_PROGRESS ON
  )
  FetchContent_MakeAvailable(cxxopts)
endif()


################################################################################
# Set up examples

add_compile_options(-Wall -Wno-maybe-uninitialized)

set(LR_SOURCES_GPU ../src/lr/lr.cu ../src/lr/coefficients.cu ../src/generic/utility.cu ../src/generic/matrix.cu ../src/generic/timer.cu ../src/generic/fft.cu ../src/generic/kernels.cu ../src/generic/netcdf.cpp)

if(${CUDA_ENABLED})
  cuda_add_executable(linear_landau linear_landau.cu)
  cuda_add_cublas_to_target(linear_landau)
  cuda_add_cufft_to_target(linear_landau)
  target_link_libraries(linear_landau ${CUDA_curand_LIBRARY})
  target_link_libraries(linear_landau ${CUDA_cusolver_LIBRARY})

  cuda_add_executable(linear_landau_2d linear_landau_2d.cu ${LR_SOURCES_GPU})
  cuda_add_cublas_to_target(linear_landau_2d)
  cuda_add_cufft_to_target(linear_landau_2d)
  target_link_libraries(linear_landau_2d ${CUDA_curand_LIBRARY})
  target_link_libraries(linear_landau_2d ${CUDA_cusolver_LIBRARY})

  cuda_add_executable(linear_landau_3d linear_landau_3d.cu ${LR_SOURCES_GPU})
  cuda_add_cublas_to_target(linear_landau_3d)
  cuda_add_cufft_to_target(linear_landau_3d)
  target_link_libraries(linear_landau_3d ${CUDA_curand_LIBRARY})
  target_link_libraries(linear_landau_3d ${CUDA_cusolver_LIBRARY})
else()
  add_executable(linear_landau linear_landau.cpp)

  add_executable(linear_landau_2d linear_landau_2d.cpp)

  add_executable(linear_landau_3d linear_landau_3d.cpp)
endif()

target_link_libraries(linear_landau    Ensign::Ensign)
target_link_libraries(linear_landau_2d Ensign::Ensign)
target_link_libraries(linear_landau_3d Ensign::Ensign)

if(${CUDA_ENABLED})
  cuda_add_executable(vlasov_poisson vlasov_poisson.cu ${LR_SOURCES_GPU})
  cuda_add_cublas_to_target(vlasov_poisson)
  cuda_add_cufft_to_target(vlasov_poisson)
  target_link_libraries(vlasov_poisson ${CUDA_curand_LIBRARY})
  target_link_libraries(vlasov_poisson ${CUDA_cusolver_LIBRARY})
else()
  add_executable(vlasov_poisson vlasov_poisson.cpp)
endif()

target_link_libraries(vlasov_poisson Ensign::Ensign cxxopts)

add_executable(alfven_waves alfven_waves.cpp)
target_link_libraries(alfven_waves Ensign::Ensign cxxopts)

if(cxxopts_FOUND)
  target_link_libraries(linear_landau_3d cxxopts::cxxopts)
  target_link_libraries(vlasov_poisson cxxopts::cxxopts)
  target_link_libraries(alfven_waves cxxopts::cxxopts)
else()
  target_link_libraries(linear_landau_3d cxxopts)
  target_link_libraries(vlasov_poisson cxxopts)
  target_link_libraries(alfven_waves cxxopts)
endif()
