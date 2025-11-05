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

if(${CUDA_ENABLED})
  set(LR_SOURCES_GPU ../src/generic/kernels.cu ../src/lr/lr.cu ../src/lr/coefficients.cu ../src/generic/utility.cu ../src/generic/matrix.cu ../src/generic/timer.cu ../src/generic/fft.cu ../src/generic/netcdf.cpp)
endif()

# Vlasov-Poisson examples

if(${CUDA_ENABLED})
  add_executable(vlasov_poisson vlasov_poisson.cu ${LR_SOURCES_GPU})
  target_link_libraries(vlasov_poisson Ensign::Ensign CUDA::curand CUDA::cusolver CUDA::cublas CUDA::cufftw cxxopts)

  add_executable(vlasov_poisson_adapt_reserve vlasov_poisson_adapt_reserve.cu ${LR_SOURCES_GPU})
  target_link_libraries(vlasov_poisson_adapt_reserve Ensign::Ensign CUDA::curand CUDA::cusolver CUDA::cublas CUDA::cufftw cxxopts)
else()
  add_executable(vlasov_poisson vlasov_poisson.cpp)
  add_dependencies(vlasov_poisson Ensign)

  add_executable(vlasov_poisson_adapt_reserve vlasov_poisson_adapt_reserve.cpp)
  add_dependencies(vlasov_poisson_adapt_reserve Ensign)

  target_link_libraries(vlasov_poisson Ensign::Ensign cxxopts)
  target_link_libraries(vlasov_poisson_adapt_reserve Ensign::Ensign cxxopts)
endif()


if(NOT ${CUDA_ENABLED})
  # Alfven waves example
  add_executable(alfven_waves alfven_waves.cpp)
  add_dependencies(alfven_waves Ensign)

  target_link_libraries(alfven_waves Ensign::Ensign cxxopts)

  # LHD instability example
  add_executable(lhd_instability lhd_instability.cpp)
  add_dependencies(lhd_instability Ensign)

  target_link_libraries(lhd_instability Ensign::Ensign cxxopts)

  # Drift-kinetics example
  add_executable(drift_kinetics drift_kinetics_interpolatory.cpp)
  add_dependencies(drift_kinetics Ensign)

  target_link_libraries(drift_kinetics Ensign::Ensign cxxopts)
endif()



if(cxxopts_FOUND)
  #target_link_libraries(linear_landau_3d cxxopts::cxxopts)
  target_link_libraries(vlasov_poisson cxxopts::cxxopts)
  target_link_libraries(vlasov_poisson_adpat_reserve cxxopts::cxxopts)
  #target_link_libraries(alfven_waves cxxopts::cxxopts)
  #target_link_libraries(lhd_instability cxxopts::cxxopts)
else()
  #target_link_libraries(linear_landau_3d cxxopts)
  target_link_libraries(vlasov_poisson cxxopts)
  target_link_libraries(vlasov_poisson_adapt_reserve cxxopts)
  #target_link_libraries(alfven_waves cxxopts)
  #target_link_libraries(lhd_instability cxxopts)
endif()
