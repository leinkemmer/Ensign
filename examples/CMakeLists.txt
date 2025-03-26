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

# Vlasov-Poisson examples

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
  add_dependencies(vlasov_poisson Ensign)
endif()


target_link_libraries(vlasov_poisson Ensign::Ensign cxxopts)


# Alfven waves example
add_executable(alfven_waves alfven_waves.cpp)
add_dependencies(alfven_waves Ensign)

target_link_libraries(alfven_waves Ensign::Ensign cxxopts)


# LHD instability example
add_executable(lhd_instability lhd_instability.cpp)
add_dependencies(lhd_instability Ensign)

target_link_libraries(lhd_instability Ensign::Ensign cxxopts)


if(cxxopts_FOUND)
  target_link_libraries(linear_landau_3d cxxopts::cxxopts)
  target_link_libraries(vlasov_poisson cxxopts::cxxopts)
  target_link_libraries(alfven_waves cxxopts::cxxopts)
  target_link_libraries(lhd_instability cxxopts::cxxopts)
else()
  target_link_libraries(linear_landau_3d cxxopts)
  target_link_libraries(vlasov_poisson cxxopts)
  target_link_libraries(alfven_waves cxxopts)
  target_link_libraries(lhd_instability cxxopts)
endif()



#add_executable(lhd_instability lhd_instability.cpp ${LR_SOURCES})
#target_link_libraries(lhd_instability ${FFTW_LIBRARIES} ${NETCDF_LIBRARIES})
#if(${MKL_ENABLED})
#  target_link_libraries(lhd_instability ${MKL_LIBRARIES})
#else()
#  target_link_libraries(lhd_instability ${OPENBLAS_LIBRARIES} Threads::Threads "-lgfortran")
#endif()
#
#add_executable(lhd_instability_test lhd_instability_test.cpp ${LR_SOURCES})
#target_link_libraries(lhd_instability_test ${FFTW_LIBRARIES} ${NETCDF_LIBRARIES})
#if(${MKL_ENABLED})
#  target_link_libraries(lhd_instability_test ${MKL_LIBRARIES})
#else()
#  target_link_libraries(lhd_instability_test ${OPENBLAS_LIBRARIES} Threads::Threads "-lgfortran")
#endif()
#add_test(NAME lhd_instability COMMAND lhd_instability_test)
