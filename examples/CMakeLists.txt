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
target_link_libraries(linear_landau_3d Ensign::Ensign cxxopts::cxxopts)

if(${CUDA_ENABLED})
  cuda_add_executable(vlasov_poisson vlasov_poisson.cu ${LR_SOURCES_GPU})
  cuda_add_cublas_to_target(vlasov_poisson)
  cuda_add_cufft_to_target(vlasov_poisson)
  target_link_libraries(vlasov_poisson ${CUDA_curand_LIBRARY})
  target_link_libraries(vlasov_poisson ${CUDA_cusolver_LIBRARY})
else()
  add_executable(vlasov_poisson vlasov_poisson.cpp)
endif()
target_link_libraries(vlasov_poisson Ensign::Ensign cxxopts::cxxopts)

add_executable(alfven_waves alfven_waves.cpp)
target_link_libraries(alfven_waves Ensign::Ensign cxxopts::cxxopts)

add_executable(alfven_waves_test alfven_waves_test.cpp)
target_link_libraries(alfven_waves_test Ensign::Ensign Catch2::Catch2WithMain)

add_test(NAME alfven_waves COMMAND alfven_waves_test)

