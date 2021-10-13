#
# storage test
#
if(${CUDA_ENABLED})
    cuda_add_executable(storage storage.cu ../src/generic/utility.cu ../src/generic/kernels.cu)
    cuda_add_cublas_to_target(storage)
else()
    add_executable(storage storage.cpp ../src/generic/utility.cpp)
endif()
target_link_libraries(storage Catch2::Catch2)
add_test(NAME storage COMMAND storage)

#
# matrix test
#
if(${CUDA_ENABLED})
    cuda_add_executable(matrix matrix.cu ../src/generic/matrix.cu ../src/generic/utility.cu ../src/generic/kernels.cu)
    cuda_add_cublas_to_target(matrix)
    cuda_add_cufft_to_target(matrix)
else()
    add_executable(matrix matrix.cpp ../src/generic/matrix.cpp ../src/generic/utility.cpp ../src/generic/timer.cpp)
endif()
target_link_libraries(matrix Catch2::Catch2)
add_test(NAME matrix COMMAND matrix)

#
# coefficient test
#
if(${CUDA_ENABLED})
    cuda_add_executable(coefficients coefficients.cu ../src/lr/coefficients.cu ../src/generic/utility.cu ../src/generic/matrix.cu ../src/generic/kernels.cu)
    cuda_add_cublas_to_target(coefficients)
    cuda_add_cufft_to_target(coefficients)
else()
    add_executable(coefficients coefficients.cpp ../src/lr/coefficients.cpp ../src/generic/matrix.cpp ../src/generic/utility.cpp ../src/generic/timer.cpp)
endif()
target_link_libraries(coefficients Catch2::Catch2)
add_test(NAME coefficients COMMAND coefficients)

#
# low-rank test
#
if(${CUDA_ENABLED})
    cuda_add_executable(lr lr.cu ../src/lr/lr.cu ../src/generic/utility.cu ../src/generic/matrix.cu ../src/generic/kernels.cu)
    cuda_add_cublas_to_target(lr)
    cuda_add_cufft_to_target(lr)
    target_link_libraries(lr ${CUDA_curand_LIBRARY})
else()
    add_executable(lr lr.cpp ../src/lr/lr.cpp ../src/generic/utility.cpp ../src/generic/matrix.cpp ../src/generic/timer.cpp)
endif()
target_link_libraries(lr Catch2::Catch2)
add_test(NAME lr COMMAND lr)

#
# Add common libraries to all tests
#
target_link_libraries(matrix ${FFTW_LIBRARIES})
target_link_libraries(coefficients ${FFTW_LIBRARIES})
target_link_libraries(lr ${FFTW_LIBRARIES})

if(${MKL_ENABLED})
  target_link_libraries(matrix       ${MKL_LIBRARIES})
  target_link_libraries(coefficients ${MKL_LIBRARIES})
  target_link_libraries(lr           ${MKL_LIBRARIES})
else()
  target_link_libraries(matrix       ${OPENBLAS_LIBRARIES} Threads::Threads "-lgfortran")
  target_link_libraries(coefficients ${OPENBLAS_LIBRARIES} Threads::Threads "-lgfortran")
  target_link_libraries(lr           ${OPENBLAS_LIBRARIES} Threads::Threads "-lgfortran")
endif()
