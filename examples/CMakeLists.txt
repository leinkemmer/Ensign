if(${CUDA_ENABLED})
    cuda_add_executable(advection advection.cu ../src/lr/lr.cu ../src/lr/coefficients.cu ../src/generic/utility.cu ../src/generic/matrix.cu)
    cuda_add_cublas_to_target(advection)
else()
    add_executable(advection advection.cpp ../src/lr/lr.cpp ../src/lr/coefficients.cpp ../src/generic/utility.cpp ../src/generic/matrix.cpp)
    target_link_libraries(advection fftw3)
endif()


include(FindBLAS)
include(FindLAPACK)

if (BLAS_FOUND)
    target_compile_options(advection PRIVATE ${BLAS_LINKER_FLAGS})
    target_link_libraries(advection ${BLAS_LIBRARIES})
endif ()
if (LAPACK_FOUND)
    target_compile_options(advection PRIVATE ${LAPACK_LINKER_FLAGS})
    target_link_libraries(advection ${LAPACK_LIBRARIES})
endif ()

target_link_libraries(advection ${LAPACKE_LIBRARIES})
