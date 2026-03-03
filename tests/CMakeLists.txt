################################################################################
# catch2

find_package(Catch2 QUIET)
if(NOT catch2_FOUND)
  FetchContent_Declare(catch2
    GIT_REPOSITORY "https://github.com/catchorg/Catch2"
    GIT_TAG "v3.7.1"
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(catch2)
  find_package(Catch2 REQUIRED)
endif()


################################################################################

#
# storage test
#
if(${CUDA_ENABLED})
    # TODO: adapt
    #cuda_add_executable(storage storage.cu ../src/generic/utility.cu ../src/generic/kernels.cu)
    #cuda_add_cublas_to_target(storage)
else()
    add_executable(storage storage.cpp)
    target_link_libraries(storage Ensign::Ensign Catch2::Catch2WithMain)
    add_test(NAME storage COMMAND storage)
    target_compile_options(storage PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(storage PRIVATE -fsanitize=address,undefined)
endif()

#
# matrix test
#
if(${CUDA_ENABLED})
    # TODO: adapt
    #cuda_add_executable(matrix matrix.cu ../src/generic/matrix.cu ../src/generic/utility.cu ../src/generic/kernels.cu)
    #cuda_add_cublas_to_target(matrix)
    #cuda_add_cufft_to_target(matrix)
    #target_link_libraries(matrix ${CUDA_cusolver_LIBRARY})
else()
    add_executable(matrix matrix.cpp)
    target_link_libraries(matrix Ensign::Ensign Catch2::Catch2WithMain)
    add_test(NAME matrix COMMAND matrix)
    target_compile_options(matrix PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(matrix PRIVATE -fsanitize=address,undefined)
endif()

#
# coefficient test
#
if(${CUDA_ENABLED})
    # TODO: adapt
    #cuda_add_executable(coefficients coefficients.cu ../src/lr/coefficients.cu ../src/generic/utility.cu ../src/generic/matrix.cu ../src/generic/kernels.cu)
    #cuda_add_cublas_to_target(coefficients)
    #cuda_add_cufft_to_target(coefficients)
    #target_link_libraries(coefficients ${CUDA_cusolver_LIBRARY})
else()
    add_executable(coefficients coefficients.cpp)
    target_link_libraries(coefficients Ensign::Ensign Catch2::Catch2WithMain)
    add_test(NAME coefficients COMMAND coefficients)
    target_compile_options(coefficients PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(coefficients PRIVATE -fsanitize=address,undefined)
endif()

#
# low-rank test
#
if(${CUDA_ENABLED})
    # TODO: adapt
    #cuda_add_executable(lr lr.cu ../src/lr/lr.cu ../src/lr/coefficients.cu ../src/generic/utility.cu ../src/generic/matrix.cu ../src/generic/kernels.cu)
    #cuda_add_cublas_to_target(lr)
    #cuda_add_cufft_to_target(lr)
    #target_link_libraries(lr ${CUDA_curand_LIBRARY})
    #target_link_libraries(lr ${CUDA_cusolver_LIBRARY})
else()
    add_executable(lr lr.cpp)
    target_link_libraries(lr Ensign::Ensign Catch2::Catch2WithMain)
    add_test(NAME lr COMMAND lr)
    target_compile_options(lr PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(lr PRIVATE -fsanitize=address,undefined)
endif()

if(NOT ${CUDA_ENABLED})
#
# Interpolatory LR test
#
add_executable(interpolatory interpolatory.cpp)
target_link_libraries(interpolatory Ensign::Ensign Catch2::Catch2WithMain)
add_test(NAME interpolatory COMMAND interpolatory)
target_compile_options(interpolatory PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(interpolatory PRIVATE -fsanitize=address,undefined)

#
# index test
#
add_executable(index index.cpp)
target_link_libraries(index Ensign::Ensign Catch2::Catch2WithMain)
add_test(NAME index COMMAND index)
target_compile_options(index PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(index PRIVATE -fsanitize=address,undefined)

#
# tensor test
#
add_executable(tensor tensor.cpp)
target_link_libraries(tensor Ensign::Ensign Catch2::Catch2WithMain)
add_test(NAME tensor COMMAND tensor)
target_compile_options(tensor PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(tensor PRIVATE -fsanitize=address,undefined)

#
# Alfven waves test
#
add_executable(alfven_waves_test alfven_waves.cpp)
target_link_libraries(alfven_waves_test Ensign::Ensign Catch2::Catch2WithMain)
add_test(NAME alfven_waves COMMAND alfven_waves_test)
target_compile_options(alfven_waves_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(alfven_waves_test PRIVATE -fsanitize=address,undefined)

#
# LHD waves test
#
add_executable(lhd_instability_test lhd_instability.cpp)
target_link_libraries(lhd_instability_test Ensign::Ensign Catch2::Catch2WithMain)
add_test(NAME lhd_instability COMMAND lhd_instability_test)
target_compile_options(lhd_instability_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(lhd_instability_test PRIVATE -fsanitize=address,undefined)

#
# Drift-kinetic test
#
add_executable(drift_kinetics_test drift_kinetics.cpp)
target_link_libraries(drift_kinetics_test Ensign::Ensign Catch2::Catch2WithMain)
add_test(NAME drift_kinetics COMMAND drift_kinetics_test)
target_compile_options(drift_kinetics_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(drift_kinetics_test PRIVATE -fsanitize=address,undefined)

endif()
