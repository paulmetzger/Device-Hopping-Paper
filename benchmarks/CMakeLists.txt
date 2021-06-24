enable_language(CXX)

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -pthread -Wall -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -Wall -pedantic")

message("Compile generated benchmarks: Enabled")

add_executable(generated_ocl_rodinia_nn_executable_with_slicing generated_rodinia_nn.cu)
target_compile_options(generated_ocl_rodinia_nn_executable_with_slicing PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_ocl_rodinia_nn_executable_with_slicing PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_rodinia_nn_executable_with_slicing LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_rodinia_nn_plastic_sliced_data_transfers SHARED generated_rodinia_nn.cu)
target_link_libraries(generated_ocl_rodinia_nn_plastic_sliced_data_transfers LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## SHOC Reduction
#############################
add_executable(generated_omp_shoc_reduction_executable_with_slicing generated_shoc_reduction.cu)
target_compile_options(generated_omp_shoc_reduction_executable_with_slicing PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_omp_shoc_reduction_executable_with_slicing PRIVATE EXECUTABLE)
target_link_libraries(generated_omp_shoc_reduction_executable_with_slicing LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_omp_shoc_reduction_plastic_sliced_data_transfers SHARED generated_shoc_reduction.cu)
target_link_libraries(generated_omp_shoc_reduction_plastic_sliced_data_transfers LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## SHOC SPMV
#############################
add_executable(generated_ocl_shoc_sspmv_scalar_executable_with_slicing generated_shoc_spmv.cu)
target_compile_options(generated_ocl_shoc_sspmv_scalar_executable_with_slicing PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_ocl_shoc_sspmv_scalar_executable_with_slicing PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_shoc_sspmv_scalar_executable_with_slicing LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_shoc_sspmv_scalar_plastic_sliced_data_transfers SHARED generated_shoc_spmv.cu)
target_link_libraries(generated_ocl_shoc_sspmv_scalar_plastic_sliced_data_transfers LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## SHOC FFT
#############################
add_executable(generated_ocl_shoc_sfft_executable_with_slicing generated_shoc_fft.cu)
target_compile_options(generated_ocl_shoc_sfft_executable_with_slicing PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_ocl_shoc_sfft_executable_with_slicing PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_shoc_sfft_executable_with_slicing LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_shoc_sfft_plastic_sliced_data_transfers SHARED generated_shoc_fft.cu)
target_link_libraries(generated_ocl_shoc_sfft_plastic_sliced_data_transfers LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## SHOC GEMM NN
#############################
add_executable(generated_ocl_shoc_sgemm_nn_executable generated_shoc_gemm_nn.cu)
target_compile_options(generated_ocl_shoc_sgemm_nn_executable PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_ocl_shoc_sgemm_nn_executable PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_shoc_sgemm_nn_executable LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_shoc_sgemm_nn_plastic SHARED generated_shoc_gemm_nn.cu)
target_link_libraries(generated_ocl_shoc_sgemm_nn_plastic LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## SHOC iFFT
#############################
add_executable(generated_ocl_shoc_sifft_executable_with_slicing generated_shoc_ifft.cu)
target_compile_options(generated_ocl_shoc_sifft_executable_with_slicing PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_ocl_shoc_sifft_executable_with_slicing PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_shoc_sifft_executable_with_slicing LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_shoc_sifft_plastic_sliced_data_transfers SHARED generated_shoc_ifft.cu)
target_link_libraries(generated_ocl_shoc_sifft_plastic_sliced_data_transfers LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## SHOC MD5Hash
#############################
add_executable(generated_ocl_shoc_md5hash_executable generated_shoc_md5hash.cu)
target_compile_options(generated_ocl_shoc_md5hash_executable PRIVATE "-Xcompiler=\"-fPIC\"")
target_compile_definitions(generated_ocl_shoc_md5hash_executable PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_shoc_md5hash_executable LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_shoc_md5hash_plastic SHARED generated_shoc_md5hash.cu)
target_link_libraries(generated_ocl_shoc_md5hash_plastic LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## Btree Find-k
#############################
add_executable(generated_ocl_rodinia_btree_find_k_executable generated_rodinia_btree_find_k.cu
                ./rodinia_src/btree/common.cpp
                ./rodinia_src/btree/kernel_gpu_cuda_wrapper.cu
                ./rodinia_src/btree/util/cuda/cuda.cu
                ./rodinia_src/btree/util/timer/timer.cpp)
target_compile_definitions(generated_ocl_rodinia_btree_find_k_executable PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_rodinia_btree_find_k_executable LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})


add_library(generated_ocl_rodinia_btree_find_k_plastic SHARED generated_rodinia_btree_find_k.cu
                ./rodinia_src/btree/common.cpp
                ./rodinia_src/btree/kernel_gpu_cuda_wrapper.cu
                ./rodinia_src/btree/util/cuda/cuda.cu
                ./rodinia_src/btree/util/timer/timer.cpp)
target_link_libraries(generated_ocl_rodinia_btree_find_k_plastic LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#############################
## Btree Find-range
#############################
add_executable(generated_ocl_rodinia_btree_find_range_executable generated_rodinia_btree_find_range.cu
                ./rodinia_src/btree/common.cpp
                ./rodinia_src/btree/kernel_gpu_cuda_wrapper_2.cu
                ./rodinia_src/btree/util/cuda/cuda.cu
                ./rodinia_src/btree/util/timer/timer.cpp)
target_compile_definitions(generated_ocl_rodinia_btree_find_range_executable PRIVATE EXECUTABLE)
target_link_libraries(generated_ocl_rodinia_btree_find_range_executable LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

add_library(generated_ocl_rodinia_btree_find_range_plastic SHARED generated_rodinia_btree_find_range.cu
                ./rodinia_src/btree/common.cpp
                ./rodinia_src/btree/kernel_gpu_cuda_wrapper_2.cu
                ./rodinia_src/btree/util/cuda/cuda.cu
                ./rodinia_src/btree/util/timer/timer.cpp)
target_link_libraries(generated_ocl_rodinia_btree_find_range_plastic LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL ${OPENMP_LIB} ${INTEL_LIBRARY_DEPENDENCIES})

#####
## Dummy application to block one of the devices.
#####
add_executable(device_hog device_hog.cpp)
target_compile_definitions(device_hog PRIVATE ORACLE)
target_link_libraries(device_hog LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL)

add_library(device_hog_for_daemon SHARED device_hog.cpp)
target_compile_definitions(device_hog_for_daemon PRIVATE PLASTIC ORACLE_WITH_DAEMON)
target_link_libraries(device_hog_for_daemon LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL)

add_library(device_hog_for_daemon_with_ocl SHARED device_hog.cpp)
target_compile_definitions(device_hog_for_daemon_with_ocl PRIVATE PLASTIC ORACLE_WITH_DAEMON OPENCL)
target_link_libraries(device_hog_for_daemon_with_ocl LINK_PUBLIC ${Boost_LIBRARIES} rt OpenCL)