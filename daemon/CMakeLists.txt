enable_language(CXX)

add_executable(ocl_and_cuda_plasticd main.cpp daemon.cpp init_cuda.cu)
target_compile_definitions(ocl_and_cuda_plasticd PRIVATE OPENCL CUDA)
target_link_libraries(ocl_and_cuda_plasticd rt OpenCL ${Boost_LIBRARIES} dl)

add_executable(cuda_plasticd main.cpp daemon.cpp init_cuda.cu)
target_compile_definitions(cuda_plasticd PRIVATE CUDA)
target_link_libraries(cuda_plasticd rt ${Boost_LIBRARIES} dl)

add_executable(client daemon_client.cpp)
target_link_libraries(client rt ${Boost_LIBRARIES})

#[[add_executable(test_application test_application.cpp)
target_link_libraries(test_application OpenCL)]]

add_library(test_application_as_library SHARED test_application.cpp)
target_link_libraries(test_application_as_library OpenCL)
