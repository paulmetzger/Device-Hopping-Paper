add_executable(destroy_ipc                   destroy_ipc.cpp)
add_executable(init_ipc                      init_ipc.cpp)
add_executable(ocl_binary_generator          ocl_binary_generator.cpp)
add_executable(read_book_keeping_data        read_book_keeping_data.cpp)
add_executable(spmv_gen_matrices             spmv_gen_matrices.cpp)
add_executable(set_device_to_used            set_device_to_used.cpp)
add_executable(set_device_to_unused          set_device_to_unused.cpp)
add_executable(rodinia_nn_hurricanegen       rodinia_nn_hurricanegen.cpp)

target_link_libraries(destroy_ipc            rt)
target_link_libraries(init_ipc               rt)
target_link_libraries(read_book_keeping_data rt)

target_link_libraries(ocl_binary_generator          LINK_PUBLIC ${Boost_LIBRARIES} OpenCL)
target_link_libraries(spmv_gen_matrices             LINK_PUBLIC ${Boost_LIBRARIES} OpenCL)

target_link_libraries(set_device_to_used   LINK_PUBLIC ${Boost_LIBRARIES} rt)
target_link_libraries(set_device_to_unused LINK_PUBLIC ${Boost_LIBRARIES} rt)
