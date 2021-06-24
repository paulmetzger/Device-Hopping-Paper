//
// Created by paul on 18/05/2020.
//
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

extern "C" int __attribute__ ((externally_visible))  hello_world(
        cl_context& cpu_context,
        cl_command_queue& cpu_queue,
        cl_context& gpu_context,
        cl_command_queue& gpu_queue,
        const unsigned int argc,
        const char* argv[],
        const high_resolution_clock::time_point& timing_start);
int __attribute__ ((externally_visible)) hello_world(
        cl_context& cpu_context,
        cl_command_queue& cpu_queue,
        cl_context& gpu_context,
        cl_command_queue& gpu_queue,
        const unsigned int argc,
        const char* argv[],
        const high_resolution_clock::time_point& timing_start) {
    const unsigned int startup_time_us = duration_cast<microseconds>(high_resolution_clock::now() - timing_start).count();
    std::cout << "Startup time in us: " << startup_time_us << std::endl;
    std::cout << "argc: " << argc << std::endl;
    std::cout << "Argv[0]: " << argv[0] << std::endl;
    std::cout << "Argv[1]: " << argv[1] << std::endl;

    std::cout << "Hello world" << std::endl;
    cl_int err = CL_SUCCESS;
    clCreateBuffer(cpu_context, CL_MEM_READ_WRITE, 100, NULL, &err);
    if (err != 0) std::cerr << "OpenCL error: " << err << std::endl;

    return EXIT_SUCCESS;
}

/*
int main(int argc, char* argv[]) {
    cl_context context;
    cl_command_queue queue;
    cl_uint num_platforms = 0;
    cl_platform_id platform[2];
    cl_device_id device;
    cl_int err = clGetPlatformIDs(2, platform, &num_platforms);
    err |= clGetDeviceIDs(
            platform[0],
            CL_DEVICE_TYPE_CPU,
            1,
            &device,
            NULL);
    if (err == -1 && num_platforms > 1) {
        std::cout << "INFO: First platform does not offer the target device. "
                     "Trying to use the second platform..." << std::endl;
        err = clGetDeviceIDs(
                platform[1],
                CL_DEVICE_TYPE_CPU,
                1,
                &device,
                NULL);
    }

    // Create the OCL context handle
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    hello_world(context, queue, high_resolution_clock::now());
}
*/
