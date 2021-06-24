//
// Created by paul on 30/10/2019.
//

#if defined(OPENCL)
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>

#include "../device_hopper/context.h"
#include "device_hog.h"

using namespace std::chrono;

void setup_opencl(cl_context& context,
                  cl_command_queue& queue,
                  cl_device_id& device,
                  cl_device_type cl_device) {
    cl_uint num_platforms = 0;
    cl_platform_id platform[2];
    cl_int err = clGetPlatformIDs(2, platform, &num_platforms);
    err |= clGetDeviceIDs(
            platform[0],
            cl_device,
            1,
            &device,
            NULL);
    if (err == -1 && num_platforms > 1) {
        std::cout << "INFO: First platform does not offer the target device. "
                     "Trying to use the second platform..." << std::endl;
        err = clGetDeviceIDs(
                platform[1],
                cl_device,
                1,
                &device,
                NULL);
    }

    // Create the OCL context handle
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
}

#if defined(PLASTIC) || defined(ORACLE_WITH_DAEMON)
extern "C" int __attribute__ ((externally_visible)) device_hog_main(
#if defined(OPENCL)
        cl_context& cpu_context,
        cl_command_queue& cpu_queue,
        cl_device_id& cpu_handle,
#endif
        int argc,
        char* argv[],
        const high_resolution_clock::time_point& start_time_point);

int device_hog_main(
#if defined(OPENCL)
        cl_context& cpu_context,
        cl_command_queue& cpu_queue,
        cl_device_id& cpu_handle,
#endif
        int argc,
        char* argv[],
        const high_resolution_clock::time_point& start_time_point) {
#else
    int main(int argc, char* argv[]) {
#endif
    boost::program_options::options_description desc("Options");
    desc.add_options() ("problem-size", boost::program_options::value<int>(),
                        "The time in ms that this application will block the device.");
    plasticity::setup::Context context(argc, argv, desc);

    int execution_time_ms = -1;
    boost::program_options::variables_map vm = context.get_variables_map();
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: The execution time is missing" << std::endl;
        context.print_cli_help();
        std::exit(EXIT_FAILURE);
    } else {
        execution_time_ms = vm["problem-size"].as<int>();
    }

    plasticity::kernels::impl::device_hog(execution_time_ms, context);

    return EXIT_SUCCESS;
}