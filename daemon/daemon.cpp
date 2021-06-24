//
// Created by paul on 18/05/2020.
//

#define CL_TARGET_OPENCL_VERSION 120

#include <chrono>

#include "daemon.h"
#include "messages.h"
#include "ipc_utils.h"

#if defined(OPENCL)
#include <CL/cl.h>
#include "../device_hopper/internal/cl_utils.h"
#endif

#if defined(CUDA)
#include "./init_cuda.cuh"
#endif

#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <csignal>
#include <syslog.h>
#include <thread>
#include <vector>

namespace plasticity {

#if defined(OPENCL)
    void warm_up_opencl(cl_context& context, cl_command_queue q, cl_device_id d) {
        std::cout << "Warming OpenCL up..." << std::endl;
        size_t global_work_size = 200;
        float* input = (float*) malloc(sizeof(float) * global_work_size);
        for (size_t i = 0; i < global_work_size; ++i) input[i] = 1.0f;
        std::string warm_up_src = "__kernel void add(__global float* a) {size_t i = get_global_id(0); a[i] *= 2;}";
        const char* src = warm_up_src.c_str();
        const size_t length = warm_up_src.length();
        cl_int err;
        cl_program p = clCreateProgramWithSource(context, 1, &src, &length, &err);
        if (err != CL_SUCCESS) utils::exit_with_err("Failed creating a program");
        err |= clBuildProgram(p, 1, &d, NULL, NULL, NULL);
        if (err) {
            std::cerr << "Program building failed: ERROR " << err << std::endl;
            // Print warnings and errors from compilation
            static char log[65536];
            memset(log, 0, sizeof(log));
            clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
            printf("-----OpenCL Compiler Output-----\n");
            if (strstr(log, "warning:") || strstr(log, "error:"))
                printf("<<<<\n%s\n>>>>\n", log);
            printf("--------------------------------\n");
            std::exit(1);
        }
        cl_kernel k = clCreateKernel(p, "add", &err);
        if (err != CL_SUCCESS) utils::exit_with_err("Failed creating a kernel");
        cl_mem b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 100 * 1024 * 1024, input, &err);
        if (err != CL_SUCCESS) utils::exit_with_err("Failed creating a device buffer");
        err = clSetKernelArg(k, 0, sizeof(cl_mem), (void*) &b);
        if (err != CL_SUCCESS) utils::exit_with_err("Failed setting kernel argument");

        err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) utils::exit_with_err("Failed launching a kernel");
        err = clFinish(q);
        if (err != CL_SUCCESS) utils::exit_with_err("Failed waiting for the warm up kernel to finish");

        clReleaseProgram(p);
        clReleaseKernel(k);
        clReleaseMemObject(b);
        std::cout << "Warm-up results:" << std::endl;
        for (size_t i = 0; i < global_work_size; ++i) std::cout << input[i] << " ";
        std::cout << std::endl;
        std::cout << "Done with the warm-up" << std::endl;
    }

    void setup_opencl(cl_context& context,
                      cl_command_queue& queue,
                      cl_device_id& device,
                      cl_device_type cl_device) {
        cl_uint num_platforms = 0;
        cl_platform_id platform[2];
        cl_int err = clGetPlatformIDs(2, platform, &num_platforms);
        err |= clGetDeviceIDs(platform[0], cl_device, 1, &device, NULL);
        if (err == -1 && num_platforms > 1) {
            std::cout << "INFO: First platform does not offer the target device. "
                         "Trying to use the second platform..." << std::endl;
            err = clGetDeviceIDs(platform[1], cl_device, 1, &device, NULL);
        }
        if (err != CL_SUCCESS) utils::exit_with_err("Could not create OpenCL device handle");

        // Create the OCL context handle
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) utils::exit_with_err("Could not create OpenCL context");
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) utils::exit_with_err("Could not create OpenCL command queue");
    }
#endif

    void worker(mqd_t queue_from_client_to_daemon,
                mqd_t queue_from_daemon_to_client,
                size_t worker_id,
#if defined(OPENCL)
                cl_context& cpu_context,
                cl_command_queue& cpu_queue,
                cl_device_id& cpu_handle,
#endif
                std::ostringstream& stream_cout) {

#if defined(CUDA)
        init_cuda();
#endif

#if defined(OPENCL)
        if (worker_id == 1) setup_opencl(cpu_context, cpu_queue, cpu_handle, CL_DEVICE_TYPE_CPU);
#endif

        daemon_command command;
        daemon_response response;
        while(command.command_type != CommandType::EXIT_DAEMON) {
#if defined(DEBUG)
            std::cout << "Worker " << worker_id << " entered the loop" << std::endl;
#endif
            if (mq_receive(queue_from_client_to_daemon, (char *) &command, sizeof(daemon_command), 0) != sizeof(daemon_command))
                utils::exit_with_err("mq_receive failed");

#if defined(DEBUG)
            std::cout << "Worker " << worker_id << " received a command" << std::endl;
#endif

            if (command.command_type == CommandType::EXECUTE_APPLICATION) {
                // We exclude this time because the comparison measurements also do not include
                // the time required to load a binary from disk.
                std::chrono::high_resolution_clock::time_point start_load_app = std::chrono::high_resolution_clock::now();
                void *lib = dlopen(command.path_to_application, RTLD_LAZY);
                if (lib == NULL) {
                    std::cerr << dlerror() << std::endl;
                    utils::exit_with_err("Could not load the library at: " + std::string(command.path_to_application));
                }
                int (*imported_function)(
#if defined(OPENCL)
                        cl_context &,
                        cl_command_queue &,
                        cl_device_id &,
#endif
                        int,
                        const char*[],
                        const high_resolution_clock::time_point &);
                imported_function = (int (*)(
#if defined(OPENCL)
                        cl_context &,
                        cl_command_queue &,
                        cl_device_id &,
#endif
                        int,
                        const char*[],
                        const high_resolution_clock::time_point &)) dlsym(lib, command.function_name);
                char *error;
                if ((error = dlerror()) != NULL) {
                    std::cerr << "Error: " << error << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                std::chrono::high_resolution_clock::time_point stop_load_app = std::chrono::high_resolution_clock::now();
                command.time_start += std::chrono::duration_cast<std::chrono::microseconds>(stop_load_app - start_load_app);

                auto iss = std::stringstream{command.application_cli_parameters};
                auto str = std::string{};
                unsigned int argc = 1;
                std::vector<std::string> cli_parameters;
                while(iss >> str) {
                    ++argc;
                    cli_parameters.push_back(str);
                }
                char** argv = (char **) malloc(argc * sizeof(char*));
                const std::string application_name = "application_in_daemon";
                argv[0] = (char*) application_name.c_str();
                for (size_t i = 0; i < cli_parameters.size(); ++i) argv[i + 1] = (char*) cli_parameters[i].c_str();

                int err = imported_function(
#if defined(OPENCL)
                        cpu_context,
                        cpu_queue,
                        cpu_handle,
#endif
                        argc,
                        (const char**) argv,
                        command.time_start);
                if(dlclose(lib)) {
                    std::cerr << "Error: " << dlerror() << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                response.return_code = err;
                if (strstr(command.function_name, "device_hog") == nullptr) strcpy(response.output, stream_cout.str().c_str());
                else response.output[0] = '\0';
            } else {
                response.return_code = 0;
                response.output[0] = '\0';
            }

            if (strstr(command.function_name, "device_hog") == nullptr) {
                if (mq_send(queue_from_daemon_to_client, (const char *) &response, sizeof(daemon_response), 0) != 0)
                    utils::exit_with_err("mq_send faild");
                stream_cout.str("");
                stream_cout.clear();
            }
        }
    }

    std::ostringstream stream_cout;
    std::ostringstream stream_cerr;
    struct sigaction abrt_default_hanlder, segfault_default_handler, term_default_handler;
    void sig_crash_handler(int sig) {
        std::string received_signal;
        if (sig == SIGSEGV) {
            received_signal = "SIGSEGV";
        } else if (sig == SIGABRT) {
            received_signal = "SIGABRT";
        } else if (sig == SIGTERM) {
            received_signal = "SIGTERM";
        } else {
            received_signal = "unknown";
        }
        int err = sigaction(SIGSEGV, &segfault_default_handler, NULL);
        err |= sigaction(SIGABRT, &abrt_default_hanlder, NULL);
        err |= sigaction(SIGTERM, &term_default_handler, NULL);
        if (err != 0) {
            syslog(LOG_ERR, "Could not set default signal handlers");
            std::exit(1);
        }

        syslog(LOG_ERR, "The plasticity daemon exited because of a %s signal", received_signal.c_str());
        syslog(LOG_DEBUG, stream_cerr.str().c_str(), "");
        syslog(LOG_DEBUG, stream_cout.str().c_str(), "");
        if (sig != SIGABRT) std::exit(1);
    }

    void Daemon::start() {
        // Setup logging of errors
        struct sigaction crash_handler_action;
        crash_handler_action.sa_handler = sig_crash_handler;
        crash_handler_action.sa_flags = SA_RESTART;
        sigemptyset(&crash_handler_action.sa_mask);
        syslog(LOG_INFO, "The plasticity daemon started...", "");

        setup_ipc(queue_from_normal_daemon_worker_to_client, false, false);
        setup_ipc(queue_from_cpu_affine_daemon_worker_to_client, false, true);
        setup_ipc(queue_from_client_to_normal_daemon_worker, true, false);
        setup_ipc(queue_from_client_to_cpu_affine_daemon_worker, true, true);

#if defined(OPENCL)
        cl_context cpu_context;
        cl_command_queue cpu_queue;
        cl_device_id cpu_handle;
#endif

#if !defined(DEBUG)
        std::streambuf *old_cerr_stream_buf = std::cout.rdbuf();
        std::cerr.rdbuf(stream_cerr.rdbuf());
        std::streambuf *old_cout_stream_buf = std::cout.rdbuf();
        std::cout.rdbuf(stream_cout.rdbuf());
#endif

        int err = 0;
        err |= sigaction(SIGSEGV, &crash_handler_action, &segfault_default_handler);
        err |= sigaction(SIGABRT, &crash_handler_action, &abrt_default_hanlder);
        err |= sigaction(SIGTERM, &crash_handler_action, &term_default_handler);
        if (err != 0) {
            syslog(LOG_ERR, "Could not set signal handlers");
            closelog();
            stop();
            std::exit(1);
        }

        std::thread first_worker(worker,
                                 queue_from_client_to_normal_daemon_worker,
                                 queue_from_normal_daemon_worker_to_client,
                                 0,
#if defined(OPENCL)
                                 std::ref(cpu_context),
                                 std::ref(cpu_queue),
                                 std::ref(cpu_handle),
#endif
                                 std::ref(stream_cout));


        worker(queue_from_client_to_cpu_affine_daemon_worker,
               queue_from_cpu_affine_daemon_worker_to_client,
               1,
#if defined(OPENCL)
               std::ref(cpu_context),
               std::ref(cpu_queue),
               std::ref(cpu_handle),
#endif
               std::ref(stream_cout));

        first_worker.join();
#if !defined(DEBUG)
        std::cout.rdbuf(old_cout_stream_buf);
        std::cerr.rdbuf(old_cerr_stream_buf);
#endif

    }

    void Daemon::stop() {
        tear_down_ipc(queue_from_client_to_normal_daemon_worker, true, false);
        tear_down_ipc(queue_from_client_to_cpu_affine_daemon_worker, true, true);
        tear_down_ipc(queue_from_normal_daemon_worker_to_client, false, false);
        tear_down_ipc(queue_from_cpu_affine_daemon_worker_to_client, false, true);
    }
}
