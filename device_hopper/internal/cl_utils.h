//
// Created by paul on 19/12/2019.
//

#ifndef PLASTICITY_CL_UTILS_H
#define PLASTICITY_CL_UTILS_H

#include <CL/cl.h>
#include <functional>
#include <future>
#include <string>

#include "../context.h"
#include "../internal/timing.h"
#include "../internal/utils.h"

using namespace plasticity::internal;

namespace plasticity { namespace cl_utils {
        // These are global variables to make sure they do not go out
        // of scope while they are used in the std::async threads.
        std::string cl_file_path_for_pref_device;
        std::string cl_file_path_for_alt_device;

        volatile bool cpu_setup_is_done = false;
        volatile bool gpu_setup_is_done = false;
        bool platform_array_has_been_populated = false;
        cl_platform_id platforms[2];

        // OpenCL handles
        struct handles {
            cl_kernel        kernel  = NULL;
            cl_command_queue queue   = NULL;
            cl_context       ctx     = NULL;
            cl_device_id     dev     = NULL;
            cl_program       program = NULL;
            Device device = Device::None;
        };
        typedef struct handles handles;

        inline cl_int convert_device_to_cl_device(Device device) {
            cl_int cl_device_type;

            if      (device == Device::GPU) cl_device_type = CL_DEVICE_TYPE_GPU;
            else if (device == Device::CPU) cl_device_type = CL_DEVICE_TYPE_CPU;
            else {
                std::cerr << "Error: Unknown device type" << std::endl;
                std::exit(EXIT_FAILURE);
            }

            return cl_device_type;
        }


        inline void cl_check_return_code(const int& return_code, const std::string& error_message) {
            if (return_code != CL_SUCCESS) {
                std::cout << "ERROR: " << error_message << std::endl;
                std::cout << "OpenCL error type: ";
                if (return_code == CL_INVALID_COMMAND_QUEUE)              std::cout << "CL_INVALID_COMMAND_QUEUE";
                else if (return_code == CL_INVALID_CONTEXT)               std::cout << "CL_INVALID_CONTEXT";
                else if (return_code == CL_INVALID_MEM_OBJECT)            std::cout << "CL_INVALID_MEM_OBJECT";
                else if (return_code == CL_INVALID_VALUE)                 std::cout << "CL_INVALID_VALUE";
                else if (return_code == CL_INVALID_EVENT_WAIT_LIST)       std::cout << "CL_INVALID_EVENT_WAIT_LIST";
                else if (return_code == CL_MEM_OBJECT_ALLOCATION_FAILURE) std::cout << "CL_MEM_OBJECT_ALLOCATION_FAILURE";
                else if (return_code == CL_OUT_OF_HOST_MEMORY)            std::cout << "CL_OUT_OF_HOST_MEMORY";
                else std::cout << return_code;
                std::cout << std::endl;

                std::exit(EXIT_FAILURE);
            }
        }

        void init_platforms() {
            // Do not query the ICD again if the platform has already been populated.
            // This is relevant if the 'preinitialise cl_utils contexts' command line flag is set.
            if (!platform_array_has_been_populated) {
                // Query the ICD for platforms.
                // Then select a platform and get the device handle.
                timing::start(timing::Type::OPENCL_PLATFORM_QUERY);
                cl_uint num_platforms = 2;
                cl_int err = clGetPlatformIDs(2, platforms, &num_platforms);
                plasticity::cl_utils::cl_check_return_code(err, "Could not initialise the OpenCL platform");
                timing::stop(timing::Type::OPENCL_PLATFORM_QUERY);
                platform_array_has_been_populated = true;
            }
        }

        void populate_cl_context_and_device_handles(struct handles& handles,
                                                    const Device target_device) {
            timing::start(timing::Type::OPENCL_DEVICE_QUERY);
            cl_int device_type;
            cl_int err = 0;
            
            if (target_device == Device::GPU) device_type = CL_DEVICE_TYPE_GPU;
            else if (target_device == Device::CPU) device_type = CL_DEVICE_TYPE_CPU;
            else {
                std::cerr << "Error: Unknown device type" << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // Find the Opencl device ID for the requested device
            err |= clGetDeviceIDs(
                    platforms[0],
                    device_type,
                    1,
                    &handles.dev,
                    NULL);
            if (err == -1) {
#ifdef DEBUG
                std::cout << "INFO: First platform does not offer the target device. "
                             "Trying to use the second platform..." << std::endl;
#endif
                err = clGetDeviceIDs(
                        platforms[1],
                        device_type,
                        1,
                        &handles.dev,
                        NULL);
            }
            plasticity::cl_utils::cl_check_return_code(err, "Could not initialise the OpenCL platform or device -- here");

/*            // Create sub-device if requested
            if (ctx.control_cpu_core_count && target_device == Device::CPU) {
                cl_device_id subdevice;
                cl_uint subdevice_count = 1;
                cl_device_partition_property properties[] = {
                        CL_DEVICE_PARTITION_BY_COUNTS,
                        ctx.cpu_core_count,
                        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
                        0
                };
                err = clCreateSubDevices(
                        handles.dev,
                        properties,
                        1,
                        &subdevice,
                        &subdevice_count);
                plasticity::cl_utils::cl_check_return_code(err, "Could not create a sub-device");
                handles.dev = subdevice;
            }*/
            timing::stop(timing::Type::OPENCL_DEVICE_QUERY);

            // Create the OCL ctx handle
            timing::Type context_creation_measurement_type = (target_device == Device::GPU ?
                                                              timing::Type::GPU_OPENCL_SETUP_CONTEXT_CREATION
                                                                                           :
                                                              timing::Type::CPU_OPENCL_SETUP_CONTEXT_CREATION);
            timing::start(context_creation_measurement_type);
            handles.ctx = clCreateContext(NULL, 1, &handles.dev, NULL, NULL, &err);
            plasticity::cl_utils::cl_check_return_code(err, "Could not initialise the OpenCL ctx");
            internal::timing::stop(context_creation_measurement_type);
        }

        void popul_kern_h(struct handles& handles,
                          const std::string& kernel_name,
                          const std::string& cl_file_path,
                          const std::string& flags,
                          const Device target_device) {
            // Load OpenCL kernel
            cl_int err = CL_SUCCESS;
            FILE *program_handle = fopen(cl_file_path.c_str(), "r");
            if (program_handle == NULL) {
                std::cerr << "Error: Could not open the OpenCL source code file" << std::endl;
                std::cerr << "File path: " << cl_file_path << std::endl;
                std::exit(EXIT_FAILURE);
            }
            err = fseek(program_handle, 0, SEEK_END);
            size_t program_size = ftell(program_handle);
            rewind(program_handle);
            char *program_buffer = (char *) malloc(program_size + 1);
            program_buffer[program_size] = '\0';
            fread(program_buffer, sizeof(char), program_size, program_handle);
            fclose(program_handle);

            cl_int binary_status = 0;
            handles.program = clCreateProgramWithBinary(handles.ctx,
                                                           1,
                                                           &handles.dev,
                                                           &program_size,
                                                           (const unsigned char **) &program_buffer,
                                                           &binary_status,
                                                           &err);
            plasticity::cl_utils::cl_check_return_code(binary_status, "Error: Wrong binary status.");
            free(program_buffer);

            // Load a precompiled executable or compile the OpenCL code
            err |= clBuildProgram(handles.program ,
                                  0,
                                  NULL,
                                  flags.c_str(),
                                  NULL,
                                  NULL);
            plasticity::cl_utils::cl_check_return_code(err, "Could not compile OpenCL code");

            // Create the kernel handle
            handles.kernel = clCreateKernel(handles.program , kernel_name.c_str(), &err);
            plasticity::cl_utils::cl_check_return_code(err, "Could not create a kernel handle");
        }

        /**
         * Does the actual setup and populates OpenCL handles.
         * Multiple instances of this function are executed when multiple devices are set up in parallel.
         * Only one instance of this function is executed with back-to-back and best-available scheduling.
         * Post condition: All OpenCL handles in the handles struct are populated
         */
        bool _setup_create_handles(
                const std::string& cl_file_path,
                const std::string& flags,
                struct handles& handles,
                const std::string& kernel_name,
                cl_platform_id (&platforms)[2],
                const Device& target_device,
                plasticity::setup::Context& ctx) {
            cl_int err = 0;
            if (handles.ctx == NULL) populate_cl_context_and_device_handles(handles, target_device);

            // Load and compile the .cl file
            timing::Type kernel_compilation_measurement_type = (target_device == Device::GPU ?
                                                                timing::Type::GPU_CL_KERN_COMPIL
                                                                                             :
                                                                timing::Type::CPU_CL_KERN_COMPIL);
            internal::timing::start(kernel_compilation_measurement_type);
            FILE *program_handle = fopen(cl_file_path.c_str(), "r");
            if (program_handle == NULL) {
                std::cerr << "Error: Could not open the OpenCL source code file" << std::endl;
                std::cerr << "File path: " << cl_file_path << std::endl;
                std::exit(EXIT_FAILURE);
            }
            err |= fseek(program_handle, 0, SEEK_END);
            size_t program_size = ftell(program_handle);
            rewind(program_handle);
            char *program_buffer = (char *) malloc(program_size + 1);
            program_buffer[program_size] = '\0';
            fread(program_buffer, sizeof(char), program_size, program_handle);
            fclose(program_handle);
            cl_program program;
            cl_int binary_status = 0;
            program = clCreateProgramWithBinary(
                        handles.ctx,
                        1,
                        &handles.dev,
                        &program_size,
                        (const unsigned char **) &program_buffer,
                        &binary_status,
                        &err);
            plasticity::cl_utils::cl_check_return_code(binary_status, "Error: Wrong binary status.");
            if (target_device == Device::CPU) {
                plasticity::cl_utils::cl_check_return_code(err, "Could not create CPU program handle");
            } else if (target_device == Device::GPU) {
                plasticity::cl_utils::cl_check_return_code(err, "Could not create GPU program handle");
            } else {
                std::cerr << "Error: Unknown device" << std::endl;
                std::exit(1);
            }
            free(program_buffer);

            // Load a precompiled executable or compile the OpenCL code
            err |= clBuildProgram(program, 0, NULL, flags.c_str(), NULL, NULL);
            plasticity::cl_utils::cl_check_return_code(err, "Could not compile OpenCL code");
            internal::timing::stop(kernel_compilation_measurement_type);

            // Create the kernel handle
            timing::Type other_setup_measurement_type = (target_device == Device::GPU ?
                                                         timing::Type::GPU_OPENCL_SETUP_OTHER :
                                                         timing::Type::CPU_OPENCL_SETUP_OTHER);
            internal::timing::start(other_setup_measurement_type);
            handles.kernel = clCreateKernel(program, kernel_name.c_str(), &err);
            plasticity::cl_utils::cl_check_return_code(err, "Could not create a kernel handle");

            // Create the queue handle
            handles.queue = clCreateCommandQueue(handles.ctx, handles.dev, 0, &err);
            plasticity::cl_utils::cl_check_return_code(err, "Could not create a command queue");
            internal::timing::stop(other_setup_measurement_type);

            // Bookkeeping that is used by the application if OpenCL setup and kernel
            // execution are overlapped.
            // This is the case with plastic scheduling if the 'preinitialise OpenCL' command line
            // parameter is not set.
            if (target_device == Device::CPU) cpu_setup_is_done = true;
            else if (target_device == Device::GPU) gpu_setup_is_done = true;
            else {
                std::cerr << "Error: Unknown device" << std::endl;
                std::exit(1);
            }
            return true;
        }


        /**
         * Sets up OpenCL for a single device. This function is called with back-to-back and best-available
         * scheduling.
         */
        inline void opencl_set_up_of_single_device(
                const std::function<std::string(bool, Device)> get_kernel_filename,
                const std::string& kernel_name,
                const Device target_device,
                struct handles& handles,
                const std::string& flags,
                plasticity::setup::Context& ctx) {
            init_platforms();

            timing::start(timing::Type::OPENCL_DEVICE_QUERY);
            const std::string cl_file_path = ctx.library_path + "/plasticity/opencl_files/" + \
                get_kernel_filename(true, target_device);
            timing::stop(timing::Type::OPENCL_DEVICE_QUERY);

            _setup_create_handles(cl_file_path, flags, handles, kernel_name, platforms, target_device, ctx);
        }
}}

#endif //PLASTICITY_CL_UTILS_H
