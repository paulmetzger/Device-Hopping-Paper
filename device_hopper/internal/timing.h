//
// Created by paul on 11/10/2019.
//
#ifndef PLASTICITY_TIMING_H
#define PLASTICITY_TIMING_H

#define CL_TARGET_OPENCL_VERSION 120

#include <chrono>
#include <numeric>
#include <iostream>
#include <CL/cl_platform.h>

namespace plasticity { namespace internal { namespace timing {
    enum class Type {
        ROI,
        CPU_CL_KERN_COMPIL,
        CPU_CLEAN_UP,
        CPU_KERNEL,
        CPU_OPENCL_SETUP_CONTEXT_CREATION,
        CPU_OPENCL_SETUP_OTHER,
        CPU_SETUP_IN_ROI,
        CPU_TIME_IN_CL_WAIT,
        CPU_WAITING_TIME,
        GPU_CL_KERN_COMPIL,
        GPU_CLEAN_UP,
        GPU_KERNEL,
        GPU_OPENCL_SETUP_CONTEXT_CREATION,
        GPU_OPENCL_SETUP_OTHER,
        GPU_SETUP_IN_ROI,
        GPU_TIME_IN_CL_WAIT,
        GPU_WAITING_TIME,
        OPENCL_PLATFORM_QUERY,
        OPENCL_DEVICE_QUERY,
        CPU_EXECUTION_TIME,
        GPU_EXECUTION_TIME,
        DAEMON_STARTUP,

        CL_CREATE_PROGRAM_WITH_BINARY,
        CL_BUILD_PROGRAM,
        CL_CREATE_KERNEL,
        CL_CREATE_COMMAND_QUEUE,
        CL_CREATE_BUFFER,
        CL_ENQUEUE_WRITE_BUFFER,
        CL_ENQUEUE_WRITE_BUFFER_PREP,
        CL_SET_KERNEL_ARG,
        CL_ENQUEUE_NDRANGE_KERNEL,
        CL_ENQUEUE_READ_BUFFER,
        CL_FINISH,
        CL_RELEASE_MEM_OBJECT,
        CL_RELEASE_KERNEL,

        POSIX_SHM_OPEN,
        POSIX_SHM_CLOSE,
        POSIX_SHM_MUNMAP,
        POSIX_SHM_MUNMAP_PTR_LOOKUP,
        POSIX_SHM_MUNMAP_PTR_MAPPING_REMOVAL,
        POSIX_SHM_MMAP,
        POSIX_SHM_MMAP_CHECK_IF_IS_MAPPED,
        POSIX_SHM_MMAP_CREATE_MAPPING,

        COUNT};
    // Create a zero intialised array that holds a timing for each timing type
    unsigned int measurements[(int) Type::COUNT] = {0};
    std::chrono::high_resolution_clock::time_point start_time[(int) Type::COUNT];
#if defined(PROFILING)
    std::vector<cl_ulong> transf_to_dev_us;
    std::vector<cl_ulong> kernel_exec_us;
    std::vector<cl_ulong> transf_from_dev_us;
#endif

    inline void start(timing::Type type) {
        start_time[(int) type] = std::chrono::high_resolution_clock::now();
    }

    inline void set_start(timing::Type type, std::chrono::high_resolution_clock::time_point start_time_point) {
        start_time[(int) type] = start_time_point;
    }

    inline void stop(timing::Type type) {
        auto end = std::chrono::high_resolution_clock::now();
        measurements[(int) type] += std::chrono::duration_cast<std::chrono::microseconds>(
                end - start_time[(int) type]).count();
    }

    inline void reset() {
        for (auto it = std::begin(measurements); it != std::end(measurements); ++it) (*it) = 0;
    }

    inline void print_results() {
        std::cout << "Measurements in ROI:             " << std::endl;
        std::cout << "CPU - OpenCL kernel compilation: " << measurements[(int) Type::CPU_CL_KERN_COMPIL] / 1000.0f << "ms" << std::endl;
        std::cout << "CPU - Clean up:                  " << measurements[(int) Type::CPU_CLEAN_UP] / 1000.0f    << "ms" << std::endl;
        std::cout << "CPU - Kernel:                    " << measurements[(int) Type::CPU_KERNEL] / 1000.0f    << "ms" << std::endl;
        std::cout << "CPU - Waiting:                   " << measurements[(int) Type::CPU_WAITING_TIME] / 1000.0f    << "ms" << std::endl;
        std::cout << "CPU - Time in clWait:            " << measurements[(int) Type::CPU_TIME_IN_CL_WAIT] / 1000.0f    << "ms" << std::endl;
        std::cout << "CPU - Setup in ROI:              " << measurements[(int) Type::CPU_SETUP_IN_ROI] / 1000.0f    << "ms" << std::endl;
        std::cout << "GPU - OpenCL kernel compilation: " << measurements[(int) Type::GPU_CL_KERN_COMPIL] / 1000.0f << "ms" << std::endl;
        std::cout << "GPU - Clean up:                  " << measurements[(int) Type::GPU_CLEAN_UP] / 1000.0f    << "ms" << std::endl;
        std::cout << "GPU - Kernel:                    " << measurements[(int) Type::GPU_KERNEL] / 1000.0f    << "ms" << std::endl;
        std::cout << "GPU - Waiting:                   " << measurements[(int) Type::GPU_WAITING_TIME] / 1000.0f    << "ms" << std::endl;
        std::cout << "GPU - Time in clWait:            " << measurements[(int) Type::GPU_TIME_IN_CL_WAIT] / 1000.0f    << "ms" << std::endl;
        std::cout << "GPU - Setup in ROI:              " << measurements[(int) Type::GPU_SETUP_IN_ROI] / 1000.0f    << "ms" << std::endl;
        std::cout << "Daemon startup:                  " << measurements[(int) Type::DAEMON_STARTUP] / 1000.0f    << "ms" << std::endl;
        std::cout << "Total:                           " << measurements[(int) Type::ROI] / 1000.0f << "ms" << std::endl;
        std::cout << "-------------" << std::endl;
        std::cout << "Measurements outside the ROI:    " << std::endl;
        std::cout << "OpenCL platform query:           " << measurements[(int) Type::OPENCL_PLATFORM_QUERY] / 1000.0f             << "ms" << std::endl;
        std::cout << "OpenCL device query:             " << measurements[(int) Type::OPENCL_DEVICE_QUERY] / 1000.0f               << "ms" << std::endl;
        std::cout << "CPU - OpenCL context creation:   " << measurements[(int) Type::CPU_OPENCL_SETUP_CONTEXT_CREATION] / 1000.0f << "ms" << std::endl;
        std::cout << "CPU - Other OpenCL setup:        " << measurements[(int) Type::CPU_OPENCL_SETUP_OTHER] / 1000.0f            << "ms" << std::endl;
        std::cout << "CPU - Thread execution time:     " << measurements[(int) Type::CPU_EXECUTION_TIME] / 1000.0f                << "ms" << std::endl;
        std::cout << "GPU - OpenCL context creation:   " << measurements[(int) Type::GPU_OPENCL_SETUP_CONTEXT_CREATION] / 1000.0f << "ms" << std::endl;
        std::cout << "GPU - Other OpenCL setup:        " << measurements[(int) Type::GPU_OPENCL_SETUP_OTHER] / 1000.0f            << "ms" << std::endl;
        std::cout << "GPU - Thread execution time:     " << measurements[(int) Type::GPU_EXECUTION_TIME] / 1000.0f                << "ms" << std::endl;
#if defined(PROFILING)
        std::cout << "-------------" << std::endl;
        std::cout << "Measurements with the OpenCL profiling API:" << std::endl;
        for (size_t i = 0; i < transf_to_dev_us.size(); ++i)
            std::cout << i << ". transfer to device: " << transf_to_dev_us[i] / 1000.0f << "ms" << std::endl;
        for (size_t i = 0; i < kernel_exec_us.size(); ++i)
            std::cout << i << ". kernel exec.: " << kernel_exec_us[i] / 1000.0f << "ms" << std::endl;
        for (size_t i = 0; i < transf_from_dev_us.size(); ++i)
            std::cout << i << ". transfer from device: " << transf_from_dev_us[i] / 1000.0f << "ms" << std::endl;

        std::cout << "Transfer to device: " << std::accumulate(transf_to_dev_us.begin(), transf_to_dev_us.end(), 0) / 1000.0f << "ms" << std::endl;
        std::cout << "Kernel exec: " << std::accumulate(kernel_exec_us.begin(), kernel_exec_us.end(), 0) / 1000.0f << "ms" << std::endl;
        std::cout << "Transfer from device: " << std::accumulate(transf_from_dev_us.begin(), transf_from_dev_us.end(), 0) / 1000.0f << "ms" << std::endl;
#endif
    }

    inline void print_cl_api_timings() {
        std::cout << "Measurements: "              << std::endl;
        std::cout << "clCreateProgramWithBinary: " << measurements[(int) Type::CL_CREATE_PROGRAM_WITH_BINARY] << "us" << std::endl;
        std::cout << "clBuildProgram:            " << measurements[(int) Type::CL_BUILD_PROGRAM] << "us" << std::endl;
        std::cout << "clCreateKernel:            " << measurements[(int) Type::CL_CREATE_KERNEL] << "us" << std::endl;
        std::cout << "clCreateCommandQueue:      " << measurements[(int) Type::CL_CREATE_COMMAND_QUEUE] << "us" << std::endl;
        std::cout << "clCreateBuffer:            " << measurements[(int) Type::CL_CREATE_BUFFER] << "us" << std::endl;
        std::cout << "clEnqueueWriteBuffer prep: " << measurements[(int) Type::CL_ENQUEUE_WRITE_BUFFER_PREP] << "us" << std::endl;
        std::cout << "clEnqueueWriteBuffer:      " << measurements[(int) Type::CL_ENQUEUE_WRITE_BUFFER] << "us" << std::endl;
        std::cout << "clSetKernelArg:            " << measurements[(int) Type::CL_SET_KERNEL_ARG] << "us" << std::endl;
        std::cout << "clEnqueueNDRangeKernel:    " << measurements[(int) Type::CL_ENQUEUE_NDRANGE_KERNEL] << "us" << std::endl;
        std::cout << "clEnqueueReadBuffer:       " << measurements[(int) Type::CL_ENQUEUE_READ_BUFFER] << "us" << std::endl;
        std::cout << "clFinish:                  " << measurements[(int) Type::CL_FINISH] << "us" << std::endl;
        std::cout << "clReleaseMemObj:           " << measurements[(int) Type::CL_RELEASE_MEM_OBJECT] << "us" << std::endl;
        std::cout << "clReleaseKernel:           " << measurements[(int) Type::CL_RELEASE_KERNEL] << "us" << std::endl;
    }

    inline void print_posix_shm_timings() {
        std::cout << "Posix shared memory related measurements: " << std::endl;
        std::cout << "shm_open:             " << measurements[(int) Type::POSIX_SHM_OPEN]   << "us" << std::endl;
        std::cout << "close:                " << measurements[(int) Type::POSIX_SHM_CLOSE]  << "us" << std::endl;
        std::cout << "mmap:                 " << measurements[(int) Type::POSIX_SHM_MMAP]   << "us" << std::endl;
        std::cout << "mmap check if mapped: " << measurements[(int) Type::POSIX_SHM_MMAP_CHECK_IF_IS_MAPPED]   << "us" << std::endl;
        std::cout << "mmap create mapping:  " << measurements[(int) Type::POSIX_SHM_MMAP_CREATE_MAPPING]   << "us" << std::endl;
        std::cout << "munmap:               " << measurements[(int) Type::POSIX_SHM_MUNMAP] << "us" << std::endl;
        std::cout << "munmap ptr lookup:    " << measurements[(int) Type::POSIX_SHM_MUNMAP_PTR_LOOKUP] << "us" << std::endl;
        std::cout << "munmap ptr removal:   " << measurements[(int) Type::POSIX_SHM_MUNMAP_PTR_MAPPING_REMOVAL] << "us" << std::endl;
    }
}}}
#endif //PLASTICITY_TIMING_H
