//
// Created by Paul Metzger on 11/06/2020.
//

#ifndef PLASTICITY_NN_H
#define PLASTICITY_NN_H

#if defined(OPENCL)
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#include <boost/asio/ip/host_name.hpp>
#include <fcntl.h>
#include <future>
#include <mutex>

#if defined(OPENMP)
#include <omp.h>
#endif

#include <signal.h>
#include <string>
#include <sched.h>
#include <thread>
#include <unistd.h>
#include <zconf.h>

#include "../device_hopper/context.h"
#if defined(OPENCL)
#include "../device_hopper/internal/cl_utils.h"
#endif
#include "../device_hopper/internal/device.h"
#include "../device_hopper/internal/timing.h"
#include "../device_hopper/internal/utils.h"

// END =========== original benchmark specifics =========

//%FORWARD_DECLARATION%

namespace timing = plasticity::internal::timing;

namespace plasticity { namespace  kernels { namespace impl {
    using namespace internal;

    std::thread pref_dev_thr;
    std::thread alt_dev_thr;

#if defined(OPENCL)
    typedef struct handles : cl_utils::handles {
        struct {
            //%OPENCL_BUFFER_HANDLES%
        } opencl;
#elif defined(OMP)
    typedef struct handles {
        Device device = Device::None;
#endif
        struct {
            //%CUDA_BUFFER_HANDLES%
        } cuda;
    } handles;
    //%BUFFER_SIZES_OF_INTERMEDIATE_BUFFERS_FOR_INDIRECT_MEMORY_ACCESSES%

#if defined(OPENCL)
    const std::string kernel_name = "opencl_kernel";

    std::string get_kernel_filename(const Device device) {
        std::string path;
        path = "/%OPENCL_FILE_NAME_PREFIX%";
        std::string suffix;
        if(device == Device::GPU) {
            suffix = "GPU";
        } else if (device == Device::CPU) {
            suffix = "CPU";
        } else {
            std::cerr << "Error: Unknown device" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::string hostname = boost::asio::ip::host_name();
        path = path + "_" + hostname + "_" + suffix + ".bin";
        //path += "_with_slicing.bin";
        return path;
    }
#endif

    //%BOILER_PLATE_CODE_FOR_TEXTURES%

    /**
     * This function creates OpenCL buffers for the kernel.
     */
    void cuda_create_buffers(
            //%BUFFERS_CONTAINING_INDICES_DECLS%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            impl::handles& h,
            plasticity::setup::Context& ctx) {
#if defined(DEBUG)
        std::cout << "cuda_create_buffers" << std::endl;
#endif
        const size_t slice_size = ctx.sched.get_gpu_x_slice_size();
        //%CUDA_MALLOC%

#if defined(DEBUG)
        std::cout << "Finished cuda_create_buffers" << std::endl;
#endif
    }

    /**
     * This function releases OpenCL buffers with clReleaseMemObject
     */
    void cleanup(void* h_in) {
        handles* h = (handles*) h_in;
        timing::stop(h->device == Device::CPU ? timing::Type::CPU_KERNEL : timing::Type::GPU_KERNEL);
        timing::start(h->device == Device::CPU ? timing::Type::CPU_CLEAN_UP : timing::Type::GPU_CLEAN_UP);
#ifdef DEBUG
        std::cout << "DEBUG: Releasing OpenCL resources... " << std::flush;
#endif

#if defined(OPENCL)
        if (h->device == Device::CPU) {
            cl_int err = CL_SUCCESS;
            //%OPENCL_RELEASE_MEM_OBJECT%

            if (h->kernel) {
                err |= clReleaseKernel(h->kernel);
                h->kernel = NULL;
            }
            if (h->program) {
                err |= clReleaseProgram(h->program);
                h->program = NULL;
            }
            plasticity::cl_utils::cl_check_return_code(err, "Could not free all OpenCL resources");
        }
#endif
        if (h->device == Device::GPU) {
            //%CUDA_RELEASE_MEM_OBJECT%
        }

#ifdef DEBUG
        std::cout << "Done" << std::endl;
#endif
        timing::stop(h->device == Device::CPU ? timing::Type::CPU_CLEAN_UP : timing::Type::GPU_CLEAN_UP);
        timing::start(h->device == Device::CPU ? timing::Type::CPU_KERNEL : timing::Type::GPU_KERNEL);
    }

    inline void transfer_random_access_buffers_from_device(
            //%BUFFER_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
            handles &h
    ) {
        //%ONE_TIME_DATA_TRANSFERS_FROM_THE_DEVICE%
    }

    inline void transfer_random_access_buffers_to_device(
            //%ABORT_SLICE_FLAG_DECL%
            //%BUFFER_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
            handles &h,
            plasticity::setup::Context& ctx
    ) {
        //%ONE_TIME_DATA_TRANSFERS_TO_THE_GPU%
    }

    //%TRANSFER_DATA_TO_DEVICE_FUNCTION_STUB%

    //%TRANSFER_DATA_FROM_DEVICE_FUNCTION_STUB%

    //%CUDA_KERNEL%

#if defined(OPENCL)
    void ocl_create_buffers(
            //%BUFFER_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
            impl::handles& h) {
#if defined(DEBUG)
        std::cout << "create_buffers" << std::endl;
#endif
        cl_int err = CL_SUCCESS;
        //%OPENCL_BUFFER_CREATION%

#if defined(DEBUG)
        std::cout << "Finished create_buffers" << std::endl;
#endif
    }

    void set_kernel_arguments(
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%SCALAR_PARAMETER_DECLS%
            impl::handles& h) {
#ifdef DEBUG
        std::cout << "DEBUG: Setting OpenCL kernel arguments that are constant for all slices... " << std::flush;
#endif
        // Set up kernel arguments
        timing::Type other_setup_measurement_type = (timing::Type::CPU_OPENCL_SETUP_OTHER);
        internal::timing::start(other_setup_measurement_type);
        cl_int err = CL_SUCCESS;
        //%OPENCL_KERNEL_ARGUMENTS%
        plasticity::cl_utils::cl_check_return_code(err, "Could not add kernel arguments");
        internal::timing::stop(other_setup_measurement_type);
#ifdef DEBUG
        std::cout << "Done" << std::endl;
#endif
    }

    inline void ocl_run_kernel(
            const size_t (&offsets)[2],
            const size_t (&slice_sizes)[2],
            handles& h,
            const plasticity::setup::Context& ctx) {
#ifdef DEBUG
        std::cout << "DEBUG: Running kernel... " << std::flush;
#endif
        cl_int err = CL_SUCCESS;
#if defined(KERNEL_IS_1D)
        const size_t local_work_size  = BLOCK_SIZE_X;
        const size_t global_work_size = slice_sizes[0] << LOG2_BLOCK_SIZE_X;
        const size_t offset           = offsets[0]     << LOG2_BLOCK_SIZE_X;
#elif defined(KERNEL_IS_2D)
        const size_t local_work_size[2]  = {BLOCK_SIZE_X, BLOCK_SIZE_Y};
        const size_t global_work_size[2] = {slice_sizes[0] << LOG2_BLOCK_SIZE_X, slice_sizes[1] << LOG2_BLOCK_SIZE_Y};
        const size_t offset[2]           = {offsets[0] << LOG2_BLOCK_SIZE_X, offsets[1] << LOG2_BLOCK_SIZE_Y};
#else
#error Define either KERNEL_IS_1D or KERNEL_IS_2D
#endif
        //%DYNAMIC_OPENCL_KERNEL_PARAMETERS%
        err = clEnqueueNDRangeKernel(
                h.queue,
                h.kernel,
#if defined(KERNEL_IS_1D)
                1,                 // work_dim
                &offset,           // global_work_offset
                &global_work_size, // global_work_size
                &local_work_size,  // local_work_size
#elif defined(KERNEL_IS_2D)
                2,                 // work_dim
                offset,           // global_work_offset
                global_work_size, // global_work_size
                local_work_size,  // local_work_size
#endif
                0,                 //num_events_in_wait_list
                nullptr,           //event_wait_list
                nullptr);
        plasticity::cl_utils::cl_check_return_code(err, "Could not launch the OpenCL kernel");

        err |= clFinish(h.queue);
        plasticity::cl_utils::cl_check_return_code(err, "Could not launch the OpenCL kernel");

#ifdef DEBUG
        std::cout << "Done" << std::endl;
#endif
    }
#else
    inline void cpu_kernel(
            //%BUFFER_PARAMETER_DECLS%
            //%SCALAR_PARAMETER_DECLS%
            const size_t (&offsets)[2],
            const size_t (&slice_sizes)[2]) {
            //%OPENMP_KERNEL_CODE%
    }
#endif

    /**
     * This function launches an OpenCL kernel and is called once for each slice.
     */
    inline void cuda_run_kernel(
            //%ABORT_SLICE_FLAG_DECL%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%SCALAR_PARAMETER_DECLS%
            const size_t (&offsets)[2],
            const size_t (&slice_sizes)[2],
            const Device& current_device,
            handles& h,
            plasticity::setup::Context& ctx) {
#ifdef DEBUG
        std::cout << "DEBUG: Running kernel... " << std::flush;
#endif
        //%CUDA_KERNEL_CALL%
        cudaDeviceSynchronize();
        //%CHECK_IF_THE_SLICE_SHOULD_BE_KILLED%


#ifdef DEBUG
        std::cout << "Done" << std::endl;
#endif
    }
}


#if defined(__INTEL_COMPILER)
    /**
     * This is a workaround for a bug in the Intel compiler.
     * https://community.intel.com/t5/Intel-C-Compiler/icpc-pthread-cancel-raises-exception-in-C-program/td-p/1137054
     * Pthread_cancel raises SIGABRT instead of exiting the target thread.
     * This handler catches SIGABRT and exits the thread as intended.
     */
    jmp_buf env;
    void workaround_sigabrt_handler(int sig) {
#if defined(DEBUG)
        std::cout << "DEBUG: Entered the handler for the workaround for the pthread_cancel bug in the Intel compiler." << std::endl;
#endif
        longjmp(env, 1);
    }
#endif

    /**
     * Multiple instance of this function are executed in parallel.
     * At the moment two instances are create. One for the CPU and one for the GPU.
     * This function waits for the device that is set via the parameter 'dev' to become available.
     * It executes some setup code, and then enters a loop that executes either the slices if the set device
     * is the alternative device or the remaining iteration space if the device is the preferred device and some
     * part of the iteration has been processed on the alternative device.
     *
     * The code below is mostly generic. You probably don't want to change it a lot.
     * Typical things that require slight changes are the data transfers.
     * Move the data transfers code outside the loop if the transfers are not slicing aware.
     * Slicing unaware data transfers are OK if the input data is very small e.g. a few 10s of MB.
     *
     * Remove code in blocks that are surrounded by 'defined(SLICED_DATA_TRANSFERS)' if the benchmark
     * does not use slicing aware data transfers.
     * It makes sometimes sense to chunk the transfer to the device but not the transfer from the device because the buffer
     * with the results that need to be transferred from the GPU to the CPU is very small.
     */
    void device_management_code(
            //%BUFFER_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
#if defined(ABANDON_SLICES)
            bool& computation_has_migrated_to_the_fast_device,
#endif
            //%SCALAR_PARAMETER_DECLS%
            const Device dev_in,
            impl::handles& pref_cuda_h,
            impl::handles& alt_cuda_h,
            plasticity::setup::Context& ctx) {
        timing::start(dev_in == Device::CPU ? timing::Type::CPU_SETUP_IN_ROI : timing::Type::GPU_SETUP_IN_ROI);

#if defined(__INTEL_COMPILER)
        if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL) != 0)
            utils::exit_with_err("Could not set the pthread cancel state");
#endif

        // Create bookkeeping variables
        alignas(64) size_t current_offsets[2] = {0, 0};
        size_t current_slice_sizes[2]         = {0, 0};

        // OpenCL setup
        const Device dev      = dev_in;
        const Device pref_dev = ctx.sched.get_preferred_device();
        pref_cuda_h.device    = pref_dev;
        alt_cuda_h.device     = pref_dev == Device::CPU ? Device::GPU : Device::CPU;

        impl::handles& h = (dev == pref_dev) ? pref_cuda_h : alt_cuda_h;

        // Block until the device becomes available
        timing::stop(dev  == Device::CPU ? timing::Type::CPU_SETUP_IN_ROI : timing::Type::GPU_SETUP_IN_ROI);
        timing::start(dev == Device::CPU ? timing::Type::CPU_WAITING_TIME : timing::Type::GPU_WAITING_TIME);
#if defined(__INTEL_COMPILER)
        if (dev ==pref_dev) {
            if (setjmp(env) == 0) {
                struct sigaction new_workaround_action, old_crash_handler_action;
                new_workaround_action.sa_handler = workaround_sigabrt_handler;
                new_workaround_action.sa_flags = SA_RESTART;
                sigemptyset(&new_workaround_action.sa_mask);
                if (sigaction(SIGABRT, &new_workaround_action, &old_crash_handler_action) != 0) {
                    std::cerr << "Could not set SIGABRT handler" << std::endl;
                    std::exit(1);
                }

                if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL) != 0)
                    utils::exit_with_err("Could not set the pthread cancel state");
                if (pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL) != 0)
                    utils::exit_with_err("Could not set the pthread cancel type");
                ctx.sched.wait_for_device(dev);
                if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL) != 0)
                    utils::exit_with_err("Could not set the pthread cancel state");
                if (sigaction(SIGABRT, &old_crash_handler_action, NULL)) {
                    std::cerr << "Could not set old SIGABRT handler" << std::endl;
                    std::exit(1);
                }
            } else {
                pthread_exit(0);
            }
        } else ctx.sched.wait_for_device(dev);
#else
        ctx.sched.wait_for_device(dev);
#endif

        timing::stop(dev  == Device::CPU ? timing::Type::CPU_WAITING_TIME : timing::Type::GPU_WAITING_TIME);
        timing::start(dev == Device::CPU ? timing::Type::CPU_SETUP_IN_ROI : timing::Type::GPU_SETUP_IN_ROI);

#if defined(DEBUG)
        std::cout << "DEBUG " << utils::convert_device_to_str(h.device) << ": thread joins the computation" << std::endl;
#endif
        if (ctx.sched.kill_current_slice(dev)) {
            impl::cleanup(&h);
            ctx.sched.free_device(h.device);
            return;
        }

        // Initialise the OpenCL buffers for the new target device
        if (dev == Device::GPU) {
            impl::cuda_create_buffers(
                //%BUFFERS_CONTAINING_INDICES%
                //%BUFFER_ELEMENT_COUNTS%
                h,
                ctx);
#if defined(OPENCL)
        } else {
            ocl_create_buffers(
                    //%BUFFERS%
                    //%BUFFER_ELEMENT_COUNTS%
                    //%BUFFER_ELEMENT_SIZES%
                    h);
            impl::set_kernel_arguments(
                    //%BUFFER_ELEMENT_COUNTS%
                    //%SCALAR_PARAMETERS%
                    h);
        }
#elif defined(OMP)
        }
#else
#error "ERROR: Define either 'OPENCL' or 'OMP'"
#endif

        timing::stop( dev == Device::CPU ? timing::Type::CPU_SETUP_IN_ROI : timing::Type::GPU_SETUP_IN_ROI);
        timing::start(dev == Device::CPU ? timing::Type::CPU_KERNEL       : timing::Type::GPU_KERNEL);
#if PREFERRED_DEVICE == CPU
        bool abort_slice = false;
#endif
        if (dev == Device::GPU) {
            transfer_random_access_buffers_to_device(
                    //%ABORT_SLICE_FLAG%
                    //%BUFFERS%
                    //%BUFFER_ELEMENT_COUNTS%
                    //%BUFFER_ELEMENT_SIZES%
                    h,
                    ctx);
            //%RETURN_IF_THE_SLICE_IS_ABORTED%
        }

// The code from here until END_GENERIC_CODE is the same for all benchmark applications and most likely
// does not need to be changed.
        ctx.sched.book_keeping.lock();

#if defined(ABANDON_SLICES)
        if (dev == pref_dev) computation_has_migrated_to_the_fast_device = true;
#endif
        if (dev == Device::GPU) {
            //%TRANSFER_RANDOM_ACCESS_OUTPUT_BUFFER_IF_THE_GPU_IS_THE_PREFERRED_DEVICE%
        }
        if (!ctx.sched.computation_is_on_preferred_device()) {
            ctx.sched.set_device(dev);
            if (dev == pref_dev) ctx.sched.set_computation_is_on_preferred_device();
            while (!ctx.sched.computation_is_done(current_slice_sizes, current_offsets)
                   && !ctx.sched.computation_has_migrated(dev)) {
                ctx.sched.get_new_offsets_and_slice_sizes(current_slice_sizes, current_offsets);

                // Signal the thread waiting for the preferred device to exit
                if (dev != pref_dev && ctx.sched.is_final_slice(current_slice_sizes, current_offsets)) {
#if defined(DEBUG)
                    std::cout << "Sending exit signal to the other thread" << std::endl;
#endif
                    pthread_cancel(impl::pref_dev_thr.native_handle());
                }
                ctx.sched.book_keeping.unlock();
//END_GENERIC_CODE

#ifdef DEBUG
                std::cout << "X offset: " << current_offsets[0] << " Y offset: " << current_offsets[1] << std::endl;
                std::cout << "X slice size: " << current_slice_sizes[0] << " Y slice size: " << current_slice_sizes[1]
                          << std::endl;
#endif
                //%TRANSFER_DATA_TO_DEVICE_FUNCTION_CALL%

                //%SET_OLD_OFFSET_AND_SLICE_SIZE_FOR_NON_IDEMPOTENT_KERNEL%

                // Launch the OpenCL kernel
                if (dev == Device::CPU) {
#if defined(OPENCL)
                    impl::ocl_run_kernel(current_offsets, current_slice_sizes, h, ctx);
#elif defined(OMP)
                    const size_t offset = current_offsets[0] << 8;
                    const size_t slice_size = current_slice_sizes[0] << 8;
                    impl::cpu_kernel(
                            //%BUFFERS%
                            //%SCALAR_PARAMETERS%
                            current_offsets,
                            current_slice_sizes);
#else
#error "Error: Set either 'OMP' or 'OpenCL'"
#endif
                } else {
                    impl::cuda_run_kernel(
                            //%ABORT_SLICE_FLAG%
                            //%BUFFER_ELEMENT_COUNTS%
                            //%SCALAR_PARAMETERS%
                            current_offsets,
                            current_slice_sizes,
                            dev,
                            h,
                            ctx);
                    //%RETURN_IF_THE_SLICE_IS_ABORTED_RIGHT_AFTER_THE_KERNEL_LAUNCH%
                    //%REDUCTION_RETURN_IF_THE_SLICE_IS_ABORTED%
                }

                //%TRANSFER_DATA_FROM_DEVICE_FUNCTION_CALL%

                ctx.sched.book_keeping.lock();
                //%SET_OLD_OFFSET_AND_SLICE_SIZE_FOR_IDEMPOTENT_KERNEL%
            }
        }
        ctx.sched.book_keeping.unlock();

        if (dev == Device::GPU) transfer_random_access_buffers_from_device(
                    //%BUFFERS%
                    //%BUFFER_ELEMENT_COUNTS%
                    //%BUFFER_ELEMENT_SIZES%
                    h);
        impl::cleanup(&h);

        timing::stop(dev == Device::CPU ? timing::Type::CPU_KERNEL : timing::Type::GPU_KERNEL);

#if defined(DEBUG)
        std::cout << "Kernel execution completed on " << (dev == Device::CPU ? "CPU" : "GPU") << std::endl;
#endif

        // Clean up
        ctx.sched.free_device(dev);

#if defined(DEBUG)
        std::cout << "The " << (dev == Device::CPU ? "CPU" : "GPU") << " thread exits" << std::endl;
#endif
    }

    /**
     * This function is called from the cpp file in the benchmarks directory.
     * This launches a management thread for each OpenCL device in binaries that are used in experiments with plasticity.
     */
    void parallel_for(
#if defined(OPENCL)
#if defined(EXECUTABLE)
        impl::handles cpu_handles,
#else
        cl_context& cpu_context,
        cl_command_queue& cpu_queue,
        cl_device_id& cpu_handle,
#endif
#endif
        const size_t iteration_space_start_x,
        const size_t iteration_space_start_y,
        const size_t iteration_space_end_x,
        const size_t iteration_space_end_y,
        //%BUFFER_PARAMETER_DECLS%
        //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
        //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
        //%SCALAR_PARAMETER_DECLS%
        plasticity::setup::Context& ctx) {

        alignas(64) impl::handles pref_h;
        alignas(64) impl::handles alt_h;
#if defined(EXECUTABLE) && defined(OPENCL)
        alt_h = pref_h = cpu_handles;
#endif
        Device pref_dev = ctx.sched.get_preferred_device();

#if !defined(EXECUTABLE) && defined(OPENCL)
        // Populate CL handles
        //const Device alt_dev = (pref_dev == Device::GPU) ? Device::CPU : Device::GPU;
        if (pref_dev == Device::GPU) {
            alt_h.queue  = cpu_queue;
            alt_h.ctx    = cpu_context;
            alt_h.dev    = cpu_handle;
        } else {
            pref_h.queue = cpu_queue;
            pref_h.ctx   = cpu_context;
            pref_h.dev   = cpu_handle;
        }

        /**
         * In the interest of fairness we do not measure the time that it takes to load the kernel.
         * The daemon benefits from caching and so measuring this time would unfairly disadvantage the baseline.
         */

        impl::handles& cpu_handles = (pref_dev == Device::CPU ? pref_h : alt_h);

        std::string cl_file_path = ctx.library_path + "/programming_model/benchmarks/opencl_files/" + impl::get_kernel_filename(Device::CPU);
        plasticity::cl_utils::popul_kern_h(cpu_handles, impl::kernel_name, cl_file_path, "", pref_dev);
#endif

        //%BUFFER_FOR_INTERMEDIATE_RESULTS_OF_REDUCTION_ON_THE_GPU%

        // The ROI starts after this barrier
        ctx.wait_for_other_benchmark_applications(AppType::BENCHM);

        // --- START OF ROI ---
        internal::timing::start(internal::timing::Type::ROI);

        ctx.sched.wait_for_device_with_back_to_back_scheduling_heuristic();
        //%SET_IF_KERNEL_IS_IDEMPOTENT%

#if defined(IS_REDUCTION)
        const size_t iteration_space_size_x = iteration_space_end_x - iteration_space_start_x;
        //const size_t iteration_space_size_y = iteration_space_end_y - iteration_space_start_y;
#else
        const size_t iteration_space_size_x = (iteration_space_end_x - iteration_space_start_x) >> LOG2_BLOCK_SIZE_X;
#if defined(KERNEL_IS_2D)
        const size_t iteration_space_size_y = (iteration_space_end_y - iteration_space_start_y) >> LOG2_BLOCK_SIZE_Y;
#endif
#endif
        //if (numRecords % 64) global_workSize[0] += 64 - (numRecords % 64);
        //std::cout << "Iteration space size: " << global_workSize[0] << std::endl;
#if defined(KERNEL_IS_1D)
        ctx.sched.set_problem_size(iteration_space_size_x, 1);
#elif defined(KERNEL_IS_2D)
        ctx.sched.set_problem_size(iteration_space_size_x, iteration_space_size_y);
#else
#error Either the KERNEL_IS_1D or KERNEL_IS_2D macro has to be set.
#endif

#if defined(ABANDON_SLICES)
        bool computation_has_migrated_to_the_fast_device = false;
#endif
#if defined(ABANDON_SLICES)
        impl::alt_dev_thr = std::thread(
            device_management_code,
            //%BUFFERS%
            //%BUFFER_ELEMENT_COUNTS%
            //%BUFFER_ELEMENT_SIZES%
            std::ref(computation_has_migrated_to_the_fast_device),
            //%SCALAR_PARAMETERS%
            pref_dev == Device::CPU ? Device::GPU : Device::CPU,
            std::ref(pref_h),
            std::ref(alt_h),
            std::ref(ctx));
#endif

        impl::pref_dev_thr = std::thread(
            device_management_code,
            //%BUFFERS%
            //%BUFFER_ELEMENT_COUNTS%
            //%BUFFER_ELEMENT_SIZES%
#if defined(ABANDON_SLICES)
            std::ref(computation_has_migrated_to_the_fast_device),
#endif
            //%SCALAR_PARAMETERS%
            pref_dev,
            std::ref(pref_h),
            std::ref(alt_h),
            std::ref(ctx));

        timing::Type wait_t;

#if !defined(ABANDON_SLICES)
        wait_t = (pref_dev == Device::CPU) ? timing::Type::GPU_EXECUTION_TIME : timing::Type::CPU_EXECUTION_TIME;
        timing::start(wait_t);
        device_management_code(
            //%BUFFERS%
            //%BUFFER_ELEMENT_COUNTS%
            //%BUFFER_ELEMENT_SIZES%
            //%SCALAR_PARAMETERS%
            pref_dev == Device::CPU ? Device::GPU : Device::CPU,
            std::ref(pref_h),
            std::ref(alt_h),
            ctx);
        timing::stop(wait_t);
#endif

        wait_t = (pref_dev == Device::CPU) ? timing::Type::CPU_EXECUTION_TIME : timing::Type::GPU_EXECUTION_TIME;
        timing::start(wait_t);
        impl::pref_dev_thr.join();
        timing::stop(wait_t);

#if defined(ABANDON_SLICES)
        if (!computation_has_migrated_to_the_fast_device) {
            wait_t = (pref_dev == Device::CPU) ? timing::Type::GPU_EXECUTION_TIME : timing::Type::CPU_EXECUTION_TIME;
            timing::start(wait_t);
            impl::alt_dev_thr.join();
            timing::stop(wait_t);
        }
#endif

        //%COMBINE_INTERMEDIATE_RESULTS_OF_REDUCTION_THAT_WERE_COMPUTED_ON_THE_CPU_AND_GPU
        internal::timing::stop(internal::timing::Type::ROI);
#if defined(ABANDON_SLICES)
        if (computation_has_migrated_to_the_fast_device) impl::alt_dev_thr.join();
#endif

        //%COMPUTE_FINAL_REDUCTION_RESULT_BASED_ON_THE_INTERMEDIATE_RESULTS%

        // Print measurements and extra information
        ctx.sched.print_info();
        internal::timing::print_results();
    }
}}

#endif //PLASTICITY_NN_H
