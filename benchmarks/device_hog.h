//
// Created by paul on 30/10/2019.
//

#ifndef PLASTICITY_DEVICE_HOG_H
#define PLASTICITY_DEVICE_HOG_H

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/opencl.h>
#include <unistd.h>

#include "../device_hopper/internal/ipc/coordination.h"

#include "../device_hopper/context.h"
#include "../device_hopper/internal/device.h"
#include "../device_hopper/internal/utils.h"

using namespace plasticity::internal;

namespace plasticity { namespace  kernels { namespace impl {
    void device_hog(
            size_t execution_time_ms,
            plasticity::setup::Context& ctx) {
        const Device dev = ctx.sched.get_preferred_device();
        ctx.sched.set_device(dev);
        ctx.sched.wait_for_device(dev);
        ctx.wait_for_other_benchmark_applications(AppType::DEVICE_HOG);

        //internal::timing::start(internal::timing::Type::ROI);

        // Tell the scheduler which kernel will be executed and the problem size
        ctx.sched.set_problem_size(1, 1);

        //ipc::coordination::device_hog_has_acquired_a_device();
#if defined(DEBUG)
        std::cout << "Device hog has acquired a device" << std::endl;
#endif
/*        while(!ctx.sched.computation_is_done()) {

            if (dev == Device::CPU) {
#ifdef DEBUG
                std::cout << "DEBUG: Device = CPU" << std::endl;
#endif
            } else if (dev == Device::GPU) {
#ifdef DEBUG
                std::cout << "DEBUG: Device = GPU" << std::endl;
#endif
            } else {
                std::cerr << "Error: Unknown device" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            ctx.sched.update_bookkeeping();
            ctx.sched.get_new_offsets_and_slice_sizes(current_offsets);
            ctx.sched.get_slice_sizes(current_slice_sizes, current_offsets);
            ctx.sched.get_block_sizes(block_sizes);*/

            // Execute on the CPU
            usleep(execution_time_ms * 1000);
/*
            ctx.sched.set_old_offset_and_slice_size(current_offsets);
        }*/
        ctx.sched.free_device(dev);
#if defined(DEBUG)
        std::cout << "Device hog has freed a device" << std::endl;
#endif
        //internal::timing::stop(internal::timing::Type::ROI);
#if defined(DEBUG)
        //ctx.sched.print_info();
        //internal::timing::print_results();
#endif
        //ipc::coordination::device_hog_has_released_a_device();
#if defined(DEBUG)
        std::cout << "Device hog has released a device" << std::endl;
#endif
    }
}}}
#endif //PLASTICITY_DEVICE_HOG_H
