//
// Created by paul on 10/10/2019.
//

#ifndef PLASTICITY_SCHEDULER_H
#define PLASTICITY_SCHEDULER_H

#include <emmintrin.h>
#include <fstream>
#include <boost/asio/ip/host_name.hpp>

#include "../context.h"
#include "ipc/coordination.h"

#include "device.h"
#include "scheduling_modes.h"
#include "slicing_modes.h"
#include "slice_size_modes.h"
#include "timing.h"
#include "utils.h"

namespace plasticity {
    namespace internal {
        class Scheduler {
        public:
            // First cache line. The fields are ordered by the access order in the while in the
            // kernel implementations.
#if defined(PLASTIC)
            alignas(64) std::mutex book_keeping;
            size_t previous_x_offset = 0;
#else
            alignas(64) size_t previous_x_offset = 0;
#endif
            size_t previous_y_offset = 0;
            size_t previous_slice_size_x = 0;
            size_t previous_slice_size_y = 0;
            size_t problem_size_x = 0;
            size_t problem_size_y = 0;
            Device current_device = Device::None; //#8
            size_t kernel_invocation_count = 0;

            Device preferred_device;
            SlicingMode slicing_mode = SlicingMode::SlicingOnAllDevices;
            size_t next_slice_size_x = 0;
            size_t next_slice_size_y = 0;
            size_t current_slice_size_x = 0;
            size_t current_slice_size_y = 0;
            bool first_kernel_launch = true;

            // End of the first cache line

            //Second cache line
#if defined(DEBUG)
            size_t debug_slices_before_migration = 0;
#endif
            bool comp_is_on_pref_dev = false;
            size_t cpu_x_slice_size = 0;
            size_t gpu_x_slice_size = 0;
            size_t cpu_y_slice_size = 0;
            size_t gpu_y_slice_size = 0;
            int kernel_invocations_on_preferred_device = 0;

            SchedulingMode scheduling_mode = SchedulingMode::BestAvailable;
            //Device alternative_device;
            Device launch_device;
            short back_to_back_scheduling_order = -1;
            //short back_to_back_scheduling_application_count = -1;
            //bool finish_row_of_slices = false;
            bool pin_to_device = false;
            bool launch_device_is_set = false;
            //bool kernel_is_idempotent = false;

            void load_tuning_parameters(Device device, bool device_has_changed) {
                if ((this->next_slice_size_x == 0 &&
                     this->next_slice_size_y == 0) ||
                    device_has_changed) {

                    //bool use_dyn_slice_sizes = (device != this->preferred_device && this->use_dynamic_slice_sizes());
                    if (device == Device::GPU) {
                        this->next_slice_size_x = this->gpu_x_slice_size;
                        this->next_slice_size_y = this->gpu_y_slice_size;
                    } else if (device == Device::CPU) {
                        this->next_slice_size_x = this->cpu_x_slice_size;
                        this->next_slice_size_y = this->cpu_y_slice_size;
                    } else {
                        std::cerr << "Error: Unknown device" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
            }

            /**
             *
             * @param preferred_slice_size_x  Preferred slice size for a kernel, device and problem size
             * @param preferred_slice_size_y
             * @param problem_size_x Global problem size
             * @param problem_size_y
             * @param current_offset_x The offset for the next iteration
             * @param current_offset_y
             * @param new_slice_sizes The slice sizes for the next iteration
             */
            inline void compute_slice_size(const size_t &preferred_slice_size_x,
                                           const size_t &preferred_slice_size_y,
                                           const size_t &problem_size_x,
                                           const size_t &problem_size_y,
                                           const size_t &current_offset_x,
                                           const size_t &current_offset_y,
                                           size_t (&new_slice_sizes)[2]) const {
                new_slice_sizes[0] = preferred_slice_size_x <= problem_size_x ? preferred_slice_size_x : problem_size_x;
                new_slice_sizes[1] = preferred_slice_size_y <= problem_size_y ? preferred_slice_size_y : problem_size_y;

                // The last condition in both if statements handles cases where the chunk size is bigger than the work size
                if (current_offset_x + preferred_slice_size_x > problem_size_x &&
                    preferred_slice_size_x < problem_size_x) {
                    new_slice_sizes[0] -= (current_offset_x + preferred_slice_size_x - problem_size_x);
                }

                if (current_offset_y + preferred_slice_size_y > problem_size_y &&
                    preferred_slice_size_y < problem_size_y) {
                    new_slice_sizes[1] -= (current_offset_y + preferred_slice_size_y - problem_size_y);
                }

                if (preferred_slice_size_x == 0) new_slice_sizes[0] = problem_size_x;
                if (preferred_slice_size_y == 0) new_slice_sizes[1] = problem_size_y;
            }

            /**
             *
             * @param slice_size_x Preferred slice size for a kernel, device and problem size
             * @param slice_size_y
             * @param problem_size_x Global problem size
             * @param kernel_invocation_count How often the kernel has been invoked
             * @param offsets Offsets for the next iteration
             */
            inline void compute_offsets(const size_t &slice_size_x,
                                        const size_t &slice_size_y,
                                        const size_t &previous_offset_x,
                                        const size_t &previous_offset_y,
                                        const size_t &problem_size_x,
                                        const size_t &kernel_invocation_count,
                                        size_t (&offsets)[2]) const {
                if (kernel_invocation_count > 1) {
                    offsets[0] = previous_offset_x + slice_size_x;
                    // The second condition handles cases in which a chunk size for the x direction is not set
                    if (offsets[0] >= problem_size_x || slice_size_x == 0) {
                        offsets[0] = 0;
                        offsets[1] = previous_offset_y + slice_size_y;
                        if (offsets[1] == problem_size_y) offsets[1] = 0;
                    } else {
                        offsets[1] = previous_offset_y;
                    }
                }
            }

            inline Device pick_device() {
                Device chosen_device = Device::None;
                if (this->scheduling_mode == SchedulingMode::BestAvailable ||
                    this->scheduling_mode == SchedulingMode::Plastic) {
                    chosen_device = pick_device_with_best_avail_heuristic();
                } else if (this->scheduling_mode == SchedulingMode::BacktoBack) {
                    chosen_device = this->preferred_device;
                } else {
                    std::cerr << "Unknown scheduling mode" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                /*bool use_dyn_slice_sizes = (chosen_device != this->preferred_device &&
                                            !this->use_dynamic_slice_sizes());*/
                if (chosen_device == Device::CPU) {
                    this->next_slice_size_x = this->cpu_x_slice_size;;
                    this->next_slice_size_y = this->cpu_y_slice_size;
                } else if (chosen_device == Device::GPU) {
                    this->next_slice_size_x = this->gpu_x_slice_size;;
                    this->next_slice_size_y = this->gpu_y_slice_size;
                } else {
                    std::cerr << "Error: Unknown device" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                return chosen_device;
            }

            inline bool iter_space_has_been_covered(const size_t current_offsets_x,
                                                    const size_t current_offsets_y,
                                                    const size_t current_slice_size_x,
                                                    const size_t current_slice_size_y,
                                                    const size_t problem_size_x,
                                                    const size_t problem_size_y) const {
                return current_offsets_x + current_slice_size_x >= problem_size_x &&
                       current_offsets_y + current_slice_size_y >= problem_size_y;
            }

            inline Device pick_device_with_best_avail_heuristic() {
                bool preferred_device_is_avail = ipc::coordination::is_avail(this->preferred_device);
                if (preferred_device_is_avail) return this->preferred_device;
                else {
                    if (this->current_device == this->preferred_device) return this->preferred_device;

                    if (this->preferred_device == Device::GPU) {
                        return Device::CPU;
                    } else if (this->preferred_device == Device::CPU) {
                        return Device::GPU;
                    } else {
                        std::cerr << "Error: Unknown device" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
            }

        public:
            Scheduler() {
                ipc::coordination::init();
            }

            ~Scheduler() {
                ipc::coordination::shutdown();
            }

            bool kill_current_slice(Device dev) const {
#if defined(DEBUG)
                std::cout << "DEBUG " << (dev == Device::CPU ? "CPU" : "GPU") << ": Called kill_current_slice" << std::endl;
#endif
                const bool result = (dev != this->preferred_device) && this->comp_is_on_pref_dev;
#if defined(DEBUG)
                std::cout << "DEBUG: " << (result ? "Kill the current slice" : "Do not kill the current slice") << std::endl;
#endif
                return result;
            }

#if defined(DEBUG)
            void set_slices_before_migration(size_t slice_cnt) {
                this->debug_slices_before_migration = slice_cnt;
            }
#endif

            void pin_computation_to_device(Device device) {
                this->pin_to_device = true;
                this->launch_device = device;
                this->current_device = device;
            }

            void set_launch_device(Device device) {
                this->launch_device_is_set = true;
                this->launch_device = device;
            }

            void set_scheduling_mode(SchedulingMode scheduling_mode,
                                     int back_to_back_scheduling_order = -1,
                                     int application_count = -1) {
                this->scheduling_mode = scheduling_mode;
                this->back_to_back_scheduling_order = back_to_back_scheduling_order;
                //this->back_to_back_scheduling_application_count = application_count;
            }

            void set_slicing_mode(SlicingMode slicing_mode) {
                this->slicing_mode = slicing_mode;
            }

            Device get_launch_device() const {
                return this->launch_device;
            }

            size_t get_cpu_x_slice_size() const {
                return this->cpu_x_slice_size;
            }

            size_t get_cpu_y_slice_size() const {
                return this->cpu_y_slice_size;
            }

            size_t get_gpu_x_slice_size() const {
                return this->gpu_x_slice_size;
            }

            size_t get_gpu_y_slice_size() const {
                return this->gpu_y_slice_size;
            }

            Device get_preferred_device() const {
                return this->preferred_device;
            }

            size_t get_y_problem_size() const {
                return this->problem_size_y;
            }

            void wait_for_device(Device dev) {
#if defined(DEBUG)
                std::cout << "Called 'wait_for_device' with " << (dev == Device::CPU ? "CPU" : "GPU") << std::endl;
                if (this->launch_device_is_set && dev != this->launch_device)
                    while (this->kernel_invocation_count < this->debug_slices_before_migration) {}
#endif
                //if (multiprog_exp) ipc::coordination::wait_for_the_device_hog_to_acquire_a_device();
                //ipc::coordination::wait_for_device(dev);
                //ipc::coordination::acquire_lock();
                ipc::coordination::use(dev, preferred_device);
                //ipc::coordination::release_lock();
            }

            bool computation_is_on_preferred_device() const {
                //return this->current_device == this->preferred_device;
                return this->comp_is_on_pref_dev;
            }

            void set_computation_is_on_preferred_device() {
                this->comp_is_on_pref_dev = true;
            }

            void set_device(Device dev) {
                //this->comp_is_on_pref_dev = (dev == this->preferred_device);
                this->current_device = dev;
                load_tuning_parameters(dev, true);
            }

            // Getters
            Device get_device() {
                ipc::coordination::acquire_lock();

                if (!this->pin_to_device) {
                    // The computation has not been pinned to a device through command line parameters

                    if (this->launch_device_is_set && this->first_kernel_launch) {
                        // Launch the kernel on a device set by the user via the command line

                        this->current_device = this->launch_device;
                    } else {
                        // Check if we want to choose a new device
                        // Choose target device based on availability and affinity

                        Device chosen_device = Device::None;
                        chosen_device = this->pick_device();

                        if (!this->first_kernel_launch) {
                            if (chosen_device != this->current_device) {
                                ipc::coordination::cease_to_use(this->current_device);
                                this->current_device = chosen_device;
                                ipc::coordination::use(this->current_device, preferred_device);
#if defined(PLASTIC) || defined(WITH_SLICING)
                                load_tuning_parameters(this->current_device, true);

#else
                                internal::timing::stop(internal::timing::Type::ROI);
                                load_tuning_parameters(this->current_device, true);
                                internal::timing::start(internal::timing::Type::ROI);
#endif
                                /*if ((this->previous_x_offset + *//*this->current_slice_size_x*//* 0) != this->problem_size_x)
                                    this->finish_row_of_slices = true;*/
                                //this->kernel_invocations_on_current_device = 0;
                            }
                        } else {
                            this->current_device = chosen_device;
                        }
                    }
                }

                if (this->first_kernel_launch) {
                    // Update bookkeeping data and load tuning parameters
                    ipc::coordination::use(this->current_device, preferred_device);
#if defined(PLASTIC) || defined(WITH_SLICING)
                    load_tuning_parameters(this->current_device, false);
#else
                    internal::timing::stop(internal::timing::Type::ROI);
                    load_tuning_parameters(this->current_device, false);
                    internal::timing::start(internal::timing::Type::ROI);
#endif
                    this->first_kernel_launch = false;
                }
                ipc::coordination::release_lock();
                return this->current_device;
            }

            void set_preferred_device(Device device) {
                this->preferred_device = device;
                //this->alternative_device = (device == Device::CPU) ? Device::GPU : Device::CPU;
            }

            void set_problem_size(size_t problem_size_x, size_t problem_size_y) {
                this->problem_size_x = problem_size_x;
                this->problem_size_y = problem_size_y;
                this->kernel_invocation_count = 0;
            }

            void set_cpu_slice_sizes(int cpu_x_slice_size, int cpu_y_slice_size) {
                this->cpu_x_slice_size = cpu_x_slice_size;
                this->cpu_y_slice_size = cpu_y_slice_size;
            }

            void set_gpu_slice_sizes(int gpu_x_slice_size, int gpu_y_slice_size) {
                this->gpu_x_slice_size = gpu_x_slice_size;
                this->gpu_y_slice_size = gpu_y_slice_size;
            }

            // Todo: Remove the parameters and update all benchmark implementations and the translator accordingly.
            inline void set_old_offset_and_slice_size(size_t (&offsets)[2], size_t (&slice_sizes)[2]) {
#if defined(DEBUG)
                std::cout << "DEBUG: Called 'set_old_offset'" << std::endl;
#endif
                this->previous_x_offset = offsets[0];
                this->previous_y_offset = offsets[1];
                this->previous_slice_size_x = slice_sizes[0];
                this->previous_slice_size_y = slice_sizes[1];
            }

            void set_kernel_is_idempotent() {
                //this->kernel_is_idempotent = true;
            }

            inline bool computation_is_done(size_t (&slices)[2], size_t (&offsets)[2]) const {
                // Check if the iteration space has been covered
                const bool result = this->iter_space_has_been_covered(
                        this->previous_x_offset,
                        this->previous_y_offset,
                        this->previous_slice_size_x, //this->current_slice_size_x,
                        this->previous_slice_size_y, //this->current_slice_size_y,
                        this->problem_size_x,
                        this->problem_size_y);
#if defined(DEBUG)
                if (result) std::cout << "DEBUG: Computation is done" << std::endl;
                else std::cout << "DEBUG: Computation is not done" << std::endl;
#endif

                return result;
            }

            inline bool computation_has_migrated(Device dev) const {
                return dev != this->current_device;
            }

            inline void get_new_offsets_and_slice_sizes(size_t (&slices)[2], size_t (&offsets)[2]) {
                // Bookkeeping that is required to compute offsets
                //if (__builtin_expect(this->first_kernel_launch, false)) this->first_kernel_launch = false;
                if (__builtin_expect(this->kernel_invocation_count < 3, false)) this->kernel_invocation_count += 1;
                if (__builtin_expect(this->current_device == this->preferred_device, false))
                    this->kernel_invocations_on_preferred_device += 1;

                // Compute offsets
                if (__builtin_expect(this->slicing_mode == SlicingMode::NoSlicingOnPreferredDevice ||
                                     this->slicing_mode == SlicingMode::SlicingOnAllDevices, true)) {
                    // Computation is not on the preferred device or we slice on all devices.
                    if (__builtin_expect(!this->comp_is_on_pref_dev ||
                        this->slicing_mode == SlicingMode::SlicingOnAllDevices, true) ) {
                        this->compute_offsets(this->next_slice_size_x,
                                              this->next_slice_size_y,
                                              this->previous_x_offset,
                                              this->previous_y_offset,
                                              this->problem_size_x,
                                              this->kernel_invocation_count,
                                              offsets);
                        this->compute_slice_size(this->next_slice_size_x,
                                                 this->next_slice_size_y,
                                                 this->problem_size_x,
                                                 this->problem_size_y,
                                                 offsets[0],
                                                 offsets[1],
                                                 slices);
                    } else {
                        if (this->kernel_invocations_on_preferred_device == 1) {
#if defined(DEBUG)
                            std::cout << "DEBUG | current_slice_size_x: "  << this->current_slice_size_x << std::endl;
                            std::cout << "DEBUG | current_slice_size_y: "  << this->current_slice_size_y << std::endl;
                            std::cout << "DEBUG | previous_slice_size_y: " << this->previous_slice_size_y << std::endl;
                            std::cout << "DEBUG | previous_slice_size_x: " << this->previous_slice_size_x << std::endl;
                            std::cout << "DEBUG | previous_slice_size_y: " << this->previous_slice_size_y << std::endl;
                            std::cout << "DEBUG | previous_x_offset: "     << this->previous_x_offset << std::endl;
                            std::cout << "DEBUG | previous_y_offset: "     << this->previous_y_offset << std::endl;
                            std::cout << "DEBUG | problem_size_x: "        << this->problem_size_x << std::endl;
                            std::cout << "DEBUG | problem_size_y: "        << this->problem_size_y << std::endl;
                            std::cout << "DEBUG | kernel_invocation_count: " << this->kernel_invocation_count << std::endl;
                            std::cout << "DEBUG | offset[0]: " << offsets[0] << std::endl;
                            std::cout << "DEBUG | offset[1]: " << offsets[1] << std::endl;
                            std::cout << "---" << std::endl;
#endif
                            if (this->kernel_invocation_count == 2) {
                                offsets[0] = this->previous_x_offset + this->previous_slice_size_x;
                                offsets[1] = this->previous_y_offset + this->previous_slice_size_y;
                            } else {
                                this->compute_offsets(
                                        this->current_slice_size_x,
                                        this->current_slice_size_y,
                                        this->previous_x_offset,
                                        this->previous_y_offset,
                                        this->problem_size_x,
                                        this->kernel_invocation_count - 1,
                                        offsets);
                            }

                            // Either do not use the code in the branch below and set the old slice size and offset
                            // in the middle of the while loop before the kernel launch, or use the code below and
                            // set the old slice size and offset only at the end of each iteration.
/*                            if (!this->kernel_is_idempotent) {
                            // Do not restart the current slice if the alternative device is the CPU
                            // At the moment we don't have a way to kill a slice on the CPU and reset the state.
                            // Restarting a slice that ran on the CPU again on the GPU creates problems if the kernel
                            // is not idempotent.
                                if (this->alternative_device == Device::CPU *//*&& (offsets[0] != 0 || offsets[1] != 0)*//*) {
#if defined(DEBUG)
                                    std::cout << "DEBUG | current_slice_size_x: " << this->current_slice_size_x << std::endl;
                                    std::cout << "DEBUG | current_slice_size_y: " << this->current_slice_size_y << std::endl;
                                    std::cout << "DEBUG | offset[0]: " << offsets[0] << std::endl;
                                    std::cout << "DEBUG | offset[1]: " << offsets[1] << std::endl;
                                    std::cout << "DEBUG | kernel_invocation_count: " << this->kernel_invocation_count << std::endl;
#endif
                                    size_t current_x_offset = this->previous_x_offset + this->previous_slice_size_x >= this->problem_size_x ? 0 : this->previous_x_offset + this->previous_slice_size_x;
                                    size_t current_y_offset = this->previous_x_offset + this->previous_slice_size_x >= this->problem_size_x ? this->previous_y_offset + this->previous_slice_size_y : this->previous_y_offset;
                                    if (this->kernel_invocation_count != 2)
                                        this->compute_offsets(
                                                this->current_slice_size_x,
                                                this->current_slice_size_y,
                                                current_x_offset,
                                                current_y_offset,
                                                this->problem_size_x,
                                                this->kernel_invocation_count,
                                                offsets);
#if defined(DEBUG)
                                    std::cout << "DEBUG | new offset[0]: " << offsets[0] << std::endl;
                                    std::cout << "DEBUG | new offset[1]: " << offsets[1] << std::endl;
#endif
                                }
                            }*/
                        } else if (this->kernel_invocations_on_preferred_device == 2) {
                            offsets[0] = 0;
                            offsets[1] += slices[1];
                        } else {
                            std::cerr << "Error: This code should be unreachable. (get_new_offsets_and_slice_sizes) "
                                      << std::endl;
                            std::cerr << "kernel_invocations_on_preferred_device: "
                                      << this->kernel_invocations_on_preferred_device << std::endl;
                            std::exit(EXIT_FAILURE);
                        }

                        // Either do the entire problem space at once or
                        // complete the current row of slices and then do
                        // the remaining problem space.
                        if (this->kernel_invocation_count == 1 || (offsets[0] == 0 && offsets[1] == 0)) {
                            slices[0] = this->problem_size_x;
                            slices[1] = this->problem_size_y;
                        } else {
                            if (this->kernel_invocations_on_preferred_device == 1) {
                                slices[0] = this->problem_size_x - offsets[0];
                                /*slices[1] = slices[1] == 0 ? this->problem_size_y
                                                           : slices[1];*/
                                slices[1] = this->current_slice_size_y;
                                if (offsets[1] + slices[1] > this->problem_size_y)
                                    slices[1] = this->problem_size_y - offsets[1];
                            } else if (this->kernel_invocations_on_preferred_device == 2) {
                                slices[0] = this->problem_size_x;
                                slices[1] = this->problem_size_y - offsets[1];
                            } else {
                                std::cerr << "Error: This code should be unreachable. (get_slice_sizes)" << std::endl;
                                std::cerr << "kernel_invocations_on_preferred_device: "
                                          << this->kernel_invocations_on_preferred_device << std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                        }
                    }
                } else if (this->slicing_mode == SlicingMode::NoSlicing) {
                    slices[0] = this->problem_size_x;
                    slices[1] = this->problem_size_y;
                } else {
                    std::cerr << "Error: Unknown slicing mode" << std::endl;
                    std::exit(EXIT_FAILURE);
                }

                /*_mm_stream_si32((int*) &this->current_slice_size_x, slices[0]);
                _mm_stream_si32((int*) &this->current_slice_size_y, slices[1]);*/
                this->current_slice_size_x = slices[0];
                this->current_slice_size_y = slices[1];
            }

            inline bool is_final_slice(size_t (&slices)[2], size_t (&offsets)[2]) const {
                return offsets[0] + slices[0] >= this->problem_size_x &&
                       offsets[1] + slices[1] >= this->problem_size_y;
            }

            Device get_preferred_device() {
                return this->preferred_device;
            }

            void free_device(Device dev) {
                // Update bookkeeping data structures in case this computation has finished
                ipc::coordination::cease_to_use(dev);

                if (this->scheduling_mode == SchedulingMode::BacktoBack) {
                    //ipc::coordination::acquire_lock();
                    ipc::coordination::finish_turn_in_back_to_back_mode(this->back_to_back_scheduling_order);
                    //ipc::coordination::release_lock();
                }
            }

            void print_info() {
                std::string problem_size_x_str = std::to_string(problem_size_x);

                //Todo: Add support for other impl
                std::string cpu_implementation = "OpenCL";
                std::string gpu_implementation = "OpenCL";

                std::cout << "Information on the experimental set up:" << std::endl;
                std::cout << "CPU x slice size: " << this->cpu_x_slice_size << std::endl;
                std::cout << "CPU y slice size: " << this->cpu_y_slice_size << std::endl;
                std::cout << "GPU x slice size: " << this->gpu_x_slice_size << std::endl;
                std::cout << "GPU y slice size: " << this->gpu_y_slice_size << std::endl;

                std::cout << "CPU implementation: " << cpu_implementation << std::endl;
                std::cout << "GPU implementation: " << gpu_implementation << std::endl;
            }

            void wait_for_device_with_back_to_back_scheduling_heuristic() {
                if (this->scheduling_mode == SchedulingMode::BacktoBack) {
                    //ipc::coordination::release_lock();
                    //internal::timing::stop(internal::timing::Type::SCHEDULING);
                    internal::timing::start(
                            this->current_device == Device::CPU ? timing::Type::CPU_WAITING_TIME
                                                                : timing::Type::GPU_WAITING_TIME);
                    ipc::coordination::wait_for_turn_in_back_to_back_mode(this->back_to_back_scheduling_order);
                    internal::timing::stop(this->current_device == Device::CPU ? timing::Type::CPU_WAITING_TIME
                                                                               : timing::Type::GPU_WAITING_TIME);
                    //internal::timing::start(internal::timing::Type::SCHEDULING);
                    //ipc::coordination::acquire_lock();
                }
            }
        };
    }
}
#endif //PLASTICITY_SCHEDULER_H
