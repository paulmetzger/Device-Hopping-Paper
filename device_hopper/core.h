/**
 * DISCLAIMER: This header does not implement the parallel backends of the device-hopper framework.
 * The parallel backends are implemented via the source-to-source compiler
 */

#ifndef PLASTICITY_CORE_H
#define PLASTICITY_CORE_H

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <istream>
#include <tuple>

#include <stdlib.h>

#define PLASTIC
#include "../device_hopper/internal/utils.h"
#include "../device_hopper/internal/cl_utils.h"
#include "../device_hopper/context.h"

//#define DEVICE_HOPPER_BUFFER_DECL std::tuple<void*, device_hopper::access_direction, int /*ACCESS_PATTERN*/>
//#define BUFFER_DECL(...) (__VA_ARGS__)

#define DEVICE_HOPPER_LAMBDA(...) (int iteration_x, int iteration_y, int group_size_x, int group_size_y)

#if defined(EXECUTABLE)
#define DEVICE_HOPPER_MAIN(...) int main(                              \
    __VA_ARGS__                                                        \
)

#if defined(SOURCE_TO_SOURCE_TRANSLATED) && defined(OPENCL)
#define DEVICE_HOPPER_SETUP \
                     timing::reset(); \
                     cl_int err = CL_SUCCESS;\
                     plasticity::kernels::impl::handles cl_handle; \
                     plasticity::cl_utils::init_platforms(); \
                     plasticity::cl_utils::populate_cl_context_and_device_handles(cl_handle, Device::CPU); \
                     cl_handle.queue = clCreateCommandQueue(cl_handle.ctx, cl_handle.dev, 0, &err); \
                     plasticity::cl_utils::cl_check_return_code(err, "Could not create a command queue"); \
                     const std::string cl_file_path = std::string(std::getenv("PLASTICITY_ROOT")) +        \
                                                      "/library/programming_model/benchmarks/opencl_files/" + \
                                                      plasticity::kernels::impl::get_kernel_filename(Device::CPU); \
                     plasticity::cl_utils::popul_kern_h( \
                         cl_handle, \
                         plasticity::kernels::impl::kernel_name, \
                         cl_file_path, \
                         "", \
                         Device::CPU);

#elif defined(SOURCE_TO_SOURCE_TRANSLATED) && !defined(OPENCL)
#define DEVICE_HOPPER_SETUP timing::reset();
#else
#define DEVICE_HOPPER_SETUP timing::reset();
#endif

#else
#if defined(OPENCL)
#define DEVICE_HOPPER_MAIN(...) extern "C" int __attribute__ ((externally_visible)) device_hopper_main( \
    cl_context& cpu_context,                                                \
    cl_command_queue& cpu_queue,                                            \
    cl_device_id& cpu_handle,                                               \
    __VA_ARGS__,                                                            \
    const std::chrono::high_resolution_clock::time_point& start_time_point               \
)
#elif defined(OMP)
#define DEVICE_HOPPER_MAIN(...) extern "C" int __attribute__ ((externally_visible)) device_hopper_main( \
    __VA_ARGS__,                                                            \
    const std::chrono::high_resolution_clock::time_point& start_time_point               \
)
#else
#error "Define either OPENCL or OMP"
#endif

#define DEVICE_HOPPER_SETUP \
    timing::reset();\
    timing::set_start(timing::Type::ROI, start_time_point); \
    timing::set_start(timing::Type::DAEMON_STARTUP, start_time_point);\
    timing::stop(timing::Type::DAEMON_STARTUP);\
    timing::stop(timing::Type::ROI);
#endif

#define GET_ITERATION() iteration_x
#define GET_BATCH_ID() iteration_x / group_size_x
#define GET_ITERATION_WITHIN_BATCH() iteration_x % group_size_x

#define GET_2D_ITERATION(DIM) iteration_##DIM
#define GET_2D_ITERATION_X iteration_x
#define GET_2D_ITERATION_Y iteration_y

#define GET_2D_ITERATION_BATCH(DIM) iteration_ ## DIM / group_size_##DIM
#define GET_2D_ITERATION_WITHIN_BATCH(DIM) iteration_ ## DIM % group_size_##DIM
#define GET_2D_ITERATION_WITHIN_BATCH_X iteration_x % group_size_x
#define GET_2D_ITERATION_WITHIN_BATCH_Y iteration_y % group_size_y

#if !defined(SOURCE_TO_SOURCE_TRANSLATED)
struct float2 {
    float x;
    float y;
};
#endif

namespace device_hopper {
    namespace pattern {
        const int ALL_OR_ANY = 1;
        const int BY_THREAD_ID = 2;
        const int BY_BLOCK_ID = 3;
        const int INDIRECT = 4;
        const int INDIRECT_RANGE = 5;
        const int RANDOM = 6;
        const int REDUCTION = 7;
        size_t SUCCESSIVE_SUBSECTIONS(const size_t elements) { return 0; }
        size_t BATCH_CONTINUOUS(std::function<size_t (size_t)> access_start,
                                std::function<size_t (size_t)> access_end) { return 0; }
        size_t BATCH_CONTINUOUS(std::function<size_t (const size_t, const size_t)> access_start,
                                std::function<size_t (const size_t, const size_t)> access_end) { return 0; }
        size_t BATCH_CONTINUOUS(std::function<size_t (size_t)> access_start,
                                std::function<size_t (const size_t, const size_t)> access_end) { return 0; }
        size_t BATCH_CONTINUOUS(std::function<size_t (const size_t, const size_t)> access_start,
                                std::function<size_t (size_t)> access_end) { return 0; }
        size_t ELEMENTS_ACCESSED_PER_THREAD_ID(const size_t elements) {return elements;}
        size_t OVERLAPPING_MEMORY_ACCESSES(const size_t elements) {return elements;}
        size_t INDIRECT_ACCESS_RANGE_START(std::function<size_t (size_t)> f, void* buffer) { return 0; }
        size_t INDIRECT_ACCESS_RANGE_END(std::function<size_t (size_t)> f, void* buffer) { return 0; }

        /**
         * These functions do not do anything except make the code
         * more self-documenting.
         * @tparam T
         * @param l
         * @return
         */
        template<typename T>
        T ACCESS_START(T l) {return l;}
        template<typename T>
        T ACCESS_END(T l) {return l;}
    }
    enum direction {IN, OUT, IN_OUT};
    enum data_kind {INTERIM_RESULTS};
    enum reduction_operation {ADD, SUB, MUL};
    enum gpu_implementation {TEXTURE};

    void batch_barrier() {}

    template<typename... Args>
    std::tuple<Args...> buf(Args... args) {
        return std::tuple_cat(std::tuple<Args...>(args...));
    }

    class parallel_for {
    private:
        const size_t begin;
        const size_t end;
        const std::tuple<size_t, size_t> begin_2d;
        const std::tuple<size_t, size_t> end_2d;
        const std::function<void (int, int, size_t, size_t)> f;
        size_t batch_size_x;
        size_t batch_size_y;
        const size_t dimensions;

    public:
        const size_t batch_size = 16;

        parallel_for(const size_t begin,
                     const size_t end,
                     std::function<void (int, int, size_t, size_t)> f) : begin(begin), end(end), f(f), batch_size_x(128),
                                                            dimensions(1) {};

        parallel_for(const std::tuple<size_t, size_t> begin_2d,
                     const std::tuple<size_t, size_t> end_2d,
                     std::function<void (int, int, size_t, size_t)> f) : begin(0), end(0), begin_2d(begin_2d),
                                                                         end_2d(end_2d), f(f), batch_size_x(128),
                                                                         batch_size_y(128), dimensions(2) {};

        template<class... args>
        void add_buffer_access_patterns( args...) {

        }

        template<class... args>
        void add_scalar_parameters( args...) {

        }

        parallel_for& opt_set_batch_size(const size_t batch_size) {
            this->batch_size_x = batch_size;
            return *this;
        }

        parallel_for& opt_set_batch_size(const size_t batch_size_x, const size_t batch_size_y) {
            this->batch_size_x = batch_size_x;
            this->batch_size_y = batch_size_y;
            return *this;
        }

        parallel_for& opt_set_simple_indices(bool t) {
            return *this;
        }

        parallel_for& opt_set_is_idempotent(bool t) {
            return *this;
        }

        template<typename T>
        void set_is_reduction(reduction_operation op, T& result) {}

        void run() {
            if (dimensions == 1) {
                for (size_t it = this->begin; it < this->end; ++it) {
                    f(it, 0, this->batch_size_x, 0);
                }
            } else if (dimensions == 2) {
                for (size_t it_y = std::get<1>(this->begin_2d); it_y < std::get<1>(this->end_2d); ++it_y) {
                    for (size_t it_x = std::get<0>(this->begin_2d); it_x < std::get<0>(this->end_2d); ++it_x) {
                        f(it_x, it_y, this->batch_size_x, this->batch_size_y);
                    }
                }
            } else {
                std::cerr << "Error: This branch should never be entered" << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    };

    void* malloc(const size_t elements, const size_t element_size) {
        const size_t s = elements * element_size;
        return (void *) std::malloc(s);
    }

    void* aligned_malloc(const size_t alignment, const size_t elements, const size_t element_size) {
        const size_t s = elements * element_size;
        return (void *) aligned_alloc(alignment, s);
    }

    void* use_existing_buffer(const size_t elements, const size_t element_size, void* buffer_start) {
        return buffer_start;
    }
}

#endif //PLASTICITY_CORE_H
