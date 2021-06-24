/**
 * Some part of the source code is from the SHOC benchmark suite.
 *
 * Original authors:
 * Lukasz Wesolowski
 *
 * Note from the original implementation:
 * Based on Bell (SC09) and Baskaran (IBM Tech Report).
 */

#include <boost/math/special_functions/next.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <tuple>

#include <omp.h>
#include <sys/mman.h>

#include "../device_hopper/core.h"
#include "shoc_src/spmv_util.h"

#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))
#define PREFERRED_DEVICE CPU

using namespace device_hopper;
using namespace device_hopper::pattern;

void read_input_data_in(
        long* cols_in,
        long* row_delimiters_in,
        float* val_in,
        float* vec_in,
        long n,
        size_t non_zero_elements) {
    std::string precision_prefix = "s";

    for (size_t i = 0; i < non_zero_elements; ++i) val_in[i] = 1.0;
    // Read from disk
    std::string path_to_column_file = (REPOSITORY_PATH + "/library/benchmarks/input_data/" + precision_prefix + "spmv_" +
                                      std::to_string(n) + "x" + std::to_string(n) + "cols.bin");
    FILE* cols_file = fopen(path_to_column_file.c_str(), "rb");
    if (cols_file == NULL) plasticity::utils::exit_with_err("Could not open SPMV columns input file");
    if (sizeof(cl_long) * non_zero_elements != fread(&cols_in[0], 1, sizeof(cl_long) * non_zero_elements, cols_file)) {
        std::cerr << "Could not read input matrix" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    fclose(cols_file);

    FILE* row_delimiters_file = fopen((REPOSITORY_PATH + "/library/benchmarks/input_data/" + precision_prefix + "spmv_" +
                                       std::to_string(n) + "x" + std::to_string(n) + "row_delimiters.bin").c_str(), "rb");
    if (cols_file == NULL) plasticity::utils::exit_with_err("Could not open SPMV row delimiters input file");
    if (sizeof(cl_long) * (n + 1) != fread(&row_delimiters_in[0], 1, sizeof(cl_long) * (n + 1), row_delimiters_file)) {
        std::cerr << "Could not read input matrix" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    fclose(row_delimiters_file);

    fill(vec_in, n, 10);
}

// ****************************************************************************
// This function is taken from the SHOC SPMV benchmark implementation.
// Function: spmvCpu
//
// Purpose:
//   Runs sparse matrix vector multiplication on the CPU
//
// Arguements:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of A
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last
//                  element of A
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: input - buffer of size dim
//        output - result from the spmv calculation
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing directly
//   out indirectly through a pointer
// ****************************************************************************
void spmvCpu(
        const float *val,
        const long *cols,
        const long *rowDelimiters,
        const float *vec,
        long dim,
        float *out) {
#pragma omp parallel for
    for (long i=0; i<dim; i++) {
        float t = 0.0;
        for (long j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++) {
            long col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

bool verifyResults(
        const float *cpuResults,
        const float *gpuResults,
        const long size) {
    bool passed = true;
    for (long i=0; i<size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i] > MAX_RELATIVE_ERROR)
        {
#ifdef VERBOSE_OUTPUT
            cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
                " dev: " << gpuResults[i] << endl;
#endif
            passed = false;
        }
    }
    return passed;
}

DEVICE_HOPPER_MAIN(int argc, char* argv[]) {
    DEVICE_HOPPER_SETUP

    // Parse CLI parameters
    boost::program_options::options_description desc("Options");
    desc.add_options() ("problem-size", boost::program_options::value<long>(), "Sample count");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    long n = 0;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        // This parameter is passed to the application by the Python scripts that automate the experiments.
        n = vm["problem-size"].as<long>();
    }

#if defined(SOURCE_TO_SOURCE_TRANSLATED)
    // This sets the number of blocks in a slice as slice size as expected by the parallel for.
    // The hand implementation uses the number of iteration points in a slice as slice size.
    // To keep the database tables and configs consistent we continue to use the old slice sizes in terms of iteration points
    // in the configs and databases but convert them then here in the source code.
    context.sched.set_cpu_slice_sizes(context.sched.get_cpu_x_slice_size() / 128, 1);
    context.sched.set_gpu_slice_sizes(context.sched.get_gpu_x_slice_size() / 128, 1);
#endif

    // 1% of entries will be non-zero (This is how the SHOC benchmark is implemented)
    long non_zero_elements = n * n / 100;
    std::string precision_prefix = "s";
    size_t data_type_size = sizeof(float);
    size_t long_data_type_size = sizeof(cl_long);
    size_t n_plus_one = n + 1;
    // Prepare in- and output buffers
    float *val_in            = (float*) device_hopper::malloc(non_zero_elements, data_type_size);
    long  *cols_in           = (long*)  device_hopper::malloc(non_zero_elements, long_data_type_size);
    long  *row_delimiters_in = (long*)  device_hopper::malloc(n_plus_one, long_data_type_size);
    float *vec_in            = (float*) device_hopper::malloc(n, data_type_size);
    float *res_out           = (float*) device_hopper::malloc(n, data_type_size);
    float *ref_out           = (float*) device_hopper::malloc(n, data_type_size);

    // Populate buffers
    read_input_data_in(cols_in, row_delimiters_in, val_in, vec_in, n, non_zero_elements);

    // Create parallel for
    parallel_for pf(0, n, [=]DEVICE_HOPPER_LAMBDA() {
        long myRow = GET_ITERATION();
        if (myRow < n) {
            float t = 0.0f;
            long start = row_delimiters_in[myRow];
            long end = row_delimiters_in[myRow+1];
            for (long j = start; j < end; j++)
            {
                long col = cols_in[j];
                t += val_in[j] * vec_in[col];
            }
            res_out[myRow] = t;
        }
    });
    // Register buffers and specify access patterns
    pf.add_buffer_access_patterns(
        device_hopper::buf(row_delimiters_in, direction::IN, pattern::BATCH_CONTINUOUS(
               ACCESS_START([](size_t batch_id, size_t batch_size) {return batch_id * batch_size;}),
               ACCESS_END(  [](size_t batch_id, size_t batch_size) {return (batch_id + 1) * batch_size;}))),
        device_hopper::buf(val_in, direction::IN, pattern::BATCH_CONTINUOUS(
               ACCESS_START([=](size_t batch_id, size_t batch_size) {return row_delimiters_in[batch_id * batch_size];}),
               ACCESS_END(  [=](size_t batch_id, size_t batch_size) {return row_delimiters_in[(batch_id + 1) * batch_size] - 1;}))),
        device_hopper::buf(cols_in, direction::IN, pattern::BATCH_CONTINUOUS(
               ACCESS_START([=](size_t batch_id, size_t batch_size) {return row_delimiters_in[batch_id * batch_size];}),
               ACCESS_END(  [=](size_t batch_id, size_t batch_size) {return row_delimiters_in[(batch_id + 1) * batch_size] - 1;}))),
        device_hopper::buf(vec_in, direction::IN, pattern::ALL_OR_ANY, gpu_implementation::TEXTURE),
        device_hopper::buf(res_out, direction::OUT, pattern::BATCH_CONTINUOUS(
               ACCESS_START([](size_t batch_id, size_t batch_size) {return batch_id * batch_size;}),
               ACCESS_END(  [](size_t batch_id, size_t batch_size) {return (batch_id + 1) * batch_size - 1;}))));
    // Add scalar kernel parameters
    pf.add_scalar_parameters(n);
    // Set optional tuning parameters and call run()
    pf.opt_set_batch_size(128).opt_set_is_idempotent(true).opt_set_simple_indices(true).run();

    // Check results
    spmvCpu(val_in, cols_in, row_delimiters_in, vec_in, n, ref_out);
    if (! verifyResults(ref_out, res_out, n)) {
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << "Info: The results are correct" << std::endl;
    }

    free(val_in);
    free(cols_in);
    free(row_delimiters_in);
    free(vec_in);
    free(res_out);
    free(ref_out);

    return EXIT_SUCCESS;
}