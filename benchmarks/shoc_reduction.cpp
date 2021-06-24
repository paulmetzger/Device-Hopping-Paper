/**
 * Some part of the source code is from the SHOC benchmark suite.
 *
 * Original authors:
 * Kyle Spafford
 */

#include <boost/math/special_functions/next.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <tuple>

#include "../device_hopper/core.h"

#include <omp.h>
#include <sys/mman.h>

using namespace device_hopper;

#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))
#define PREFERRED_DEVICE CPU

DEVICE_HOPPER_MAIN(int argc, char* argv[]) {
    DEVICE_HOPPER_SETUP

    // Parse CLI parameters
    boost::program_options::options_description desc("Options");
    desc.add_options() ("problem-size", boost::program_options::value<long>(), "Sample count");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    size_t problem_size = 0;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        // This parameter is passed to the application by the Python scripts that automate the experiments.
        problem_size = vm["problem-size"].as<long>();
    }

    double *in  = (double *) device_hopper::malloc(problem_size, sizeof(double));
    for (size_t i = 0; i < problem_size; i++) in[i] = i % 3;
    double result = 0.0;

    // Create parallel for
    parallel_for pf(0, problem_size, [&]DEVICE_HOPPER_LAMBDA() {
        result += in[GET_ITERATION()];
    });
    // Register buffers and specify access patterns
    pf.add_buffer_access_patterns(device_hopper::buf(in, direction::IN, pattern::REDUCTION));
    // Tell the translator and RT system that this a reduction
    pf.set_is_reduction(device_hopper::reduction_operation::ADD, result);
    // Execute the parallel for
    pf.run();

    // Check if the results are correct
    const double threshold = 1.0e-8;
    double ref_result = 0.0;
    for (int i = 0; i < problem_size; ++i) ref_result += in[i];

    free(in);

    if (abs(result - ref_result) > threshold) {
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << "Info: The results are correct" << std::endl;
        return EXIT_SUCCESS;
    }
}