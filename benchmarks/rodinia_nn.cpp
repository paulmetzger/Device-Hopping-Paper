/**
 * Some part of the source code is from the Rodinia benchmark suite.
 * Link quoted in the README of the original implementation: http://weather.unisys.com/hurricane/
 */

#include <boost/math/special_functions/next.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <tuple>

#include "../device_hopper/core.h"

#include <omp.h>
#include <sys/mman.h>

#define RODINIA_NN_EPSILON 10e-05
#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))
#define PREFERRED_DEVICE CPU

using namespace device_hopper;

typedef struct latLong
{
    float lat;
    float lng;
} LatLong;

void gen_data(struct latLong *locations, int numRecods) {
    for (size_t record_i = 0; record_i < numRecods; ++record_i) {
        locations[record_i].lat = ((float) (7 + rand() % 63)) + ((float) rand() / (float) 0x7fffffff);
        locations[record_i].lng = ((float) (rand() % 358)) + ((float) rand() / (float) 0x7fffffff);
    }
}

bool verifyResults(
        float* ocl_recordDistances,
        int numRecords,
        struct latLong *locations,
        float latitude,
        float longitude) {

    float *ref_distances = (float *)malloc(sizeof(float) * numRecords);

    // calculate distances on CPU
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < numRecords; i++) {
        const struct latLong location = locations[i];
        ref_distances[i] = (float) sqrt((latitude  - location.lat) * (latitude  - location.lat) +
                                        (longitude - location.lng) * (longitude - location.lng));
    }

    // compare to ocl result
    bool results_are_correct = true;
    float largest_difference = 0;
    float current_difference = 0;
    for (int i = 0; i < numRecords; i++) {
        current_difference = abs(ref_distances[i] - ocl_recordDistances[i]);
        if (current_difference > largest_difference) largest_difference = current_difference;
        if (current_difference > RODINIA_NN_EPSILON) {
            if (results_are_correct) {
                    std::cout << "Distance mismatch at index " << i << " by: " <<
                              abs(ref_distances[i] - ocl_recordDistances[i]) << "\n";
                    std::cout << "OCL: " << std::setprecision(10) << ocl_recordDistances[i] << "\n";
                    std::cout << "Ref: " << std::setprecision(10) << ref_distances[i] << "\n";
                    //std::cout << "Record: " << records[i].recString << std::endl;
                    results_are_correct = false;
            }
        }
    }
    std::cout << "Largest epsilon: " << largest_difference << std::endl;

    free(ref_distances);
    return results_are_correct;
}

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

    // TODO The data paths are currently all hard coded
    // Choose the problem size
    std::string data_dir;
    std::string data_file;
    int numRecords = 0;
    if (problem_size == 1) {
        data_dir = "small";
        data_file = "list81920k_4.txt";
        numRecords = 81920 * 1000;
    } else if (problem_size == 2) {
        data_dir = "medium";
        data_file = "list148480k_8.txt";
        numRecords = 148480 * 1000;
    } else if (problem_size == 3) {
        data_dir = "large";
        data_file = "list368640k_8.txt";
        numRecords = 368640 * 1000;
    } else {
        std::cerr << "Error: Unknown problem size" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // TODO this is currently not used
    std::string benchmark_data_root = REPOSITORY_PATH + "/benchmarks/input_data/rodinia/nn/" + data_dir;

    // Allocate input buffer
    struct latLong *locations = (struct latLong *) device_hopper::malloc(numRecords, sizeof(struct latLong));

    // Load input data
    gen_data(locations, numRecords);

    // Allocate output buffer
    float *recordDistances = (float *) device_hopper::malloc(numRecords, sizeof(float));

    // Default kernel parameters
    float latitude=30.0;
    float longitude=90.0;

    // Create parallel for
    parallel_for pf(0, numRecords, [=]DEVICE_HOPPER_LAMBDA() {
        int i = GET_ITERATION();
        struct latLong *latLong = locations+i;
        if (i < numRecords) { // Not necessary with our programming model
            float *dist = recordDistances + i;
            *dist = (float) sqrt((latitude-latLong->lat)  * (latitude-latLong->lat) +
                                 (longitude-latLong->lng) * (longitude-latLong->lng));
        }
    });
    // Register buffers and describe accesses
    pf.add_buffer_access_patterns(
        device_hopper::buf(locations,       direction::IN,  pattern::SUCCESSIVE_SUBSECTIONS(pf.batch_size)),
        device_hopper::buf(recordDistances, direction::OUT, pattern::SUCCESSIVE_SUBSECTIONS(pf.batch_size)));
    // Add scalar kernel parameters
    pf.add_scalar_parameters(latitude, longitude, numRecords);
    // Set optional tuning parameters and call run()
    pf.opt_set_batch_size(256).opt_set_simple_indices(true).opt_set_is_idempotent(true).run();

    // Check if the results are correct
    if (!verifyResults(
            recordDistances,
            numRecords,
            locations,
            latitude,
            longitude)) {
        // Do not change these messages and the return codes.
        // The Python scripts that automate the experiments expect them.
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << "Info: The results are correct" << std::endl;
    }

    free(locations);
    free(recordDistances);

    return EXIT_SUCCESS;
}