//
// Created by paul on 22/11/2019.
//

#define CL_TARGET_OPENCL_VERSION 120

#include <boost/program_options.hpp>
#include <CL/opencl.h>
#include <iostream>

#include "../benchmarks/shoc_src/spmv_util.h"

int main(int argc, char* argv[]) {
    boost::program_options::options_description desc("Options");
    desc.add_options()
            ("path-to-library", boost::program_options::value<std::string>(), "Path to the root directory of the plasticity library")
            ("problem-size",    boost::program_options::value<long>(), "Sample count")
            ("precision",       boost::program_options::value<std::string>(), "Options: Single, Double");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    long problem_size = -1;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        problem_size = vm["problem-size"].as<long>();
    }

    std::string precision_prefix;
    if (vm.count("precision")) {
        std::string precision = vm["precision"].as<std::string>();
        if (precision == "DOUBLE") {
            precision_prefix = "d";
        } else if (precision == "SINGLE") {
            precision_prefix = "s";
        } else {
            std::cerr << "Error: Unknown precision" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    } else {
        std::cerr << "Error: Precision is not set" << std::endl;
        std::cout << desc << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string path_to_the_library;
    if (vm.count("path-to-library")) {
        path_to_the_library = vm["path-to-library"].as<std::string>();
    } else {
        std::cerr << "Error: Path to the library is not set" << std::endl;
        std::cout << desc << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Setup buffers
    long non_zero_elements = problem_size * problem_size / 100;
    // Prepare in- and output buffers
    // cl_long is the correct data type for single and double precision input data
    cl_long*  cols_in           = (cl_long*)  aligned_alloc(4096,sizeof(cl_long) * non_zero_elements);
    cl_long*  row_delimiters_in = (cl_long*)  aligned_alloc(4096,sizeof(cl_long) * (problem_size + 1));

    // Generate matrix
    initRandomMatrix(cols_in, row_delimiters_in, non_zero_elements, problem_size);

    // Write to disk

    FILE* cols_file = fopen((path_to_the_library +
                            "/benchmarks/input_data/" + precision_prefix + "spmv_" +
                            std::to_string(problem_size) + "x" + std::to_string(problem_size) +
                            "cols.bin").c_str(), "wb");
    fwrite(&cols_in[0], 1, sizeof(cl_long) * non_zero_elements, cols_file);
    fclose(cols_file);

    FILE* row_delimiters_file = fopen((path_to_the_library +
                             "/benchmarks/input_data/" + precision_prefix + "spmv_" +
                             std::to_string(problem_size) + "x" + std::to_string(problem_size) +
                             "row_delimiters.bin").c_str(), "wb");
    fwrite(&row_delimiters_in[0], 1, sizeof(cl_long) * (problem_size + 1), row_delimiters_file);
    fclose(row_delimiters_file);

    std::cout << "Done" << std::endl;
}