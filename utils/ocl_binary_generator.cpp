//
// Created by paul on 25/10/2019.
//

#define CL_TARGET_OPENCL_VERSION 120

#include <boost/program_options.hpp>
#include <boost/asio/ip/host_name.hpp>
#include <CL/cl.h>
#include <iostream>

namespace boost_po = boost::program_options;

int main(int argc, char *argv[]) {
    // Parse command line parameters
    boost::program_options::options_description desc("Options");
    desc.add_options()
            ("add-to-filename",     boost::program_options::value<std::string>(), "Add this to the filename (Optional)")
            ("kernel-name",         boost::program_options::value<std::string>(), "Kernel function name")
            ("cl-compiler-flags",   boost::program_options::value<std::string>(),
             "Flags for the OpenCL compiler (Optional)")
            ("cl-file",             boost::program_options::value<std::string>(), "Path to the cl_utils file")
            ("cl-file-name",        boost::program_options::value<std::string>(), ".cl file name")
            ("output-directory",    boost::program_options::value<std::string>(),  "Output directory for precompiled kernels")
            ("target-device", boost::program_options::value<std::string>(), "Target device. Options: CPU, GPU")
            ("no-hostname", "Do not add the hostname to the file name");

    boost_po::variables_map vm;
    boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
    boost_po::notify(vm);

    if (vm.count("help") ||
        //vm.count("config-file") == 0 ||
        vm.count("cl-file") == 0 ||
        vm.count("cl-file-name") == 0 ||
        vm.count("target-device") == 0 ||
        vm.count("kernel-name") == 0 ||
        vm.count("output-directory") == 0) {
        std::cout << desc << std::endl;
        std::exit(EXIT_SUCCESS);
    }
    bool no_hostname = vm.count("no-hostname") == 1;
    //std::string path_to_config_file = vm["config-file"].as<std::string>();
    std::string path_to_opencl_file = vm["cl-file"].as<std::string>();
    std::string opencl_file_name = vm["cl-file-name"].as<std::string>();
    std::string target_device = vm["target-device"].as<std::string>();
    std::string kernel_name = vm["kernel-name"].as<std::string>();
    std::string output_directory = vm["output-directory"].as<std::string>();
    cl_int device_type = -1;
    if (target_device == "GPU") {
        device_type = CL_DEVICE_TYPE_GPU;
    } else if (target_device == "CPU") {
        device_type = CL_DEVICE_TYPE_CPU;
    } else {
        std::cerr << "Error: Unknown device type" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string add_to_filename = "";
    if (vm.count("add-to-filename")) {
        add_to_filename = vm["add-to-filename"].as<std::string>();
    }

    std::string opencl_compiler_flags = "";
    if (vm.count("cl-compiler-flags")) {
        opencl_compiler_flags = vm["cl-compiler-flags"].as<std::string>();
    }

    // Create OpenCL ctx
    std::cout << "Creating OpenCL ctx..." << std::endl;
    cl_context cl_context = NULL;
    cl_int cl_err = 0;
    cl_uint cl_num_platforms = -1;
    cl_platform_id cl_platform[2];
    cl_device_id cl_device;
    cl_err |= clGetPlatformIDs(
            2,
            cl_platform,
            &cl_num_platforms);
    cl_err |= clGetDeviceIDs(
            cl_platform[0],
            device_type,
            1,
            &cl_device,
            NULL);
    if (cl_err == -1 && cl_num_platforms > 1) {
        std::cout << "INFO: First platform does not offer the target device. "
                     "Trying to use the second platform..." << std::endl;
        cl_err = clGetDeviceIDs(
                cl_platform[1],
                device_type,
                1,
                &cl_device,
                NULL);
        if (cl_err) return EXIT_FAILURE;
    }

    cl_context = clCreateContext(
            NULL,
            1,
            &cl_device,
            NULL,
            NULL,
            &cl_err);
    if (cl_err) return EXIT_FAILURE;
    FILE *program_handle = fopen((path_to_opencl_file + opencl_file_name).c_str(), "r");
    if (program_handle == NULL) {
        std::cerr << "Error: Could not open the OpenCL source code file" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cl_err |= fseek(program_handle, 0, SEEK_END);
    if (cl_err) return EXIT_FAILURE;
    size_t program_size = ftell(program_handle);
    rewind(program_handle);
    char *program_buffer = (char *) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // Compile OCL binary
    std::cout << "Compiling kernel..." << std::endl;
    cl_program program = clCreateProgramWithSource(
            cl_context,
            1,
            (const char **) &program_buffer,
            &program_size,
            &cl_err);
    if (cl_err) {
        std::cerr << "Program creation failed: ERROR " << cl_err << std::endl;
        return EXIT_FAILURE;
    }

    cl_err |= clBuildProgram(
            program,
            1,
            &cl_device,
            opencl_compiler_flags.c_str(),
            NULL,
            NULL);
    if (cl_err) {
        std::cerr << "Program building failed: ERROR " << cl_err << std::endl;
        // Print warnings and errors from compilation
        static char log[65536];
        memset(log, 0, sizeof(log));
        clGetProgramBuildInfo(program,
                              cl_device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(log) - 1,
                              log,
                              NULL);
        printf("OpenCL Compiler Output:\n");
        if (strstr(log, "warning:") || strstr(log, "error:"))
            printf("\n%s\n", log);
        return EXIT_FAILURE;
    }
    clCreateKernel(program, kernel_name.c_str(), &cl_err);
    if (cl_err) {
        std::cerr << "Kernel creation failed: ERROR " << cl_err << std::endl;
        return EXIT_FAILURE;
    }
    //free(program_buffer);

    // Write OCL binary to disk
    std::cout << "Writing OpenCL binary to a buffer..." << std::endl;
    cl_uint program_num_devices;
    clGetProgramInfo(
            program,
            CL_PROGRAM_NUM_DEVICES,
            sizeof(cl_uint),
            &program_num_devices,
            NULL);
    if (program_num_devices == 0) {
        std::cerr << "no valid binary was found" << std::endl;
        return EXIT_FAILURE;
    }

    size_t *binaries_sizes = (size_t *) malloc(sizeof(size_t) * program_num_devices);
    cl_err = clGetProgramInfo(
            program,
            CL_PROGRAM_BINARY_SIZES,
            program_num_devices * sizeof(size_t),
            binaries_sizes,
            NULL);
    if (cl_err) {
        std::cerr << "Retrieving program information failed: ERROR " << cl_err << std::endl;
        return EXIT_FAILURE;
    }

    char **binaries = new char *[program_num_devices];
    uint64_t n_size_total = 0;
    for (size_t i = 0; i < program_num_devices; i++) {
        binaries[i] = new char[binaries_sizes[i] + 1];
        n_size_total += binaries_sizes[i];
    }

    int n_result;
    size_t n_ret_size = 0;
    if ((n_result = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                     size_t(n_size_total), &binaries[0], &n_ret_size)) != CL_SUCCESS) {
        fprintf(stderr, "error: the first clGetProgramInfo(CL_PROGRAM_BINARIES) failed with:"
                        " %d ", n_result);
        return EXIT_FAILURE;
    }

    std::cout << "Writing OpenCL binary to disk" << std::endl;
    std::string hostname = boost::asio::ip::host_name();
    std::string out_file_name =
            opencl_file_name.replace(opencl_file_name.begin() + opencl_file_name.length() - 3, opencl_file_name.end(),
                                     "") +
            //std::string out_file_name = regex_replace(opencl_file_name, std::regex(".cl"), "") +
            (no_hostname ? "" : "_" + hostname) +
            "_" + target_device +
            (add_to_filename == "" ? "" : "_" + add_to_filename) +
            ".bin";
    FILE *f = fopen((output_directory + out_file_name).c_str(), "w");
    fwrite(binaries[0], n_size_total, 1, f);
    fclose(f);

    std::cout << "Cleaning up" << std::endl;
    for (size_t i = 0; i < program_num_devices; i++)
        free(binaries[i]);
    free(binaries);

    clReleaseProgram(program);
    clReleaseContext(cl_context);

    free(binaries_sizes);

    return EXIT_SUCCESS;
}
