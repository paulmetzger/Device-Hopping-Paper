//
// Created by paul on 18/05/2020.
//

#include "messages.h"
#include "ipc_utils.h"

#include <boost/program_options.hpp>
#include <chrono>
#include <mqueue.h>
#include <iostream>

namespace boost_po = boost::program_options;
using namespace plasticity;

int main(int argc, char *argv[]) {
    mqd_t queue_to_normal_daemon_worker;
    mqd_t queue_to_cpu_affine_daemon_worker;
    mqd_t queue_from_normal_daemon_worker;
    mqd_t queue_from_cpu_affine_daemon_worker;
    setup_ipc(queue_to_normal_daemon_worker, true, false);
    setup_ipc(queue_to_cpu_affine_daemon_worker, true, true);
    setup_ipc(queue_from_normal_daemon_worker, false, false);
    setup_ipc(queue_from_cpu_affine_daemon_worker, false, true);

    boost_po::options_description desc("Options");
    desc.add_options()
            ("exit-daemon", "Stop the daemon")
            ("function-name", boost_po::value<std::string>(), "Name of the function inside the shared library")
            ("path-to-the-application", boost_po::value<std::string>(), "Path to the application that will be executed by the daemon")
            ("cli-parameters", boost_po::value<std::string>(), "CLI parameters for the application that will be executed by the daemon");
    boost_po::variables_map vm;
    boost_po::store(boost_po::parse_command_line(argc, argv, desc), vm);
    boost_po::notify(vm);

    daemon_command command;
    if (vm.count("exit-daemon") == 1) {
        command.command_type = CommandType::EXIT_DAEMON;
        std::cerr << "Not implemented" << std::endl;
        std::exit(1);
        /*for (size_t i = 0; i < 2; ++i) {
            if (mq_send(queue_to_daemon, (const char *) &command, sizeof(daemon_command), 0) != 0)
                utils::exit_with_err("Could not send a message to the daemon");

            daemon_response response;
            if (mq_receive(queue_from_daemon, (char *) &response, sizeof(daemon_response), 0) != sizeof(daemon_response))
                utils::exit_with_err("Could not receive a message from the daemon");

            if (response.return_code != 0) utils::exit_with_err("An error occured while exiting the daemon.");
        }*/
    } else {
        std::string application_path;
        if (vm.count("path-to-the-application") == 1) {
            application_path = vm["path-to-the-application"].as<std::string>();
        } else {
            std::cerr << "Error: No path to an executable is set." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::string cli_parameters;
        if (vm.count("cli-parameters") == 1) {
            cli_parameters = vm["cli-parameters"].as<std::string>();
        } else {
            std::cerr << "Error: No CLI parameters for the application set." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::string function_name;
        if (vm.count("function-name") == 1) {
            function_name = vm["function-name"].as<std::string>();
        } else {
            std::cerr << "Error: No function name set" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        bool use_cpu_affine_worker = (function_name.find("device_hog") == std::string::npos);

        command.command_type = CommandType::EXECUTE_APPLICATION;
        std::strcpy(command.application_cli_parameters, cli_parameters.c_str());
        std::strcpy(command.path_to_application,        application_path.c_str());
        std::strcpy(command.function_name,              function_name.c_str());
        command.time_start = std::chrono::high_resolution_clock::now();

        if (mq_send(
                use_cpu_affine_worker ? queue_to_cpu_affine_daemon_worker : queue_to_normal_daemon_worker,
                (const char*) &command,
                sizeof(daemon_command),
                0) != 0) utils::exit_with_err("Could not send a message to the daemon");

        if (strstr(command.function_name, "device_hog") == nullptr) {
            daemon_response response;
            if (mq_receive(
                    use_cpu_affine_worker ? queue_from_cpu_affine_daemon_worker : queue_from_normal_daemon_worker,
                    (char *) &response,
                    sizeof(daemon_response),
                    0) != sizeof(daemon_response)) utils::exit_with_err("Could not receive a message from the daemon");
            std::cout << response.output << std::endl;
            std::cout << "Return code: " << response.return_code << std::endl;
            return response.return_code;
        }

        mq_close(queue_to_normal_daemon_worker);
        mq_close(queue_to_cpu_affine_daemon_worker);
        mq_close(queue_from_normal_daemon_worker);
        mq_close(queue_from_cpu_affine_daemon_worker);
    }

    //tear_down_ipc(queue_to_daemon, true);
    //tear_down_ipc(queue_from_daemon, false);
    return EXIT_SUCCESS;
}