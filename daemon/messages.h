//
// Created by paul on 18/05/2020.
//

#ifndef PLASTICITY_MESSAGES_H
#define PLASTICITY_MESSAGES_H

#include <chrono>

using namespace std::chrono;

enum CommandType {EXECUTE_APPLICATION, EXIT_DAEMON, NONE};

typedef struct DaemonCommand {
    CommandType command_type = CommandType::NONE;
    char application_cli_parameters[1024*5];
    char function_name[1024];
    char path_to_application[1024];
    high_resolution_clock::time_point time_start;
} daemon_command;

typedef struct DaemonResponse {
    char output[2048*100];
    int return_code;
} daemon_response;

#endif //PLASTICITY_MESSAGES_H
