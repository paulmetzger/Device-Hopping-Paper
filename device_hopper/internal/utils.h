//
// Created by paul on 10/10/2019.
//

#ifndef PLASTICITY_UTILS_H
#define PLASTICITY_UTILS_H

#include <cerrno>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <string>

#include "../internal/device.h"

namespace plasticity { namespace utils {

        inline bool file_exists(std::string file_name) {
            std::ifstream infile(file_name);
            return infile.good();
        }

        inline void exit_with_err(std::string message) {
            std::cerr << "Error: " << message << std::endl;
            std::cerr << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        inline std::string convert_device_to_str(Device device) {
            std::string result;
            if (device == Device::CPU) {
                result = "CPU";
            } else if (device == Device::GPU) {
                result = "GPU";
            } else {
                std::cerr << "Error: Unknown device" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            return result;
        }
}}

#endif //PLASTICITY_UTILS_H
