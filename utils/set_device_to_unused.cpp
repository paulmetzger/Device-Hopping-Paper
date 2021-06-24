//
// Created by s1576303 on 05/03/2020.
//

#include <boost/program_options.hpp>

#include "../device_hopper/internal/device.h"
#include "../device_hopper/internal/ipc/coordination.h"

int main(int argc, char* argv[]) {
    boost::program_options::options_description desc("Options");
    desc.add_options()
            ("device", boost::program_options::value<std::string>(), "Device that should be set as used. Options: CPU, GPU");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    Device device;
    if (vm.count("device") == 0) {
        std::cerr << "Error: Device not set" << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        device = (vm["device"].as<std::string>() == "CPU" ? Device::CPU : Device::GPU);
    }

    plasticity::internal::ipc::coordination::init();
    if(!plasticity::internal::ipc::coordination::is_avail(device)) {
        plasticity::internal::ipc::coordination::cease_to_use(device);
    } else {
        std::cerr << "Error: The device is not in use" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
