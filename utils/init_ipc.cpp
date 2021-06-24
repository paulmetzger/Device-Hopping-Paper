//
// Created by s1576303 on 05/09/2019.
//

#include <iostream>

#include "../device_hopper/internal/ipc/coordination.h"

int main(int argc, char* argv[]) {
    std::cout << "Intialising IPC mechanisms..." << std::endl;
    std::cout << "Initialising semaphores..." << std::endl;
    plasticity::internal::ipc::coordination::create_semaphores();
    plasticity::internal::ipc::coordination::shutdown();
    std::cout << "Finished" << std::endl;
}