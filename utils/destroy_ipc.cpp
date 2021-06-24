//
// Created by s1576303 on 05/09/2019.
//

#include <iostream>

#include "../device_hopper/internal/ipc/coordination.h"

int main(int argc, char* argv[]) {
    std::cout << "Destroying IPC mechanisms..." << std::endl;
    plasticity::internal::ipc::coordination::destroy();
    std::cout << "Finished" << std::endl;
}