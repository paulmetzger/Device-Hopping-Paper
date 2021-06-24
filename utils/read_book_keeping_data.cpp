//
// Created by s1576303 on 10/09/2019.
//

#include <iostream>
#include "../device_hopper/internal/ipc/coordination.h"
#include "../device_hopper/internal/utils.h"

using namespace plasticity::internal::ipc;

int main(int argc, char* argv[]) {
    coordination::init();
    int cpu_val = 0;
    int gpu_val = 0;
    int err = sem_getvalue(coordination::book_keeping_data.cpu_sem,  &cpu_val);
    err    |= sem_getvalue(coordination::book_keeping_data.gpu_sem, &gpu_val);
    if (err) plasticity::utils::exit_with_err("Could not read semaphore values");

    std::cout << "CPU semaphore value: " << cpu_val << std::endl;
    std::cout << "GPU semaphore value: " << gpu_val << std::endl;
    coordination::shutdown();
}