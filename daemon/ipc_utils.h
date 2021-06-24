//
// Created by paul on 18/05/2020.
//

#ifndef PLASTICITY_IPC_UTILS_H
#define PLASTICITY_IPC_UTILS_H

#include <mqueue.h>
#include <string>

#include "messages.h"
#include "../device_hopper/internal/utils.h"

#define QUEUE_TO_NORMAL_DAEMON_WORKER_NAME "/queue_to_normal_daemon_worker"
#define QUEUE_TO_CPU_AFFINE_DAEMON_WORKER_NAME "/queue_to_cpu_affine_daemon_worker"
#define QUEUE_FROM_NORMAL_DAEMON_WORKER_NAME "/queue_from_normal_daemon_worker"
#define QUEUE_FROM_CPU_AFFINE_DAEMON_WORKER_NAME "/queue_from_cpu_affine_daemon_worker"

void setup_ipc(mqd_t &queue, bool to_daemon, bool queue_for_cpu_affine_worker) {
    mq_attr queue_attr;
    queue_attr.mq_maxmsg = 1;
    std::string queue_name;
    if (to_daemon) {
        queue_attr.mq_msgsize = sizeof(daemon_command);
        queue_name = queue_for_cpu_affine_worker ? QUEUE_TO_CPU_AFFINE_DAEMON_WORKER_NAME : QUEUE_TO_NORMAL_DAEMON_WORKER_NAME;
    } else {
        queue_attr.mq_msgsize = sizeof(daemon_response);
        queue_name = queue_for_cpu_affine_worker ? QUEUE_FROM_CPU_AFFINE_DAEMON_WORKER_NAME : QUEUE_FROM_NORMAL_DAEMON_WORKER_NAME;
    }

    queue = mq_open(queue_name.c_str(), O_CREAT | O_RDWR, 0666, &queue_attr);
    if (-1 == queue) plasticity::utils::exit_with_err("Could not set up queues");
}

void tear_down_ipc(mqd_t queue, bool to_daemon, bool queue_for_cpu_affine_worker) {
    size_t err = 0;
    err |= mq_close(queue);
    std::string queue_name;
    if (to_daemon) queue_name = queue_for_cpu_affine_worker ? QUEUE_TO_CPU_AFFINE_DAEMON_WORKER_NAME : QUEUE_TO_NORMAL_DAEMON_WORKER_NAME;
    else queue_name = queue_for_cpu_affine_worker ? QUEUE_FROM_CPU_AFFINE_DAEMON_WORKER_NAME : QUEUE_FROM_NORMAL_DAEMON_WORKER_NAME;
    err |= mq_unlink(queue_name.c_str());
    if (err != 0) plasticity::utils::exit_with_err("Could not tear down IPC");
}

#endif //PLASTICITY_IPC_UTILS_H
