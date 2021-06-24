//
// Created by paul on 18/05/2020.
//

#ifndef PLASTICITY_DAEMON_H
#define PLASTICITY_DAEMON_H

#include <mqueue.h>

namespace plasticity {
    class Daemon {
    private:
        mqd_t queue_from_client_to_normal_daemon_worker;
        mqd_t queue_from_client_to_cpu_affine_daemon_worker;
        mqd_t queue_from_normal_daemon_worker_to_client;
        mqd_t queue_from_cpu_affine_daemon_worker_to_client;

        //static void sigterm_handler(int sig);
    public:
        void start();
        void stop();
    };
}

#endif //PLASTICITY_DAEMON_H
