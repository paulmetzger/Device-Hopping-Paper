//
// Created by s1576303 on 05/09/2019.
//
#ifndef PLASTICITY_IPC_COORDINATION_H
#define PLASTICITY_IPC_COORDINATION_H

#include <csetjmp>
#include <csignal>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../device.h"
#include "../utils.h"

#define BACK_TO_BACK_FIRST_APP_SEMAPHORE "btb_first_app_semaphore"
#define BACK_TO_BACK_SECOND_APP_SEMAPHORE "btb_second_app_semaphore"
#define BOOK_KEEPING_MUTEX "book_keeping_mutex"
#define CPU_SEMAPHORE "cpu_semaphore"
#define GPU_SEMAPHORE "gpu_semaphore"

namespace plasticity { namespace internal { namespace ipc { namespace coordination {
    typedef struct {
        sem_t* gpu_sem;
        sem_t* cpu_sem;
        sem_t* back_to_back_first_app_sem;
        sem_t* back_to_back_second_app_sem;
        sem_t* book_keeping_mutex;
    } shared_book_keeping_data;
    shared_book_keeping_data book_keeping_data;

    void acquire_lock() {
        if (sem_wait(book_keeping_data.book_keeping_mutex) != 0) utils::exit_with_err("Could not acquire book keeping lock");
    }

    void cease_to_use(Device device) {
#ifdef DEBUG
        std::cout << "DEBUG: Called book_keeping::cease_to_use ";
#endif
        if (device == Device::GPU) {
#ifdef DEBUG
            std::cout << "GPU";
#endif
            int err = sem_post(book_keeping_data.gpu_sem);
            if (err) plasticity::utils::exit_with_err("sem_post failed in an attempt to free the GPU");
        } else if (device == Device::CPU) {
#ifdef DEBUG
            std::cout << "CPU";
#endif
            int err = sem_post(book_keeping_data.cpu_sem);
            if (err) utils::exit_with_err("sem_post failed in an attempt to free the CPU");
        } else {
            utils::exit_with_err("Unknown device. This code should be unreachable.");
        }
#ifdef DEBUG
        std::cout << std::endl;
#endif
    }

    void destroy() {
        int err = 0;
        if (utils::file_exists("/dev/shm/sem." BACK_TO_BACK_FIRST_APP_SEMAPHORE)) err   = sem_unlink(BACK_TO_BACK_FIRST_APP_SEMAPHORE);
        if (utils::file_exists("/dev/shm/sem." BACK_TO_BACK_SECOND_APP_SEMAPHORE)) err |= sem_unlink(BACK_TO_BACK_SECOND_APP_SEMAPHORE);
        if (utils::file_exists("/dev/shm/sem." BOOK_KEEPING_MUTEX)) err                |= sem_unlink(BOOK_KEEPING_MUTEX);
        if (utils::file_exists("/dev/shm/sem." CPU_SEMAPHORE)) err                     |= sem_unlink(CPU_SEMAPHORE);
        if (utils::file_exists("/dev/shm/sem." GPU_SEMAPHORE)) err                     |= sem_unlink(GPU_SEMAPHORE);
        if (err != 0 && err != ENONET) utils::exit_with_err("Could not unlink all semaphores");
    }

    void finish_turn_in_back_to_back_mode(const int turn) {
        int err = 0;

        if (turn == 0) {
            err |= sem_post(book_keeping_data.back_to_back_first_app_sem);
            err |= sem_post(book_keeping_data.back_to_back_second_app_sem);
        } else if (turn == 1) ; // No action is required when the second application finishes its turn.
        else {
            std::cerr << "Turn: " << turn << std::endl;
            utils::exit_with_err("Back-to-back scheduling does not support more than two applications");
        }

        if (err != 0) utils::exit_with_err("Could not post on one of the semaphores");
    }

    void init() {
        book_keeping_data.back_to_back_first_app_sem  = sem_open(BACK_TO_BACK_FIRST_APP_SEMAPHORE, 0);
        if (book_keeping_data.back_to_back_first_app_sem == SEM_FAILED)
            utils::exit_with_err("Could not open: " BACK_TO_BACK_FIRST_APP_SEMAPHORE);
        book_keeping_data.back_to_back_second_app_sem = sem_open(BACK_TO_BACK_SECOND_APP_SEMAPHORE, 0);
        if (book_keeping_data.back_to_back_second_app_sem == SEM_FAILED)
            utils::exit_with_err("Could not open: " BACK_TO_BACK_SECOND_APP_SEMAPHORE);

        book_keeping_data.book_keeping_mutex = sem_open(BOOK_KEEPING_MUTEX, 0);
        if (book_keeping_data.book_keeping_mutex == SEM_FAILED) utils::exit_with_err("Could not open: " BOOK_KEEPING_MUTEX);

        book_keeping_data.cpu_sem = sem_open(CPU_SEMAPHORE, 0);
        if (book_keeping_data.cpu_sem == SEM_FAILED) utils::exit_with_err("Could not open: " CPU_SEMAPHORE);
        book_keeping_data.gpu_sem = sem_open(GPU_SEMAPHORE, 0);
        if (book_keeping_data.gpu_sem == SEM_FAILED) utils::exit_with_err("Could not open: " GPU_SEMAPHORE);
    }

    void create_semaphores() {
        // Sometimes the semaphore is not immediately removed. The while loop is a workaround for this.
        while((book_keeping_data.back_to_back_first_app_sem = sem_open(BACK_TO_BACK_FIRST_APP_SEMAPHORE, O_CREAT | O_EXCL, 0666, 1)) == 0);
        if (book_keeping_data.back_to_back_first_app_sem == SEM_FAILED) utils::exit_with_err("Could not create " BACK_TO_BACK_FIRST_APP_SEMAPHORE);
        book_keeping_data.back_to_back_second_app_sem = sem_open(BACK_TO_BACK_SECOND_APP_SEMAPHORE, O_CREAT, 0666, 0);
        if (book_keeping_data.back_to_back_second_app_sem == SEM_FAILED)
            utils::exit_with_err("Could not create " BACK_TO_BACK_SECOND_APP_SEMAPHORE);

        book_keeping_data.book_keeping_mutex = sem_open(BOOK_KEEPING_MUTEX, O_CREAT, 0666, 1);
        if (book_keeping_data.book_keeping_mutex == SEM_FAILED) utils::exit_with_err("Could not create" BOOK_KEEPING_MUTEX);

        book_keeping_data.cpu_sem = sem_open(CPU_SEMAPHORE, O_CREAT, 0666, 1);
        if (book_keeping_data.cpu_sem == SEM_FAILED) utils::exit_with_err("Could not create " CPU_SEMAPHORE);
        book_keeping_data.gpu_sem = sem_open(GPU_SEMAPHORE, O_CREAT, 0666, 1);
        if (book_keeping_data.gpu_sem == SEM_FAILED) utils::exit_with_err("Could not create " GPU_SEMAPHORE);
    }

    bool is_avail(Device device) {
#ifdef DEBUG
        std::cout << "DEBUG: Called book_keeping::isAvail" << std::endl;
#endif

        int value = 0;
        int err = 0;

        if      (device == Device::GPU) err = sem_getvalue(book_keeping_data.gpu_sem, &value);
        else if (device == Device::CPU) err = sem_getvalue(book_keeping_data.cpu_sem, &value);
        else utils::exit_with_err("Unknown device in htrop::book_keeping::is_avail");

        if (err != 0) utils::exit_with_err("Could not get the value of one of the semaphores.");
        return value == 1;
    }

    void release_lock() {
        if (sem_post(book_keeping_data.book_keeping_mutex) != 0) utils::exit_with_err("Could not release book keeping lock");
    }

    void shutdown() {
        int err = sem_close(book_keeping_data.back_to_back_first_app_sem);
        err |= sem_close(book_keeping_data.back_to_back_second_app_sem);
        err |= sem_close(book_keeping_data.book_keeping_mutex);
        err |= sem_close(book_keeping_data.cpu_sem);
        err |= sem_close(book_keeping_data.gpu_sem);
        if (err != 0) utils::exit_with_err("Could not unlink all semaphores");
    }

    /**
     * This is a workaround for a bug in the Intel compiler.
     * https://community.intel.com/t5/Intel-C-Compiler/icpc-pthread-cancel-raises-exception-in-C-program/td-p/1137054
     * Pthread_cancel raises SIGABRT instead of exiting the target thread.
     * This handler catches SIGABRT and exits the thread as intended.
     */
    jmp_buf env;
    void workaround_sigabrt_handler(int sig) {
#if defined(DEBUG)
        std::cout << "DEBUG: Entered the handler for the workaround for the pthread_cancel bug in the Intel compiler." << std::endl;
#endif
        longjmp(env, 1);
    }

    void use(Device device, Device pref_device) {
#ifdef DEBUG
        std::cerr << "DEBUG: Called book_keeping::use with "  << (device == Device::CPU ? "CPU" : "GPU") << std::endl;
#endif
        // See the comment above 'workaround_sigabrt_handler' regarding a bug in the Intel compiler.
/*        struct sigaction new_workaround_action, old_crash_handler_action;
        new_workaround_action.sa_handler = workaround_sigabrt_handler;
        new_workaround_action.sa_flags = SA_RESTART;
        sigemptyset(&new_workaround_action.sa_mask);

        if (setjmp(env) == 0) {
            if (device == pref_device) {
                if (sigaction(SIGABRT, &new_workaround_action, &old_crash_handler_action) != 0) {
                    std::cerr << "Could not set SIGABRT handler" << std::endl;
                    std::exit(1);
                }

                if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL) != 0)
                    utils::exit_with_err("Could not set the pthread cancel state");
                if (pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL) != 0)
                    utils::exit_with_err("Could not set the pthread cancel type");
            }*/
            if (device == Device::GPU) {
                int err;
                while ((err = sem_wait(book_keeping_data.gpu_sem)) == EINTR);
                if (err) utils::exit_with_err("Failed to wait for the GPU");
            } else if (device == Device::CPU) {
                int err;
                while ((err = sem_wait(book_keeping_data.cpu_sem)) == EINTR);
                if (err) utils::exit_with_err("Failed to wait for the CPU");
            } else utils::exit_with_err("Unknown device in 'void use(Device device)'");
/*            if (device == pref_device) {
                if (sigaction(SIGABRT, &old_crash_handler_action, NULL)) {
                    std::cerr << "Could not set old SIGABRT handler" << std::endl;
                    std::exit(1);
                }

                if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL) != 0)
                    utils::exit_with_err("Could not set the pthread cancel state");
            }*/
        /*} else {
#if defined(DEBUG)
            std::cout << "DEBUG: Restoring the old sigabrt handler..." << std::endl;
#endif
            if (sigaction(SIGABRT, &old_crash_handler_action, NULL))
                std::cerr << "Could not set old SIGABRT handler" << std::endl;
            //pthread_exit(0);
            if (computation_completed != NULL) *computation_completed = true;
#if defined(DEBUG)
            std::cout << "DEBUG: Done" << std::endl;
#endif
        }*/

#ifdef DEBUG
        std::cerr << "DEBUG: Leaving book_keeping::use with "  << (device == Device::CPU ? "CPU" : "GPU") << std::endl;
#endif
    }

    void wait_for_turn_in_back_to_back_mode(const int turn) {
        int err = 0;

        if (turn == 0) err = sem_wait(book_keeping_data.back_to_back_first_app_sem);
        else if (turn == 1) err = sem_wait(book_keeping_data.back_to_back_second_app_sem);
        else {
            std::cerr << "Turn: " << turn << std::endl;
            utils::exit_with_err("Back-to-back scheduling does not support more than two applications");
        }

        if (err != 0) utils::exit_with_err("Could not wait on one of the semaphores");
    }

/*    void wait_for_device(Device device) {
        if      (device == Device::GPU) sem_wait(&book_keeping_data->gpu_sem); //while(book_keeping_data->gpu_users != 0) {}
        else if (device == Device::CPU) sem_wait(&book_keeping_data->cpu_sem); //while(book_keeping_data->cpu_users != 0) {}
        else {
            std::cerr << "ERROR: Unknown device in htrop::book_keeping::is_avail" << std::endl;
            exit(EXIT_FAILURE);
        }
    }*/
}}}}
#endif //PLASTICITY_SCHEDULER_H

