#!/bin/sh
export PLASTICITY_ROOT=`pwd`/

#Clean up
pkill plasticd
./cmake-build-debug/utils/destroy_ipc
./cmake-build-debug/utils/init_ipc

# Run a benchmark
# First start the daemon
./cmake-build-debug/daemon/ocl_and_cuda_plasticd --stay-in-foreground

