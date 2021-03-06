#!/bin/sh
# Check if ICC is installed
which icc
if [ $? -ne 0 ] ;
then
  echo "Error: icc is not installed."
  exit
fi

# Set environment variable
export PLASTICITY_ROOT=`pwd`/
echo $PLASTICITY_ROOT

# Build a generate for OpenCL binaries
mkdir cmake-build-debug
cd cmake-build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make ocl_binary_generator

# Translate the high-level implementations that use our parallel_for
cd ../source_to_source_translator
./translate_benchmarks.sh

# Build everything
cd ../cmake-build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_BENCHMARKS=True ..
make -j`nproc`

# Set up IPC
./utils/destroy_ipc
./utils/init_ipc