#!/bin/sh
export PLASTICITY_ROOT=`pwd`/

# First tell the daemon to run the dummy application to block one of the devices.
# Which device is blocked is set via "--preferred-device".
# How long the device is blocked is set via "--problem-size". In this case the CPU is blocked for 42ms.
./cmake-build-debug/daemon/client  \
  --function-name="device_hog_main" \
  --path-to-the-application=$PLASTICITY_ROOT/cmake-build-debug/benchmarks/libdevice_hog_for_daemon_with_ocl.so \
  --cli-parameters="--cpu-x-slice-size=0
                    --gpu-x-slice-size=0
                    --cpu-y-slice-size=0
                    --gpu-y-slice-size=0
                    --path-to-library=/home/paul/phd/my_papers/plasticity/library
                    --preferred-device=CPU
                    --problem-size=42
                    --slicing-mode=NoSlicingOnPreferredDevice
                    --sync-applications
                    --benchmark-applications=2"

./cmake-build-debug/daemon/client  \
  --function-name="device_hopper_main" \
  --path-to-the-application=$PLASTICITY_ROOT/cmake-build-debug/benchmarks/libgenerated_ocl_shoc_sfft_plastic_sliced_data_transfers.so \
  --cli-parameters="--benchmark-applications=2
                    --cpu-y-slice-size=256000
                    --gpu-x-slice-size=6400
                    --cpu-x-slice-size=256000
                    --gpu-y-slice-size=256000
                    --launch-device=GPU
                    --path-to-library=/home/paul/phd/my_papers/plasticity/library
                    --preferred-device=CPU
                    --problem-size=256000
                    --sync-applications
                    --slicing-mode=NoSlicingOnPreferredDevice
                    --data-transf-slice-size-mb=256"

