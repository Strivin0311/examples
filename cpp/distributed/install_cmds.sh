#!/bin/bash

# ---   build dist-mnist --- #

export TORCH_HOME="/usr/local/lib/python3.12/dist-packages/torch/"

mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=$TORCH_HOME ..
# or directly:
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

make

# ---   run dist-mnist --- #

# the following env vars are necessary for initializing NCCL process group's store
# but no need for MPI process group

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23456}

# LAUNCHER=mpirun
LAUNCHER=torchrun

NUM_PROCS=8

if [[ $LAUNCHER == "mpirun" ]]; then
    CMD="mpirun \
        --allow-run-as-root \
        -np $NUM_PROCS \
        ./dist-mnist \
        -l $LAUNCHER
    "
elif [[ $LAUNCHER == "torchrun" ]]; then
    # NOTE:
    #   1. we have to use --no-python flag to run binary executable file
    #   2. when using torchrun, the MPI backend won't set the global rank as expected (see issue: https://github.com/pytorch/pytorch/issues/31762)
    #     thus we can only use `mpirun` with MPI backend
    CMD="torchrun \
        --nproc_per_node=$NUM_PROCS \
        --nnodes=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --no-python \
        ./dist-mnist \
        -l $LAUNCHER
    "
fi

$CMD > dist-mnist.log 2>&1
exit

# --- profile dist-mnist --- #

nsys profile \
    --force-overwrite true \
    -o dist-mnist.nsys-rep \
    --capture-range=cudaProfilerApi \
    $CMD > dist-mnist.log 2>&1