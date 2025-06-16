#!/bin/bash

# ---   build dist-mnist --- #

export TORCH_HOME="/usr/local/lib/python3.12/dist-packages/torch/"

mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=$TORCH_HOME ..
# or directly:
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

make

# ---   run dist-mnist --- #

NUM_PROCS=8

CMD="mpirun --allow-run-as-root -np $NUM_PROCS ./dist-mnist"

$CMD


# --- profile dist-mnist --- #

nsys profile \
    --force-overwrite true \
    -o dist-mnist.nsys-rep \
    --capture-range=cudaProfilerApi \
    $CMD