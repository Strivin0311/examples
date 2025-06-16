#!/bin/bash

# ---   build dist-mnist --- #

export TORCH_HOME="/usr/local/lib/python3.12/dist-packages/torch/"

mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=$TORCH_HOME ..

make

# ---   run dist-mnist --- #

NUM_PROCS=8

mpirun --allow-run-as-root -np $NUM_PROCS ./dist-mnist