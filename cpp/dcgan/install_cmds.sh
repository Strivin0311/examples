#!/bin/bash

# ---   install argparse --- #

# first cd into somewhere else
git clone https://github.com/p-ranav/argparse

cd argparse

mkdir build

cd build

cmake -DARGPARSE_BUILD_SAMPLES=off -DARGPARSE_BUILD_TESTS=off ..

sudo make install

# ---   build dcgan --- #

# cd back to examples/cpp/dcgan/

export TORCH_HOME="/usr/local/lib/python3.12/dist-packages/torch/"

mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=$TORCH_HOME ..

make

# ---   run dcgan --- #

./dcgan --epochs 10


# ---  run prediction --- #

cd ..

python display_samples.py -i ./build/dcgan-sample-10.pt

# the predicted image will be saved as `out.png`