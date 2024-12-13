#!/bin/bash

set -e

echo "# Build zspool"
(cd zspool && make)

echo "# Building server"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd ..