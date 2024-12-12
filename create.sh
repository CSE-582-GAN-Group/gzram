#!/bin/bash

set -e

CAPACITY=${1:-2147483648}

(cd build && make)
echo 1 | sudo tee -a /sys/class/zspool/zspool0/reset
echo $CAPACITY | sudo tee -a /sys/class/zspool/zspool0/disksize
sudo ./build/server/gzram_server $CAPACITY /dev/zspool0