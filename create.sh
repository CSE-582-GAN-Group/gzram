#!/bin/bash

set -e

(cd build && make)
echo 1 | sudo tee -a /sys/class/zspool/zspool0/reset
echo 2G | sudo tee -a /sys/class/zspool/zspool0/disksize
sudo ./build/server/gzram_server