#!/bin/bash

echo "# Remove ulbk"
sudo modprobe -r ublk_drv

echo "# Remove zspool"
sudo rmmod zspool_drv.ko

echo "# Build zspool"
(cd zspool && make)

echo "# Insert zspool"
sudo insmod zspool/zspool_drv.ko

echo "# Insert ublk"
sudo modprobe ublk_drv