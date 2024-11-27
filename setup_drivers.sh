#!/bin/bash
sudo modprobe -r ublk_drv
sudo rmmod zspool_drv.ko
(cd zspool && make)
sudo modprobe ublk_drv
sudo insmod zspool/zspool_drv.ko