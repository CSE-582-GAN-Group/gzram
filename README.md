# gzram

gzram is a compressed memory block device that mimics the functionality of the [Linux zram kernel module](https://docs.kernel.org/admin-guide/blockdev/zram.html) while offloading large compression requests to userspace GPUs to accelerate those requests.

## Repository structure

The repository is organized into the following subdirectories:

- `gpu` - our GPU compression/decompression library built on top of nvCOMP in CUDA C
- `server` - the userspace daemon server for the userspace block device
- `zspool` - our zspool Linux kernel module for storing compressed pages
- `tests` - programs to perform simple tests on the block device
- `profiling` - scripts for profiling the block device

## Prerequisites

### System

At minimum, the system must be running Linux kernel version >= 6.0 (preferably on a Debian-based distribution) and be equipped with an NVIDIA GPU. For our tests, we used a node of type `gpu_p100` on [ChameleonCloud](https://www.chameleoncloud.org/).

### Kernel

This project was developed on Linux kernel version 6.11.5, but version that supports ublk (version 6.0 and up) should work.

The kernel should be built with the "userspace block device" (ublk) module enabled. Additionally, if you want to compare gzram to the stock kernel zram, the kernel should be built with the "Compressed RAM block device support" (zram) module enabled.

### Basic libraries

The following libraries must be installed

- LZ4
- liburing 2.2

These can be installed with

```shell
$ sudo apt install -y liblz4-dev liburing-dev
```

However, sometimes this only installs liburing 2.0, which is not sufficient for ublksrv. To install liburing 2.2, uninstall the current version, and then install it from source as such

```shell
$ git clone https://github.com/axboe/liburing.git && cd liburing
$ sudo make install
```

### `libublksrv`

Follow the instructions on the [ublksrv repository](https://github.com/ublk-org/ublksrv) to install `libublksrv`. We summarize the steps here

```shell
$ git clone https://github.com/ublk-org/ublksrv && cd ublksrv 
$ sudo apt install -y pkg-config libtool
$ autoreconf -i
$ ./configure
$ sudo make install
```

### GPU and CUDA

The system must be equipped with an NVIDIA GPU and have the respective GPU drivers and CUDA version installed. The easiest method is to visit the [CUDA download page](https://developer.nvidia.com/cuda-downloads) and select the "runfile" download option. For example, to installed CUDA 12.6 on Linux:

```shell
$ wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
$ sudo sh cuda_12.6.3_560.35.05_linux.run
```

### nvCOMP

The nvCOMP GPU compression library can be installed from the [nvCOMP download page](https://developer.nvidia.com/nvcomp-downloads). For example, to install on Linux for CUDA 12:

```shell
$ wget https://developer.download.nvidia.com/compute/nvcomp/4.1.1/local_installers/nvcomp-local-repo-debian12-4.1.1_4.1.1-1_amd64.deb
$ sudo dpkg -i nvcomp-local-repo-debian12-4.1.1_4.1.1-1_amd64.deb
$ sudo cp /var/cuda-repo-debian12-4-1-local/nvcomp-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install nvcomp
```

## Building

### Userspace daemon

The userspace daemon component is built with cmake. Run the following in the root directory of this repository

```shell
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build .
```

### zspool kernel module

The zspool kernel module is built separately. Build it as such

```shell
$ cd zspool
$ make
```

Warning: do not run `make` with `sudo`; if you do, you may need to install your Linux headers.

This will build a `zspool_drv.ko` module in the `zspool` directory that can be loaded with

```shell
$ sudo insmod zspool_drv.ko
```

This will create the device `/dev/zspool0`.

### Build script

Alternatively, run `build.sh` to build both components.

## Running

Run `./setup_driver.sh` to ensure the necessary kernel modules are loaded, namely ublk and zspool.

Then, run `./create.sh` to start the userspace daemon. The daemon will be connected to the zspool device at `/dev/zspool0`, but this can be changed in the script. The default capacity of the device is 2G, but this could be changed by running `./create.sh <capacity_in_bytes>`.

The gzram block device will now be available on `/dev/ublkbX` where `X` is the device ID (typically X is 0).

### Manual setup

You can also set up the device manually. First, initialize the zspool device. For example, if creating it with capacity of 2GB, do

```bash
echo 1 | sudo tee -a /sys/class/zspool/zspool0/reset # reset the zspool device
echo 2147483648 | sudo tee -a /sys/class/zspool/zspool0/disksize # set the capacity of the pool
```

You can also create another zspool device by `sudo cat /sys/class/zspool-control/hot_add` and remove a device by `echo <device_id> | sudo tee -a /sys/class/zspool-control/hot_remove`.

Then, start the server with `sudo ./build/server/gzram_server <capacity> <zspool_device_path>`. For example, `sudo ./build/server/gzram_server 2147483648 /dev/zspool0`. The capacity set here should match the capacity of the zspool device.

If the server crashes, the stored data will still be recoverable from the `/dev/zspool0` device. So, you can simply restart the server without resetting the zspool device. 

## Testing

### Writing and reading data

You can use `dd` to perform basic tests. Set `bs` to the size the request should be, and set `count` to the number of requests. GPU is used for compression requests above 10MB. For example, the following command tests writing 128MB to gzram

```bash
$ sudo dd if=my_file of=/dev/ublkb0 bs=128M count=1
```

You can also use the provided `test_write_read` test program to write data to the device and then read it back. This is somewhat better for getting write/read timing information, since it only measures the time for the actual block device operation, while `dd` also includes the time to read/write the file. For example, to write and then read 500MB from `my_file` into `/dev/ublk0`,

```bash
$ ./build/tests/test_write_read /dev/ublkb0 my_file 134217728
```

You can also discard data from the block device as usual, which will release the pages from the zspool. For example, to discard all data,

```bash
$ blkdiscard /dev/ublkb0
```

### Getting memory stats

You can get memory stats from the following file

```shell
$ cat /sys/class/zspool/zspool0/mm_stat
```

The first number outputted is the total amount of memory being represented (i.e. # of pages * page size), and the second number is the actual amount of compressed memory. Thus, dividing the first number by the second gives the compression ratio. See [the kernel docs on zram](https://docs.kernel.org/admin-guide/blockdev/zram.html#stats) for more information on the `mm_stat` file format.

### Running stock zram

To run stock Linux zram with the same configuration as gzram for comparison, you can use the commands

```shell
$ sudo modprobe zram
$ echo 1 | sudo tee -a /sys/block/zram0/reset
$ echo lz4 | sudo tee -a /sys/block/zram0/comp_algorithm
$ echo 2G | sudo tee -a /sys/block/zram0/disksize
```

### Data corpus

The [Silesia lossless data compression corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) provides a good variety of file types to test compression. Download it with the following

```shell
$ wget https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip
$ unzip silesia.zip
```

You may want to expand the size of the silesia files for certain tests. We provide a script to expand them to 1GB each by simply repeating the file contents. This method of expansion does not erroneously impact the compression results because despite the fact we are repeating the whole file, compression acts on each 4KB block of the file independently. The expansion command can be used as such

```shell
$ python3 expand_files.py <path_to_silesia> <output_path> 
```