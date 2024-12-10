# gzram

gzram is a compressed memory block device that mimics the functionality of the [Linux zram kernel module](https://docs.kernel.org/admin-guide/blockdev/zram.html) while offloading large compression requests to userspace GPUs to accelerate those requests.

## Prerequisites

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

However, sometimes this only installs liburing 2.0, which is not sufficient for ublksrv. To install liburing2.2, uninstall the current version, and then install it from source as such

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

### nvComp

TODO

## Building

### Userspace daemon

The userspace daemon component is built with cmake. Run the following in the root directory of this repository

```
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

This will build a `zspool_drv.ko` module in the `zspool` directory that can be loaded with

```shell
$ sudo insmod zspool_drv.ko
```

### Running

Run `./setup_driver.sh` to ensure the necessary kernel modules are loaded, namely ublk and zspool.

Then, run `./create.sh` to start the userspace daemon.

The gzram block device will now be available on `/dev/ublkbX` where `X` is the device ID.

If the server crashes, the stored data will still be recoverable from the `/dev/zspool0` device.

## Testing

### Writing and reading data

You can use `dd` to perform basic tests. Set `bs` to the size the request should be, and set `count` to the number of requests. GPU is used for requests above 10MB. For example, the following command tests writing 500MB to gzram

```bash
$ sudo dd if=my_file of=/dev/ublkb0 bs=500M count=1
```

You can also use the provided `test_write_read` test program to write data to the device and then read it back. This is somewhat better for getting write/read timing information, since it only measures the time for the actual block device operation, while `dd` also includes the time to read/write the file. For example, to write and then read 500MB from `my_file` into `/dev/ublk0`,

```bash
$ ./build/tests/test_write_read /dev/ublkb0 my_file 524288000
```

### Getting memory stats

You can get memory stats from the following file

```shell
$ cat /sys/class/zspool/zspool0/mm_stat
```

The first number outputted is the total amount of memory being represented (i.e. # of pages * page size), and the second number is the actual amount of compressed memory. Thus, dividing the first number by the second gives the compression ratio. See [the kernel docs on zram](https://docs.kernel.org/admin-guide/blockdev/zram.html#stats) for more information on the `mm_stat` file format.

### Running stock zram

To run stock Linux zram with the same configuration as gzram, you can use the commands

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