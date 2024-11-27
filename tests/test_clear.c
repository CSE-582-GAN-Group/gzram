#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

struct discard_ioctl_data {
  size_t offset;
  size_t nr_pages;
};

#define DISCARD_IOCTL_IN _IOW('z', 0x8F, struct discard_ioctl_data)

#define QUEUE_DEPTH 1
#define BLOCK_SIZE 4096
#define BLOCKS 8

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <device_path>\n", argv[0]);
    return 1;
  }

  const char *device_path = argv[1];

  // Open the block device
  int fd = open(device_path, O_RDWR);
  if (fd < 0) {
    perror("open");
    exit(1);
  }

//  for(int i = 0; i < BLOCKS; ++i) {
//    pwrite(fd, 0, 0, i);
//  }

//  for(int i = 0; i < BLOCKS; ++i) {
//    int ret = fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, i, BLOCK_SIZE);
//    if(ret != 0) {
//      perror("fallocate");
//    }
//  }

  struct discard_ioctl_data data;
  data.offset = 0;
  data.nr_pages = BLOCKS;
  if (ioctl(fd, DISCARD_IOCTL_IN, &data) < 0) {
    perror("ioctl");
    close(fd);
    return 1;
  }

  close(fd);

  return 0;
}
