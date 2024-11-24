#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define QUEUE_DEPTH 1
#define BLOCK_SIZE 4096

void error_exit(const char *msg) {
  perror(msg);
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <device_path>\n", argv[0]);
    return 1;
  }

  const char *device_path = argv[1];

  // Open the block device
  int fd = open(device_path, O_RDWR);
  if (fd < 0) {
    error_exit("open");
  }

  char *buffer;
  buffer = malloc(BLOCK_SIZE);

  int amount_read = pread(fd, buffer, 4096, 57);

  if(amount_read < 0) {
    perror("Read");
    free(buffer);
    exit(1);
  }

  printf("Read amount: %d\n", amount_read);

  free(buffer);
  close(fd);

  return 0;
}
