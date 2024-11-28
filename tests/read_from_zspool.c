#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#define QUEUE_DEPTH 1
#define BLOCK_SIZE 4096

int main(int argc, char *argv[]) {
  int ret = 1;

  if (argc < 4) {
    fprintf(stderr, "Usage: %s <device_path> <page> <output_path>\n", argv[0]);
    return 1;
  }

  const char *device_path = argv[1];
  const char *page = argv[2];
  const char *output_path = argv[3];

  int fd = open(device_path, O_RDWR);
  if (fd < 0) {
    perror("open device");
    exit(1);
  }

  char *buffer;
  buffer = malloc(BLOCK_SIZE);

  ssize_t amount_read = pread(fd, buffer, 4096, atoi(page));

  if(amount_read < 0) {
    perror("Read");
    free(buffer);
    exit(1);
  }

  printf("Read amount: %zd\n", amount_read);

  int fd_output = open(output_path, O_CREAT | O_WRONLY);
  if (fd_output < 0) {
    perror("open");
    goto clean;
  }
  write(fd_output, buffer, amount_read);

  ret = 0;

  close(fd_output);

clean:
  free(buffer);
  close(fd);

  return ret;
}
