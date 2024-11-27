#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#define BLOCK_SIZE 4096
#define DEFAULT_SIZE (500 * 1024 * 1024)

void generate_data(unsigned char *buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    buffer[i] = i % 256;
  }
}

void compare_buffers(const unsigned char *buf1, const unsigned char *buf2, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (buf1[i] != buf2[i]) {
      printf("Mismatch at page number: %lu\n", i / BLOCK_SIZE);
      exit(1);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <device_path> (size)\n", argv[0]);
    return 1;
  }

  const char *device_path = argv[1];

  size_t data_size = DEFAULT_SIZE;
  if (argc > 2) {
    data_size = strtoull(argv[2], NULL, 10);
  }

  int fd = open(device_path, O_RDWR);
  if (fd < 0) {
    perror("Error opening block device");
    return 1;
  }

  unsigned char *write_buffer = malloc(data_size);
  unsigned char *read_buffer = malloc(data_size);
  if (!write_buffer || !read_buffer) {
    perror("Error allocating memory");
    close(fd);
    return 1;
  }

  generate_data(write_buffer, data_size);

  if (write(fd, write_buffer, data_size) != data_size) {
    perror("Error writing to block device");
    free(write_buffer);
    free(read_buffer);
    close(fd);
    return 1;
  }

  if (lseek(fd, 0, SEEK_SET) < 0) {
    perror("Error seeking block device");
    free(write_buffer);
    free(read_buffer);
    close(fd);
    return 1;
  }

  if (read(fd, read_buffer, data_size) != data_size) {
    perror("Error reading from block device");
    free(write_buffer);
    free(read_buffer);
    close(fd);
    return 1;
  }

  compare_buffers(write_buffer, read_buffer, data_size);

  printf("Success\n");

  free(write_buffer);
  free(read_buffer);
  close(fd);
  return 0;
}
