#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>

#define BLOCK_SIZE 4096
#define DEFAULT_SIZE (500 * 1024 * 1024)


static long elapsed_time_ms(struct timespec start, struct timespec end) {
  long seconds = end.tv_sec - start.tv_sec;
  long nanoseconds = end.tv_nsec - start.tv_nsec;
  return (seconds * 1000) + (nanoseconds / 1000000);
}

static double elapsed_time_s(struct timespec start, struct timespec end) {
  long seconds = end.tv_sec - start.tv_sec;
  long nanoseconds = end.tv_nsec - start.tv_nsec;
  return seconds + (nanoseconds / (double)1e9);
}

size_t get_file_size(const char *filename) {
  struct stat st;
  if (stat(filename, &st) == 0) {
    return st.st_size;
  }
  perror("stat");
  exit(1);
}

void read_data_from_file(const char *filename, unsigned char *buffer, size_t size) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror("Error opening file");
    exit(1);
  }

  size_t read_size = fread(buffer, 1, size, file);
  if (read_size != size) {
    fprintf(stderr, "File too small.\n");
    fclose(file);
    exit(1);
  }

  fclose(file);
}

void print_buffer_hex(const unsigned char *buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    printf("%02X ", buffer[i]);
  }
  printf("\n");
}

int compare_buffers(const unsigned char *buf1, const unsigned char *buf2, size_t size) {
  int fail = 0;
  printf("size=%lu\n", size);
  int num_pages = size / BLOCK_SIZE;
  for (size_t i = 0; i < num_pages; i++) {
    for(size_t j = 0; j < BLOCK_SIZE; j++) {
      int index = BLOCK_SIZE*i+j;
      if (buf1[index] != buf2[index]) {
        printf("Mismatch at page number: %lu\n", i);
//      printf("Data written:\n");
//      print_buffer_hex(buf1 + i, BLOCK_SIZE);
//      printf("Data read:\n");
//      print_buffer_hex(buf2 + i, BLOCK_SIZE);
//      exit(1);
        fail = 1;
        return 1;
        break;
      }
    }
  }
  return fail;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <device_path> <file_path> [size]\n", argv[0]);
    return 1;
  }

  const char *device_path = argv[1];
  const char *file_path = argv[2];

  size_t data_size = DEFAULT_SIZE;
  if (argc > 3) {
    data_size = strtoull(argv[3], NULL, 10);
  }

  if (get_file_size(file_path) < data_size) {
    fprintf(stderr, "File too small.\n");
    return 1;
  }

  unsigned char *write_buffer = malloc(data_size);
  unsigned char *read_buffer = malloc(data_size);
  if (!write_buffer || !read_buffer) {
    perror("malloc");
    return 1;
  }

  read_data_from_file(file_path, write_buffer, data_size);

  int fd = open(device_path, O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open");
    free(write_buffer);
    free(read_buffer);
    return 1;
  }

  // Write data
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  if (pwrite(fd, write_buffer, data_size, 0) != data_size) {
    perror("pwrite");
    free(write_buffer);
    free(read_buffer);
    close(fd);
    return 1;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  printf("write time: %f\n", elapsed_time_s(start, end));

  close(fd);

  fd = open(device_path, O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open");
    free(write_buffer);
    free(read_buffer);
    return 1;
  }

  clock_gettime(CLOCK_MONOTONIC, &start);

  if (pread(fd, read_buffer, data_size, 0) != data_size) {
    perror("pread");
    free(write_buffer);
    free(read_buffer);
    close(fd);
    return 1;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  printf("read time: %f\n", elapsed_time_s(start, end));

  if(compare_buffers(write_buffer, read_buffer, data_size)) {
    printf("Fail\n");
    return 1;
  }

  printf("Success\n");

  free(write_buffer);
  free(read_buffer);
  close(fd);
  return 0;
}
