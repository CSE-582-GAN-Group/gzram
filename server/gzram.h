#pragma once

#include <sys/ioctl.h>

#include "ublksrv.h"
#include "ublksrv_utils.h"

#include "../gpu/naive.cuh"

#include "lz4.h"

#define DISCARD_SIZE 0

#define PAGE_SIZE 4096

#define PAGE_SHIFT  12
#define SECTOR_SHIFT  9

#define SECTORS_PER_PAGE_SHIFT  (PAGE_SHIFT - SECTOR_SHIFT)
#define SECTORS_PER_PAGE	(1 << SECTORS_PER_PAGE_SHIFT)

struct gzram {
  long request_proc_time;
  long gpu_compression_time;
  long cpu_decompression_time;
  long zspool_write_time;
  long zspool_read_time;
};

static struct gzram gzram;

struct discard_ioctl_data {
  size_t offset;
  size_t nr_pages;
};

#define DISCARD_IOCTL_IN _IOW('z', 0x8F, struct discard_ioctl_data)

static long elapsed_time_ms(struct timespec start, struct timespec end) {
  long seconds = end.tv_sec - start.tv_sec;
  long nanoseconds = end.tv_nsec - start.tv_nsec;
  return (seconds * 1000) + (nanoseconds / 1000000);
}

void gzram_init_stats() {
  gzram.request_proc_time = 0;
  gzram.gpu_compression_time = 0;
  gzram.cpu_decompression_time = 0;
  gzram.zspool_write_time = 0;
  gzram.zspool_read_time = 0;
}

int open_zspool(char* path) {
  int fd = open(path, O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open zspool");
    return -1;
  }
  return fd;
}

static int zspool_fd(const struct ublksrv_queue *q) {
  return q->dev->tgt.fds[1];
}

static void* iod_page_addr(const struct ublksrv_io_desc *iod, int index) {
  return (void*)(iod->addr + (index << PAGE_SHIFT));
}

static int iod_num_bytes(const struct ublksrv_io_desc *iod) {
  return (int)(iod->nr_sectors << SECTOR_SHIFT);
}

static bool page_same_filled(void *ptr, unsigned long *element) {
  unsigned long *page;
  unsigned long val;
  unsigned int pos, last_pos = PAGE_SIZE / sizeof(*page) - 1;

  page = (unsigned long *)ptr;
  val = page[0];

  if (val != page[last_pos])
    return false;

  for (pos = 1; pos < last_pos; pos++) {
    if (val != page[pos])
      return false;
  }

  *element = val;

  return true;
}

static int gzram_handle_write_test(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  for(int i = 0; i < nr_pages; ++i) {
    ssize_t ret = pwrite(fd, (const void*)(iod->addr + (i << PAGE_SHIFT)), PAGE_SIZE, index + i);
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
      return (int)ret;
    }
  }

  fsync(fd);

  return iod_num_bytes(iod);
}

static int gzram_handle_write_cpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  printf("CPU Write, index=%d, nr_pages=%d\n", index, nr_pages);

  char *buf = malloc(LZ4_compressBound(PAGE_SIZE));
  for (int i = 0; i < nr_pages; ++i)
  {
    int size = LZ4_compress_default(iod_page_addr(iod, i), buf, PAGE_SIZE, LZ4_compressBound(PAGE_SIZE));
    if(size == 0) {
      printf("error compressing page %d\n", index + i);
      return -1;
    }
    ssize_t ret;
    if(size >= PAGE_SIZE) {
      printf("writing page %d incompressible\n", index + i);
      pwrite(fd, iod_page_addr(iod, i), PAGE_SIZE, index + i);
    } else {
      pwrite(fd, buf, size, index + i);
    }
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
      return (int)ret;
    }
  }
  free(buf);

  fsync(fd);

  return iod_num_bytes(iod);
}

static int gzram_handle_write_gpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  struct timespec start, end;

  printf("GPU Write, index=%d, nr_pages=%d\n", index, nr_pages);

  clock_gettime(CLOCK_MONOTONIC, &start);

  CompressedData *compressed = NULL;
  ErrorCode error = compress_pipelined((void*)iod->addr, iod->nr_sectors << SECTOR_SHIFT, &compressed);
  if (error != SUCCESS)
  {
    fprintf(stderr, "Compression failed with error code: %d\n", error);
    return -1;
  }
  assert(compressed->num_pages == nr_pages);

  clock_gettime(CLOCK_MONOTONIC, &end);
  gzram.gpu_compression_time += elapsed_time_ms(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);

  size_t compressed_size = 0;
  for (int i = 0; i < compressed->num_pages; ++i)
  {
    CompressedPage *comp_page = &compressed->compressed_pages[i];
    ssize_t ret;
    unsigned long element;
    if(comp_page->size >= PAGE_SIZE) {
      printf("page %d incompressible\n", index + i);
      ret = pwrite(fd, iod_page_addr(iod, i), PAGE_SIZE, index + i);
    } else {
//      if(page_same_filled(iod_page_addr(iod, i), &element)) {
//        printf("zero page\n");
//      }
//      printf("pwrite offset=%ud size=%lu\n", index+i, comp_page->size);
      ret = pwrite(fd, comp_page->data, comp_page->size, index + i);
      if(ret < comp_page->size) {
        printf("write offset=%ud\n", index + i);
        return -1;
      }
    }
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
      return (int)ret;
    }
    compressed_size += comp_page->size;
  }

  free_compressed_data(compressed);

//  printf("\n");:

  fsync(fd);

  clock_gettime(CLOCK_MONOTONIC, &end);
  gzram.zspool_write_time += elapsed_time_ms(start, end);

  return iod_num_bytes(iod);
}

static int gzram_handle_read_test(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  for(int i = 0; i < nr_pages; ++i) {
    char *page = malloc(4096);
    ssize_t len = pread(fd, page, PAGE_SIZE, index + i);
    if(len < 0) {
      printf("pread, offest=%d\n", index + i);
      perror("Read error");
      return (int)len;
    }
    memcpy((void*)(iod->addr + (i << PAGE_SHIFT)), page, 4096);
    free(page);
  }

  return iod_num_bytes(iod);
}

static int gzram_handle_read_cpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  printf("CPU Read, index=%d, nr_pages=%d\n", index, nr_pages);

  struct timespec start, end;

  char *buf = malloc(PAGE_SIZE);
  for (int i = 0; i < nr_pages; ++i)
  {
    clock_gettime(CLOCK_MONOTONIC, &start);

    ssize_t comp_size = pread(fd, buf, PAGE_SIZE, index + i);
    if(comp_size < 0) {
      printf("read offset=%ud\n", index + i);
      perror("read error");
      free(buf);
      return (int)comp_size;
    }
    if(comp_size == 0) {
      memset(iod_page_addr(iod, i), 0, PAGE_SIZE);
      continue;
    }
    if(comp_size == PAGE_SIZE) {
      printf("reading page %d incompressible\n", index + i);
//      memset(iod_page_addr(iod, i), 0, PAGE_SIZE);
      memcpy(iod_page_addr(iod, i), buf, PAGE_SIZE);
      continue;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    gzram.zspool_read_time += elapsed_time_ms(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);

    int ret = LZ4_decompress_safe(buf, iod_page_addr(iod, i), (int)comp_size, PAGE_SIZE);
    if(ret < 0) {
      printf("error decompressing page %ud\n", index + i);
      free(buf);
      return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    gzram.cpu_decompression_time += elapsed_time_ms(start, end);
  }
  free(buf);

  return iod_num_bytes(iod);
}

static int gzram_handle_read_gpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  printf("Read, index=%d, nr_pages=%d\n", index, nr_pages);
  CompressedPage* compressed_pages = malloc(sizeof(CompressedPage) * nr_pages);
  int* page_indices = malloc(sizeof(int) * nr_pages);
  int* incompressible_pages = malloc(sizeof(int) * nr_pages);
  int num_incompressible_pages = 0;

  int page_index = 0;

  for(int i = 0; i < nr_pages; ++i) {
    void* page_data = malloc(PAGE_SIZE);
    if(page_data == NULL) {
      perror("malloc");
      free(compressed_pages);
      free(incompressible_pages);
      return -1;
    }
    ssize_t len = pread(fd, page_data, PAGE_SIZE, index + i);
    if(len == 0) {
      // Zero page
      free(page_data);
      continue;
    }
    if(len == PAGE_SIZE) {
      printf("incompressible page\n");
      incompressible_pages[num_incompressible_pages] = i;
      ++num_incompressible_pages;
    }
    if(len < 0) {
      printf("pread, offset=%d\n", index + i);
      perror("Read error");
      free(compressed_pages);
      free(incompressible_pages);
      return (int)len;
    }
    compressed_pages[page_index].size = len;
    compressed_pages[page_index].data = page_data;
    page_indices[page_index] = i;
    ++page_index;
  }

  int nr_pages_to_decompress = page_index;

  if(nr_pages_to_decompress == 0) {
    printf("All zero pages\n");
    memset((void*)iod->addr, 0, iod_num_bytes(iod));
    free(compressed_pages);
    free(incompressible_pages);
    return iod_num_bytes(iod);
  }

  CompressedData compressed = {
          .compressed_pages = compressed_pages,
          .num_pages = nr_pages_to_decompress,
          .original_size = nr_pages_to_decompress*PAGE_SIZE,
  };

  char *decomp_data;
  size_t output_size;
  ErrorCode error = decompress_naive(&compressed, &decomp_data, &output_size);
  if (error != SUCCESS)
  {
    fprintf(stderr, "Compression failed with error code: %d\n", error);
    free(compressed_pages);
    free(incompressible_pages);
    return -1;
  }

  for(int i = 0; i < nr_pages_to_decompress; ++i) {
    free(compressed_pages[i].data);
  }
  free(compressed_pages);
  assert(output_size == nr_pages*PAGE_SIZE);

  memset((void*)iod->addr, 0, nr_pages*PAGE_SIZE);
  for(int i = 0; i < nr_pages_to_decompress; ++i) {
    memcpy(iod_page_addr(iod, page_indices[i]), decomp_data + (i << PAGE_SHIFT), PAGE_SIZE);
  }

  for(int i = 0; i < num_incompressible_pages; ++i) {
    int page = incompressible_pages[i];
    int len = pread(fd, iod_page_addr(iod, page), PAGE_SIZE, page);
    if(len < 0) {
      printf("pread, offset=%d\n", index + i);
      perror("Read error");
      free(decomp_data);
      free(incompressible_pages);
      return (int)len;
    }
  }

  free(decomp_data);
  free(incompressible_pages);

  return iod_num_bytes(iod);
}

static int gzram_handle_discard(int fd, unsigned int index, unsigned int nr_pages) {
//  printf("Discard, index=%d, nr_pages=%d\n", index, nr_pages);
  struct discard_ioctl_data data;
  data.offset = index;
  data.nr_pages = nr_pages;
  int ret = ioctl(fd, DISCARD_IOCTL_IN, &data);
  if (ret < 0) {
    perror("ioctl");
    return ret;
  }
  return 0;
}

int gzram_handle_io(const struct ublksrv_queue *q, const struct ublk_io_data *data)
{
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  const struct ublksrv_io_desc *iod = data->iod;
  int fd = zspool_fd(q);

  assert(iod->start_sector % SECTORS_PER_PAGE == 0);
  assert(iod->nr_sectors % SECTORS_PER_PAGE == 0);

  unsigned int index = iod->start_sector >> SECTORS_PER_PAGE_SHIFT;
  unsigned int nr_pages = iod->nr_sectors >> SECTORS_PER_PAGE_SHIFT;

  int ret = 0;

  printf("start_sector=%llud, nr_sectors=%ud \n", iod->start_sector, iod->nr_sectors);

  switch (ublksrv_get_op(iod)) {
    case UBLK_IO_OP_DISCARD:
      ret = gzram_handle_discard(fd, index, nr_pages);
      break;
    case UBLK_IO_OP_FLUSH:
//      printf("Flush\n");
      break;
    case UBLK_IO_OP_WRITE:
      if(iod_num_bytes(iod) >= 10*1024*1024) {
        ret = gzram_handle_write_gpu(iod, fd, index, nr_pages);
      } else {
        ret = gzram_handle_write_cpu(iod, fd, index, nr_pages);
      }
      break;
    case UBLK_IO_OP_READ:
      ret = gzram_handle_read_cpu(iod, fd, index, nr_pages);
      break;
    default:
      break;
  }

  ublksrv_complete_io(q, data->tag, ret);

  clock_gettime(CLOCK_MONOTONIC, &end);

  gzram.request_proc_time += elapsed_time_ms(start, end);

//  printf("request_proc_time=%lu\n", gzram.request_proc_time);
//  printf("gpu_compression_time=%lu\n", gzram.gpu_compression_time);
//  printf("cpu_decompression_time=%lu\n", gzram.cpu_decompression_time);
//  printf("zspool_read_time=%lu\n", gzram.zspool_read_time);
//  printf("zspool_write_time=%lu\n", gzram.zspool_write_time);

  return 0;
}