#include <unistd.h>
#include <sys/ioctl.h>

#include "ublksrv.h"
#include "ublksrv_utils.h"
#include "lz4.h"

#include "../gpu/gpu_comp.cuh"
#include "time_util.h"

#include "gzram.h"

#define GPU_COMPRESSION_THRESHOLD_BYTES (10*1024*1024)

struct gzram {
  long request_proc_time;
  long cpu_compression_time;
  long gpu_compression_time;
  long cpu_decompression_time;
  long gpu_decompression_time;
  long zspool_write_time;
  long zspool_read_time;
  long read_copy_overhead_time;
};

static struct gzram gzram = {0};

struct discard_ioctl_data {
  size_t offset;
  size_t nr_pages;
};

#define DISCARD_IOCTL_IN _IOW('z', 0x8F, struct discard_ioctl_data)

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

static int gzram_handle_write_no_comp(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
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
//  printf("CPU Write, index=%d, nr_pages=%d\n", index, nr_pages);

  struct timespec start, end;
  struct timespec zspool_time = {}, compression_time = {};

  char *buf = malloc(LZ4_compressBound(PAGE_SIZE));
  for (int i = 0; i < nr_pages; ++i)
  {
    clock_gettime(CLOCK_MONOTONIC, &start);
    int size = LZ4_compress_default(iod_page_addr(iod, i), buf, PAGE_SIZE, LZ4_compressBound(PAGE_SIZE));
    if(size == 0) {
      printf("error compressing page %d\n", index + i);
      return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    update_timer(&compression_time, start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    ssize_t ret;
    if(size >= PAGE_SIZE) {
//      printf("writing page %d incompressible\n", index + i);
      ret = pwrite(fd, iod_page_addr(iod, i), PAGE_SIZE, index + i);
    } else {
      ret = pwrite(fd, buf, size, index + i);
    }
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
      return (int)ret;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    update_timer(&zspool_time, start, end);
  }
  free(buf);

  clock_gettime(CLOCK_MONOTONIC, &start);
  fsync(fd);
  clock_gettime(CLOCK_MONOTONIC, &end);
  update_timer(&zspool_time, start, end);

  gzram.zspool_write_time += time_ms(zspool_time);
  gzram.cpu_compression_time += time_ms(compression_time);

  return iod_num_bytes(iod);
}

static int gzram_handle_write_gpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  struct timespec start, end;

//  printf("GPU Write, index=%d, nr_pages=%d\n", index, nr_pages);

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
  unsigned long num_zero_pages = 0;
  for (int i = 0; i < compressed->num_pages; ++i)
  {
    CompressedPage *comp_page = &compressed->compressed_pages[i];
    ssize_t ret;
    unsigned long element;
    if(comp_page->size >= PAGE_SIZE) {
//      printf("page %d incompressible\n", index + i);
      ret = pwrite(fd, iod_page_addr(iod, i), PAGE_SIZE, index + i);
    } else {
      if(page_same_filled(iod_page_addr(iod, i), &element)) {
        ++num_zero_pages;
      }
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

//  printf("Number of zero pages: %lu\n", num_zero_pages);

  return iod_num_bytes(iod);
}

static int gzram_handle_read_no_decomp(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
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
//  printf("CPU Read, index=%d, nr_pages=%d\n", index, nr_pages);

  struct timespec start, end;
//  uint64_t zspool_time_us = 0;
//  uint64_t decompression_time_us = 0;
  struct timespec zspool_time = {}, decompression_time = {};

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
//      printf("reading page %d incompressible\n", index + i);
//      memset(iod_page_addr(iod, i), 0, PAGE_SIZE);
      memcpy(iod_page_addr(iod, i), buf, PAGE_SIZE);
      continue;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    update_timer(&zspool_time, start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);

    int ret = LZ4_decompress_safe(buf, iod_page_addr(iod, i), (int)comp_size, PAGE_SIZE);
    if(ret < 0) {
      printf("error decompressing page %ud\n", index + i);
      free(buf);
      return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    update_timer(&decompression_time, start, end);
  }
  free(buf);

  gzram.zspool_read_time += time_ms(zspool_time);
  gzram.cpu_decompression_time += time_ms(decompression_time);

  return iod_num_bytes(iod);
}

static int gzram_handle_read_gpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  struct timespec start, end;

//  printf("Read, index=%d, nr_pages=%d\n", index, nr_pages);
  CompressedPage* compressed_pages = malloc(sizeof(CompressedPage) * nr_pages);
  int* page_indices = malloc(sizeof(int) * nr_pages);
  int* incompressible_pages = malloc(sizeof(int) * nr_pages);
  int num_incompressible_pages = 0;

  int page_index = 0;

  clock_gettime(CLOCK_MONOTONIC, &start);

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
//      printf("incompressible page\n");
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

  clock_gettime(CLOCK_MONOTONIC, &end);
  gzram.zspool_read_time += elapsed_time_ms(start, end);

  int nr_pages_to_decompress = page_index;

  if(nr_pages_to_decompress == 0) {
//    printf("All zero pages\n");
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

  clock_gettime(CLOCK_MONOTONIC, &start);

  char *decomp_data;
  size_t output_size;
  ErrorCode error = decompress_pipelined(&compressed, &decomp_data, &output_size);
  if (error != SUCCESS)
  {
    fprintf(stderr, "Compression failed with error code: %d\n", error);
    free(compressed_pages);
    free(incompressible_pages);
    return -1;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  gzram.gpu_decompression_time += elapsed_time_ms(start, end);

  for(int i = 0; i < nr_pages_to_decompress; ++i) {
    free(compressed_pages[i].data);
  }
  free(compressed_pages);
  assert(output_size == nr_pages*PAGE_SIZE);

  clock_gettime(CLOCK_MONOTONIC, &start);

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
      cuda_free(decomp_data);
      free(incompressible_pages);
      return (int)len;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  gzram.read_copy_overhead_time += elapsed_time_ms(start, end);

  cuda_free(decomp_data);
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

//  printf("start_sector=%llud, nr_sectors=%ud \n", iod->start_sector, iod->nr_sectors);

  switch (ublksrv_get_op(iod)) {
    case UBLK_IO_OP_DISCARD:
      ret = gzram_handle_discard(fd, index, nr_pages);
      break;
    case UBLK_IO_OP_FLUSH:
//      printf("Flush\n");
      break;
    case UBLK_IO_OP_WRITE:
//      printf("Write, tag=%d\n", data->tag);
      // ret = iod_num_bytes(iod);
      if(iod_num_bytes(iod) >= GPU_COMPRESSION_THRESHOLD_BYTES) {
        ret = gzram_handle_write_gpu(iod, fd, index, nr_pages);
      } else {
        ret = gzram_handle_write_cpu(iod, fd, index, nr_pages);
      }
      break;
    case UBLK_IO_OP_READ:
//      printf("Read, tag=%d\n", data->tag);
      // memset((void*)iod->addr, 0, iod_num_bytes(iod));
      // ret = iod_num_bytes(iod);
      if(false) {
        ret = gzram_handle_read_gpu(iod, fd, index, nr_pages);
      } else {
        ret = gzram_handle_read_cpu(iod, fd, index, nr_pages);
      }
      break;
    default:
      break;
  }

  ublksrv_complete_io(q, data->tag, ret);

  clock_gettime(CLOCK_MONOTONIC, &end);

  gzram.request_proc_time += elapsed_time_ms(start, end);

//  printf("-- stats --\n");
//  printf("request_proc_time=%lu\n", gzram.request_proc_time);
//  printf("-- write times --\n");
//  printf("cpu_compression_time=%lu\n", gzram.cpu_compression_time);
//  printf("gpu_compression_time=%lu\n", gzram.gpu_compression_time);
//  printf("zspool_write_time=%lu\n", gzram.zspool_write_time);
//  printf("-- read times --\n");
//  printf("zspool_read_time=%lu\n", gzram.zspool_read_time);
//  printf("gpu_decompression_time=%lu\n", gzram.gpu_decompression_time);
//  printf("cpu_decompression_time=%lu\n", gzram.cpu_decompression_time);
//  printf("read_copy_overhead_time=%lu\n", gzram.read_copy_overhead_time);
//  printf("\n");

  return 0;
}