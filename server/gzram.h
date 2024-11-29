#pragma once

#include <sys/ioctl.h>

#include "ublksrv.h"
#include "ublksrv_utils.h"

#include "../gpu/naive.cuh"
#include "../gpu/example.cuh"

#include "lz4.h"

#define DISCARD_SIZE 0

#define PAGE_SIZE 4096

#define PAGE_SHIFT  12
#define SECTOR_SHIFT  9

#define SECTORS_PER_PAGE_SHIFT  (PAGE_SHIFT - SECTOR_SHIFT)
#define SECTORS_PER_PAGE	(1 << SECTORS_PER_PAGE_SHIFT)

struct discard_ioctl_data {
  size_t offset;
  size_t nr_pages;
};

#define DISCARD_IOCTL_IN _IOW('z', 0x8F, struct discard_ioctl_data)

int open_zspool(char* path) {
  int fd = open(path, O_RDWR);
  if (fd < 0) {
    perror("open zspool");
    exit(1);
  }
  return fd;
}

static int zspool_fd(const struct ublksrv_queue *q) {
  return q->dev->tgt.fds[1];
}

static int iod_num_bytes(const struct ublksrv_io_desc *iod) {
  return (int)(iod->nr_sectors << SECTOR_SHIFT);
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

  return iod_num_bytes(iod);
}

static int gzram_handle_write_cpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  char *buf = malloc(LZ4_compressBound(PAGE_SIZE));
  for (int i = 0; i < nr_pages; ++i)
  {
    int size = LZ4_compress_default((const void*)(iod->addr + (i << PAGE_SHIFT)), buf, PAGE_SIZE, LZ4_compressBound(PAGE_SIZE));
    if(size > PAGE_SIZE) {
      printf("page %d incompressible\n", index + i);
    }
    if(size == 0) {
      printf("error compressing page %d\n", index + i);
      return -1;
    }
    ssize_t ret = pwrite(fd, buf, size, index + i);
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
      return (int)ret;
    }
  }
  free(buf);

  return iod_num_bytes(iod);
}

static int gzram_handle_write(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  printf("Write, index=%d, nr_pages=%d\n", index, nr_pages);
  CompressedData *compressed = NULL;
  ErrorCode error = compress((void*)iod->addr, iod->nr_sectors << SECTOR_SHIFT, &compressed);
  if (error != SUCCESS)
  {
    fprintf(stderr, "Compression failed with error code: %d\n", error);
  }
  assert(compressed->num_pages == nr_pages);
  size_t compressed_size = 0;
  for (int i = 0; i < compressed->num_pages; ++i)
  {
    CompressedPage *comp_page = &compressed->compressed_pages[i];
    if(comp_page->size > PAGE_SIZE) {
      printf("page %d incompressible\n", index + i);
    }
    ssize_t ret = pwrite(fd, comp_page->data, comp_page->size, index + i);
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
      return (int)ret;
    }
    compressed_size += comp_page->size;
  }
  printf("Compression Results:\n");
  printf("Original size: %zu bytes\n", compressed->original_size);
  printf("Compressed size: %zu bytes\n", compressed_size);
  printf("Compression ratio: %.2f\n", (float)compressed->original_size / compressed_size);
  free_compressed_data(compressed);

  printf("\n");

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
  char *buf = malloc(PAGE_SIZE);
  for (int i = 0; i < nr_pages; ++i)
  {
    ssize_t comp_size = pread(fd, buf, PAGE_SIZE, index + i);
    if(comp_size < 0) {
      printf("read offset=%ud\n", index + i);
      perror("read error");
      free(buf);
      return (int)comp_size;
    }
    LZ4_decompress_safe(buf, (void*)(iod->addr + (i << PAGE_SHIFT)), (int)comp_size, PAGE_SIZE);
  }
  free(buf);

  return iod_num_bytes(iod);
}

static int gzram_handle_read(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  printf("Read, index=%d, nr_pages=%d\n", index, nr_pages);
  CompressedPage* compressed_pages = malloc(sizeof(CompressedPage) * nr_pages);
  int* page_indices = malloc(sizeof(int) * nr_pages);

  int page_index = 0;

  for(int i = 0; i < nr_pages; ++i) {
    void* page_data = malloc(PAGE_SIZE);
    if(page_data == NULL) {
      perror("malloc");
      free(compressed_pages);
      return -1;
    }
    ssize_t len = pread(fd, page_data, PAGE_SIZE, index + i);
    if(len == 0) {
      // Zero page
      free(page_data);
      continue;
    }
    if(len < 0) {
      printf("pread, offset=%d\n", index + i);
      perror("Read error");
      free(compressed_pages);
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
    return iod_num_bytes(iod);
  }

  CompressedData compressed = {
          .compressed_pages = compressed_pages,
          .num_pages = nr_pages_to_decompress,
          .original_size = nr_pages_to_decompress*PAGE_SIZE,
  };

  char *decomp_data;
  size_t output_size;
  decompress(&compressed, &decomp_data, &output_size);
  for(int i = 0; i < nr_pages_to_decompress; ++i) {
    free(compressed_pages[i].data);
  }
  free(compressed_pages);
  assert(output_size == nr_pages*PAGE_SIZE);

  memset((void*)iod->addr, 0, nr_pages*PAGE_SIZE);
  for(int i = 0; i < nr_pages_to_decompress; ++i) {
    memcpy((void*)(iod->addr + (page_indices[i] << PAGE_SHIFT)), decomp_data + (i << PAGE_SHIFT), PAGE_SIZE);
  }

  free(decomp_data);

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
      ret = gzram_handle_write_cpu(iod, fd, index, nr_pages);
      break;
    case UBLK_IO_OP_READ:
      ret = gzram_handle_read(iod, fd, index, nr_pages);
      break;
    default:
      break;
  }

  ublksrv_complete_io(q, data->tag, ret);

  return 0;
}