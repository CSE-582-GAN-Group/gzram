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

static void gzram_handle_write_test(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  for(int i = 0; i < nr_pages; ++i) {
    ssize_t ret = pwrite(fd, (const void*)(iod->addr + (i << PAGE_SHIFT)), PAGE_SIZE, index + i);
    if(ret < 0) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
    }
  }
}

static void gzram_handle_write_cpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  char *buf = malloc(LZ4_compressBound(4096));
  for (int i = 0; i < nr_pages; ++i)
  {
    int size = LZ4_compress_default((const void*)(iod->addr + (i << PAGE_SHIFT)), buf, 4096, LZ4_compressBound(4096));
    ssize_t ret = pwrite(fd, buf, size, index + i);
    if(ret < size) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
    }
  }
  free(buf);
}

static void gzram_handle_write(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
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
    ssize_t ret = pwrite(fd, comp_page->data, comp_page->size, index + i);
    if(ret < comp_page->size) {
      printf("write offset=%ud\n", index + i);
      perror("write error");
    }
    compressed_size += comp_page->size;
  }
  printf("Compression Results:\n");
  printf("Original size: %zu bytes\n", compressed->original_size);
  printf("Compressed size: %zu bytes\n", compressed_size);
  printf("Compression ratio: %.2f\n", (float)compressed->original_size / compressed_size);
  free_compressed_data(compressed);

  printf("\n");
}

static void gzram_handle_read_test(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  for(int i = 0; i < nr_pages; ++i) {
    char *page = malloc(4096);
    ssize_t len = pread(fd, page, PAGE_SIZE, index + i);
    if(len < 0) {
      printf("pread, offest=%d\n", index + i);
      perror("Read error");
    }
    memcpy((void*)(iod->addr + (i << PAGE_SHIFT)), page, 4096);
    free(page);
  }
}

static void gzram_handle_read_cpu(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  char *buf = malloc(4096);
  for (int i = 0; i < nr_pages; ++i)
  {
    ssize_t comp_size = pread(fd, buf, 4096, index + i);
    if(comp_size < 0) {
      printf("read offset=%ud\n", index + i);
      perror("read error");
    }
    LZ4_decompress_safe(buf, (void*)(iod->addr + (i << PAGE_SHIFT)), (int)comp_size, 4096);
  }
  free(buf);
}

static void gzram_handle_read(const struct ublksrv_io_desc *iod, int fd, unsigned int index, unsigned int nr_pages) {
  printf("Read, index=%d, nr_pages=%d\n", index, nr_pages);
  CompressedPage* compressed_pages = malloc(sizeof(CompressedPage) * nr_pages);

  for(int i = 0; i < nr_pages; ++i) {
    void* page_data = malloc(4096);
    if(page_data == NULL) {
      perror("malloc");
      goto clean;
    }
    ssize_t len = pread(fd, page_data, 4096, index + i);
    if(len < 0) {
      printf("pread, offset=%d\n", index + i);
      perror("Read error");
      goto clean;
    }
    compressed_pages[i].size = len;
    compressed_pages[i].data = page_data;
  }

  CompressedData compressed = {
          .compressed_pages = compressed_pages,
          .num_pages = nr_pages,
          .original_size = nr_pages*4096,
  };

  char *decomp_data;
  size_t output_size;
  decompress(&compressed, &decomp_data, &output_size);

  for(int i = 0; i < nr_pages; ++i) {
    free(compressed_pages[i].data);
  }

  assert(output_size == nr_pages*4096);

  memcpy((void*)iod->addr, decomp_data, nr_pages*4096);

  free(decomp_data);

clean:
  free(compressed_pages);
}

static void gzram_handle_discard(int fd, unsigned int index, unsigned int nr_pages) {
//  printf("Discard, index=%d, nr_pages=%d\n", index, nr_pages);
  struct discard_ioctl_data data;
  data.offset = index;
  data.nr_pages = nr_pages;
  int ret = ioctl(fd, DISCARD_IOCTL_IN, &data);
  if (ret < 0) {
    perror("ioctl");
    return;
  }
}

int gzram_handle_io(const struct ublksrv_queue *q, const struct ublk_io_data *data)
{
  const struct ublksrv_io_desc *iod = data->iod;
  int fd = zspool_fd(q);

  assert(iod->start_sector % SECTORS_PER_PAGE == 0);
  assert(iod->nr_sectors % SECTORS_PER_PAGE == 0);

  unsigned int index = iod->start_sector >> SECTORS_PER_PAGE_SHIFT;
  unsigned int nr_pages = iod->nr_sectors >> SECTORS_PER_PAGE_SHIFT;

//  printf("start_sector=%llud, nr_sectors=%ud \n", iod->start_sector, iod->nr_sectors);

  switch (ublksrv_get_op(iod)) {
    case UBLK_IO_OP_DISCARD:
      gzram_handle_discard(fd, index, nr_pages);
      break;
    case UBLK_IO_OP_FLUSH:
//      printf("Flush\n");
      break;
    case UBLK_IO_OP_WRITE:
      gzram_handle_write_cpu(iod, fd, index, nr_pages);
      break;
    case UBLK_IO_OP_READ:
      gzram_handle_read_cpu(iod, fd, index, nr_pages);
      break;
    default:
      break;
  }

  ublksrv_complete_io(q, data->tag, iod->nr_sectors << SECTOR_SHIFT);

  return 0;
}