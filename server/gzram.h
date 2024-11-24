#pragma once

#include "ublksrv.h"
#include "ublksrv_utils.h"

#define DISCARD_SIZE 0

#define PAGE_SIZE 4096

#define PAGE_SHIFT  12
#define SECTOR_SHIFT  9

#define SECTORS_PER_PAGE_SHIFT  (PAGE_SHIFT - SECTOR_SHIFT)
#define SECTORS_PER_PAGE	(1 << SECTORS_PER_PAGE_SHIFT)

char buf[4096];

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
//      printf("Discard\n");
      for(int i = 0; i < nr_pages; ++i) {
        pwrite(fd, NULL, DISCARD_SIZE, index + i);
      }
      break;
    case UBLK_IO_OP_FLUSH:
//      printf("Flush\n");
      break;
    case UBLK_IO_OP_WRITE:
//      printf("Write, index=%d, nr_pages=%d\n", index, nr_pages);
      for(int i = 0; i < nr_pages; ++i) {
        pwrite(fd, (const void*)(iod->addr + (i << PAGE_SHIFT)), PAGE_SIZE, index + i);
      }
      break;
    case UBLK_IO_OP_READ:
//      printf("Read, index=%d, nr_pages=%d\n", index, nr_pages);
      for(int i = 0; i < nr_pages; ++i) {
//        printf("pread, offest=%d\n", index + i);
        ssize_t len = pread(fd, (void*)buf, PAGE_SIZE, index + i);
        if(len < 0) {
          perror("Read error");
        } else {
          memcpy((void*)(iod->addr + (i << PAGE_SHIFT)), buf, PAGE_SIZE);
        }
        (void)len;
      }
      break;
    default:
      break;
  }

  ublksrv_complete_io(q, data->tag, iod->nr_sectors << SECTOR_SHIFT);

  return 0;
}