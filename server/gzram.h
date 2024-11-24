#pragma once

#include "ublksrv.h"
#include "ublksrv_utils.h"

int open_zspool(char* path) {
  int fd = open(path, O_RDWR);
  if (fd < 0) {
    perror("open zspool");
    exit(1);
  }
  return fd;
}

int gzram_handle_io(const struct ublksrv_queue *q, const struct ublk_io_data *data)
{
  const struct ublksrv_io_desc *iod = data->iod;

  int fd = q->dev->tgt.fds[1];

//  char buf[100];
//  memset(buf, 0, 100);
//  pwrite(fd, buf, 100, 0);

  ublksrv_complete_io(q, data->tag, iod->nr_sectors << 9);

  return 0;
}