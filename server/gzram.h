#pragma once

#define PAGE_SIZE 4096

#define PAGE_SHIFT  12
#define SECTOR_SHIFT  9

#define SECTORS_PER_PAGE_SHIFT  (PAGE_SHIFT - SECTOR_SHIFT)
#define SECTORS_PER_PAGE	(1 << SECTORS_PER_PAGE_SHIFT)

int open_zspool(char* path);
int gzram_handle_io(const struct ublksrv_queue *q, const struct ublk_io_data *data);