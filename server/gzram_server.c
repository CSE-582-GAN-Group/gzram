// SPDX-License-Identifier: MIT or GPL-2.0-only

#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <pthread.h>
#include <getopt.h>
#include <stdarg.h>
#include <errno.h>
#include <error.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>

#include "ublksrv.h"
#include "ublksrv_utils.h"

#include "../gpu/naive.cuh"

#include "gzram.h"

#define UBLKSRV_TGT_TYPE_GZRAM  0

struct gzram_queue_info {
  const struct ublksrv_dev *dev;
  int qid;
  pthread_t thread;
};

static struct ublksrv_ctrl_dev *this_dev;

static pthread_mutex_t jbuf_lock;
static char jbuf[4096];

static void sig_handler(int sig)
{
  fprintf(stderr, "got signal %d\n", sig);
  ublksrv_ctrl_stop_dev(this_dev);
}

/*
 * io handler for each ublkdev's queue
 *
 * Just for showing how to build ublksrv target's io handling, so callers
 * can apply these APIs in their own thread context for making one ublk
 * block device.
 */
static void *gzram_io_handler_fn(void *data)
{
  struct gzram_queue_info *info = (struct gzram_queue_info *)data;
  const struct ublksrv_dev *dev = info->dev;
  const struct ublksrv_ctrl_dev_info *dinfo =
          ublksrv_ctrl_get_dev_info(ublksrv_get_ctrl_dev(dev));
  unsigned dev_id = dinfo->dev_id;
  unsigned short q_id = info->qid;
  const struct ublksrv_queue *q;

  sched_setscheduler(getpid(), SCHED_RR, NULL);

  pthread_mutex_lock(&jbuf_lock);
  ublksrv_json_write_queue_info(ublksrv_get_ctrl_dev(dev), jbuf, sizeof jbuf,
                                q_id, ublksrv_gettid());
  pthread_mutex_unlock(&jbuf_lock);
  q = ublksrv_queue_init(dev, q_id, NULL);
  if (!q) {
    fprintf(stderr, "ublk dev %d queue %d init queue failed\n",
            dinfo->dev_id, q_id);
    return NULL;
  }

  fprintf(stdout, "tid %d: ublk dev %d queue %d started\n",
          ublksrv_gettid(),
          dev_id, q->q_id);
  do {
    if (ublksrv_process_io(q) < 0)
      break;
  } while (1);

  fprintf(stdout, "ublk dev %d queue %d exited\n", dev_id, q->q_id);
  ublksrv_queue_deinit(q);
  return NULL;
}

static void gzram_set_parameters(struct ublksrv_ctrl_dev *cdev,
                                     const struct ublksrv_dev *dev)
{
  const struct ublksrv_ctrl_dev_info *info =
          ublksrv_ctrl_get_dev_info(cdev);
  struct ublk_params p = {
          .types = UBLK_PARAM_TYPE_BASIC | UBLK_PARAM_TYPE_DISCARD,
          .basic = {
                  .logical_bs_shift	= PAGE_SHIFT,
                  .physical_bs_shift	= PAGE_SHIFT,
                  .io_opt_shift		= PAGE_SHIFT,
                  .io_min_shift		= PAGE_SHIFT,
                  .max_sectors		= info->max_io_buf_bytes >> SECTOR_SHIFT,
                  .dev_sectors		= dev->tgt.dev_size >> SECTOR_SHIFT,
          },
          .discard = {
                  .max_discard_sectors	= UINT_MAX >> SECTOR_SHIFT,
                  .max_discard_segments	= 1,
                  .discard_granularity = PAGE_SIZE,
          },
  };
  int ret;

  pthread_mutex_lock(&jbuf_lock);
  ublksrv_json_write_params(&p, jbuf, sizeof jbuf);
  pthread_mutex_unlock(&jbuf_lock);

  ret = ublksrv_ctrl_set_params(cdev, &p);
  if (ret)
    fprintf(stderr, "dev %d set basic parameter failed %d\n",
            info->dev_id, ret);
}

static int gzram_io_handler(struct ublksrv_ctrl_dev *ctrl_dev)
{
  int ret, i;
  const struct ublksrv_dev *dev;
  struct gzram_queue_info *info_array;
  void *thread_ret;
  const struct ublksrv_ctrl_dev_info *dinfo =
          ublksrv_ctrl_get_dev_info(ctrl_dev);

  info_array = (struct gzram_queue_info *)
          calloc(sizeof(struct gzram_queue_info), dinfo->nr_hw_queues);
  if (!info_array)
    return -ENOMEM;

  dev = ublksrv_dev_init(ctrl_dev);
  if (!dev) {
    free(info_array);
    return -ENOMEM;
  }

  for (i = 0; i < dinfo->nr_hw_queues; i++) {
    info_array[i].dev = dev;
    info_array[i].qid = i;
    pthread_create(&info_array[i].thread, NULL,
                   gzram_io_handler_fn,
                   &info_array[i]);
  }

  gzram_set_parameters(ctrl_dev, dev);

  /* everything is fine now, start us */
  ret = ublksrv_ctrl_start_dev(ctrl_dev, getpid());
  if (ret < 0)
    goto fail;

  ublksrv_ctrl_get_info(ctrl_dev);
  ublksrv_ctrl_dump(ctrl_dev, jbuf);

  // Increase true queue size
  struct ublk_params params;
  ublksrv_ctrl_get_params(ctrl_dev, &params);
  char cmd[4096];
  sprintf(cmd, "echo %u > /sys/class/block/ublkb%d/queue/max_sectors_kb", params.basic.max_sectors, dinfo->dev_id);
  system(cmd);

  /* wait until we are terminated */
  for (i = 0; i < dinfo->nr_hw_queues; i++)
    pthread_join(info_array[i].thread, &thread_ret);
  fail:
  ublksrv_dev_deinit(dev);

  free(info_array);

  return ret;
}

static int ublksrv_start_daemon(struct ublksrv_ctrl_dev *ctrl_dev)
{
  int ret;

  if (ublksrv_ctrl_get_affinity(ctrl_dev) < 0)
    return -1;

  ret = gzram_io_handler(ctrl_dev);

  return ret;
}



static int gzram_init_tgt(struct ublksrv_dev *dev, int type, int argc,
                         char *argv[])
{
  const struct ublksrv_ctrl_dev_info *info =
          ublksrv_ctrl_get_dev_info(ublksrv_get_ctrl_dev(dev));
  struct ublksrv_tgt_info *tgt = &dev->tgt;
  struct ublksrv_tgt_base_json tgt_json = {
          .type = type,
  };
  strcpy(tgt_json.name, "gzram");

  if(argc != 3) {
    fprintf(stderr, "Usage: %s <dev_size> <zspool_path>\n", argv[0]);
    return -1;
  }

  char *endptr;
  unsigned long long int dev_size = strtoll(argv[1], &endptr, 10);
  if(*endptr != '\0') {
    fprintf(stderr, "Invalid device size\n");
    return -1;
  }

  if(dev_size < 4096) {
    fprintf(stderr, "Device size must be at least 4KB\n");
    return -1;
  }

  char *zspool_path = argv[2];

  if (type != UBLKSRV_TGT_TYPE_GZRAM)
    return -1;

  int zspool_fd = open_zspool(zspool_path);
  if(zspool_fd < 0) {
    return -1;
  }

  tgt_json.dev_size = tgt->dev_size = dev_size;
  tgt->tgt_ring_depth = info->queue_depth;
  tgt->nr_fds = 1;
  tgt->fds[1] = zspool_fd;

  ublksrv_json_write_dev_info(ublksrv_get_ctrl_dev(dev), jbuf, sizeof jbuf);
  ublksrv_json_write_target_base_info(jbuf, sizeof jbuf, &tgt_json);

  cuda_initialize();

  return 0;
}

static void gzram_deinit_tgt(const struct ublksrv_dev *dev)
{
  close(dev->tgt.fds[1]);
}

static int gzram_handle_io_async(const struct ublksrv_queue *q,
                                const struct ublk_io_data *data)
{
  return gzram_handle_io(q, data);
}

void *null_alloc_io_buf(const struct ublksrv_queue *q, int tag, int size)
{
  return malloc(size);
}

void null_free_io_buf(const struct ublksrv_queue *q, void *buf, int tag)
{
  free(buf);
}

static struct ublksrv_tgt_type gzram_tgt_type = {
        .type	= UBLKSRV_TGT_TYPE_GZRAM,
        .name	=  "gzram",
        .init_tgt = gzram_init_tgt,
        .deinit_tgt = gzram_deinit_tgt,
        .handle_io_async = gzram_handle_io_async,
        //.alloc_io_buf = null_alloc_io_buf,
        //.free_io_buf = null_free_io_buf,
};

int main(int argc, char *argv[])
{
  struct ublksrv_dev_data data = {
          .dev_id = -1,
          .max_io_buf_bytes = INT_MAX,
          .nr_hw_queues = DEF_NR_HW_QUEUES,
          .queue_depth = DEF_QD,
          .tgt_type = "gzram",
          .tgt_ops = &gzram_tgt_type,
          .flags = 0,
  };
  struct ublksrv_ctrl_dev *dev;
  int ret;
  static const struct option longopts[] = {
          { "buf",		1,	NULL, 'b' },
          { "need_get_data",	1,	NULL, 'g' },
          { NULL }
  };
  int opt;
  bool use_buf = false;

  while ((opt = getopt_long(argc, argv, ":bg",
                            longopts, NULL)) != -1) {
    switch (opt) {
      case 'b':
        use_buf = true;
        break;
      case 'g':
        data.flags |= UBLK_F_NEED_GET_DATA;
        break;
    }
  }

  data.tgt_argc = argc;
  data.tgt_argv = argv;

  if (signal(SIGTERM, sig_handler) == SIG_ERR)
    error(EXIT_FAILURE, errno, "signal");
  if (signal(SIGINT, sig_handler) == SIG_ERR)
    error(EXIT_FAILURE, errno, "signal");

  if (use_buf) {
    gzram_tgt_type.alloc_io_buf = null_alloc_io_buf;
    gzram_tgt_type.free_io_buf = null_free_io_buf;
  }

  pthread_mutex_init(&jbuf_lock, NULL);
  dev = ublksrv_ctrl_init(&data);
  if (!dev)
    error(EXIT_FAILURE, ENODEV, "ublksrv_ctrl_init");
  /* ugly, but signal handler needs this_dev */
  this_dev = dev;

  ret = ublksrv_ctrl_add_dev(dev);
  if (ret < 0) {
    error(0, -ret, "can't add dev %d", data.dev_id);
    goto fail;
  }

  ret = ublksrv_start_daemon(dev);
  if (ret < 0) {
    error(0, -ret, "can't start daemon");
    goto fail_del_dev;
  }

  ublksrv_ctrl_del_dev(dev);
  ublksrv_ctrl_deinit(dev);
  exit(EXIT_SUCCESS);

  fail_del_dev:
  ublksrv_ctrl_del_dev(dev);
  fail:
  ublksrv_ctrl_deinit(dev);

  exit(EXIT_FAILURE);
}