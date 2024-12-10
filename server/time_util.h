#pragma once

#include "time.h"

static long time_ms(struct timespec time) {
  return (time.tv_sec * 1000) + (time.tv_nsec / 1000000);
}

static long elapsed_time_ms(struct timespec start, struct timespec end) {
  long seconds = end.tv_sec - start.tv_sec;
  long nanoseconds = end.tv_nsec - start.tv_nsec;
  return (seconds * 1000) + (nanoseconds / 1000000);
}

struct timespec add_timespec(struct timespec t1, struct timespec t2) {
  struct timespec result;

  result.tv_sec = t1.tv_sec + t2.tv_sec;
  result.tv_nsec = t1.tv_nsec + t2.tv_nsec;

  if (result.tv_nsec >= 1000000000L) {
    result.tv_sec += result.tv_nsec / 1000000000L;
    result.tv_nsec %= 1000000000L;
  }

  return result;
}

struct timespec subtract_timespec(struct timespec t1, struct timespec t2) {
  struct timespec result;

  result.tv_sec = t1.tv_sec - t2.tv_sec;
  result.tv_nsec = t1.tv_nsec - t2.tv_nsec;

  if (result.tv_nsec < 0) {
    result.tv_sec -= 1;
    result.tv_nsec += 1000000000L;
  }

  return result;
}

void update_timer(struct timespec *t, struct timespec start, struct timespec end) {
  *t = add_timespec(*t, subtract_timespec(end, start));
}