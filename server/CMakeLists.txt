cmake_minimum_required(VERSION 3.28)
project(gzram)

find_library(UBLKSRV ublksrv REQUIRED)
find_library(LIBURING_LIBRARY uring REQUIRED)

add_executable(gzram_server gzram_server.c)

target_link_libraries(gzram_server PRIVATE ${LIBURING_LIBRARY} ${UBLKSRV})


