find_library(UBLKSRV ublksrv REQUIRED)
find_library(LIBURING_LIBRARY uring REQUIRED)

add_executable(gzram_server gzram_server.c)

target_link_libraries(gzram_server PRIVATE ${LIBURING_LIBRARY} ${UBLKSRV})
