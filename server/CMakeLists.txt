set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

find_library(UBLKSRV ublksrv REQUIRED)
find_library(LIBURING_LIBRARY uring REQUIRED)

include_directories(../gpu)

add_executable(gzram_server gzram_server.c)

target_link_libraries(gzram_server PRIVATE ${LIBURING_LIBRARY} ${UBLKSRV})
target_link_libraries(gzram_server PRIVATE cuda_example)
set_target_properties(gzram_server PROPERTIES CUDA_STANDARD 11)