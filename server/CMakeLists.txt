set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

find_library(UBLKSRV ublksrv REQUIRED)
find_library(LIBURING_LIBRARY uring REQUIRED)

find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)

include_directories(../gpu)

add_executable(gzram_server gzram_server.c gzram.c)

target_link_libraries(gzram_server PRIVATE ${LIBURING_LIBRARY} ${UBLKSRV})
target_link_libraries(gzram_server PRIVATE gpu_comp)
target_include_directories(gzram_server PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(gzram_server PRIVATE ${LZ4_LIBRARY})
target_compile_definitions(gzram_server PUBLIC _GNU_SOURCE)

set_target_properties(gzram_server PROPERTIES CUDA_STANDARD 11)