cmake_minimum_required(VERSION 3.22)
project(gzram LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

add_subdirectory(server)
add_subdirectory(tests)
if(CUDA_FOUND)
    add_subdirectory(gpu)
endif()