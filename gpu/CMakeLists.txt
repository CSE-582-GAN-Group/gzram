set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

find_package(CUDA REQUIRED)
find_package(nvCOMP REQUIRED)

find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)

include_directories(${CUDA_INCLUDE_DIRS} ${nvCOMP_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../gpu)

add_library(gpu_comp gpu_comp.cu)
target_link_libraries(gpu_comp ${CUDA_LIBRARIES} nvcomp::nvcomp)
set_target_properties(gpu_comp PROPERTIES CUDA_STANDARD 11)

add_executable(test test.cu)
target_link_libraries(test PRIVATE gpu_comp)
target_include_directories(test PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(test PRIVATE ${LZ4_LIBRARY})
set_target_properties(test PROPERTIES CUDA_STANDARD 11)
