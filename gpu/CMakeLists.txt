set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

find_package(CUDA REQUIRED)
find_package(nvCOMP REQUIRED)

find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)

include_directories(${CUDA_INCLUDE_DIRS} ${nvCOMP_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../gpu)

add_library(cuda_comp_naive naive.cu)
target_link_libraries(cuda_comp_naive ${CUDA_LIBRARIES} nvcomp::nvcomp)
set_target_properties(cuda_comp_naive PROPERTIES CUDA_STANDARD 11)

add_library(cuda_example example.cu)
target_link_libraries(cuda_example ${CUDA_LIBRARIES} nvcomp::nvcomp)
set_target_properties(cuda_example PROPERTIES CUDA_STANDARD 11)

add_executable(main main.cu naive.cu)
target_link_libraries(main PRIVATE cuda_comp_naive)
target_include_directories(main PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(main PRIVATE ${LZ4_LIBRARY})
set_target_properties(main PROPERTIES CUDA_STANDARD 11)
