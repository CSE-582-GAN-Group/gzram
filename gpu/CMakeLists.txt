cmake_minimum_required(VERSION 3.12)
project(MyCUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

# Find CUDA
find_package(CUDA REQUIRED)

# Find nvCOMP library
find_package(nvCOMP REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} ${nvCOMP_INCLUDE_DIRS})

# Add executable
# add_executable(main main.cu)
add_executable(main naive.cu)

# Link libraries
target_link_libraries(main ${CUDA_LIBRARIES} nvcomp::nvcomp)

# Specify CUDA standard
set_target_properties(main PROPERTIES CUDA_STANDARD 11)