# CMakeLists.txt for knapsack_solver
cmake_minimum_required(VERSION 3.18)
project(knapsack_solver LANGUAGES CXX CUDA)

# Jetson Orin (Ampere GPU = 8.7)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Correct usage of build-time config values
set(POPULATION_SIZE "512" CACHE STRING "Size of population")
set(MAX_CAPACITY "20" CACHE STRING "Vehicle capacity")

add_compile_definitions(
    POPULATION_SIZE=${POPULATION_SIZE}
    MAX_CAPACITY=${MAX_CAPACITY}
)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  include
  ../knapsack-library/src    # Add this line for knapsack_c.h
)

file(GLOB_RECURSE SOURCES "src/*.cpp")

file(GLOB_RECURSE KERNELS
  "kernels/*.cu"
)

# Add the knapsack_c.cpp file from knapsack-library
set(KNAPSACK_C_FILE "../knapsack-library/src/knapsack_c.cpp")

add_executable(knapsack_solver
  ${SOURCES}
  ${KNAPSACK_C_FILE}    # Add this line
  ${KERNELS}
)
