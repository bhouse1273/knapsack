# CMakeLists.txt for knapsack_solver
cmake_minimum_required(VERSION 3.18)

if(APPLE)
  # Apple Silicon / macOS: build a C++ + ObjC++ binary without CUDA, link Metal frameworks
  project(knapsack_solver LANGUAGES CXX OBJCXX)

  set(CMAKE_CXX_STANDARD 17)
  set(POPULATION_SIZE "512" CACHE STRING "Size of population")
  set(MAX_CAPACITY "20" CACHE STRING "Vehicle capacity")
  add_compile_definitions(
      POPULATION_SIZE=${POPULATION_SIZE}
      MAX_CAPACITY=${MAX_CAPACITY}
  )

  include_directories(
    include
    kernels/metal
  )

  file(GLOB_RECURSE SOURCES "src/*.cpp")
  add_executable(knapsack_solver
    ${SOURCES}
    kernels/metal/metal_api.mm
  )

  # Link required Apple frameworks for Metal runtime compilation
  target_link_libraries(knapsack_solver
    "-framework Metal"
    "-framework Foundation"
  )

else()
  # Jetson / CUDA-enabled builds
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
  file(GLOB_RECURSE KERNELS "kernels/*.cu")

  add_executable(knapsack_solver
    ${SOURCES}
    ${KERNELS}
  )
endif()
