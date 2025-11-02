cmake_minimum_required(VERSION 3.18)
message(STATUS "Configuring knapsack from: ${CMAKE_CURRENT_LIST_DIR} (CMake ${CMAKE_VERSION})")
project(knapsack LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_CUDA "Build with CUDA (NVIDIA/Jetson only)" OFF)
option(USE_METAL "Enable Metal backend on Apple" ON)
option(LINK_INSTALLED_LIBKNAPSACK "Link against an installed libknapsack (avoids embedding code twice)" OFF)

if(APPLE AND USE_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()

if(BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures")
  # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Tuning knobs
set(POPULATION_SIZE "512" CACHE STRING "Size of population")
set(MAX_CAPACITY "20" CACHE STRING "Vehicle capacity")

# Sources
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
set(SOURCES ${CPP_SOURCES})

if(APPLE AND USE_METAL)
  list(APPEND SOURCES kernels/metal/metal_api.mm)
endif()

if(BUILD_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES "kernels/*.cu")
  list(APPEND SOURCES ${CUDA_SOURCES})
endif()

add_executable(knapsack_solver ${SOURCES})

# Preprocessor defs
target_compile_definitions(knapsack_solver PRIVATE
  POPULATION_SIZE=${POPULATION_SIZE}
  MAX_CAPACITY=${MAX_CAPACITY}
)

# Includes
target_include_directories(knapsack_solver PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(APPLE AND USE_METAL)
  target_include_directories(knapsack_solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal
  )
  target_compile_options(knapsack_solver PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  )
  target_link_libraries(knapsack_solver PRIVATE
    "-framework Metal"
    "-framework Foundation"
  )
endif()

if(LINK_INSTALLED_LIBKNAPSACK)
  find_path(KNAPSACK_INCLUDE_DIR knapsack_c.h PATHS /usr/local/include)
  find_library(KNAPSACK_LIBRARY NAMES knapsack PATHS /usr/local/lib)
  if(KNAPSACK_INCLUDE_DIR AND KNAPSACK_LIBRARY)
    target_include_directories(knapsack_solver PRIVATE ${KNAPSACK_INCLUDE_DIR})
    target_link_libraries(knapsack_solver PRIVATE ${KNAPSACK_LIBRARY})
  endif()
endif()

install(TARGETS knapsack_solver RUNTIME DESTINATION bin)
