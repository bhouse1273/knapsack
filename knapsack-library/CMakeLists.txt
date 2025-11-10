cmake_minimum_required(VERSION 3.18)
project(knapsack_library LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_METAL "Enable Metal backend on Apple" ON)
option(BUILD_CPU_ONLY "Build CPU-only version without Metal support" OFF)
option(BUILD_CUDA "Build with CUDA support (NVIDIA GPUs)" OFF)

# Force CPU-only build on non-Apple platforms or when explicitly requested
if(BUILD_CPU_ONLY OR NOT APPLE)
  set(USE_METAL OFF)
endif()

# CUDA and Metal are mutually exclusive
if(BUILD_CUDA)
  set(USE_METAL OFF)
  set(BUILD_CPU_ONLY OFF)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(APPLE AND USE_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# Sources in this library
file(GLOB LIB_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
set(LIB_SOURCES ${LIB_CPP})

# Include V2 core sources from the main project to avoid code duplication
set(PROJ_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
list(APPEND LIB_SOURCES
  "${PROJ_ROOT}/src/v2/Data.cpp"
  "${PROJ_ROOT}/src/v2/EvalCPU.cpp"
  "${PROJ_ROOT}/src/v2/BeamSearch.cpp"
  "${PROJ_ROOT}/src/v2/Preprocess.cpp"
  "${PROJ_ROOT}/src/v2/Config_validate.cpp"
)

# Add Metal bridge (Objective-C++) on Apple
if(APPLE AND USE_METAL)
  set(KERNELS_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/metal")
  list(APPEND LIB_SOURCES "${KERNELS_METAL_DIR}/metal_api.mm")
  list(APPEND LIB_SOURCES "${PROJ_ROOT}/src/v2/Config.mm")
# Add CUDA sources on CUDA builds
elseif(BUILD_CUDA)
  set(KERNELS_CUDA_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/cuda")
  list(APPEND LIB_SOURCES "${KERNELS_CUDA_DIR}/cuda_api.cu")
  list(APPEND LIB_SOURCES "${PROJ_ROOT}/src/v2/Config_json.cpp")
else()
  list(APPEND LIB_SOURCES "${PROJ_ROOT}/src/v2/Config_json.cpp")
endif()

add_library(knapsack STATIC ${LIB_SOURCES})

# Set platform-specific output name
if(BUILD_CPU_ONLY)
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_cpu")
  message(STATUS "Library output name: libknapsack_cpu.a")
elseif(BUILD_CUDA)
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_cuda")
  message(STATUS "Library output name: libknapsack_cuda.a")
elseif(USE_METAL)
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_metal")
  message(STATUS "Library output name: libknapsack_metal.a")
else()
  # Fallback for any other case
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_generic")
  message(STATUS "Library output name: libknapsack_generic.a")
endif()

# Set Metal directory path (always needed for headers, even on Linux)
set(KERNELS_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/metal")

# Compile definitions based on build type
if(BUILD_CPU_ONLY)
  target_compile_definitions(knapsack PRIVATE KNAPSACK_CPU_ONLY)
  message(STATUS "Building CPU-only knapsack library (no GPU support)")
elseif(BUILD_CUDA)
  target_compile_definitions(knapsack PRIVATE KNAPSACK_CUDA_SUPPORT)
  message(STATUS "Building knapsack library with CUDA support")
elseif(USE_METAL)
  target_compile_definitions(knapsack PRIVATE KNAPSACK_METAL_SUPPORT)
  message(STATUS "Building knapsack library with Metal support")
endif()

# Public headers for consumers + private includes for our sources
# Note: Include PROJ_ROOT (../) not (../third_party) so that
# source files can use #include "third_party/..." paths
target_include_directories(knapsack
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${PROJ_ROOT}
)

# Only include Metal headers if Metal support is enabled
if(USE_METAL)
  target_include_directories(knapsack PRIVATE ${KERNELS_METAL_DIR})
endif()

# Only include CUDA headers if CUDA support is enabled
if(BUILD_CUDA)
  target_include_directories(knapsack PRIVATE ${KERNELS_CUDA_DIR})
endif()

# ARC and Apple frameworks for Metal
if(APPLE AND USE_METAL)
  target_compile_options(knapsack PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(knapsack PRIVATE "-framework Metal" "-framework Foundation")
endif()

# Install library and public header
install(TARGETS knapsack
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/knapsack_c.h DESTINATION include)

# Simple CLI to run V2 from a JSON config
add_executable(knapsack_v2_cli ${CMAKE_CURRENT_LIST_DIR}/src/cli_v2.cpp)
target_include_directories(knapsack_v2_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(knapsack_v2_cli PRIVATE knapsack)
if(APPLE AND USE_METAL)
  target_compile_options(knapsack_v2_cli PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(knapsack_v2_cli PRIVATE "-framework Foundation" "-framework Metal")
endif()
install(TARGETS knapsack_v2_cli RUNTIME DESTINATION bin)
