cmake_minimum_required(VERSION 3.18)
project(knapsack_library LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_METAL "Enable Metal backend on Apple" ON)

if(APPLE AND USE_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# Sources in this library
file(GLOB LIB_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
set(LIB_SOURCES ${LIB_CPP})

# Add Metal bridge (Objective-C++) on Apple
if(APPLE AND USE_METAL)
  set(KERNELS_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/metal")
  list(APPEND LIB_SOURCES "${KERNELS_METAL_DIR}/metal_api.mm")
endif()

add_library(knapsack STATIC ${LIB_SOURCES})

# Public headers for consumers + private includes for our sources
target_include_directories(knapsack
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    $<$<BOOL:${APPLE}>:${KERNELS_METAL_DIR}>
)

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
