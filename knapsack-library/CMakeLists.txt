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

# Include V2 core sources from the main project to avoid code duplication
set(PROJ_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
list(APPEND LIB_SOURCES
  "${PROJ_ROOT}/src/v2/Data.cpp"
  "${PROJ_ROOT}/src/v2/EvalCPU.cpp"
  "${PROJ_ROOT}/src/v2/BeamSearch.cpp"
  "${PROJ_ROOT}/src/v2/Preprocess.cpp"
)

# Add Metal bridge (Objective-C++) on Apple
if(APPLE AND USE_METAL)
  set(KERNELS_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/metal")
  list(APPEND LIB_SOURCES "${KERNELS_METAL_DIR}/metal_api.mm")
  list(APPEND LIB_SOURCES "${PROJ_ROOT}/src/v2/Config.mm")
else()
  list(APPEND LIB_SOURCES "${PROJ_ROOT}/src/v2/Config_json.cpp")
endif()

add_library(knapsack STATIC ${LIB_SOURCES})

# Public headers for consumers + private includes for our sources
target_include_directories(knapsack
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
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

# Simple CLI to run V2 from a JSON config
add_executable(knapsack_v2_cli ${CMAKE_CURRENT_LIST_DIR}/src/cli_v2.cpp)
target_include_directories(knapsack_v2_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(knapsack_v2_cli PRIVATE knapsack)
if(APPLE AND USE_METAL)
  target_compile_options(knapsack_v2_cli PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(knapsack_v2_cli PRIVATE "-framework Foundation" "-framework Metal")
endif()
install(TARGETS knapsack_v2_cli RUNTIME DESTINATION bin)
