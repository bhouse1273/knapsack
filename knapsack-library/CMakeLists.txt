project(knapsack_library VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(knapsack STATIC src/knapsack_c.cpp)

set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack")