project(knapsack_library VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Public headers for the C API
include_directories(
	${CMAKE_CURRENT_LIST_DIR}/include
	${CMAKE_CURRENT_LIST_DIR}/..
	${CMAKE_CURRENT_LIST_DIR}/../include
)

# Core C API implementation plus minimal CPU helpers
add_library(knapsack STATIC
	src/knapsack_solve.cpp
	../src/InputModule.cpp
	../src/RouteUtils.cpp
	src/metal_bridge.mm
)

set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack")

# On Apple, link Metal frameworks and (optionally) the prebuilt Metal static lib
if(APPLE)
	target_include_directories(knapsack PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../kernels/metal)
	target_link_libraries(knapsack PRIVATE "-framework Metal" "-framework Foundation")
	# Compile the Metal C API directly into the library to avoid a prebuilt static lib dependency.
	target_sources(knapsack PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../kernels/metal/metal_api.mm)
endif()