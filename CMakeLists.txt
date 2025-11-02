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
option(BUILD_LIB "Build and install knapsack static library and V2 CLI" ON)

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
  # Use Objective-C++ Config loader on Apple; exclude picojson-based loader
  list(FILTER SOURCES EXCLUDE REGEX ".*/src/v2/Config_json.cpp$")
  list(APPEND SOURCES src/v2/Config.mm)
  list(APPEND SOURCES kernels/metal/metal_api.mm)
endif()

if(BUILD_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES "kernels/*.cu")
  list(APPEND SOURCES ${CUDA_SOURCES})
endif()
# ---- V2 CUDA multi-constraint parity sanity (Linux/Jetson) ----
if(BUILD_CUDA)
  set(V2_CUDA_PARITY_SOURCES tools/v2_multi_constraint_parity_cuda.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp kernels/cuda/cuda_api.cu)
  if(APPLE AND USE_METAL)
    list(APPEND V2_CUDA_PARITY_SOURCES src/v2/Config.mm)
  else()
    list(APPEND V2_CUDA_PARITY_SOURCES src/v2/Config_json.cpp)
  endif()
  add_executable(v2_multi_constraint_parity_cuda ${V2_CUDA_PARITY_SOURCES})
  target_include_directories(v2_multi_constraint_parity_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  if(APPLE AND USE_METAL)
    target_compile_options(v2_multi_constraint_parity_cuda PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(v2_multi_constraint_parity_cuda PRIVATE "-framework Foundation")
  endif()
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

# Library and V2 CLI (installs lib, headers, and knapsack_v2_cli)
if(BUILD_LIB)
  add_subdirectory(knapsack-library)
endif()

# ---- V2 config sanity tool (preview) ----
set(V2_SANITY_SOURCES tools/v2_config_sanity.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_SANITY_SOURCES src/v2/Config.mm)
else()
  list(APPEND V2_SANITY_SOURCES src/v2/Config_json.cpp)
endif()
add_executable(v2_config_sanity ${V2_SANITY_SOURCES})
target_include_directories(v2_config_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_config_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_config_sanity PRIVATE "-framework Foundation")
endif()

# ---- V2 SoA sanity tool ----
set(V2_SOA_SANITY_SOURCES tools/v2_soa_sanity.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_SOA_SANITY_SOURCES src/v2/Config.mm)
else()
  list(APPEND V2_SOA_SANITY_SOURCES src/v2/Config_json.cpp)
endif()
add_executable(v2_soa_sanity ${V2_SOA_SANITY_SOURCES})
target_include_directories(v2_soa_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_soa_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_soa_sanity PRIVATE "-framework Foundation")
endif()

# ---- V2 Eval sanity tool (CPU) ----
set(V2_EVAL_SANITY_SOURCES tools/v2_eval_sanity.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_EVAL_SANITY_SOURCES src/v2/Config.mm)
else()
  list(APPEND V2_EVAL_SANITY_SOURCES src/v2/Config_json.cpp)
endif()
add_executable(v2_eval_sanity ${V2_EVAL_SANITY_SOURCES})
target_include_directories(v2_eval_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_eval_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_eval_sanity PRIVATE "-framework Foundation")
endif()

# ---- V2 Metal parity sanity tool ----
set(V2_METAL_PARITY_SOURCES tools/v2_metal_parity_sanity.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_METAL_PARITY_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
endif()
add_executable(v2_metal_parity_sanity ${V2_METAL_PARITY_SOURCES})
target_include_directories(v2_metal_parity_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_metal_parity_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_metal_parity_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- V2 Assign-mode parity sanity ----
set(V2_ASSIGN_SANITY_SOURCES tools/v2_assign_sanity.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_ASSIGN_SANITY_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
endif()
add_executable(v2_assign_sanity ${V2_ASSIGN_SANITY_SOURCES})
target_include_directories(v2_assign_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_assign_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_assign_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- V2 Beam (select-mode) sanity ----
set(V2_BEAM_SANITY_SOURCES tools/v2_beam_sanity.cpp src/v2/BeamSearch.cpp src/v2/Data.cpp src/v2/EvalCPU.cpp src/v2/Preprocess.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_BEAM_SANITY_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
else()
  list(APPEND V2_BEAM_SANITY_SOURCES src/v2/Config_json.cpp)
endif()

# ---- V2 Multi-constraint Metal parity sanity ----
set(V2_MULTI_CONS_SANITY_SOURCES tools/v2_multi_constraint_parity.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_MULTI_CONS_SANITY_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
else()
  list(APPEND V2_MULTI_CONS_SANITY_SOURCES src/v2/Config_json.cpp)
endif()

# ---- V2 Multi-constraint Beam sanity ----
set(V2_MULTI_CONS_BEAM_SOURCES tools/v2_multi_constraint_beam_sanity.cpp src/v2/BeamSearch.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp src/v2/Preprocess.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_MULTI_CONS_BEAM_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
else()
  list(APPEND V2_MULTI_CONS_BEAM_SOURCES src/v2/Config_json.cpp)
endif()
add_executable(v2_multi_constraint_beam_sanity ${V2_MULTI_CONS_BEAM_SOURCES})
target_include_directories(v2_multi_constraint_beam_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_multi_constraint_beam_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_multi_constraint_beam_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()
add_executable(v2_multi_constraint_parity ${V2_MULTI_CONS_SANITY_SOURCES})
target_include_directories(v2_multi_constraint_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_multi_constraint_parity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_multi_constraint_parity PRIVATE "-framework Foundation" "-framework Metal")
endif()
add_executable(v2_beam_sanity ${V2_BEAM_SANITY_SOURCES})
target_include_directories(v2_beam_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_options(v2_beam_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_beam_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()

