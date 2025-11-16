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
option(BUILD_ONNX "Build RL support with ONNX Runtime inference (requires onnxruntime)" OFF)

if(BUILD_ONNX)
  message(STATUS "ONNX Runtime support enabled for RL")
  include(FetchContent)
  # Use prebuilt ONNX Runtime for macOS/Linux
  # For production: download release binaries or use find_package if installed
  # Here we'll use a minimal approach: expect user to install onnxruntime via brew/apt or provide path
  # Alternative: FetchContent can download prebuilt releases
  find_package(onnxruntime QUIET)
  if(NOT onnxruntime_FOUND)
    message(STATUS "ONNX Runtime not found via find_package; attempting manual locate")
    # Fallback: check common brew path on macOS
    if(APPLE)
      set(ONNXRUNTIME_INCLUDE_DIR "/opt/homebrew/include" CACHE PATH "ONNX Runtime include dir")
      set(ONNXRUNTIME_LIB_DIR "/opt/homebrew/lib" CACHE PATH "ONNX Runtime lib dir")
    else()
      set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/include" CACHE PATH "ONNX Runtime include dir")
      set(ONNXRUNTIME_LIB_DIR "/usr/local/lib" CACHE PATH "ONNX Runtime lib dir")
    endif()
    # Check for ONNX Runtime headers (correct path for brew installation)
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/onnxruntime_cxx_api.h")
      message(STATUS "Found ONNX Runtime headers at ${ONNXRUNTIME_INCLUDE_DIR}")
      set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
      find_library(ONNXRUNTIME_LIBRARIES NAMES onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)
      if(ONNXRUNTIME_LIBRARIES)
        message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")
      else()
        message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR}. Install via: brew install onnxruntime (macOS) or build from source.")
      endif()
    else()
      message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/onnxruntime_cxx_api.h. Install ONNX Runtime or disable BUILD_ONNX.")
    endif()
  else()
    message(STATUS "ONNX Runtime found via find_package")
    # When using find_package, the imported target onnxruntime::onnxruntime provides include dirs and libs
    # However, the target sets INTERFACE_INCLUDE_DIRECTORIES to .../include/onnxruntime
    # We need .../include to use #include <onnxruntime/...> syntax
    get_target_property(ONNX_IMPORT_LOCATION onnxruntime::onnxruntime LOCATION)
    get_filename_component(ONNX_LIB_DIR "${ONNX_IMPORT_LOCATION}" DIRECTORY)
    get_filename_component(ONNX_PREFIX "${ONNX_LIB_DIR}/.." ABSOLUTE)
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNX_PREFIX}/include")
    set(ONNXRUNTIME_LIBRARIES onnxruntime::onnxruntime)
    message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIRS}")
  endif()
endif()

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
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(APPLE AND USE_METAL)
  target_compile_definitions(knapsack_solver PRIVATE KNAPSACK_METAL_SUPPORT)
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

# ---- Synthetic select-mode JSON generator ----
add_executable(gen_synth_select tools/gen_synth_select.cpp)
target_compile_features(gen_synth_select PRIVATE cxx_std_17)

# ---- Converter: samples/small_mkp.txt -> v2 select JSON ----
add_executable(convert_small_mkp_to_v2 tools/convert_small_mkp_to_v2.cpp)
target_compile_features(convert_small_mkp_to_v2 PRIVATE cxx_std_17)

# ---- V2 config sanity tool (preview) ----
set(V2_SANITY_SOURCES tools/v2_config_sanity.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_SANITY_SOURCES src/v2/Config.mm)
else()
  list(APPEND V2_SANITY_SOURCES src/v2/Config_json.cpp)
endif()
add_executable(v2_config_sanity ${V2_SANITY_SOURCES})
target_include_directories(v2_config_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(v2_config_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_config_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
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
target_include_directories(v2_soa_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_soa_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
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
target_include_directories(v2_eval_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_eval_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
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
target_include_directories(v2_metal_parity_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_metal_parity_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(v2_metal_parity_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_metal_parity_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- Benchmark: CPU vs Metal performance tool ----
set(BENCHMARK_SOURCES tools/benchmark_cpu_vs_metal.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND BENCHMARK_SOURCES src/v2/Config.mm src/v2/EvalMetal.mm kernels/metal/metal_api.mm)
else()
  list(APPEND BENCHMARK_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(benchmark_cpu_vs_metal ${BENCHMARK_SOURCES})
target_include_directories(benchmark_cpu_vs_metal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(benchmark_cpu_vs_metal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(benchmark_cpu_vs_metal PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(benchmark_cpu_vs_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(benchmark_cpu_vs_metal PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- Benchmark: Batch CPU vs Metal (where GPU wins!) ----
set(BATCH_BENCHMARK_SOURCES tools/benchmark_batch_cpu_vs_metal.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND BATCH_BENCHMARK_SOURCES src/v2/Config.mm src/v2/EvalMetal.mm kernels/metal/metal_api.mm)
else()
  list(APPEND BATCH_BENCHMARK_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(benchmark_batch_cpu_vs_metal ${BATCH_BENCHMARK_SOURCES})
target_include_directories(benchmark_batch_cpu_vs_metal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(benchmark_batch_cpu_vs_metal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(benchmark_batch_cpu_vs_metal PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(benchmark_batch_cpu_vs_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(benchmark_batch_cpu_vs_metal PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- Benchmark: Batch CPU (assign mode) ----
set(BATCH_ASSIGN_BENCH_SOURCES tools/benchmark_batch_cpu_assign.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND BATCH_ASSIGN_BENCH_SOURCES src/v2/Config.mm)
else()
  list(APPEND BATCH_ASSIGN_BENCH_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(benchmark_batch_cpu_assign ${BATCH_ASSIGN_BENCH_SOURCES})
target_include_directories(benchmark_batch_cpu_assign PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(benchmark_batch_cpu_assign PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(benchmark_batch_cpu_assign PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(benchmark_batch_cpu_assign PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(benchmark_batch_cpu_assign PRIVATE "-framework Foundation")
endif()

# ---- V2 Assign-mode parity sanity ----
set(V2_ASSIGN_SANITY_SOURCES tools/v2_assign_sanity.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_ASSIGN_SANITY_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
else()
  list(APPEND V2_ASSIGN_SANITY_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(v2_assign_sanity ${V2_ASSIGN_SANITY_SOURCES})
target_include_directories(v2_assign_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_assign_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(v2_assign_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_assign_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- V2 Beam (select-mode) sanity ----
set(V2_BEAM_SANITY_SOURCES tools/v2_beam_sanity.cpp src/v2/BeamSearch.cpp src/v2/Data.cpp src/v2/EvalCPU.cpp src/v2/Preprocess.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_BEAM_SANITY_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
else()
  list(APPEND V2_BEAM_SANITY_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
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
  list(APPEND V2_MULTI_CONS_BEAM_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(v2_multi_constraint_beam_sanity ${V2_MULTI_CONS_BEAM_SOURCES})
target_include_directories(v2_multi_constraint_beam_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(v2_multi_constraint_beam_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_multi_constraint_beam_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(v2_multi_constraint_beam_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_multi_constraint_beam_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()
add_executable(v2_multi_constraint_parity ${V2_MULTI_CONS_SANITY_SOURCES})
target_include_directories(v2_multi_constraint_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(v2_multi_constraint_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_multi_constraint_parity PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(v2_multi_constraint_parity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_multi_constraint_parity PRIVATE "-framework Foundation" "-framework Metal")
endif()
add_executable(v2_beam_sanity ${V2_BEAM_SANITY_SOURCES})
target_include_directories(v2_beam_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(v2_beam_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_beam_sanity PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(v2_beam_sanity PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_beam_sanity PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- V2 Scout mode demo ----
set(V2_SCOUT_DEMO_SOURCES tools/v2_scout_demo.cpp src/v2/BeamSearch.cpp src/v2/Data.cpp src/v2/EvalCPU.cpp src/v2/Preprocess.cpp)
if(APPLE AND USE_METAL)
  list(APPEND V2_SCOUT_DEMO_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
else()
  list(APPEND V2_SCOUT_DEMO_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(v2_scout_demo ${V2_SCOUT_DEMO_SOURCES})
target_include_directories(v2_scout_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(v2_scout_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(v2_scout_demo PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(v2_scout_demo PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(v2_scout_demo PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- NBA Beam example (select-mode) ----
set(NBA_BEAM_EXAMPLE_SOURCES tools/nba_beam_example.cpp src/v2/EvalCPU.cpp src/v2/Data.cpp)
if(APPLE AND USE_METAL)
  list(APPEND NBA_BEAM_EXAMPLE_SOURCES src/v2/Config.mm src/v2/EvalMetal.mm kernels/metal/metal_api.mm)
else()
  list(APPEND NBA_BEAM_EXAMPLE_SOURCES src/v2/Config_json.cpp src/v2/Config_validate.cpp)
endif()
add_executable(nba_beam_example ${NBA_BEAM_EXAMPLE_SOURCES})
target_include_directories(nba_beam_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(nba_beam_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(APPLE AND USE_METAL)
  target_compile_definitions(nba_beam_example PRIVATE KNAPSACK_METAL_SUPPORT)
  target_compile_options(nba_beam_example PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(nba_beam_example PRIVATE "-framework Foundation" "-framework Metal")
endif()

# ---- RL support libraries (static + shared for bindings) ----
add_library(rl_support STATIC rl/rl_api.cpp)
target_include_directories(rl_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rl)
if(BUILD_ONNX)
  target_compile_definitions(rl_support PRIVATE RL_ONNX_ENABLED)
  target_include_directories(rl_support PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
  target_link_libraries(rl_support PRIVATE ${ONNXRUNTIME_LIBRARIES})
endif()

add_library(rl_support_shared SHARED rl/rl_api.cpp)
target_include_directories(rl_support_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rl)
if(BUILD_ONNX)
  target_compile_definitions(rl_support_shared PRIVATE RL_ONNX_ENABLED)
  target_include_directories(rl_support_shared PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
  target_link_libraries(rl_support_shared PRIVATE ${ONNXRUNTIME_LIBRARIES})
endif()
set_target_properties(rl_support_shared PROPERTIES OUTPUT_NAME "rl_support")

# Link RL support into example (static preferred)
target_link_libraries(nba_beam_example PRIVATE rl_support)

# ---- Python bindings ----
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)

if(BUILD_PYTHON_BINDINGS)
  # Add pybind11 subdirectory
  add_subdirectory(third_party/pybind11)
  
  # Collect sources for the Python module (same as BeamSearch needs)
  set(PY_BINDING_SOURCES 
    bindings/python/knapsack_py.cpp
    src/v2/BeamSearch.cpp
    src/v2/Data.cpp
    src/v2/EvalCPU.cpp
    src/v2/Preprocess.cpp
  )
  
  if(APPLE AND USE_METAL)
    list(APPEND PY_BINDING_SOURCES src/v2/Config.mm kernels/metal/metal_api.mm)
  else()
    list(APPEND PY_BINDING_SOURCES src/v2/Config_json.cpp)
  endif()
  
  # Create the Python extension module (named knapsack_py to avoid conflict)
  pybind11_add_module(knapsack_py ${PY_BINDING_SOURCES})
  
  target_include_directories(knapsack_py PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal 
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  )
  
  if(APPLE AND USE_METAL)
    target_compile_definitions(knapsack_py PRIVATE KNAPSACK_METAL_SUPPORT)
    target_compile_options(knapsack_py PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(knapsack_py PRIVATE "-framework Foundation" "-framework Metal")
  endif()
  
  # Set the output name to 'knapsack' (Python will import as 'knapsack')
  set_target_properties(knapsack_py PROPERTIES OUTPUT_NAME "knapsack")
  
  # Install the Python module to a location pip can find
  install(TARGETS knapsack_py LIBRARY DESTINATION .)
endif()

# ---- Testing ----
option(BUILD_TESTS "Build unit tests with Catch2" ON)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/v2)
  message(STATUS "Unit tests enabled - run 'make test' or 'ctest' to execute")
endif()
