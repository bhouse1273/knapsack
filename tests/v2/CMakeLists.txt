# CMakeLists.txt for v2 tests
cmake_minimum_required(VERSION 3.18)

# Catch2 test framework (amalgamated single-file version)
set(CATCH2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/catch2")
add_library(catch2 STATIC ${CATCH2_DIR}/catch_amalgamated.cpp)
target_include_directories(catch2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_features(catch2 PUBLIC cxx_std_17)

# Common test dependencies
set(TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Test executable for Config validation
add_executable(test_config_validate test_config_validate.cpp)
target_include_directories(test_config_validate PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_config_validate PRIVATE catch2 knapsack)
target_compile_features(test_config_validate PRIVATE cxx_std_17)

# Test executable for data ingestion helpers
add_executable(test_data_ingest test_data_ingest.cpp)
target_include_directories(test_data_ingest PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_data_ingest PRIVATE catch2 knapsack)
if(KNAPSACK_ARROW_ENABLED)
    target_link_libraries(test_data_ingest PRIVATE ${KNAPSACK_ARROW_LIBS})
endif()
target_compile_features(test_data_ingest PRIVATE cxx_std_17)

# Test executable for Beam Search
add_executable(test_beam_search test_beam_search.cpp)
target_include_directories(test_beam_search PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_beam_search PRIVATE catch2 knapsack)
target_compile_features(test_beam_search PRIVATE cxx_std_17)

# Test executable for CPU Evaluation
add_executable(test_eval_cpu test_eval_cpu.cpp)
target_include_directories(test_eval_cpu PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_eval_cpu PRIVATE catch2 knapsack)
target_compile_features(test_eval_cpu PRIVATE cxx_std_17)

# Test executable for RL API
add_executable(test_rl_api test_rl_api.cpp)
target_include_directories(test_rl_api PRIVATE ${TEST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../rl)
target_link_libraries(test_rl_api PRIVATE catch2 rl_support)
target_compile_features(test_rl_api PRIVATE cxx_std_17)
if(BUILD_ONNX)
  target_compile_definitions(test_rl_api PRIVATE RL_ONNX_ENABLED)
endif()

# Copy ONNX model files to build directory for RL tests
if(BUILD_ONNX)
    set(ONNX_MODEL_FILES
        tiny_linear_8.onnx
        tiny_linear_12.onnx
        dummy_model.onnx
    )
    foreach(model_file ${ONNX_MODEL_FILES})
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${model_file}
            ${CMAKE_CURRENT_BINARY_DIR}/${model_file}
            COPYONLY
        )
    endforeach()
    message(STATUS "  - Copied ONNX model files to build directory")
endif()

# Test executable for Metal Evaluation (Apple + Metal only)
if(APPLE AND USE_METAL)
    set(TEST_METAL_SOURCES test_eval_metal.cpp)
    add_executable(test_eval_metal ${TEST_METAL_SOURCES})
    target_include_directories(test_eval_metal PRIVATE 
        ${TEST_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../kernels/metal
    )
    target_link_libraries(test_eval_metal PRIVATE catch2 knapsack)
    target_compile_features(test_eval_metal PRIVATE cxx_std_17)
    target_compile_options(test_eval_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(test_eval_metal PRIVATE "-framework Foundation" "-framework Metal")
    
    message(STATUS "  - test_eval_metal (Metal GPU tests)")
endif()

# Enable testing
enable_testing()

# Add tests to CTest
add_test(NAME config_validate COMMAND test_config_validate)
add_test(NAME data_ingest COMMAND test_data_ingest)
add_test(NAME beam_search COMMAND test_beam_search)
add_test(NAME eval_cpu COMMAND test_eval_cpu)
add_test(NAME rl_api COMMAND test_rl_api)

if(APPLE AND USE_METAL)
    add_test(NAME eval_metal COMMAND test_eval_metal)
    set_tests_properties(eval_metal PROPERTIES
        LABELS "v2;eval;metal;gpu"
        TIMEOUT 120
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# Optional: Set test properties
set_tests_properties(config_validate PROPERTIES
    LABELS "v2;validation"
    TIMEOUT 60
)

set_tests_properties(data_ingest PROPERTIES
    LABELS "v2;data"
    TIMEOUT 60
)
if(KNAPSACK_TESTDATA_ROOT)
    set_tests_properties(data_ingest PROPERTIES
        ENVIRONMENT "KNAPSACK_TESTDATA_ROOT=${KNAPSACK_TESTDATA_ROOT}"
    )
endif()

set_tests_properties(beam_search PROPERTIES
    LABELS "v2;solver"
    TIMEOUT 120
)

set_tests_properties(eval_cpu PROPERTIES
    LABELS "v2;eval"
    TIMEOUT 60
)

set_tests_properties(rl_api PROPERTIES
    LABELS "v2;rl"
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Custom target to build all tests
if(APPLE AND USE_METAL)
    add_custom_target(build_tests
        DEPENDS test_config_validate test_data_ingest test_beam_search test_eval_cpu test_rl_api test_eval_metal
    )
else()
    add_custom_target(build_tests
        DEPENDS test_config_validate test_data_ingest test_beam_search test_eval_cpu test_rl_api
    )
endif()

# Custom target to run all tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS build_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Configured v2 tests:")
message(STATUS "  - test_config_validate")
message(STATUS "  - test_data_ingest")
message(STATUS "  - test_beam_search")
message(STATUS "  - test_eval_cpu")
message(STATUS "  - test_rl_api")
if(APPLE)
    message(STATUS "  - test_eval_metal (Metal GPU)")
endif()
message(STATUS "Run 'make test' or 'ctest' to execute tests")
