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
add_test(NAME beam_search COMMAND test_beam_search)
add_test(NAME eval_cpu COMMAND test_eval_cpu)

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

set_tests_properties(beam_search PROPERTIES
    LABELS "v2;solver"
    TIMEOUT 120
)

set_tests_properties(eval_cpu PROPERTIES
    LABELS "v2;eval"
    TIMEOUT 60
)

# Custom target to build all tests
if(APPLE AND USE_METAL)
    add_custom_target(build_tests
        DEPENDS test_config_validate test_beam_search test_eval_cpu test_eval_metal
    )
else()
    add_custom_target(build_tests
        DEPENDS test_config_validate test_beam_search test_eval_cpu
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
message(STATUS "  - test_beam_search")
message(STATUS "  - test_eval_cpu")
if(APPLE)
    message(STATUS "  - test_eval_metal (Metal GPU)")
endif()
message(STATUS "Run 'make test' or 'ctest' to execute tests")
