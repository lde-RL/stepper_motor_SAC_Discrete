cmake_minimum_required(VERSION 3.16)
project(pipe_inspection_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# Header-only core library (graph_map.hpp + cpp_planner.hpp)
add_library(pipe_core INTERFACE)
target_include_directories(pipe_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# CLI simulator
add_executable(pipe_simulator src/simulator_main.cpp)
target_link_libraries(pipe_simulator PRIVATE pipe_core)

# Unit tests (plain asserts, no framework needed)
add_executable(pipe_tests src/tests_main.cpp)
target_link_libraries(pipe_tests PRIVATE pipe_core)

enable_testing()
add_test(NAME pipe_tests COMMAND pipe_tests)
