cmake_minimum_required(VERSION 3.12...3.18)
project(threefry_avx)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -march=native -mavx2 -mfma -mtune=native")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(threefry_avx threefry_avx.cpp)
