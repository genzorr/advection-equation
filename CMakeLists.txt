cmake_minimum_required(VERSION 3.20)
project(lab1advection)

set(CMAKE_CXX_STANDARD 17)
include_directories({MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

# Linear program
#add_executable(lab1advection src/advection.cpp)

# Parallel program
add_executable(lab1advection src/advection_parallel.cpp)