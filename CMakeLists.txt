cmake_minimum_required(VERSION 3.10)
project(heat_diffusion C)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(test_cpu test_cpu.c)
target_include_directories(
    test_cpu
    PRIVATE
    ${MPI_C_INCLUDE_PATH}
)
target_link_libraries(
    test_cpu
    ${MPI_C_LIBRARIES}
    OpenMP::OpenMP_C
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -foffload=nvptx-none")
add_executable(test_gpu test_gpu.c)
target_include_directories(
    test_gpu
    PRIVATE
    ${MPI_C_INCLUDE_PATH}
)
target_link_libraries(
    test_gpu
    ${MPI_C_LIBRARIES}
    OpenMP::OpenMP_C
)
