cmake_minimum_required(VERSION 3.10)
project(heat_diffusion C)

find_package(MPI REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main jacobi.c)
target_include_directories(
    main
    PRIVATE
    ${MPI_C_INCLUDE_PATH}
    ${SDL2_INCLUDE_DIRS}
)
target_link_libraries(
    main
    ${MPI_C_LIBRARIES}
    OpenMP::OpenMP_C
    SDL2::SDL2
)

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