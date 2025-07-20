cmake_minimum_required(VERSION 3.16)
project(VVM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++ for portability
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend for Kokkos" FORCE)
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend for Kokkos" FORCE)

# Find MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI CXX include path: ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI CXX libraries: ${MPI_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "MPI not found. Please ensure MPI is installed and discoverable by CMake.")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found and configured for CXX.")
else()
    message(FATAL_ERROR "OpenMP not found. Please ensure OpenMP is installed and discoverable by CMake.")
endif()

# Find Kokkos
find_package(Kokkos CONFIG) # Try to find Kokkos externally
if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
    message(STATUS "Kokkos not found externally. Fetching via FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.tar.gz
    )
    FetchContent_MakeAvailable(Kokkos)
endif()



add_subdirectory(src)

