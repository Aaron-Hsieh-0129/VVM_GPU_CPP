cmake_minimum_required(VERSION 3.16)
project(VVM LANGUAGES CXX C)

cmake_policy(SET CMP0135 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++ for portability

set(HDF5_DIR "/work/aaron900129/hdf5_install/lib/cmake/hdf5")

set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend for Kokkos" FORCE)
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend for Kokkos" FORCE)
option(BUILD_TESTS "Build the unit and integration tests" OFF)

message(STATUS "Build tests: ${BUILD_TESTS}")


# Find MPI
find_package(MPI REQUIRED COMPONENTS CXX C)
if (MPI_FOUND)

    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI CXX include path: ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI CXX libraries: ${MPI_CXX_LIBRARIES}")
    
    if(TARGET MPI::MPI_C AND NOT TARGET MPI::MPI_C)
        add_library(MPI::MPI_C ALIAS MPI::MPI_C)
    endif()
else()
    message(FATAL_ERROR "MPI not found. Please ensure MPI is installed and discoverable by CMake.")
endif()

# Find OpenMP
find_package(OpenMP)
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

set(PNETCDF_INSTALL_DIR "/work/aaron900129/Pnetcdf_install")
set(PNETCDF_INCLUDE_DIR "${PNETCDF_INSTALL_DIR}/include")
set(PNETCDF_LIBRARIES "-L${PNETCDF_INSTALL_DIR}/lib -lpnetcdf -lm")


# Find ADIOS2
find_package(ADIOS2 REQUIRED CXX MPI)
if (ADIOS2_FOUND)
    message(STATUS "ADIOS2 found.")
else()
    message(FATAL_ERROR "ADIOS2 not found.")
endif()
if(TARGET adios2::cxx11_mpi)
    target_link_libraries(adios2::cxx11_mpi INTERFACE MPI::MPI_CXX)
    message(STATUS "Manually linking MPI::MPI_CXX to adios2::cxx11_mpi target.")
endif()

find_package(HDF5 REQUIRED)
if (HDF5_FOUND)
    message(STATUS "HDF5 found at: ${HDF5_DIR}")
else()
    message(FATAL_ERROR "HDF5 not found at the specified path.")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)


# Build tests if requested
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    # This makes the GTest::gtest_main target available
    FetchContent_MakeAvailable(googletest)

    # Add the tests subdirectory, which will now find GTest
    add_subdirectory(tests)
endif()
