cmake_minimum_required(VERSION 3.16)

project(VVM LANGUAGES CXX C Fortran)

cmake_policy(SET CMP0135 NEW)

# ------------------------------
# Compiler-specific settings
# ------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    message(STATUS "Applying NVHPC dependency flag workaround.")
    # Disable automatic dependency generation flags that NVHPC might set
    set(CMAKE_CXX_DEPENDS "")
    set(CMAKE_C_DEPENDS "")
    set(CMAKE_DEPFILE_FLAGS_CXX "")
    set(CMAKE_DEPFILE_FLAGS_C "")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -mp" CACHE STRING "CUDA OpenMP flags" FORCE)
endif()

# ------------------------------
# Standard C++ settings
# ------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------
# HDF5
# ------------------------------
set(HDF5_DIR "/work/aaron900129/hdf5_install/lib/cmake/hdf5")

# ------------------------------
# Kokkos
# ------------------------------
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend for Kokkos" FORCE)
set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Enable Kokkos RDC")
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend for Kokkos" FORCE)
# set(EAMXX_ENABLE_GPU ON CACHE BOOL "Enable P3 GPU" FORCE)

# NVHPC + Kokkos CUDA arch
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(Kokkos_CUDA_OPTIONS "enable_lambda;enable_ldg;cuda_arch=70" CACHE STRING "Kokkos CUDA options" FORCE)
endif()

set(SCREAM_DOUBLE_PRECISIION ON CACHE BOOL "Enable scream double precision" FORCE)
option(BUILD_TESTS "Build the unit and integration tests" OFF)
message(STATUS "Build tests: ${BUILD_TESTS}")

# ------------------------------
# Fortran flags
# ------------------------------
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -fdefault-real-8 -fdefault-double-8 -Wno-line-truncation -Wno-argument-mismatch")
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mallocatable=03 -Mdclchk -Mfreeform -Mextend")
endif()

# ------------------------------
# MPI
# ------------------------------
set(MPI_Fortran_FOUND TRUE)
set(MPI_Fortran_WORKS TRUE)
find_package(MPI REQUIRED COMPONENTS CXX C Fortran)
if (MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
else()
    message(FATAL_ERROR "MPI not found.")
endif()

# ------------------------------
# Kokkos
# ------------------------------
find_package(Kokkos CONFIG)
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

# ------------------------------
# NetCDF
# ------------------------------
find_path(NETCDF_C_INCLUDE_DIR NAMES netcdf.h HINTS "/work/aaron900129/netcdf-c_install/include" NO_DEFAULT_PATH)
find_library(NETCDF_C_LIBRARY NAMES netcdf HINTS "/work/aaron900129/netcdf-c_install/lib" NO_DEFAULT_PATH)
find_path(NETCDF_FORTRAN_INCLUDE_DIR NAMES netcdf.mod HINTS "/work/aaron900129/netcdf-fortran_install/include" NO_DEFAULT_PATH)
find_library(NETCDF_FORTRAN_LIBRARY NAMES netcdff HINTS "/work/aaron900129/netcdf-fortran_install/lib" NO_DEFAULT_PATH)

if(NOT NETCDF_C_INCLUDE_DIR OR NOT NETCDF_C_LIBRARY OR NOT NETCDF_FORTRAN_INCLUDE_DIR OR NOT NETCDF_FORTRAN_LIBRARY)
    message(FATAL_ERROR "Can't find NetCDF")
endif()

# Create imported targets
if (NOT TARGET NetCDF::NetCDF_C)
    add_library(NetCDF::NetCDF_C UNKNOWN IMPORTED)
    set_property(TARGET NetCDF::NetCDF_C PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_C_INCLUDE_DIR}"
        IMPORTED_LOCATION "${NETCDF_C_LIBRARY}")
endif()

if (NOT TARGET NetCDF::NetCDF_Fortran)
    add_library(NetCDF::NetCDF_Fortran UNKNOWN IMPORTED)
    set_property(TARGET NetCDF::NetCDF_Fortran PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_FORTRAN_INCLUDE_DIR}"
        IMPORTED_LOCATION "${NETCDF_FORTRAN_LIBRARY}"
        INTERFACE_LINK_LIBRARIES NetCDF::NetCDF_C)
endif()

# ------------------------------
# EKAT
# ------------------------------
# find_package(Ekat QUIET)

# if (Ekat_FOUND)
#     message(STATUS "Found pre-installed Ekat at: ${Ekat_DIR}")
# else()
# set(EKAT_ENABLE_TESTS OFF CACHE BOOL "")
# set(EKAT_ENABLE_EXAMPLES OFF CACHE BOOL "")
# set(EKAT_ENABLE_NETCDF ON CACHE BOOL "")
# set(EKAT_ENABLE_PNETCDF ON CACHE BOOL "")
# set(EKAT_ENABLE_YAML ON CACHE BOOL "")
# set(EKAT_TPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals CACHE PATH "")
# message(STATUS "Pre-installed Ekat not found. Building from source at 'extern/ekat'.")
# set(EKAT_ENABLE_CUDA ON CACHE BOOL "Pass VVM CUDA setting to Ekat" FORCE)
# if (DEFINED Kokkos_DIR)
#     set(Kokkos_DIR ${Kokkos_DIR} CACHE PATH "Point Ekat to VVM's Kokkos" FORCE)
# endif()
#
# if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/externals/ekat)
#     add_subdirectory(externals/ekat)
# else()
#     message(FATAL_ERROR "Could not find pre-installed Ekat, AND "
#                         "the fallback directory 'extern/ekat' does not exist.")
# endif()
# endif()

message(STATUS "[VVM] Forcing Ekat to build from source.")
message(STATUS "[VVM] VVM (parent) found Kokkos at: ${Kokkos_DIR}")

set(EKAT_ENABLE_TESTS OFF CACHE BOOL "")
set(EKAT_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(EKAT_ENABLE_NETCDF ON CACHE BOOL "")
set(EKAT_ENABLE_PNETCDF ON CACHE BOOL "")
set(EKAT_ENABLE_YAML ON CACHE BOOL "")
set(EKAT_TPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals CACHE PATH "")

set(EKAT_ENABLE_CUDA ${Kokkos_ENABLE_CUDA} CACHE BOOL "Pass VVM CUDA setting to Ekat" FORCE)

set(Kokkos_DIR ${Kokkos_DIR} CACHE PATH "FORCE Ekat to use VVM's Kokkos" FORCE)

message(STATUS "[VVM] Forcing Ekat's Kokkos_DIR to: ${Kokkos_DIR}")
message(STATUS "[VVM] Forcing Ekat's EKAT_ENABLE_CUDA to: ${EKAT_ENABLE_CUDA}")

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/externals/ekat)
    add_subdirectory(externals/ekat)
else()
    message(FATAL_ERROR "Could not find 'externals/ekat' directory.")
endif()

message(STATUS "Aggressively cleaning global OpenMP flags set by subdirectories (e.g., EKAT)...")
# 1. Use REGEX to remove '-mp' as a standalone word (handles spaces correctly)
string(REGEX REPLACE " -mp( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE " -mp( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REGEX REPLACE " -mp( |$)" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
# 2. Handle if '-mp' is the *first* flag
string(REGEX REPLACE "^-mp( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "^-mp( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REGEX REPLACE "^-mp( |$)" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
# 3. Force-reset the OpenMP cache variables that ekat might have set
set(OpenMP_CXX_FLAGS "" CACHE STRING "Reset OpenMP CXX flags" FORCE)
set(OpenMP_C_FLAGS "" CACHE STRING "Reset OpenMP C flags" FORCE)
set(OpenMP_Fortran_FLAGS "" CACHE STRING "Reset OpenMP Fortran flags" FORCE)

function(CLEAN_FLAG_VAR var_name flag_to_remove)
    string(REPLACE " " ";" flag_list "${${var_name}}")
    list(REMOVE_ITEM flag_list "${flag_to_remove}")
    string(REPLACE ";" " " flag_string "${flag_list}")
    set(${var_name} "${flag_string}" PARENT_SCOPE)
endfunction()

clean_flag_var(CMAKE_CXX_FLAGS "-Wext-lambda-captures-this")


# ------------------------------
# ADIOS2
# ------------------------------
find_package(ADIOS2 REQUIRED CXX MPI)
if(TARGET adios2::cxx11_mpi)
    target_link_libraries(adios2::cxx11_mpi INTERFACE MPI::MPI_CXX)
endif()

# ------------------------------
# HDF5
# ------------------------------
find_path(HDF5_INCLUDE_DIR hdf5.h HINTS /work/aaron900129/hdf5_install/include NO_DEFAULT_PATH)
find_library(HDF5_LIBRARY NAMES hdf5 HINTS /work/aaron900129/hdf5_install/lib NO_DEFAULT_PATH)

if(HDF5_INCLUDE_DIR AND HDF5_LIBRARY)
    add_library(HDF5::HDF5 UNKNOWN IMPORTED)
    set_target_properties(HDF5::HDF5 PROPERTIES
                          IMPORTED_LOCATION "${HDF5_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")
    find_library(Z_LIBRARY z)
    find_library(DL_LIBRARY dl)
    find_library(M_LIBRARY m)
    target_link_libraries(HDF5::HDF5 INTERFACE ${Z_LIBRARY} ${DL_LIBRARY} ${M_LIBRARY} MPI::MPI_CXX)
endif()

# ------------------------------
# Project sources
# ------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)

set(PNETCDF_INSTALL_DIR "/work/aaron900129/pnetcdf_install" CACHE PATH "Path to PnetCDF installation")
find_path(PNETCDF_INCLUDE_DIR pnetcdf.h
          HINTS ${PNETCDF_INSTALL_DIR}/include
          NO_DEFAULT_PATH)
find_library(PNETCDF_LIBRARY NAMES pnetcdf
             HINTS ${PNETCDF_INSTALL_DIR}/lib
             NO_DEFAULT_PATH)

if (PNETCDF_INCLUDE_DIR AND PNETCDF_LIBRARY)
    add_library(PnetCDF::pnetcdf UNKNOWN IMPORTED GLOBAL)
    set_target_properties(PnetCDF::pnetcdf PROPERTIES
                          IMPORTED_LOCATION "${PNETCDF_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${PNETCDF_INCLUDE_DIR}")
    find_library(M_LIBRARY m)
    target_link_libraries(PnetCDF::pnetcdf INTERFACE MPI::MPI_CXX ${M_LIBRARY})
    set(PNETCDF_FOUND TRUE)
    message(STATUS "PnetCDF found manually at: ${PNETCDF_LIBRARY}")
else()
    message(FATAL_ERROR "PnetCDF library or includes not found at the specified path: ${PNETCDF_INSTALL_DIR}")
endif()

# Find netcdf-c
set(NETCDF_C_INSTALL_DIR "/work/aaron900129/netcdf-c_install" CACHE PATH "Path to NetCDF-C installation")
find_path(NETCDF_C_INCLUDE_DIR netcdf.h
          HINTS ${NETCDF_C_INSTALL_DIR}/include
          NO_DEFAULT_PATH)
find_library(NETCDF_C_LIBRARY NAMES netcdf
             HINTS ${NETCDF_C_INSTALL_DIR}/lib
             NO_DEFAULT_PATH)

if (NETCDF_C_INCLUDE_DIR AND NETCDF_C_LIBRARY)
    add_library(NetCDF::netcdf UNKNOWN IMPORTED GLOBAL)
    set_target_properties(NetCDF::netcdf PROPERTIES
                          IMPORTED_LOCATION "${NETCDF_C_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_C_INCLUDE_DIR}")
    set(NETCDF_C_FOUND TRUE)
    message(STATUS "NetCDF-C (for fallback) found manually at: ${NETCDF_C_LIBRARY}")
else()
    message(FATAL_ERROR "NetCDF-C library or includes not found at the specified path: ${NETCDF_C_INSTALL_DIR}")
endif()


# Build tests if requested
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()

message(STATUS "=== Compiler flags summary ===")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
message(STATUS "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
message(STATUS "Fortran Flags: ${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "==============================")
