# Note that the user needs to specify the nvhpc mpi path in their environment variables or do it here 
# e.g. /work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/ompi/bin/mpicc
set(CMAKE_C_COMPILER mpicc CACHE STRING "C compiler for NVHPC")
set(CMAKE_CXX_COMPILER mpic++ CACHE STRING "C++ compiler for NVHPC")
set(CMAKE_Fortran_COMPILER mpifort CACHE STRING "Fortran compiler for NVHPC")
message(STATUS "Using NVHPC Toolchain: C=${CMAKE_C_COMPILER}, CXX=${CMAKE_CXX_COMPILER}")
