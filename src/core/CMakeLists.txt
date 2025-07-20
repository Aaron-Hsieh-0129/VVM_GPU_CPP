add_library(vvm_core STATIC
    State.cpp
    Grid.cpp
)

target_include_directories(vvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${MPI_INCLUDE_PATH}
)

find_library(NVCPUMATH_LIBRARY nvcpumath
    HINTS /work/aaron900129/nvhpc_23_9/Linux_x86_64/23.9/compilers/lib
)

if(NOT NVCPUMATH_LIBRARY)
    message(FATAL_ERROR "Could not find NVIDIA CPU Math Library (libnvcpumath.so)")
else()
    message(STATUS "Found NVIDIA CPU Math Library: ${NVCPUMATH_LIBRARY}")
endif()

target_link_libraries(vvm_core
    PUBLIC
        MPI::MPI_CXX
        ${NVCPUMATH_LIBRARY}
    PRIVATE
        Kokkos::kokkos
        ${MPI_LIBRARIES}
        ${OpenMP_LIBRARIES}
        nvToolsExt
)
