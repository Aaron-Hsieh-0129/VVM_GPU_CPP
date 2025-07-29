set(VVM_CORE_SOURCES
    State.cpp
    Grid.cpp
    Initializer.cpp
    BoundaryConditionManager.cpp
)

set(VVM_CORE_HEADERS
    State.hpp
    Grid.hpp
    HaloExchanger.hpp
    Initializer.hpp
    BoundaryConditionManager.hpp
    Parameters.hpp
    Field.hpp
)

add_library(vvm_core STATIC
    ${VVM_CORE_SOURCES}
    ${VVM_CORE_HEADERS}
)

find_library(NVTOOLSEXT_LIBRARY nvToolsExt
    HINTS /work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/cuda/lib64
)
find_library(NVCPUMATH_LIBRARY nvcpumath
    HINTS /work/aaron900129/nvhpc_23_9/Linux_x86_64/23.9/compilers/lib
)

if(NOT NVCPUMATH_LIBRARY)
    message(FATAL_ERROR "Could not find NVIDIA CPU Math Library (libnvcpumath.so)")
else()
    message(STATUS "Found NVIDIA CPU Math Library: ${NVCPUMATH_LIBRARY}")
endif()

if(NOT NVTOOLSEXT_LIBRARY)
    message(FATAL_ERROR "Could not find NVIDIA Tools Extension Library (libnvToolsExt.so)")
else()
    message(STATUS "Found NVIDIA Tools Extension Library: ${NVTOOLSEXT_LIBRARY}")
endif()


target_include_directories(vvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)


target_link_libraries(vvm_core
    PUBLIC
        vvm_utils
        MPI::MPI_CXX
	    ${NVCPUMATH_LIBRARY}
        ${PNETCDF_LIBRARY}
    PRIVATE
        Kokkos::kokkos
        ${MPI_LIBRARIES}
        ${OpenMP_LIBRARIES}
	    ${NVTOOLSEXT_LIBRARY}
)
