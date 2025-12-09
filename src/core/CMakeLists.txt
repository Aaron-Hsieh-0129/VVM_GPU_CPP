set(VVM_CORE_SOURCES
    State.cpp
    Grid.cpp
    Initializer.cpp
    BoundaryConditionManager.cpp
    Parameters.cpp
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

target_include_directories(vvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${NCCL_INCLUDE_DIR}
)


target_link_libraries(vvm_core
    PUBLIC
        vvm_utils
        MPI::MPI_CXX
        ${NVCPUMATH_LIBRARY}
        ${PNETCDF_LIBRARY}
        ${NCCL_LIBRARY}
    PRIVATE
        Kokkos::kokkos
        ${OpenMP_LIBRARIES}
	${NVTOOLSEXT_LIBRARY}
        PnetCDF::pnetcdf
)
