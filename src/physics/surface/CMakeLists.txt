add_library(vvm_surface
    SurfaceProcess.cpp
)

target_link_libraries(vvm_surface
    PUBLIC
    vvm_core
    vvm_utils
    Kokkos::kokkos
)

target_include_directories(vvm_surface
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NCCL_INCLUDE_DIR}
)
