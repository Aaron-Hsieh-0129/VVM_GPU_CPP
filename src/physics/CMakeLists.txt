add_subdirectory(p3)
add_subdirectory(rrtmgp)

add_library(vvm_physics STATIC
    p3/VVM_p3_process_interface.cpp
    rrtmgp/VVM_rrtmgp_process_interface.cpp
)

target_link_libraries(vvm_physics PUBLIC
    vvm_core
    scream_share
    p3 
    ekat
    Kokkos::kokkos
    scream_rrtmgp
)
target_compile_definitions(vvm_physics PUBLIC 
    -DYAKL_INLINE=KOKKOS_INLINE_FUNCTION
    -DYAKL_LAMBDA=KOKKOS_LAMBDA
)

target_include_directories(vvm_physics PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/share
)

