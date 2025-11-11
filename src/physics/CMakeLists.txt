add_subdirectory(p3)

add_library(vvm_physics STATIC
    p3/VVM_p3_process_interface.cpp
)

target_link_libraries(vvm_physics PUBLIC
    vvm_core
    scream_share
    p3 
    ekat
    Kokkos::kokkos
)

target_include_directories(vvm_physics PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/share
)

