file(GLOB_RECURSE VVM_DYNAMICS_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE VVM_DYNAMICS_HEADERS CONFIGURE_DEPENDS "*.hpp")

add_library(vvm_dynamics STATIC
    ${VVM_DYNAMICS_SOURCES}
    ${VVM_DYNAMICS_HEADERS}
)

target_include_directories(vvm_dynamics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(vvm_dynamics
    PUBLIC
        vvm_core
        vvm_utils
        Kokkos::kokkos
)
