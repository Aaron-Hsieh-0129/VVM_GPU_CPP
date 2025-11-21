set(SCREAM_PACK_SIZE 1 CACHE STRING "The number of scalars in a scream::pack::Pack")
set(SCREAM_SMALL_PACK_SIZE 1 CACHE STRING "The number of scalars in a scream::pack::SmallPack")
set(SCREAM_POSSIBLY_NO_PACK_SIZE 1 CACHE STRING "The number of scalars in a possibly-no-pack")

set(SCREAM_NUM_VERTICAL_LEV 33 CACHE STRING "Default number of vertical levels")


add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(dynamics)
add_subdirectory(share)
add_subdirectory(physics)


add_executable(vvm 
    main.cpp
)

# Make the output binary under build directory
set_target_properties(vvm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(vvm
    PUBLIC
        MPI::MPI_CXX
        Kokkos::kokkos

        vvm_utils
        vvm_core
        vvm_io
        vvm_dynamics
        scream_share
        eamxx_physics
        # p3
        # ekat
)
