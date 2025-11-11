set(SCREAM_PACK_SIZE 8 CACHE STRING "The number of scalars in a scream::pack::Pack")
set(SCREAM_SMALL_PACK_SIZE 8 CACHE STRING "The number of scalars in a scream::pack::SmallPack")
set(SCREAM_POSSIBLY_NO_PACK_SIZE 1 CACHE STRING "The number of scalars in a possibly-no-pack")

set(SCREAM_NUM_VERTICAL_LEV 72 CACHE STRING "Default number of vertical levels")


include_directories(BEFORE
  "${CMAKE_SOURCE_DIR}/externals/ekat/src"
  "${CMAKE_BINARY_DIR}/externals/ekat/src"
)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/scream_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/scream_config.h"
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/scream_config.f" "")
include_directories(BEFORE
    "${CMAKE_CURRENT_BINARY_DIR}"
)


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
        vvm_physics
        scream_share
        p3
)
