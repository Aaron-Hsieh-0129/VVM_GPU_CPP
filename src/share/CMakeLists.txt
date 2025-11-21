add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(physics)
# add_subdirectory(property_checks)

add_library(scream_share INTERFACE)
target_link_libraries(scream_share INTERFACE
    eamxx_core
    eamxx_utils
    eamxx_physics_share
    # eamxx_property_checks
)
