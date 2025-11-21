add_library (eamxx_physics_share
  eamxx_trcmix.cpp
)

# Add ETI source files if not on CUDA/HIP
# if (NOT EAMXX_ENABLE_GPU)
#   target_sources(eamxx_physics_share PRIVATE
#     physics_saturation.cpp)
# endif()

set_target_properties(eamxx_physics_share PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
)
target_include_directories(eamxx_physics_share PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/modules
)
target_link_libraries(eamxx_physics_share PUBLIC
  eamxx_core
  eamxx_utils
  ekat::Pack)
