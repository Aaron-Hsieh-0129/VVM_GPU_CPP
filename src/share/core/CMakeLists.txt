# Generate eamxx_config.h and eamxx_config.f
include (EkatUtils)
EkatConfigFile(${CMAKE_CURRENT_SOURCE_DIR}/eamxx_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/eamxx_config.h
               F90_FILE ${CMAKE_CURRENT_BINARY_DIR}/eamxx_config.f)

# Generate eamxx_version.h
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eamxx_version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/eamxx_version.h)

add_library(eamxx_core
  eamxx_config.cpp
  eamxx_session.cpp
)

target_include_directories(eamxx_core PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR} # for config/version headers
  ${SCREAM_SRC_DIR}           # for all eamxx headers
)

# Attach custom f90 flags, so they are picked up by any EAMxx target
target_compile_options(eamxx_core PUBLIC
  $<$<COMPILE_LANGUAGE:Fortran>:${SCREAM_Fortran_FLAGS}>
)

# We have some issues with RDC and repeated libraries in the link line.
# It's a known issue, and nvcc_wrapper has a flag for handling this.
if (Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
  if(CMAKE_VERSION VERSION_LESS "3.13.0")
    target_link_libraries(eamxx_core PUBLIC "--remove-duplicate-link-files")
  else()
    target_link_options(eamxx_core PUBLIC "--remove-duplicate-link-files")
  endif()
endif()

# Only link what this library need. Other eamxx libs will link other tpls
target_link_libraries(eamxx_core PUBLIC ekat::Core ekat::KokkosUtils)
