add_library(eamxx_utils
  eamxx_utils.cpp
)

set_target_properties(eamxx_utils PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
target_include_directories (eamxx_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/modules)

target_link_libraries (eamxx_utils PUBLIC
  eamxx_core
  ekat::Logging
  ekat::Pack)
