add_library(vvm_utils STATIC
    ConfigurationManager.cpp
)

target_include_directories(vvm_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/lib/json
)