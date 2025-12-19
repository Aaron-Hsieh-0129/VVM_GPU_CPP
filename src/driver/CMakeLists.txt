# src/driver/CMakeLists.txt

add_library(vvm_driver
    Model.cpp
)

target_link_libraries(vvm_driver
    PUBLIC
    vvm_core
    vvm_dynamics
    eamxx_physics
    vvm_io
    vvm_utils
)

target_include_directories(vvm_driver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
