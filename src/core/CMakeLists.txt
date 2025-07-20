add_library(vvm_core STATIC
    Grid.cpp
    Field.cpp
    HaloExchanger.cpp
)

target_include_directories(vvm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${MPI_INCLUDE_PATH}
)

target_link_libraries(vvm_core
    PRIVATE
        Kokkos::kokkos
        ${MPI_LIBRARIES}
        ${OpenMP_LIBRARIES}
        nvToolsExt
)
