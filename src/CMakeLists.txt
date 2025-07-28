add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(io)

add_executable(vvm 
    main.cpp
)

# Make the output binary under build directory
set_target_properties(vvm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_compile_options(vvm PRIVATE "--extended-lambda")

target_link_libraries(vvm
    PUBLIC
        MPI::MPI_CXX
        Kokkos::kokkos
        OpenMP::OpenMP_CXX

        vvm_utils
        vvm_core
        vvm_io
)
