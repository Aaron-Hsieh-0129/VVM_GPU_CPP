find_package(GTest REQUIRED)

set(VVM_TEST_LIBS
    PRIVATE
        GTest::gtest_main
        vvm_core
        vvm_utils
        Kokkos::kokkos
        ${MPI_LIBRARIES}
        ${OpenMP_LIBRARIES}
        nvToolsExt
    PUBLIC
        MPI::MPI_CXX
        ${NVCPUMATH_LIBRARY}
)

# Halo Exchanger Tests
add_executable(run_halo_tests
    halo_exchanger_test.cpp
)
target_link_libraries(run_halo_tests ${VVM_TEST_LIBS})

# Field and State Tests
add_executable(run_field_state_tests
    field_state_test.cpp
)
target_link_libraries(run_field_state_tests ${VVM_TEST_LIBS})

# Halo Exchanger Tests
add_test(
    NAME HaloExchange.Periodic.2x2
    COMMAND mpirun -np 4 $<TARGET_FILE:run_halo_tests> --gtest_filter=HaloExchangerTest.Periodic2x2Grid
)

# Field State Tests
add_test(
    NAME FieldAndState.UnitTests
    COMMAND mpirun -np 1 $<TARGET_FILE:run_field_state_tests>
)