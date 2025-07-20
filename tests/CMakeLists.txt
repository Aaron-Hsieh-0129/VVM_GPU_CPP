find_package(GTest REQUIRED)

add_executable(run_tests
    halo_exchanger_test.cpp
)

target_link_libraries(run_tests PRIVATE
    GTest::gtest_main 
    PUBLIC
        MPI::MPI_CXX
        ${NVCPUMATH_LIBRARY}
    PRIVATE
        Kokkos::kokkos
        ${MPI_LIBRARIES}
        ${OpenMP_LIBRARIES}
        nvToolsExt
    vvm_core
    vvm_utils
)

add_test(
    NAME HaloExchange.Periodic.2x2
    COMMAND mpirun -np 4 $<TARGET_FILE:run_tests> --gtest_filter=HaloExchangerTest.Periodic2x2Grid
)