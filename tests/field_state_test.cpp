#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "../src/utils/ConfigurationManager.hpp"
#include "../src/core/Grid.hpp"
#include "../src/core/Field.hpp"
#include "../src/core/State.hpp"

// Test Fixture for Field and State tests
class FieldStateTest : public ::testing::Test {
protected:

};

// --- Field Class Tests ---

TEST_F(FieldStateTest, Constructor) {
    VVM::Utils::ConfigurationManager config("../../data/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::Field test_field(grid, "TestField");

    // Check dimensions
    const int expected_nx = grid.get_local_physical_points_x() + 2 * grid.get_halo_cells();
    const int expected_ny = grid.get_local_physical_points_y() + 2 * grid.get_halo_cells();
    const int expected_nz = grid.get_local_physical_points_z() + 2 * grid.get_halo_cells();

    ASSERT_EQ(test_field.get_device_data().extent(0), expected_nz);
    ASSERT_EQ(test_field.get_device_data().extent(1), expected_ny);
    ASSERT_EQ(test_field.get_device_data().extent(2), expected_nx);

    ASSERT_EQ(test_field.get_name(), "TestField");
}

TEST_F(FieldStateTest, InitializeToZero) {
    VVM::Utils::ConfigurationManager config("../../data/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::Field test_field(grid, "ZeroField");

    Kokkos::deep_copy(test_field.get_mutable_device_data(), 42.0);
    test_field.initialize_to_zero();
    auto host_data = test_field.get_host_data();

    for (int k = 0; k < host_data.extent(0); ++k) {
        for (int j = 0; j < host_data.extent(1); ++j) {
            for (int i = 0; i < host_data.extent(2); ++i) {
                EXPECT_DOUBLE_EQ(host_data(k, j, i), 0.0);
            }
        }
    }
}

TEST_F(FieldStateTest, GetHostData) {
    VVM::Utils::ConfigurationManager config("../../data/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::Field test_field(grid, "HostDataTestField");
    
    Kokkos::deep_copy(test_field.get_mutable_device_data(), 123.45);
    auto host_data = test_field.get_host_data();

    ASSERT_EQ(host_data.extent(0), test_field.get_device_data().extent(0));
    ASSERT_EQ(host_data.extent(1), test_field.get_device_data().extent(1));
    ASSERT_EQ(host_data.extent(2), test_field.get_device_data().extent(2));
    
    EXPECT_DOUBLE_EQ(host_data(0, 0, 0), 123.45);
    EXPECT_DOUBLE_EQ(host_data(host_data.extent(0)-1, host_data.extent(1)-1, host_data.extent(2)-1), 123.45);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    int final_result;
    MPI_Allreduce(&result, &final_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


    Kokkos::finalize();
    MPI_Finalize();
    return final_result;
}