#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "../src/utils/ConfigurationManager.hpp"
#include "../src/core/Grid.hpp"
#include "../src/core/Field.hpp"
#include "../src/core/State.hpp"
#include "../src/core/Parameters.hpp"

// Test Fixture for Field and State tests
class FieldStateTest : public ::testing::Test {
protected:
    // Common setup can go here if needed
};

// --- Field Class Tests ---

TEST_F(FieldStateTest, FieldConstructor) {
    VVM::Utils::ConfigurationManager config("../../rundata/input_configs/default_config.json");
    VVM::Core::Grid grid(config);

    // --- Test 2D Field ---
    VVM::Core::Field<2> field_2d("TestField2D", {
        grid.get_local_total_points_y(), 
        grid.get_local_total_points_x()
    });
    ASSERT_EQ(field_2d.get_device_data().extent(0), grid.get_local_total_points_y());
    ASSERT_EQ(field_2d.get_device_data().extent(1), grid.get_local_total_points_x());
    ASSERT_EQ(field_2d.get_name(), "TestField2D");

    // --- Test 3D Field ---
    VVM::Core::Field<3> field_3d("TestField3D", {
        grid.get_local_total_points_z(), 
        grid.get_local_total_points_y(), 
        grid.get_local_total_points_x()
    });
    ASSERT_EQ(field_3d.get_device_data().extent(0), grid.get_local_total_points_z());
    ASSERT_EQ(field_3d.get_device_data().extent(1), grid.get_local_total_points_y());
    ASSERT_EQ(field_3d.get_device_data().extent(2), grid.get_local_total_points_x());
    ASSERT_EQ(field_3d.get_name(), "TestField3D");

    // --- Test 4D Field ---
    const int w_dim = 5;
    VVM::Core::Field<4> field_4d("TestField4D", {
        w_dim,
        grid.get_local_total_points_z(), 
        grid.get_local_total_points_y(), 
        grid.get_local_total_points_x()
    });
    ASSERT_EQ(field_4d.get_device_data().extent(0), w_dim);
    ASSERT_EQ(field_4d.get_device_data().extent(1), grid.get_local_total_points_z());
    ASSERT_EQ(field_4d.get_name(), "TestField4D");
}

TEST_F(FieldStateTest, InitializeToZero) {
    VVM::Core::Field<3> test_field("ZeroField", {3, 4, 5});

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
    VVM::Core::Field<3> test_field("HostDataTestField", {3, 4, 5});
    
    Kokkos::deep_copy(test_field.get_mutable_device_data(), 123.45);
    auto host_data = test_field.get_host_data();

    ASSERT_EQ(host_data.extent(0), 3);
    ASSERT_EQ(host_data.extent(1), 4);
    ASSERT_EQ(host_data.extent(2), 5);
    
    EXPECT_DOUBLE_EQ(host_data(0, 0, 0), 123.45);
    EXPECT_DOUBLE_EQ(host_data(2, 3, 4), 123.45);
}

// --- State Class Tests ---

TEST_F(FieldStateTest, StateOperations) {
    VVM::Utils::ConfigurationManager config("../../rundata/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::ModelParameters params(grid, config);
    VVM::Core::State state(config, params);

    // Add fields of different dimensions
    state.add_field<3>("temp_3d", {10, 20, 30});
    state.add_field<2>("pressure_2d", {20, 30});
    state.add_field<4>("chem_4d", {5, 10, 20, 30});

    // Check successful retrieval
    ASSERT_NO_THROW({
        auto& f3d = state.get_field<3>("temp_3d");
        EXPECT_EQ(f3d.get_device_data().extent(0), 10);
    });
    ASSERT_NO_THROW({
        auto& f2d = state.get_field<2>("pressure_2d");
        EXPECT_EQ(f2d.get_device_data().extent(0), 20);
    });
    ASSERT_NO_THROW({
        auto& f4d = state.get_field<4>("chem_4d");
        EXPECT_EQ(f4d.get_device_data().extent(0), 5);
    });

    // Check retrieval with incorrect dimension (should throw)
    ASSERT_THROW(state.get_field<2>("temp_3d"), std::runtime_error);
    ASSERT_THROW(state.get_field<4>("pressure_2d"), std::runtime_error);
    ASSERT_THROW(state.get_field<3>("chem_4d"), std::runtime_error);

    // Check retrieval of non-existent field (should throw)
    ASSERT_THROW(state.get_field<3>("non_existent_field"), std::runtime_error);
}

// --- Main function for the test executable ---

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    // Finalize MPI and Kokkos
    Kokkos::finalize();
    MPI_Finalize();
    
    return result;
}
