#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>

// 包含您需要測試的類別
#include "../src/utils/ConfigurationManager.hpp"
#include "../src/core/Grid.hpp"
#include "../src/core/Field.hpp"
#include "../src/core/HaloExchanger.hpp"

class HaloExchangerTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    // Execute at the end
    void TearDown() override {
        // Clean resources if needed
    }

    // 輔助函式，用於初始化 Field 資料
    void initialize_field(VVM::Core::Field& field, const VVM::Core::Grid& grid) {
        auto field_data_host = Kokkos::create_mirror_view(field.get_mutable_device_data());
        
        const int nz = grid.get_local_physical_points_z();
        const int ny = grid.get_local_physical_points_y();
        const int nx = grid.get_local_physical_points_x();
        const int h = grid.get_halo_cells();
        
        // 僅填滿物理區域
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    // 計算這一點的全域座標
                    int global_y = grid.get_local_physical_start_y() + j;
                    int global_x = grid.get_local_physical_start_x() + i;
                    
                    // 使用獨一無二的公式賦值: rank * 1M + global_y * 1k + global_x
                    // 這樣我們就可以從 halo 點的值反推出它應該來自哪個 rank 的哪個座標
                    double value = rank * 100 + global_y * 10 + global_x;
                    field_data_host(k + h, j + h, i + h) = value;
                }
            }
        }
        
        Kokkos::deep_copy(field.get_mutable_device_data(), field_data_host);
        Kokkos::fence();
    }
    
    // 輔助函式，用於計算期望值
    double get_expected_value(int neighbor_rank, int global_y, int global_x) {
        return neighbor_rank * 100 + global_y * 10 + global_x;
    }

    int rank;
    int size;
};

// Test case: global grid size 8x8, 2x2 MPI ranks
TEST_F(HaloExchangerTest, Periodic2x2Grid) {
    if (size != 4) {
        GTEST_SKIP() << "Skipping test, requires exactly 4 MPI ranks.";
    }

    // 1. Construct Grid and Field
    VVM::Utils::ConfigurationManager config("../../data/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::Field field(grid, "TestField_8x8_2x2");
    
    // Check grid dimensions
    ASSERT_EQ(grid.get_local_physical_points_x(), 4);
    ASSERT_EQ(grid.get_local_physical_points_y(), 4);

    initialize_field(field, grid);

    // // 2. Halo Exchange
    VVM::Core::HaloExchanger exchanger(grid);
    exchanger.exchange_halos(field);

    // // 3. Check halo points values 
    auto field_mirror = field.get_host_data();
    Kokkos::fence();

    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    
    const int total_nx = grid.get_global_points_x();
    const int total_ny = grid.get_global_points_y();

    const int start_gx = grid.get_local_physical_start_x();
    const int start_gy = grid.get_local_physical_start_y();

    MPI_Comm cart_comm = grid.get_cart_comm();
    int neighbor_rank_source, neighbor_rank_dest;

    // --- Check WEST (left) Halo ---
    MPI_Cart_shift(cart_comm, 1, -1, &neighbor_rank_dest, &neighbor_rank_source);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < h; ++i) {
                int global_y = start_gy + j;
                int global_x = (start_gx - h + i + total_nx) % total_nx;
                double expected = get_expected_value(neighbor_rank_source, global_y, global_x);
                double actual = field_mirror(k + h, j + h, i);
                EXPECT_DOUBLE_EQ(actual, expected) << "Mismatch at WEST halo for rank " << rank << " from rank " << neighbor_rank_source;
            }
        }
    }

    // --- Check EAST (right) Halo ---
    MPI_Cart_shift(cart_comm, 1, 1, &neighbor_rank_dest, &neighbor_rank_source);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < h; ++i) {
                int global_y = start_gy + j;
                int global_x = (start_gx + nx + i) % total_nx;
                double expected = get_expected_value(neighbor_rank_source, global_y, global_x);
                double actual = field_mirror(k + h, j + h, i + nx + h);
                EXPECT_DOUBLE_EQ(actual, expected) << "Mismatch at EAST halo for rank " << rank << " from rank " << neighbor_rank_source;
            }
        }
    }

    // --- Check SOUTH (bottom) Halo ---
    MPI_Cart_shift(cart_comm, 0, -1, &neighbor_rank_dest, &neighbor_rank_source);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < nx; ++i) {
                int global_y = (start_gy - h + j + total_ny) % total_ny;
                int global_x = start_gx + i;
                double expected = get_expected_value(neighbor_rank_source, global_y, global_x);
                double actual = field_mirror(k + h, j, i + h);
                EXPECT_DOUBLE_EQ(actual, expected) << "Mismatch at SOUTH halo for rank " << rank << " from rank " << neighbor_rank_source;
            }
        }
    }

    // --- Check NORTH (up) Halo ---
    MPI_Cart_shift(cart_comm, 0, 1, &neighbor_rank_dest, &neighbor_rank_source);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < nx; ++i) {
                int global_y = (start_gy + ny + j) % total_ny;
                int global_x = start_gx + i;
                double expected = get_expected_value(neighbor_rank_source, global_y, global_x);
                double actual = field_mirror(k + h, j + ny + h, i + h);
                EXPECT_DOUBLE_EQ(actual, expected) << "Mismatch at NORTH halo for rank " << rank << " from rank " << neighbor_rank_source;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    omp_set_num_threads(16);

    Kokkos::initialize(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    
    int result = RUN_ALL_TESTS();

    int final_result;
    MPI_Allreduce(&result, &final_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return final_result;
}