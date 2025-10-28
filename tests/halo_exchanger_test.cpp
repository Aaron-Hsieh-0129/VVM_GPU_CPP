#include <gtest/gtest.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <numeric>

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

    // Helper to initialize a 4D field
    void initialize_field_4d(VVM::Core::Field<4>& field, const VVM::Core::Grid& grid) {
        auto field_data_host = Kokkos::create_mirror_view(field.get_mutable_device_data());
        const int nw = field.get_device_data().extent(0); // 4th dimension
        const int nz = grid.get_local_physical_points_z();
        const int ny = grid.get_local_physical_points_y();
        const int nx = grid.get_local_physical_points_x();
        const int h = grid.get_halo_cells();

        for (int w = 0; w < nw; ++w) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        int global_y = grid.get_local_physical_start_y() + j;
                        int global_x = grid.get_local_physical_start_x() + i;
                        // Unique value for each point
                        double value = rank * 10000 + w * 1000 + k * 100 + global_y * 10 + global_x;
                        field_data_host(w, k + h, j + h, i + h) = value;
                    }
                }
            }
        }
        Kokkos::deep_copy(field.get_mutable_device_data(), field_data_host);
        Kokkos::fence();
    }

    // Helper to initialize a 3D field
    void initialize_field_3d(VVM::Core::Field<3>& field, const VVM::Core::Grid& grid) {
        auto field_data_host = Kokkos::create_mirror_view(field.get_mutable_device_data());
        const int nz = grid.get_local_physical_points_z();
        const int ny = grid.get_local_physical_points_y();
        const int nx = grid.get_local_physical_points_x();
        const int h = grid.get_halo_cells();

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int global_y = grid.get_local_physical_start_y() + j;
                    int global_x = grid.get_local_physical_start_x() + i;
                    double value = rank * 1000 + k * 100 + global_y * 10 + global_x;
                    field_data_host(k + h, j + h, i + h) = value;
                }
            }
        }
        Kokkos::deep_copy(field.get_mutable_device_data(), field_data_host);
        Kokkos::fence();
    }

    // Helper to initialize a 2D field
    void initialize_field_2d(VVM::Core::Field<2>& field, const VVM::Core::Grid& grid) {
        auto field_data_host = Kokkos::create_mirror_view(field.get_mutable_device_data());
        const int ny = grid.get_local_physical_points_y();
        const int nx = grid.get_local_physical_points_x();
        const int h = grid.get_halo_cells();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int global_y = grid.get_local_physical_start_y() + j;
                int global_x = grid.get_local_physical_start_x() + i;
                double value = rank * 100 + global_y * 10 + global_x;
                field_data_host(j + h, i + h) = value;
            }
        }
        Kokkos::deep_copy(field.get_mutable_device_data(), field_data_host);
        Kokkos::fence();
    }

    int rank;
    int size;
};

// --- 4D Field ---
TEST_F(HaloExchangerTest, Periodic4DField) {
    if (size != 4) {
        GTEST_SKIP() << "Skipping 4D test, requires exactly 4 MPI ranks for a 2x2 grid.";
    }

    VVM::Utils::ConfigurationManager config("../../rundata/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    const int num_species = 2;
    VVM::Core::Field<4> field("TestField4D", {
        num_species,
        grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()
    });
    
    initialize_field_4d(field, grid);

    VVM::Core::HaloExchanger exchanger(grid);
    exchanger.exchange_halos(field);

    auto field_mirror = field.get_host_data();
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    
    const int total_nx = grid.get_global_points_x();
    const int total_ny = grid.get_global_points_y();
    const int start_gx = grid.get_local_physical_start_x();
    const int start_gy = grid.get_local_physical_start_y();

    MPI_Comm cart_comm = grid.get_cart_comm();
    int neighbor_rank_source, neighbor_rank_dest;

    // --- Check WEST (left) Halo ---
    MPI_Cart_shift(cart_comm, 1, -1, &neighbor_rank_dest, &neighbor_rank_source);
    for(int w = 0; w < num_species; ++w) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < h; ++i) {
                    int global_y = start_gy + j;
                    int global_x = (start_gx - h + i + total_nx) % total_nx;
                    double expected = neighbor_rank_source * 10000 + w * 1000 + k * 100 + global_y * 10 + global_x;
                    double actual = field_mirror(w, k + h, j + h, i);
                    EXPECT_DOUBLE_EQ(actual, expected) << "4D MISMATCH @ WEST HALO";
                }
            }
        }
    }
}


// --- 測試案例：3D Field ---
TEST_F(HaloExchangerTest, Periodic3DField) {
    if (size != 4) {
        GTEST_SKIP() << "Skipping 3D test, requires exactly 4 MPI ranks for a 2x2 grid.";
    }

    VVM::Utils::ConfigurationManager config("../../rundata/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::Field<3> field("TestField3D", {
        grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()
    });
    
    initialize_field_3d(field, grid);

    VVM::Core::HaloExchanger exchanger(grid);
    exchanger.exchange_halos(field);

    auto field_mirror = field.get_host_data();
    const int h = grid.get_halo_cells();
    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    
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
                double expected = neighbor_rank_source * 1000 + k * 100 + global_y * 10 + global_x;
                double actual = field_mirror(k + h, j + h, i);
                EXPECT_DOUBLE_EQ(actual, expected) << "3D MISMATCH @ WEST HALO";
            }
        }
    }
}

// --- 2D Field ---
TEST_F(HaloExchangerTest, Periodic2DField) {
    if (size != 4) {
        GTEST_SKIP() << "Skipping 2D test, requires exactly 4 MPI ranks for a 2x2 grid.";
    }

    VVM::Utils::ConfigurationManager config("../../rundata/input_configs/default_config.json");
    VVM::Core::Grid grid(config);
    VVM::Core::Field<2> field("TestField2D", {
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()
    });
    
    initialize_field_2d(field, grid);

    VVM::Core::HaloExchanger exchanger(grid);
    exchanger.exchange_halos(field);

    auto field_mirror = field.get_host_data();
    const int h = grid.get_halo_cells();
    const int ny = grid.get_local_physical_points_y();
    
    const int total_nx = grid.get_global_points_x();
    const int start_gx = grid.get_local_physical_start_x();
    const int start_gy = grid.get_local_physical_start_y();

    MPI_Comm cart_comm = grid.get_cart_comm();
    int neighbor_rank_source, neighbor_rank_dest;

    // --- Check WEST (left) Halo ---
    MPI_Cart_shift(cart_comm, 1, -1, &neighbor_rank_dest, &neighbor_rank_source);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < h; ++i) {
            int global_y = start_gy + j;
            int global_x = (start_gx - h + i + total_nx) % total_nx;
            double expected = neighbor_rank_source * 100 + global_y * 10 + global_x;
            double actual = field_mirror(j + h, i);
            EXPECT_DOUBLE_EQ(actual, expected) << "2D MISMATCH @ WEST HALO";
        }
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    omp_set_num_threads(8);
    Kokkos::initialize(argc, argv);


    ::testing::InitGoogleTest(&argc, argv);
    
    int result = RUN_ALL_TESTS();

    int final_result;
    MPI_Allreduce(&result, &final_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();

    return final_result;
}
