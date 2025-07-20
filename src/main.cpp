#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>

#include "utils/ConfigurationManager.hpp"
#include "core/Grid.hpp"
#include "core/Field.hpp"
#include "core/HaloExchanger.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"

template <class ExecutionSpace>
void matrix_multiply(Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace> A,
                     Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace> B,
                     Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace> C,
                     int N) {
    using Policy = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>;
    Kokkos::parallel_for("MatrixMultiply",
                         Policy({0, 0}, {N, N}),
                         KOKKOS_LAMBDA(const int i, const int j) {
                             double sum = 0.0;
                             for (int k = 0; k < N; k++) {
                                 sum += A(i,k) * B(k,j);
                             }
                             C(i,j) = sum;
                         });
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omp_set_num_threads(16);

    if (rank == 0) {
        std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
        std::cout << "OpenMP Proc Bind: " << omp_get_proc_bind() << std::endl;
    }

    Kokkos::initialize(argc, argv);
    {
        if (rank == 0) {
            std::cout << "VVM Model Simulation Started." << std::endl;
        }

        // Load configuration file
        std::string config_file_path = "../data/input_configs/default_config.json";
        if (argc > 1) {
            config_file_path = argv[1]; // Allow command line override
        }

        VVM::Utils::ConfigurationManager config(config_file_path);
        if (rank == 0) {
            config.print_config(); // Print loaded configuration
        }

        

        // Create a VVM model instance and run the simulation
        VVM::Core::Grid grid(config);
        VVM::Core::ModelParameters model_params(grid, config);
        grid.print_info();

        const int nz_total = grid.get_local_total_points_z();
        const int ny_total = grid.get_local_total_points_y();
        const int nx_total = grid.get_local_total_points_x();


        // 3D Field and its halo exchange test
        VVM::Core::State state(config, model_params);
        state.add_field<3>("eta", {nz_total, ny_total, nx_total});

        auto& var_eta = state.get_field<3>("eta");
        auto eta_mutable = var_eta.get_mutable_device_data();

        Kokkos::parallel_for("EtaHaloTest",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz_total, ny_total, nx_total}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                eta_mutable(k, j, i) = static_cast<double>(100*rank + 10*j + i);
            }
        );
        Kokkos::fence(); 
        if (rank == 0) {
            std::cout << "\n--- Field State BEFORE Halo Exchange ---" << std::endl;
        }
        var_eta.print_slice_z_at_k(grid, 0, grid.get_halo_cells());


        if (rank == 0) {
            std::cout << "\n--- Field State AFTER Halo Exchange ---" << std::endl;
        }
        VVM::Core::HaloExchanger halo_exchanger(grid);
        halo_exchanger.exchange_halos(state);

        var_eta.print_slice_z_at_k(grid, 0, grid.get_halo_cells());
        Kokkos::fence(); 
        
        // 2D field test
        if (rank == 0) {
            std::cout << "\n--- Testing 2D Field ---" << std::endl;
        }
        state.add_field<2>("htflx_sfc", {
            grid.get_local_total_points_y(),
            grid.get_local_total_points_x()
        });
        auto& htflx_sfc = state.get_field<2>("htflx_sfc");
        auto htflx_sfc_mutable = htflx_sfc.get_mutable_device_data();
        Kokkos::parallel_for("InitHeatFluxField",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny_total, nx_total}),
            KOKKOS_LAMBDA(const int j, const int i) {
                htflx_sfc_mutable(j, i) = static_cast<double>(100*rank + 10*j + i);
            }
        );
        if (rank == 0) {
            std::cout << "\n--- Field State BEFORE Halo Exchange ---" << std::endl;
        }

        htflx_sfc.print_slice_z_at_k(grid, 0, grid.get_halo_cells());

        halo_exchanger.exchange_halos(htflx_sfc);

        if (rank == 0) {
            std::cout << "\n--- Field State AFTER Halo Exchange ---" << std::endl;
        }

        htflx_sfc.print_slice_z_at_k(grid, 0, grid.get_halo_cells());
        Kokkos::fence(); 


        // 4D field test
        if (rank == 0) {
            std::cout << "\n--- Testing 4D Field ---" << std::endl;
        }
        state.add_field<4>("d_eta", {
            2,
            grid.get_local_total_points_z(),
            grid.get_local_total_points_y(),
            grid.get_local_total_points_x()
        });
        auto& d_eta = state.get_field<4>("d_eta");
        auto d_eta_mutable = d_eta.get_mutable_device_data();
        Kokkos::parallel_for("InitDetaField",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {2, grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
            KOKKOS_LAMBDA(const int N, const int k, const int j, const int i) {
                d_eta_mutable(N, k, j, i) = static_cast<double>(100*rank + 10*j + i);
            }
        );
        if (rank == 0) {
            std::cout << "\n--- Field State BEFORE Halo Exchange ---" << std::endl;
        }

        d_eta.print_slice_z_at_k(grid, 0, 0);

        halo_exchanger.exchange_halos(d_eta);

        if (rank == 0) {
            std::cout << "\n--- Field State AFTER Halo Exchange ---" << std::endl;
        }

        d_eta.print_slice_z_at_k(grid, 1, 0);


        // Example of matrix multiplication using Kokkos
        if (rank == 0) {
            // Matrix dimensions
            const int N = 512;

            // Define views for CPU and GPU
            using HostSpace = Kokkos::HostSpace;
            using DeviceSpace = Kokkos::DefaultExecutionSpace;

            Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> A_host("A_host", N, N);
            Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> B_host("B_host", N, N);
            Kokkos::View<double**, Kokkos::LayoutRight, HostSpace> C_host("C_host", N, N);

            Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> A_device("A_device", N, N);
            Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> B_device("B_device", N, N);
            Kokkos::View<double**, Kokkos::LayoutRight, DeviceSpace> C_device("C_device", N, N);

            // Initialize matrices with random values
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A_host(i,j) = static_cast<double>(rand()) / RAND_MAX;
                    B_host(i,j) = static_cast<double>(rand()) / RAND_MAX;
                }
            }

            // Copy data to device
            Kokkos::deep_copy(A_device, A_host);
            Kokkos::deep_copy(B_device, B_host);

            // CPU execution with OpenMP
            auto start_cpu = std::chrono::high_resolution_clock::now();
            matrix_multiply<Kokkos::OpenMP>(A_host, B_host, C_host, N);
            Kokkos::fence();
            auto end_cpu = std::chrono::high_resolution_clock::now();
            auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();

            // GPU execution
            auto start_gpu = std::chrono::high_resolution_clock::now();
            matrix_multiply<DeviceSpace>(A_device, B_device, C_device, N);
            Kokkos::fence();
            auto end_gpu = std::chrono::high_resolution_clock::now();
            auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count();

            // Output results
            std::cout << "Matrix size: " << N << " x " << N << std::endl;
            std::cout << "CPU (OpenMP) execution time: " << cpu_duration << " ms" << std::endl;
            std::cout << "GPU execution time: " << gpu_duration << " ms" << std::endl;
            std::cout << "Speedup (CPU/GPU): " << static_cast<double>(cpu_duration) / gpu_duration << "x" << std::endl;
        }

    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}