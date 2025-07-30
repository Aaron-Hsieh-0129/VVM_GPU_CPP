#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>

#include "utils/ConfigurationManager.hpp"
#include "io/OutputManager.hpp"
#include "core/Grid.hpp"
#include "core/Field.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "core/Initializer.hpp"

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

    omp_set_num_threads(72/size);

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
        VVM::Core::Parameters parameters(config, grid);
        grid.print_info();
        VVM::IO::OutputManager output_manager(config, grid, MPI_COMM_WORLD);
        VVM::Core::State state(config, parameters);
        VVM::Core::HaloExchanger halo_exchanger(grid);
        VVM::Core::BoundaryConditionManager bc_manager(grid, config);

        const int ny_total = grid.get_local_total_points_y();
        const int nx_total = grid.get_local_total_points_x();
        auto& htflx_sfc = state.get_field<2>("htflx_sfc");
        auto htflx_sfc_mutable = htflx_sfc.get_mutable_device_data();
        Kokkos::parallel_for("InitHeatFluxField",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny_total, nx_total}),
            KOKKOS_LAMBDA(const int j, const int i) {
                htflx_sfc_mutable(j, i) = static_cast<double>(100*rank + 10*j + i);
            }
        );

        VVM::Core::Initializer init(config, grid, parameters, state);
        init.initialize_state(state);

        // 1D field
        if (rank == 0) state.get_field<1>("thbar").print_profile(grid, 0, 0, 0);
        // if (rank == 0) parameters.dz_mid.print_profile(grid, 0, 0, 0);
        // if (rank == 0) parameters.flex_height_coef_mid.print_profile(grid, 0, 0, 0);
        output_manager.write(state, 0.0);


        // VVM::Dynamics::Takacs takacs_scheme;
        //
        std::cout << "params_nx: " << parameters.get_value_host(parameters.nx) << std::endl;
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
