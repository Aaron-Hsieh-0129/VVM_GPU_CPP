#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath> // For std::exp

#include "utils/ConfigurationManager.hpp"
#include "io/OutputManager.hpp"
#include "core/Grid.hpp"
#include "core/Field.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "core/Initializer.hpp"
#include "dynamics/DynamicalCore.hpp"

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
        VVM::IO::OutputManager output_manager(config, grid, parameters, MPI_COMM_WORLD);
        VVM::Core::State state(config, parameters);
        VVM::Core::HaloExchanger halo_exchanger(grid);
        VVM::Core::BoundaryConditionManager bc_manager(grid);

        const int nz_total = grid.get_local_total_points_z();
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


        // B.C. process
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("thbar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("rhobar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("rhobar_up"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("pbar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("pibar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("U"));

        auto& th = state.get_field<3>("th").get_mutable_device_data();
        auto& thbar = state.get_field<1>("thbar").get_device_data();
        auto& xi = state.get_field<3>("xi").get_mutable_device_data();
        auto& eta = state.get_field<3>("eta").get_mutable_device_data();
        auto& zeta = state.get_field<3>("zeta").get_mutable_device_data();
        auto& w = state.get_field<3>("w").get_mutable_device_data();
        
        Kokkos::parallel_for("th_init_with_perturbation", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz_total, ny_total, nx_total}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                double base_th = thbar(k);
                
                th(k,j,i) = 0.;
                // if (j == ny_total/2 && i == nx_total/2) th(k,j,i) += 5;
                if (k == 3 && j == 3 && i == 3) xi(k,j,i) += 5;
                if (k == 10 && j == 3 && i == 3) th(k,j,i) += 5;
                if (j == ny_total/2 && i == nx_total/2) eta(k,j,i) += 5;
                if (j == ny_total/2 && i == nx_total/2) zeta(k,j,i) += 5;

                if (k == 0 || k == nz_total-1 || k == nz_total-2) w(k,j,i) = 0;
        });

        halo_exchanger.exchange_halos(state.get_field<3>("th"));

        if (rank == 0) {
            std::cout << "\n--- Field State AFTER Halo Exchange ---" << std::endl;
        }

        // state.get_field<3>("th").print_slice_z_at_k(grid, 0, 1);
        

        // 1D field
        if (rank == 0) state.get_field<1>("thbar").print_profile(grid, 0, 0, 0);
        // if (rank == 0) parameters.dz_mid.print_profile(grid, 0, 0, 0);
        // if (rank == 0) parameters.flex_height_coef_mid.print_profile(grid, 0, 0, 0);
        output_manager.write(state, 0.0);


        VVM::Dynamics::DynamicalCore dynamical_core(config, grid, parameters, state);

        // Simulation loop parameters
        double total_time = config.get_value<double>("simulation.total_time_s");
        double dt = parameters.get_value_host(parameters.dt);
        double output_interval = config.get_value<double>("simulation.output_interval_s");
        double current_time = 0.0;
        double next_output_time = output_interval;

        // Simulation loop
        while (current_time < total_time) {
            dynamical_core.step(state, dt);
            current_time += dt;
            halo_exchanger.exchange_halos(state);

            std::cout << current_time << std::endl;

            // Output data at specified intervals
            if (current_time >= next_output_time) {
                output_manager.write(state, current_time);
                next_output_time += output_interval;
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
