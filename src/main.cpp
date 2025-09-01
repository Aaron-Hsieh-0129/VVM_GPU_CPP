#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath> // For std::exp

#include "utils/ConfigurationManager.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"
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
        VVM::Utils::Timer total_timer("total vvm");
        VVM::Utils::TimingManager::get_instance().start_timer("initialize");

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
        VVM::Core::State state(config, parameters, grid);
        VVM::Core::HaloExchanger halo_exchanger(grid);
        VVM::Core::BoundaryConditionManager bc_manager(grid);

        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int h = grid.get_halo_cells();
        auto& htflx_sfc = state.get_field<2>("htflx_sfc");
        auto htflx_sfc_mutable = htflx_sfc.get_mutable_device_data();
        auto& zeta_field = state.get_field<3>("zeta");
        auto& zeta = state.get_field<3>("zeta").get_mutable_device_data();

        Kokkos::parallel_for("InitHeatFluxField",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
            KOKKOS_LAMBDA(const int j, const int i) {
                htflx_sfc_mutable(j, i) = static_cast<double>(100*rank + 10*j + i);
            }
        );
        // if (rank == 0) zeta_field.print_slice_z_at_k(grid, 0, nz-h-1);
        // Kokkos::fence();
        // halo_exchanger.exchange_halos_top_slice(state.get_field<3>("zeta"));
        // if (rank == 0) zeta_field.print_slice_z_at_k(grid, 0, nz-h-1);
        // Kokkos::fence();
        // exit(1);

        double heat_flux_mean = state.calculate_horizontal_mean(htflx_sfc);
        if (rank == 0) {
            std::cout << "Average of heat flux is: " << heat_flux_mean << std::endl;
        }

        VVM::Core::Initializer init(config, grid, parameters, state);
        init.initialize_state();


        // B.C. process
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("thbar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("rhobar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("rhobar_up"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("pbar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("pibar"));
        bc_manager.apply_z_bcs_to_field(state.get_field<1>("U"));


        if (rank == 0) state.get_field<1>("rhobar_up").print_profile(grid, 0, 0, 0);
        if (rank == 0) parameters.z_mid.print_profile(grid, 0, 0, 0);
        if (rank == 0) parameters.z_up.print_profile(grid, 0, 0, 0);

        auto& th = state.get_field<3>("th").get_mutable_device_data();
        auto& thbar = state.get_field<1>("thbar").get_device_data();
        auto& xi = state.get_field<3>("xi").get_mutable_device_data();
        auto& eta = state.get_field<3>("eta").get_mutable_device_data();
        auto& u = state.get_field<3>("u").get_mutable_device_data();
        auto& v = state.get_field<3>("v").get_mutable_device_data();
        auto& w = state.get_field<3>("w").get_mutable_device_data();
        auto& w_field = state.get_field<3>("w");

        const int global_start_j = grid.get_local_physical_start_y();
        const int global_start_i = grid.get_local_physical_start_x();

        const auto& dx = parameters.dx;
        const auto& dy = parameters.dy;
        const auto& z_mid = parameters.z_mid.get_device_data();

        
        Kokkos::parallel_for("th_init_with_perturbation", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int local_j = j - h;
                const int local_i = i - h;

                const int global_j = global_start_j + local_j;
                const int global_i = global_start_i + local_i;

                double base_th = thbar(k);
                
                th(k,j,i) = 300;
                // if (k == 3 && j == ny/2 && i == nx/2 && (rank == 0 || rank == 1)) th(k,j,i) += 50;

                /*
                if ((32/2-3-1 <= global_j && 32/2+3-1 >= global_j) && (32/2-3-1 <= global_i && 32/2+3-1 >= global_i)) {
                    if (k == h+15) {
                        xi(k,j,i) = 50;
                        eta(k,j,i) = 50;
                        th(k,j,i) += 50;
                    }
                    else if (k == nz-h-1) {
                        zeta(k,j,i) = 50;
                    }
                    else if (k == nz-h-2) {
                        eta(k,j,i) = 50;
                    }
                }

                if (global_j >= 4 && global_j <= 32-1-3 && global_i >= 4 && global_i <= 32-1-3) {
                    // u(k,j,i) = 32./2. - global_i - 1;
                    // v(k,j,i) = 32./2. - global_j - 1;
                    // u(k,j,i) = 32./2. - global_j - 1;
                    u(k,j,i) = 32./2. - (global_i+global_j)/2. - 1;
                    v(k,j,i) = 32./2. - (global_i+global_j)/2. - 1;

                    u(k,j,i) /= 8;
                    v(k,j,i) /= 8;
                }
                // if (k == 0 || k == nz-1 || k == nz-2) w(k,j,i) = 0;
                */

                double radius_norm = std::sqrt(
                             std::pow(((global_i+1)-32./2.)*dx()/2000., 2) + std::pow((z_mid(k)-3000.)/2000., 2) 
                           + std::pow(((global_j+1)-32./2.)*dy()/2000., 2)
                          );
                if (radius_norm <= 1) {
                    th(k,j,i) = th(k,j,i) + 5.*(std::cos(3.14159265*0.5*radius_norm));
                }
            }
        );
        // if (rank == 0) {
        //     std::cout << "\n--- Field State BEFORE Halo Exchange ---" << std::endl;
        //     state.get_field<3>("v").print_slice_z_at_k(grid, 0, 18);
        // }
        halo_exchanger.exchange_halos(state.get_field<3>("th"));
        halo_exchanger.exchange_halos(state.get_field<3>("xi"));
        halo_exchanger.exchange_halos(state.get_field<3>("eta"));
        halo_exchanger.exchange_halos(state.get_field<3>("zeta"));
        halo_exchanger.exchange_halos(state.get_field<3>("u"));
        halo_exchanger.exchange_halos(state.get_field<3>("v"));

        // if (rank == 0) {
        //     std::cout << "\n--- Field State AFTER Halo Exchange ---" << std::endl;
        //     state.get_field<3>("v").print_slice_z_at_k(grid, 0, 18);
        // }

        
        // if (rank == 0) w_field.print_xz_cross_at_j(grid, 0, 3);

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

        VVM::Utils::TimingManager::get_instance().stop_timer("initialize");
        // Simulation loop
        while (current_time < total_time) {
            dynamical_core.step(state, dt);
            halo_exchanger.exchange_halos(state);
            current_time += dt;

            std::cout << current_time << std::endl;

            // Output data at specified intervals
            if (current_time >= next_output_time) {
                output_manager.write(state, current_time);
                next_output_time += output_interval;
            }
        }
    }
    VVM::Utils::TimingManager::get_instance().print_timings(MPI_COMM_WORLD);

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
