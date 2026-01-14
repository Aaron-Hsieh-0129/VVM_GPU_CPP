#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <memory>
#include <nccl.h>
#include <cuda_runtime.h>

#include "core/Field.hpp"
#include "core/Grid.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Parameters.hpp"
#include "core/State.hpp"
#include "io/OutputManager.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"

#include "driver/Model.hpp"

#if defined(ENABLE_NCCL)
void init_nccl(ncclComm_t* comm, int rank, int size) {
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(comm, size, id, rank);
}
#endif

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omp_set_num_threads(72 / size);

    if (rank == 0) {
        std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
        std::cout << "OpenMP Proc Bind: " << omp_get_proc_bind() << std::endl;
    }

    Kokkos::initialize(argc, argv);

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm;
    init_nccl(&nccl_comm, rank, size);

    int device_id;
    cudaGetDevice(&device_id);
#endif

    {
        VVM::Utils::Timer total_timer("total vvm");
        VVM::Utils::TimingManager::get_instance().start_timer("initialize");

        if (rank == 0) std::cout << "VVM Model Simulation Started." << std::endl;

        // Load configuration file
        std::string config_file_path = "../rundata/input_configs/default_config.json";
        if (argc > 1) {
            config_file_path = argv[1]; // Allow command line override
        }

        VVM::Utils::ConfigurationManager config(config_file_path);
        // if (rank == 0) config.print_config(); // Print loaded configuration


        cudaStream_t stream = Kokkos::Cuda().cuda_stream();

        // Create a VVM model instance and run the simulation
        VVM::Core::Grid grid(config);
        VVM::Core::Parameters parameters(config, grid);
        grid.print_info();

#if defined(ENABLE_NCCL)
        VVM::Core::State state(config, parameters, grid, nccl_comm, stream);
        VVM::Core::HaloExchanger halo_exchanger(config, grid, nccl_comm, stream);
#else
        VVM::Core::State state(config, parameters, grid);
        VVM::Core::HaloExchanger halo_exchanger(grid);
#endif
        VVM::Driver::Model model(config, parameters, grid, state, halo_exchanger);
        model.init();

        VVM::Utils::TimingManager::get_instance().stop_timer("initialize");

        VVM::IO::OutputManager output_manager(config, grid, parameters, state, MPI_COMM_WORLD);
        output_manager.write(0, 0.0);
        output_manager.write_static_topo_file();

        // Simulation loop parameters
        double total_time = config.get_value<double>("simulation.total_time_s");
        double dt = parameters.get_value_host(parameters.dt);
        double output_interval = config.get_value<double>("simulation.output_interval_s");
        double next_output_time = output_interval;

        // Simulation loop
        while (state.get_time() < total_time) {
            model.run_step(dt);

            state.increment_step();
            state.advance_time(dt);

            std::cout << state.get_time() << std::endl;

             // Output data at specified intervals
            if (state.get_time() >= next_output_time) {
                output_manager.write(state.get_step(), state.get_time());
                next_output_time += output_interval;
            }
        }
        VVM::Utils::TimingManager::get_instance().stop_timer("total vvm");
        VVM::Utils::TimingManager::get_instance().print_timings(MPI_COMM_WORLD);
        model.finalize();

        Kokkos::fence();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
