#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <memory>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h>

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
#include "io/IOServer.hpp"

#if defined(ENABLE_NCCL)
void init_nccl(ncclComm_t* comm, int rank, int size, MPI_Comm mpi_comm) {
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
    ncclCommInitRank(comm, size, id, rank);
}
#endif

int get_io_tasks(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--io-tasks" && i + 1 < argc) {
            return std::stoi(argv[i + 1]);
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    long total_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int threads_per_rank = total_cores / node_size;
    if (threads_per_rank < 1) threads_per_rank = 1;
    omp_set_num_threads(threads_per_rank);
    if (world_rank == 0) {
        std::cout << "[System] Node Cores: " << total_cores 
                  << " | Ranks per Node: " << node_size 
                  << " | OpenMP Threads per Rank: " << threads_per_rank << std::endl;
    }

    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0) {
        num_gpus = 1; 
    }
    int gpu_id = node_rank % num_gpus;
    Kokkos::InitializationSettings args;
    args.set_device_id(gpu_id);

    Kokkos::initialize(args);

    // Load configuration file
    std::string config_file_path = "../rundata/input_configs/default_config.json";
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--io-tasks") { i++; continue; } 
        if(arg[0] != '-') config_file_path = arg;
    }
    VVM::Utils::ConfigurationManager config(config_file_path);

    if (world_rank == 0) {
        std::string engine = config.get_value<std::string>("output.engine", "HDF5");
        if (engine == "SST") {
            std::string prefix = config.get_value<std::string>("output.output_filename_prefix");
            std::string sst_path = prefix + ".sst";
            
            std::cout << "[Main] Global Rank 0 cleaning stale SST: " << sst_path << std::endl;
            std::string cmd = "rm -rf " + sst_path;
            system(cmd.c_str());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int num_io_tasks = get_io_tasks(argc, argv);
    int num_sim_tasks = world_size - num_io_tasks;

    if (num_sim_tasks <= 0) {
        if (world_rank == 0) std::cerr << "Error: Not enough ranks for simulation!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int color = (world_rank < num_sim_tasks) ? 0 : 1;
    
    MPI_Comm split_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &split_comm);

    int split_rank, split_size;
    MPI_Comm_rank(split_comm, &split_rank);
    MPI_Comm_size(split_comm, &split_size);

    if (color == 1) {
        VVM::IO::run_io_server(split_comm, config);

        MPI_Comm_free(&split_comm);
        MPI_Finalize();
        return 0;
    }

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm;
    init_nccl(&nccl_comm, split_rank, split_size, split_comm);

    int device_id;
    cudaGetDevice(&device_id);
#endif

    {
        VVM::Utils::Timer total_timer("total vvm");
        VVM::Utils::TimingManager::get_instance().start_timer("initialize");

        if (split_rank == 0) std::cout << "VVM Model Simulation Started." << std::endl;

        // if (rank == 0) config.print_config(); // Print loaded configuration


        cudaStream_t stream = Kokkos::Cuda().cuda_stream();

        // Create a VVM model instance and run the simulation
        VVM::Core::Grid grid(config, split_comm);
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

        auto output_manager = std::make_unique<VVM::IO::OutputManager>(
            config, grid, parameters, state, split_comm
        );

        output_manager->write(0, 0.0);
        // output_manager->write_static_topo_file();

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
                output_manager->write(state.get_step(), state.get_time());
                next_output_time += output_interval;
            }
        }
        VVM::Utils::TimingManager::get_instance().stop_timer("total vvm");
        VVM::Utils::TimingManager::get_instance().print_timings(split_comm);
        model.finalize();

        Kokkos::fence();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
