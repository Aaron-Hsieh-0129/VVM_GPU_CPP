#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

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

    // Load configuration file
    std::string config_file_path = TOSTRING(VVM_ROOT_DIR) "/rundata/input_configs/default_config.json";
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--io-tasks") { i++; continue; } 
        if(arg[0] != '-') config_file_path = arg;
    }
    VVM::Utils::ConfigurationManager config(config_file_path);

    // clean existing SST file to prevent errors
    if (world_rank == 0) {
        std::string engine = config.get_value<std::string>("output.engine", "HDF5");
        if (engine == "SST") {
            std::string output_dir = config.get_value<std::string>("output.output_dir");
            std::string prefix = config.get_value<std::string>("output.output_filename_prefix");
            std::string sst_path = output_dir + "/" + prefix + ".sst";
            
            std::cout << "[Main] Global Rank 0 cleaning stale SST: " << sst_path << std::endl;
            std::string cmd = "rm -rf " + sst_path;
            system(cmd.c_str());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Split Compute and IO jobs
    int num_io_tasks = get_io_tasks(argc, argv);

    // IO server is only meaningful for SST engine. Abort early if the job was
    // submitted with --io-tasks N for a non-SST engine, because those IO server
    // ranks would either hang waiting for an SST stream that never opens, or
    // compete for CUDA devices with compute ranks in CUDA exclusive mode.
    {
        std::string engine = config.get_value<std::string>("output.engine", "HDF5");
        if (engine != "SST" && num_io_tasks > 0) {
            if (world_rank == 0) {
                std::cerr << "[Main] ERROR: output.engine=\"" << engine
                          << "\" but job was submitted with --io-tasks " << num_io_tasks << ".\n"
                          << "  IO server ranks are only used with SST engine.\n"
                          << "  Re-submit without --io-tasks (or with --io-tasks 0)." << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

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

    int threads_per_rank = 1;
    if (const char* env_p = std::getenv("OMP_NUM_THREADS")) {
        threads_per_rank = std::stoi(env_p);
    }
    omp_set_num_threads(threads_per_rank);

    int num_gpus = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&num_gpus);
    if (cuda_err != cudaSuccess || num_gpus <= 0) {
        std::cerr << "[Rank " << world_rank
                  << "] ERROR: no visible CUDA device. CUDA_VISIBLE_DEVICES="
                  << (std::getenv("CUDA_VISIBLE_DEVICES")
                          ? std::getenv("CUDA_VISIBLE_DEVICES")
                          : "<unset>")
                  << " cudaGetDeviceCount error="
                  << cudaGetErrorString(cuda_err)
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    Kokkos::InitializationSettings args;
    // core_run.sh constrains each process to one intended GPU.
    args.set_device_id(0);
    Kokkos::initialize(args);

    if (world_rank == 0) {
        std::cout << "[System] CPU Threads per Rank set to: "
                  << threads_per_rank << std::endl;
    }
    std::cout << "[KokkosInit]"
              << " world_rank=" << world_rank
              << " role=" << ((color == 0) ? "compute" : "io")
              << " CUDA_VISIBLE_DEVICES="
              << (std::getenv("CUDA_VISIBLE_DEVICES")
                      ? std::getenv("CUDA_VISIBLE_DEVICES")
                      : "<unset>")
              << " visible_num_gpus=" << num_gpus
              << " kokkos_device_id=0"
              << std::endl;

    // All ranks have now initialized Kokkos/CUDA before ADIOS2 starts SST.
    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: IO server branch.
    if (color == 1) {
        VVM::IO::run_io_server(split_comm, config);
        MPI_Comm_free(&split_comm);
        Kokkos::fence();
        Kokkos::finalize();
        MPI_Finalize();
        return 0;
    }

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm;
    init_nccl(&nccl_comm, split_rank, split_size, split_comm);
#endif

    {
        auto& timing = VVM::Utils::TimingManager::get_instance();

        timing.configure(
            config.get_value<bool>("performance.timing.enable", true),
            config.get_value<bool>("performance.timing.fence_gpu", false),
            config.get_value<int>("performance.timing.warmup_steps", 0)
        );

        const int timing_print_interval_steps =
            config.get_value<int>("performance.timing.print_interval_steps", 0);

        const bool timing_reset_after_interval_print =
            config.get_value<bool>("performance.timing.reset_after_interval_print", false);

        timing.start_timer("total_vvm");

        timing.start_timer("initialize");

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

        timing.stop_timer("initialize");

        auto output_manager = std::make_unique<VVM::IO::OutputManager>(
            config, grid, parameters, state, split_comm
        );

        const bool restart_enabled = config.get_value<bool>("restart.enable", false);
        const bool output_initial_step =
            config.get_value<bool>("output.output_initial_step", true);
        if (output_initial_step) {
            VVM::Utils::Timer timer("io");
            output_manager->write(0, 0.0);
        } else if (split_rank == 0) {
            std::cout << "[Output] Skipping initial full output." << std::endl;
        }
        // output_manager->write_static_topo_file();

        // Simulation loop parameters
        double total_time = config.get_value<double>("simulation.total_time_s");
        double dt = parameters.get_value_host(parameters.dt);
        double output_interval = config.get_value<double>("simulation.output_interval_s");
        double next_output_time = output_interval;
        if (restart_enabled) {
            next_output_time = (std::floor(state.get_time() / output_interval) + 1.0) * output_interval;
        }

        // Simulation loop
        while (state.get_time() < total_time) {
            model.run_step(dt);

            state.increment_step();
            state.advance_time(dt);

            if (split_rank == 0) std::cout << state.get_time() << std::endl;

             // Output data at specified intervals
            if (state.get_time() >= next_output_time) {
                {
                    VVM::Utils::Timer timer("io");
                    output_manager->write(state.get_step(), state.get_time());
                }
                next_output_time += output_interval;
            }
            if (timing_print_interval_steps > 0 &&
                state.get_step() % static_cast<size_t>(timing_print_interval_steps) == 0) {
                timing.print_timings(split_comm, timing_reset_after_interval_print);
            }
        }
        timing.stop_timer("total_vvm");
        timing.print_timings(split_comm, false);

        model.finalize();
        Kokkos::fence();
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
