#include "IOServer.hpp"
#include <iostream>
#include <adios2.h>
#include <thread>
#include <chrono>

namespace VVM {
namespace IO {

void run_io_server(MPI_Comm io_comm) {
    int rank, size;
    MPI_Comm_rank(io_comm, &rank);
    MPI_Comm_size(io_comm, &size);
    
    // Create a coloring for terminal output to distinguish IO from Sim
    std::string prefix = "  [IO-Server " + std::to_string(rank) + "]\033[0m ";

    if (rank == 0) {
        std::cout << prefix << "Started on " << size << " ranks." << std::endl;
        std::cout << prefix << "Waiting for Simulation data..." << std::endl;
    }

    // Initialize ADIOS2 on the I/O communicator
    adios2::ADIOS adios(io_comm);

    // --- PHASE 2 TODO: Add SST Reading Logic Here ---
    // For Step 1 test, we just wait a bit so the Sim can run, then exit.
    // In production, this will be a while loop reading steps.
    
    // Simulate work to keep the server alive while Sim initializes
    MPI_Barrier(io_comm);
    
    if (rank == 0) std::cout << prefix << "Shutting down." << std::endl;
}

} // namespace IO
} // namespace VVM
