#include "TimingManager.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>

namespace VVM {
namespace Utils {

TimingManager& TimingManager::get_instance() {
    static TimingManager instance;
    return instance;
}

void TimingManager::start_timer(const std::string& name) {
    Kokkos::Profiling::pushRegion(name);
}

void TimingManager::stop_timer(const std::string& name) {
    Kokkos::Profiling::popRegion();
}


void TimingManager::print_timings(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        std::cout << "\n======================================================\n"
                  << " TIMINGS: Delegated to Kokkos Profiling API\n"
                  << "------------------------------------------------------\n"
                  << " For accurate MPI+GPU timing analysis, please use:\n"
                  << " 1. NVIDIA Nsight Systems (nsys profile)\n"
                  << " 2. Kokkos Tools (e.g., kp_kernel_timer.so)\n"
                  << "Use command like this: nsys profile --trace=mpi,cuda,nvtx,osrt -o vvm_profile_report mpirun -np 2 ./vvm --io-tasks 1\n"
                  << "======================================================\n";
    }
}

} // namespace Utils
} // namespace VVM
