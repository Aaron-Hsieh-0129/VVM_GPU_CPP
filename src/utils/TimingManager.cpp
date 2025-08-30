#include "TimingManager.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

namespace VVM {
namespace Utils {

TimingManager& TimingManager::get_instance() {
    static TimingManager instance;
    return instance;
}

void TimingManager::start_timer(const std::string& name) {
    if (!timings_[name].running) {
        wall_starts_[name] = std::chrono::high_resolution_clock::now();
        cpu_starts_[name] = clock();
        timings_[name].running = true;
    }
}

void TimingManager::stop_timer(const std::string& name) {
    if (timings_[name].running) {
        auto wall_end = std::chrono::high_resolution_clock::now();
        auto cpu_end = clock();

        std::chrono::duration<double> wall_duration = wall_end - wall_starts_[name];
        double cpu_duration = static_cast<double>(cpu_end - cpu_starts_[name]) / CLOCKS_PER_SEC;

        timings_[name].wall_time += wall_duration.count();
        timings_[name].cpu_time += cpu_duration;
        timings_[name].call_count++;
        timings_[name].running = false;
    }
}

void TimingManager::print_timings(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all event names on rank 0
    std::vector<std::string> all_event_names;
    if (rank == 0) {
        for (const auto& pair : timings_) {
            all_event_names.push_back(pair.first);
        }

        for (int i = 1; i < size; ++i) {
            int name_count;
            MPI_Recv(&name_count, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
            for (int j = 0; j < name_count; ++j) {
                int name_len;
                MPI_Recv(&name_len, 1, MPI_INT, i, 1, comm, MPI_STATUS_IGNORE);
                std::vector<char> name_buf(name_len + 1);
                MPI_Recv(name_buf.data(), name_len, MPI_CHAR, i, 2, comm, MPI_STATUS_IGNORE);
                name_buf[name_len] = '\0';
                std::string event_name(name_buf.data());
                bool found = false;
                for(const auto& existing_name : all_event_names){
                    if(existing_name == event_name){
                        found = true;
                        break;
                    }
                }
                if(!found){
                    all_event_names.push_back(event_name);
                }
            }
        }
    } else {
        int name_count = timings_.size();
        MPI_Send(&name_count, 1, MPI_INT, 0, 0, comm);
        for (const auto& pair : timings_) {
            int name_len = pair.first.length();
            MPI_Send(&name_len, 1, MPI_INT, 0, 1, comm);
            MPI_Send(pair.first.c_str(), name_len, MPI_CHAR, 0, 2, comm);
        }
    }

    // Rank 0 prints the header
    if (rank == 0) {
        std::cout << "\n TIMINGS (process:event,running,cpu,wall)\n";
    }

    // Each process reports its timings for all events
    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            for (const auto& event_name : all_event_names) {
                auto it = timings_.find(event_name);
                if (it != timings_.end()) {
                    const auto& data = it->second;
                    std::cout << "      " << std::setw(3) << rank << " : "
                              << std::left << std::setw(20) << event_name
                              << (data.running ? 'T' : 'F')
                              << std::fixed << std::setprecision(2)
                              << std::setw(15) << data.cpu_time
                              << std::setw(15) << data.wall_time << '\n';
                } else {
                     std::cout << "      " << std::setw(3) << rank << " : "
                              << std::left << std::setw(20) << event_name
                              << 'F'
                              << std::fixed << std::setprecision(2)
                              << std::setw(15) << 0.0
                              << std::setw(15) << 0.0 << '\n';
                }
            }
        }
        MPI_Barrier(comm);
    }
}


} // namespace Utils
} // namespace VVM
