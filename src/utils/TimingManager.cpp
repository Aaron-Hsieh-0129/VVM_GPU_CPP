#include "TimingManager.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>

namespace VVM {
namespace Utils {

TimingManager& TimingManager::get_instance() {
    static TimingManager instance;
    return instance;
}

void TimingManager::configure(bool enabled, bool fence_gpu, int warmup_steps) {
    enabled_ = enabled;
    fence_gpu_ = fence_gpu;
    warmup_steps_ = static_cast<std::size_t>(std::max(0, warmup_steps));
}

void TimingManager::set_step(std::size_t step) {
    current_step_ = step;
}

bool TimingManager::should_accumulate() const {
    return enabled_ && current_step_ >= warmup_steps_;
}

void TimingManager::start_timer(const std::string& name) {
    // Keep Kokkos profiling regions for Nsight/Kokkos Tools.
    Kokkos::Profiling::pushRegion(name);

    if (!should_accumulate()) return;

    if (fence_gpu_) {
        Kokkos::fence("timer_start_" + name);
    }

    active_stack_.push_back({name, MPI_Wtime()});
}

void TimingManager::stop_timer(const std::string& name) {
    if (should_accumulate() && !active_stack_.empty()) {
        if (fence_gpu_) {
            Kokkos::fence("timer_stop_" + name);
        }

        const double stop_s = MPI_Wtime();

        ActiveTimer active = active_stack_.back();
        active_stack_.pop_back();

        const double elapsed_s = stop_s - active.start_s;
        auto& rec = records_[active.name];
        rec.total_s += elapsed_s;
        rec.last_s = elapsed_s;
        rec.count += 1;
    }

    Kokkos::Profiling::popRegion();
}

void TimingManager::reset() {
    records_.clear();
    active_stack_.clear();
}

std::vector<std::string> TimingManager::collect_timer_names(MPI_Comm comm) const {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::string local_packed;
    for (const auto& kv : records_) {
        local_packed += kv.first;
        local_packed += '\n';
    }

    const int local_len = static_cast<int>(local_packed.size());

    std::vector<int> recv_counts;
    std::vector<int> displs;

    if (rank == 0) {
        recv_counts.resize(size, 0);
        displs.resize(size, 0);
    }

    MPI_Gather(
        &local_len, 1, MPI_INT,
        rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT,
        0, comm
    );

    int total_len = 0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            displs[r] = total_len;
            total_len += recv_counts[r];
        }
    }

    std::string gathered;
    if (rank == 0) {
        gathered.resize(total_len);
    }

    MPI_Gatherv(
        local_packed.data(), local_len, MPI_CHAR,
        rank == 0 ? gathered.data() : nullptr,
        rank == 0 ? recv_counts.data() : nullptr,
        rank == 0 ? displs.data() : nullptr,
        MPI_CHAR, 0, comm
    );

    std::string union_packed;

    if (rank == 0) {
        std::set<std::string> names;
        std::istringstream iss(gathered);
        std::string line;

        while (std::getline(iss, line)) {
            if (!line.empty()) names.insert(line);
        }

        for (const auto& n : names) {
            union_packed += n;
            union_packed += '\n';
        }
    }

    int union_len = static_cast<int>(union_packed.size());
    MPI_Bcast(&union_len, 1, MPI_INT, 0, comm);

    if (rank != 0) {
        union_packed.resize(union_len);
    }

    MPI_Bcast(union_packed.data(), union_len, MPI_CHAR, 0, comm);

    std::vector<std::string> names;
    std::istringstream iss(union_packed);
    std::string line;

    while (std::getline(iss, line)) {
        if (!line.empty()) names.push_back(line);
    }

    return names;
}

void TimingManager::print_timings(MPI_Comm comm, bool reset_after_print) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (!enabled_) {
        if (rank == 0) {
            std::cout << "[TimingManager] Internal timing disabled.\n";
        }
        return;
    }

    const auto names = collect_timer_names(comm);

    struct Row {
        std::string name;
        long long count_max = 0;
        double total_min = 0.0;
        double total_mean = 0.0;
        double total_max = 0.0;
        double avg_call_ms = 0.0;
        double pct_total = 0.0;
    };

    std::vector<Row> rows;
    rows.reserve(names.size());

    double total_vvm_max = 0.0;

    auto total_it = records_.find("total_vvm");
    const double local_total_vvm =
        total_it == records_.end() ? 0.0 : total_it->second.total_s;

    MPI_Reduce(&local_total_vvm, &total_vvm_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    for (const auto& name : names) {
        auto it = records_.find(name);

        const double local_total = it == records_.end() ? 0.0 : it->second.total_s;
        const long long local_count = it == records_.end() ? 0 : it->second.count;

        double min_total = 0.0;
        double max_total = 0.0;
        double sum_total = 0.0;
        long long max_count = 0;

        MPI_Reduce(&local_total, &min_total, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&local_total, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&local_count, &max_count, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);

        if (rank == 0) {
            Row row;
            row.name = name;
            row.count_max = max_count;
            row.total_min = min_total;
            row.total_mean = sum_total / static_cast<double>(size);
            row.total_max = max_total;
            row.avg_call_ms =
                max_count > 0 ? 1000.0 * max_total / static_cast<double>(max_count) : 0.0;
            row.pct_total =
                total_vvm_max > 0.0 ? 100.0 * max_total / total_vvm_max : 0.0;

            rows.push_back(row);
        }
    }

    if (rank == 0) {
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            return a.total_max > b.total_max;
        });

        std::cout << "\n============================================================\n";
        std::cout << " VVM INTERNAL TIMING SUMMARY\n";
        std::cout << " GPU fence timing: " << (fence_gpu_ ? "ON" : "OFF") << "\n";
        std::cout << " Warmup skipped steps: " << warmup_steps_ << "\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(28) << "component"
                  << std::right
                  << std::setw(10) << "calls"
                  << std::setw(14) << "max_s"
                  << std::setw(14) << "mean_s"
                  << std::setw(14) << "min_s"
                  << std::setw(14) << "ms/call"
                  << std::setw(10) << "%total"
                  << "\n";

        std::cout << "------------------------------------------------------------\n";

        for (const auto& r : rows) {
            std::cout << std::left
                      << std::setw(28) << r.name
                      << std::right
                      << std::setw(10) << r.count_max
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.total_max
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.total_mean
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.total_min
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.avg_call_ms
                      << std::setw(10) << std::fixed << std::setprecision(2) << r.pct_total
                      << "\n";
        }

        std::cout << "============================================================\n";
    }

    if (reset_after_print) {
        reset();
    }
}

} // namespace Utils
} // namespace VVM
