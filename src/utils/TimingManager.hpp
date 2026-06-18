#ifndef VVM_UTILS_TIMING_MANAGER_HPP
#define VVM_UTILS_TIMING_MANAGER_HPP

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace VVM {
namespace Utils {

class TimingManager {
public:
    static TimingManager& get_instance();

    void configure(bool enabled, bool fence_gpu, int warmup_steps = 0);
    void set_step(std::size_t step);

    void start_timer(const std::string& name);
    void stop_timer(const std::string& name);

    void reset();
    void print_timings(MPI_Comm comm, bool reset_after_print = false);

    TimingManager(const TimingManager&) = delete;
    TimingManager& operator=(const TimingManager&) = delete;

private:
    struct Record {
        double total_s = 0.0;
        double last_s = 0.0;
        long long count = 0;
    };

    struct ActiveTimer {
        std::string name;
        double start_s = 0.0;
    };

    TimingManager() = default;
    ~TimingManager() = default;

    bool should_accumulate() const;
    std::vector<std::string> collect_timer_names(MPI_Comm comm) const;

    bool enabled_ = true;
    bool fence_gpu_ = false;
    std::size_t current_step_ = 0;
    std::size_t warmup_steps_ = 0;

    std::map<std::string, Record> records_;
    std::vector<ActiveTimer> active_stack_;
};

} // namespace Utils
} // namespace VVM

#endif
