#ifndef VVM_UTILS_TIMING_MANAGER_HPP
#define VVM_UTILS_TIMING_MANAGER_HPP

#include <string>
#include <map>
#include <chrono>
#include <vector>
#include <mpi.h>

namespace VVM {
namespace Utils {

struct TimingData {
    double wall_time = 0.0;
    double cpu_time = 0.0;
    long long call_count = 0;
    bool running = false;
};

class TimingManager {
public:
    static TimingManager& get_instance();

    void start_timer(const std::string& name);
    void stop_timer(const std::string& name);
    void print_timings(MPI_Comm comm);

    TimingManager(const TimingManager&) = delete;
    void operator=(const TimingManager&) = delete;

private:
    TimingManager() = default;
    ~TimingManager() = default;

    std::map<std::string, TimingData> timings_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> wall_starts_;
    std::map<std::string, clock_t> cpu_starts_;
};

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_TIMING_MANAGER_HPP
