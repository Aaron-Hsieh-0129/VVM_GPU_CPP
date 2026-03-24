#ifndef VVM_UTILS_TIMING_MANAGER_HPP
#define VVM_UTILS_TIMING_MANAGER_HPP

#include <string>
#include <mpi.h>

namespace VVM {
namespace Utils {

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
};

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_TIMING_MANAGER_HPP
