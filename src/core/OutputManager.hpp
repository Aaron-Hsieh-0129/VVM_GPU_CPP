#ifndef VVM_CORE_OUTPUTMANAGER_HPP
#define VVM_CORE_OUTPUTMANAGER_HPP

#include <string>
#include <vector>
#include <map>
#include <pnetcdf.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <variant>

#include "Grid.hpp"
#include "State.hpp"
#include "../utils/ConfigurationManager.hpp"

namespace VVM {
namespace Core {

using AnyDeviceView = std::variant<
    std::monostate,
    Kokkos::View<const double*>,
    Kokkos::View<const double**>,
    Kokkos::View<const double***>,
    Kokkos::View<const double****>
>;

struct OutputJob {
    double time;
    std::map<std::string, AnyDeviceView> fields_to_write;
};


class OutputManager {
public:
    OutputManager(const Utils::ConfigurationManager& config, const Grid& grid, MPI_Comm comm);
    ~OutputManager();

    // Disable copy and move semantics to prevent accidental copying of MPI communicators and PnetCDF file handles.
    OutputManager(const OutputManager&) = delete;
    OutputManager& operator=(const OutputManager&) = delete;
    OutputManager(OutputManager&&) = delete;
    OutputManager& operator=(OutputManager&&) = delete;

    void write_output(const State& state, double time);

private:
    void io_thread_function(); 
    void write_job_to_netcdf(const OutputJob& job);
    void define_dimensions_and_vars(const State& state);
    void handle_pnetcdf_error(int status, const std::string& message) const;

    const Grid& grid_;
    std::string output_dir_;
    std::string filename_prefix_;
    std::vector<std::string> fields_to_output_;

    // Global output grid specifications
    MPI_Offset g_x_start_, g_x_end_, x_stride_;
    MPI_Offset g_y_start_, g_y_end_, y_stride_;
    MPI_Offset g_z_start_, g_z_end_, z_stride_;

    int rank_;
    int mpi_size_;
    MPI_Comm comm_;

    bool enabled_ = false;
    int ncid_ = -1; // NetCDF file ID

    // Dimension and variable IDs for NetCDF
    int time_dim_id_, z_dim_id_, y_dim_id_, x_dim_id_, nv_dim_id_;
    int time_var_id_, z_var_id_, y_var_id_, x_var_id_;
    std::map<std::string, int> field_var_ids_;
    MPI_Offset time_idx_ = 0;

    // Threading for non-blocking I/O
    std::thread io_thread_;
    std::queue<OutputJob> job_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool finished_ = false;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_OUTPUTMANAGER_HPP