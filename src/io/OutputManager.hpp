#ifndef VVM_IO_OutputManager_HPP
#define VVM_IO_OutputManager_HPP

#include <string>
#include <vector>
#include <map>
#include <adios2.h>
#include <Kokkos_Core.hpp>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"

namespace VVM {
namespace IO {

class OutputManager {
public:
    OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, VVM::Core::State& state, MPI_Comm comm);
    ~OutputManager();

    OutputManager(const OutputManager&) = delete;
    OutputManager& operator=(const OutputManager&) = delete;
    OutputManager(OutputManager&&) = delete;
    OutputManager& operator=(OutputManager&&) = delete;

    void write(int step, double time);
    void write_static_data();
    void write_static_topo_file();

private:
    const VVM::Core::Grid& grid_;
    const VVM::Core::Parameters& params_;
    VVM::Core::State& state_;
    std::string output_dir_;
    std::string filename_prefix_;
    std::vector<std::string> fields_to_output_;
    double output_interval_s_;
    double total_time_;

    int rank_;
    int mpi_size_;
    MPI_Comm comm_;
    adios2::ADIOS adios_;
    adios2::IO io_;
    adios2::Engine writer_;
    std::map<std::string, adios2::Variable<double>> field_variables_;

    size_t output_x_start_, output_y_start_, output_z_start_;
    size_t output_x_end_, output_y_end_, output_z_end_;
    size_t output_x_stride_, output_y_stride_, output_z_stride_;

    bool variables_defined_ = false;
    adios2::Variable<double> var_time_;

    void define_variables();
    void grads_ctl_file();
};

} // namespace IO
} // namespace VVM

#endif // VVM_CORE_OutputManager_HPP
