#ifndef VVM_IO_OutputManager_HPP
#define VVM_IO_OutputManager_HPP

#include <string>
#include <vector>
#include <map>
#include <adios2.h>
#include <Kokkos_Core.hpp>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace IO {

class OutputManager {
public:
    OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, MPI_Comm comm);
    ~OutputManager();

    OutputManager(const OutputManager&) = delete;
    OutputManager& operator=(const OutputManager&) = delete;
    OutputManager(OutputManager&&) = delete;
    OutputManager& operator=(OutputManager&&) = delete;

    void write(const VVM::Core::State& state, double time);

private:
    const VVM::Core::Grid& grid_;
    std::string output_dir_;
    std::string filename_prefix_;
    std::vector<std::string> fields_to_output_;

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

    void define_variables(const VVM::Core::State& state);
};

} // namespace IO
} // namespace VVM

#endif // VVM_CORE_OutputManager_HPP
