#ifndef VVM_READERS_PNETCDFREADER_HPP
#define VVM_READERS_PNETCDFREADER_HPP

#include "io/Reader.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include "core/HaloExchanger.hpp"

#include <string>
#include <map>
#include <vector>
#include <stdexcept>
#include <mpi.h>
#include <pnetcdf.h>

namespace VVM {
namespace IO {

class PnetcdfReader : public Reader {
public:
    PnetcdfReader(const std::string& filepath, 
                  const VVM::Core::Grid& grid, 
                  const VVM::Core::Parameters& params, 
                  const VVM::Utils::ConfigurationManager& config);
    
    ~PnetcdfReader() override;

    void read_and_initialize(VVM::Core::State& state) override;

private:
    void check_ncmpi_error(int status, const std::string& msg) const;

    template<size_t Dim>
    void read_variable_1d(int ncid, const std::string& var_name, VVM::Core::Field<Dim>& field);

    template<size_t Dim>
    void read_variable_2d(int ncid, const std::string& var_name, VVM::Core::Field<Dim>& field);

    template<size_t Dim>
    void read_variable_3d(int ncid, const std::string& var_name, VVM::Core::Field<Dim>& field);

    std::map<std::string, MPI_Offset> get_file_dimensions(int ncid) const;
    void validate_dimensions(const std::map<std::string, MPI_Offset>& file_dims) const;

    std::string source_file_;
    const VVM::Core::Grid& grid_;
    const VVM::Core::Parameters& params_;
    const VVM::Utils::ConfigurationManager& config_;
    
    MPI_Comm comm_;
    int rank_;
    int ncid_; // NetCDF file ID
};

} // namespace IO
} // namespace VVM

#endif // VVM_READERS_PNETCDFREADER_HPP
