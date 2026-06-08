#ifndef VVM_CORE_INITIALIZER_HPP
#define VVM_CORE_INITIALIZER_HPP

#include <memory>
#include "Grid.hpp"
#include "Parameters.hpp"
#include "State.hpp"
#include "utils/ConfigurationManager.hpp"
#include "io/Reader.hpp"
#include "HaloExchanger.hpp"

namespace VVM {
namespace Core {

class Initializer {
public:
    Initializer(const Utils::ConfigurationManager& config, 
                const Grid& grid, 
                Parameters& parameters, 
                State &state,
                HaloExchanger& halo_exchanger);
    void initialize_state() const;
    void initialize_grid() const;
    void initialize_topo() const;
    void initialize_poisson() const;
    void assign_vars() const;
    void initialize_perturbation() const;
    void initialize_zeta_factor_for_twisting() const;
    void load_restart() const;
    VVM::Real get_restart_time_from_filename(const std::string& source_file) const;

private:
    std::unique_ptr<VVM::IO::Reader> reader_;
    std::unique_ptr<VVM::IO::Reader> pnetcdf_reader_;
    std::unique_ptr<VVM::IO::Reader> restart_reader_;

    const Utils::ConfigurationManager& config_;
    const Grid& grid_;
    Parameters& parameters_;
    State& state_;
    Core::HaloExchanger& halo_exchanger_;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_INITIALIZER_HPP
