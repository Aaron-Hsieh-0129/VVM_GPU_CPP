#ifndef VVM_DYNAMICS_FORCINGS_LATERAL_BOUNDARY_NUDGING_HPP
#define VVM_DYNAMICS_FORCINGS_LATERAL_BOUNDARY_NUDGING_HPP

#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Field.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Dynamics {

class LateralBoundaryNudging {
public:
    LateralBoundaryNudging(const Utils::ConfigurationManager& config, 
                           const Core::Grid& grid, 
                           const Core::Parameters& params,
                           Core::State& state);

    void initialize(Core::State& state);

    template<size_t Dim>
    void calculate_tendencies(Core::State& state, 
                              const std::string& var_name, 
                              Core::Field<Dim>& out_tendency) const;

    void update_large_scale_forcing(Core::State& state, double current_time);

    const std::vector<std::string>& get_target_vars() const { return target_vars_; }

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;

    std::vector<std::string> target_vars_;
    std::map<std::string, std::string> name_T1_;
    std::map<std::string, std::string> name_T2_;

    std::string data_dir_;
    bool time_varying_;
    std::string file_name_;
    std::string file_prefix_;

    double tau_b_;
    double inv_tau_b_;
    double sigma_;
    double xc_;
    bool   enable_;
    double offset_;
    double width_;
    double radius_;

    bool nudge_W_; // West  (x = 0)
    bool nudge_E_; // East  (x = xsize)
    bool nudge_S_; // South (y = 0)
    bool nudge_N_; // North (y = ysize)
    
    double time_T1_;
    double time_T2_;
    double update_interval_;

    void load_forcing_data(Core::State& state, const std::string& filepath, bool is_constant);
    void check_ncmpi_error(int status, const std::string& msg) const;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_FORCINGS_LATERAL_BOUNDARY_NUDGING_HPP
