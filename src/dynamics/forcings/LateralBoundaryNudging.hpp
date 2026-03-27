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
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_FORCINGS_LATERAL_BOUNDARY_NUDGING_HPP
