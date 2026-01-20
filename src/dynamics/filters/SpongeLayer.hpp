#ifndef VVM_PHYSICS_SPONGE_LAYER_HPP
#define VVM_PHYSICS_SPONGE_LAYER_HPP

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

class SpongeLayer {
public:
    SpongeLayer(const Utils::ConfigurationManager& config, 
                const Core::Grid& grid, 
                const Core::Parameters& params,
                Core::HaloExchanger& halo_exchanger,
                Core::State& state);

    template<size_t Dim>
    void calculate_tendencies(Core::State& state, 
                              const std::string& var_name, 
                              Core::Field<Dim>& out_tendency);

    void initialize(Core::State& state);

    const std::vector<std::string>& get_thermodynamics_vars() const { return thermodynamics_vars_; }
    const std::vector<std::string>& get_dynamics_vars() const { return dynamics_vars_; }

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    Core::HaloExchanger& halo_exchanger_;

    std::vector<std::string> dynamics_vars_;
    std::vector<std::string> thermodynamics_vars_;

    double CRAD_;
    int k_start_thermo_;
    int k_start_vort_;
    Kokkos::View<const double*> ref_profile_;
    double sponge_base_height_;
    double max_damping_coeff_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_PHYSICS_SPONGE_LAYER_HPP
