#include "BuoyancyTerm.hpp"

namespace VVM {
namespace Dynamics {

BuoyancyTerm::BuoyancyTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name, VVM::Core::HaloExchanger& halo_exchanger)
    : scheme_(std::move(scheme)), variable_name_(std::move(var_name)), halo_exchanger_(halo_exchanger) {}

BuoyancyTerm::~BuoyancyTerm() = default;

void BuoyancyTerm::compute_tendency(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency) const {
    
    if (variable_name_ == "xi") {
        scheme_->calculate_buoyancy_tendency_x(state, grid, params, out_tendency);
    } 
    else if (variable_name_ == "eta") {
        scheme_->calculate_buoyancy_tendency_y(state, grid, params, out_tendency);
    }
}

} // namespace Dynamics
} // namespace VVM
