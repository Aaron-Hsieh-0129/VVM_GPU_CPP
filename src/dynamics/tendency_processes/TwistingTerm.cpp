#include "TwistingTerm.hpp"

namespace VVM {
namespace Dynamics {

TwistingTerm::TwistingTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name)
    : scheme_(std::move(scheme)), variable_name_(std::move(var_name)) {}

TwistingTerm::~TwistingTerm() = default;

void TwistingTerm::compute_tendency(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency) const {
    
    if (variable_name_ == "xi") {
        scheme_->calculate_twisting_tendency_x(state, grid, params, out_tendency, variable_name_);
    } 
    else if (variable_name_ == "eta") {
        scheme_->calculate_twisting_tendency_y(state, grid, params, out_tendency, variable_name_);
    } 
    else if (variable_name_ == "zeta") {
        scheme_->calculate_twisting_tendency_z(state, grid, params, out_tendency, variable_name_);
    }
}

} // namespace Dynamics
} // namespace VVM
