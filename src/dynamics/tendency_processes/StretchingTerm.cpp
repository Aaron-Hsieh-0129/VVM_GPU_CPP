#include "StretchingTerm.hpp"

namespace VVM {
namespace Dynamics {

StretchingTerm::StretchingTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name)
    : scheme_(std::move(scheme)), variable_name_(std::move(var_name)) {}

StretchingTerm::~StretchingTerm() = default;

void StretchingTerm::compute_tendency(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency) const {
    
    if (variable_name_ == "xi") {
        scheme_->calculate_stretching_tendency_x(state, grid, params, out_tendency, variable_name_);
    } 
    else if (variable_name_ == "eta") {
        scheme_->calculate_stretching_tendency_y(state, grid, params, out_tendency, variable_name_);
    } 
    else if (variable_name_ == "zeta") {
        scheme_->calculate_stretching_tendency_z(state, grid, params, out_tendency, variable_name_);
    }
}

} // namespace Dynamics
} // namespace VVM
