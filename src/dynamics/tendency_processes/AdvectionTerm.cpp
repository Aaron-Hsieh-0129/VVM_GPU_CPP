#include "AdvectionTerm.hpp"

namespace VVM {
namespace Dynamics {

AdvectionTerm::AdvectionTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name)
    : scheme_(std::move(scheme)), variable_name_(std::move(var_name)) {}

AdvectionTerm::~AdvectionTerm() = default;


void AdvectionTerm::compute_tendency(
    const Core::State& state, 
    Core::State& tendencies,
    const Core::Grid& grid,
    const Core::Parameters& params) const {
    
    // 1. 從 state 中取得需要的場
    // 這裡以 variable_name_ (例如 "th") 作為要計算傾向的變數
    const auto& scalar_field = state.get_field<3>(variable_name_);
    const auto& u_field = state.get_field<3>("u");
    const auto& w_field = state.get_field<3>("w");

    auto& target_tendency_field = tendencies.get_field<3>(variable_name_);

    std::cout << "Computed advection tendency for variable: " << variable_name_ << std::endl;
    scheme_->calculate_flux_divergence_x(
        scalar_field, u_field, grid, params, target_tendency_field
    );
}

} // namespace Dynamics
} // namespace VVM
