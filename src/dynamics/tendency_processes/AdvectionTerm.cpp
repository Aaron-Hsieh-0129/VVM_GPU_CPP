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
    // 2. 呼叫 scheme_->calculate_flux_divergence(...)
    // 3. 將結果累加到 tendencies 物件中
}

} // namespace Dynamics
} // namespace VVM
