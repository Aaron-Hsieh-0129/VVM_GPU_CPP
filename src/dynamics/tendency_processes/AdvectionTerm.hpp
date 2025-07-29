#ifndef VVM_DYNAMICS_ADVECTION_TERM_HPP
#define VVM_DYNAMICS_ADVECTION_TERM_HPP

#include "TendencyTerm.hpp"
#include "../spatial_schemes/SpatialScheme.hpp" // 注意路徑
#include <memory>
#include <string>

namespace VVM {
namespace Dynamics {

class AdvectionTerm : public TendencyTerm {
public:
    AdvectionTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name);
    ~AdvectionTerm() override;

    void compute_tendency(
        const Core::State& state, 
        Core::State& tendencies, 
        const Core::Grid& grid,
        const Core::Parameters& params) const override;
private:
    std::unique_ptr<SpatialScheme> scheme_;
    std::string variable_name_;
};

} // namespace Dynamics
} // namespace VVM
#endif
