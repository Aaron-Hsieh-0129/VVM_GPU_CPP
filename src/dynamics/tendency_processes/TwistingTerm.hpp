#ifndef VVM_DYNAMICS_TWISTING_TERM_HPP
#define VVM_DYNAMICS_TWISTING_TERM_HPP

#include "TendencyTerm.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include <string>

namespace VVM {
namespace Dynamics {

class TwistingTerm : public TendencyTerm {
public:
    explicit TwistingTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name);
    ~TwistingTerm() override;

    void compute_tendency(
        const Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        Core::Field<3>& out_tendency) const override;

private:
    std::unique_ptr<SpatialScheme> scheme_;
    std::string variable_name_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_TWISTING_TERM_HPP
