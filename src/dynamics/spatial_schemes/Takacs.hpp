#ifndef VVM_DYNAMICS_TAKACS_HPP
#define VVM_DYNAMICS_TAKACS_HPP

#include "SpatialScheme.hpp"

namespace VVM {
namespace Dynamics {

class Takacs : public SpatialScheme {
public:
    Core::Field<3> calculate_flux_divergence(
        const Core::Field<3>& scalar, const Core::Field<3>& u, const Core::Field<3>& w,
        const Core::Grid& grid, const Core::Parameters& params) const override;
};

} // namespace Dynamics
} // namespace VVM
#endif
