#ifndef VVM_DYNAMICS_SPATIAL_SCHEME_HPP
#define VVM_DYNAMICS_SPATIAL_SCHEME_HPP

#include "core/Field.hpp"
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include <string>

namespace VVM {
namespace Dynamics {

class SpatialScheme {
public:
    virtual ~SpatialScheme() = default;

    virtual Core::Field<3> calculate_flux_divergence(
        const Core::Field<3>& scalar, const Core::Field<3>& u, const Core::Field<3>& w,
        const Core::Grid& grid, const Core::Parameters& params) const = 0;
};

} // namespace Dynamics
} // namespace VVM
#endif
