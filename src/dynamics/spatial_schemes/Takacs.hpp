#ifndef VVM_DYNAMICS_TAKACS_HPP
#define VVM_DYNAMICS_TAKACS_HPP

#include "SpatialScheme.hpp"

namespace VVM {
namespace Dynamics {

class Takacs : public SpatialScheme {
public:
    void calculate_flux_convergence_x(
        const Core::Field<3>& scalar, const Core::Field<3>& u,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_flux_convergence_y(
        const Core::Field<3>& scalar, const Core::Field<3>& v,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_flux_convergence_z(
        const Core::Field<3>& scalar, const Core::Field<1>& rhobar_up_field, const Core::Field<3>& w,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
};

} // namespace Dynamics
} // namespace VVM
#endif
