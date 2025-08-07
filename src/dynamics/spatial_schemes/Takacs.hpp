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

    void calculate_stretching_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_stretching_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_stretching_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;

    void calculate_twisting_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_twisting_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;
    void calculate_twisting_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const override;

    void calculate_R_xi(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_xi) const override;
    void calculate_R_eta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_eta) const override;
    void calculate_R_zeta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_zeta) const override;
};

} // namespace Dynamics
} // namespace VVM
#endif
