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

    // ∇⋅(ρvφ)
    virtual void calculate_flux_convergence_x(
        const Core::Field<3>& scalar, const Core::Field<3>& u,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;
    virtual void calculate_flux_convergence_y(
        const Core::Field<3>& scalar, const Core::Field<3>& v,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;
    virtual void calculate_flux_convergence_z(
        const Core::Field<3>& scalar, const Core::Field<1>& rhobar_divide_field, const Core::Field<3>& w,
        const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;

    // ρω∇⋅(v)
    virtual void calculate_stretching_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;
    virtual void calculate_stretching_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;
    virtual void calculate_stretching_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;

    // Twisting Term
    // 0.5ρ(eta*Rzeta + zeta*Reta)
    virtual void calculate_twisting_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;
    // 0.5ρ(xi*Rzeta + zeta*Rxi)
    virtual void calculate_twisting_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;
    // 0.5ρ(xi*Reta + eta*Rxi)
    virtual void calculate_twisting_tendency_z(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const = 0;

    // Rotation (Diagnostic)
    virtual void calculate_R_xi(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_xi) const = 0;
    virtual void calculate_R_eta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_eta) const = 0;
    virtual void calculate_R_zeta(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_R_zeta) const = 0;

    // Vorticity divergence
    virtual void calculate_vorticity_divergence(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_field) const = 0;

    // Buoyancy Term
    virtual void calculate_buoyancy_tendency_x(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const = 0;
    virtual void calculate_buoyancy_tendency_y(
        const Core::State& state, const Core::Grid& grid,
        const Core::Parameters& params, Core::Field<3>& out_tendency) const = 0;
};

} // namespace Dynamics
} // namespace VVM
#endif
