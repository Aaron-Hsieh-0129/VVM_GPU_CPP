#ifndef VVM_CORE_PARAMETERS_HPP
#define VVM_CORE_PARAMETERS_HPP

#include "Grid.hpp"
#include "utils/ConfigurationManager.hpp"
#include "Field.hpp"

namespace VVM {
namespace Core {

// This class holds all the constant parameters for the simulation.
// It is initialized once and then passed by const reference to where it's needed.
class Parameters {
public:
    // Physical constants
    Kokkos::View<VVM::Real> gravity;
    Kokkos::View<VVM::Real> Rd;
    Kokkos::View<VVM::Real> PSFC;
    Kokkos::View<VVM::Real> P0;
    Kokkos::View<VVM::Real> Cp;
    Kokkos::View<VVM::Real> Lv;
    int solver_iteration;

    // Grid-derived parameters
    Kokkos::View<VVM::Real> nx;
    Kokkos::View<VVM::Real> ny;
    Kokkos::View<VVM::Real> nz;
    Kokkos::View<VVM::Real> dx;
    Kokkos::View<VVM::Real> dy;
    Kokkos::View<VVM::Real> dz;
    Kokkos::View<VVM::Real> dt;
    Kokkos::View<VVM::Real> rdx;
    Kokkos::View<VVM::Real> rdy;
    Kokkos::View<VVM::Real> rdz;
    Kokkos::View<VVM::Real> rdx2;
    Kokkos::View<VVM::Real> rdy2;
    Kokkos::View<VVM::Real> rdz2;
    Kokkos::View<VVM::Real> WRXMU;
    int max_topo_idx;

    Field<0> fact1_zeta;
    Field<0> fact2_zeta;

    // 1D Kokkos Views
    Field<1> z_mid;
    Field<1> z_up;
    Field<1> flex_height_coef_mid;
    Field<1> flex_height_coef_up;
    Field<1> dz_mid;
    Field<1> dz_up;
    Field<1> fact1_xi_eta;
    Field<1> fact2_xi_eta;
    Field<1> AGAU;
    Field<1> BGAU;
    Field<1> CGAU;
    Field<1> bn_new;
    Field<1> cn_new;

    Parameters(const Utils::ConfigurationManager& config, const Grid& grid);

    VVM::Real get_value_host(const Kokkos::View<VVM::Real>& device_view) const;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_PARAMETERS_HPP
