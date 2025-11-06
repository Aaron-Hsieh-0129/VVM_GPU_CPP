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
    Kokkos::View<double> gravity;
    Kokkos::View<double> Rd;
    Kokkos::View<double> PSFC;
    Kokkos::View<double> P0;
    Kokkos::View<double> Cp;
    Kokkos::View<double> Lv;
    int solver_iteration;

    // Grid-derived parameters
    Kokkos::View<double> nx;
    Kokkos::View<double> ny;
    Kokkos::View<double> nz;
    Kokkos::View<double> dx;
    Kokkos::View<double> dy;
    Kokkos::View<double> dz;
    Kokkos::View<double> dt;
    Kokkos::View<double> rdx;
    Kokkos::View<double> rdy;
    Kokkos::View<double> rdz;
    Kokkos::View<double> rdx2;
    Kokkos::View<double> rdy2;
    Kokkos::View<double> rdz2;
    Kokkos::View<double> WRXMU;
    int max_topo_idx;

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

    double get_value_host(const Kokkos::View<double>& device_view) const;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_PARAMETERS_HPP
