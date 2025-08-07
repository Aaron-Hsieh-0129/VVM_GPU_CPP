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

    // 1D Kokkos Views
    Field<1> z_mid;
    Field<1> z_up;
    Field<1> flex_height_coef_mid;
    Field<1> flex_height_coef_up;
    Field<1> dz_mid;
    Field<1> dz_up;
    Field<1> fact1_xi_eta;
    Field<1> fact2_xi_eta;

    Parameters(const Utils::ConfigurationManager& config, const Grid& grid);

    double get_value_host(const Kokkos::View<double>& device_view) const;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_PARAMETERS_HPP
