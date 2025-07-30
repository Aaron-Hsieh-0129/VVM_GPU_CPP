#ifndef VVM_CORE_PARAMETERS_HPP
#define VVM_CORE_PARAMETERS_HPP

#include "Grid.hpp"
#include "utils/ConfigurationManager.hpp"
#include "Field.hpp"

namespace VVM {
namespace Core {

// This struct holds all the constant parameters for the simulation.
// It is initialized once and then passed by const reference to where it's needed.
struct Parameters {
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

    Parameters(const Utils::ConfigurationManager& config, const Grid& grid) :
        gravity("gravity"),
        Rd("Rd"),
        z_mid("z_mid", {grid.get_local_total_points_z()}),
        z_up("z_up", {grid.get_local_total_points_z()}),
        flex_height_coef_mid("flex_height_coef_mid", {grid.get_local_total_points_z()}),
        flex_height_coef_up("flex_height_coef_up", {grid.get_local_total_points_z()}),
        dz_mid("dz_mid", {grid.get_local_total_points_z()}),
        dz_up("dz_up", {grid.get_local_total_points_z()})
    {
        Kokkos::deep_copy(gravity, config.get_value<double>("constants.gravity"));
        Kokkos::deep_copy(Rd, config.get_value<double>("constants.Rd"));
        
        std::cout << "nx test0: " << config.get_value<double>("grid.nx") << std::endl;
        Kokkos::deep_copy(nx, config.get_value<double>("grid.nx"));
        Kokkos::deep_copy(ny, config.get_value<double>("grid.ny"));
        Kokkos::deep_copy(nz, config.get_value<double>("grid.nz"));
        Kokkos::deep_copy(dx, config.get_value<double>("grid.dx"));
        Kokkos::deep_copy(dy, config.get_value<double>("grid.dy"));
        Kokkos::deep_copy(dz, config.get_value<double>("grid.dz"));
        Kokkos::deep_copy(rdx, 1./config.get_value<double>("grid.dx"));
        Kokkos::deep_copy(rdy, 1./config.get_value<double>("grid.dy"));
        Kokkos::deep_copy(rdz, 1./config.get_value<double>("grid.dz"));

        double nx_h = 0.;
        Kokkos::deep_copy(nx_h, nx);
        std::cout << "nx test: " << nx_h << std::endl;

        Kokkos::fence();
    }

    double get_value_host(const Kokkos::View<double>& device_view) const {
        double host_value;
        // 將設備端數據深層複製到主機變數
        Kokkos::deep_copy(host_value, device_view);
        return host_value;
    }
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_PARAMETERS_HPP
