#include "Parameters.hpp"
#include <iostream>

namespace VVM {
namespace Core {

Parameters::Parameters(const Utils::ConfigurationManager& config, const Grid& grid) :
    gravity("gravity"),
    Rd("Rd"),
    dx("dx"),
    dy("dy"),
    dz("dz"),
    dt("dt"),
    rdx("rdx"),
    rdy("rdy"),
    rdz("rdz"),
    rdx2("rdx2"),
    rdy2("rdy2"),
    rdz2("rdz2"),
    WRXMU("WRXMU"),
    z_mid("z_mid", {grid.get_local_total_points_z()}),
    z_up("z_up", {grid.get_local_total_points_z()}),
    flex_height_coef_mid("flex_height_coef_mid", {grid.get_local_total_points_z()}),
    flex_height_coef_up("flex_height_coef_up", {grid.get_local_total_points_z()}),
    dz_mid("dz_mid", {grid.get_local_total_points_z()}),
    dz_up("dz_up", {grid.get_local_total_points_z()}),
    fact1_xi_eta("fact1_xi_eta", {grid.get_local_total_points_z()}),
    fact2_xi_eta("fact2_xi_eta", {grid.get_local_total_points_z()}),
    AGAU("AGAU", {grid.get_local_total_points_z()}),
    BGAU("BGAU", {grid.get_local_total_points_z()}),
    CGAU("CGAU", {grid.get_local_total_points_z()}),
    bn_new("bn_new", {grid.get_local_total_points_z()}),
    cn_new("cn_new", {grid.get_local_total_points_z()})
{
    Kokkos::deep_copy(gravity, config.get_value<double>("constants.gravity"));
    Kokkos::deep_copy(Rd, config.get_value<double>("constants.Rd"));

    double dx_val = config.get_value<double>("grid.dx");
    double dy_val = config.get_value<double>("grid.dy");
    double dz_val = config.get_value<double>("grid.dz");
    double dt_val = config.get_value<double>("simulation.dt_s");

    Kokkos::deep_copy(dx, dx_val);
    Kokkos::deep_copy(dy, dy_val);
    Kokkos::deep_copy(dz, dz_val);
    Kokkos::deep_copy(dt, dt_val);

    Kokkos::deep_copy(rdx, 1.0 / dx_val);
    Kokkos::deep_copy(rdy, 1.0 / dy_val);
    Kokkos::deep_copy(rdz, 1.0 / dz_val);
    
    Kokkos::deep_copy(rdx2, 1.0 / (dx_val * dx_val));
    Kokkos::deep_copy(rdy2, 1.0 / (dy_val * dy_val));
    Kokkos::deep_copy(rdz2, 1.0 / (dz_val * dz_val));

    double WRXMU_val = config.get_value<double>("dynamics.solver.WRXMU");
    Kokkos::deep_copy(WRXMU, WRXMU_val);

    solver_iteration = config.get_value<int>("dynamics.solver.iteration");

    Kokkos::fence();

    if (Kokkos::HostSpace::execution_space::concurrency() > 0) {
        double nx_h = 0.;
        Kokkos::deep_copy(nx_h, nx);
        std::cout << "Parameters initialized. nx test: " << nx_h << std::endl;
    }
}

double Parameters::get_value_host(const Kokkos::View<double>& device_view) const {
    double host_value;
    Kokkos::deep_copy(host_value, device_view);
    return host_value;
}

} // namespace Core
} // namespace VVM
