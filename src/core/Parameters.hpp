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
    const double gravity;
    const double Rd;

    // Grid-derived parameters
    const double rdx;
    const double rdy;
    const double rdz;
    const double rdx2;
    const double rdy2;
    const double rdz2;

    // 1D Kokkos Views
    Field<1> z_mid;
    Field<1> z_up;
    Field<1> flex_height_coef_mid;
    Field<1> flex_height_coef_up;
    Field<1> dz_mid;
    Field<1> dz_up;

    // This is put on CPU side only
    Parameters(const Utils::ConfigurationManager& config, const Grid& grid) : 
        gravity(config.get_value<double>("constants.gravity")),
        Rd(config.get_value<double>("constants.Rd")),
        rdx(1.0 / config.get_value<double>("grid.dx")),
        rdy(1.0 / config.get_value<double>("grid.dy")),
        rdz(1.0 / config.get_value<double>("grid.dz")),
        rdx2(rdx * rdx), rdy2(rdy * rdy), rdz2(rdz * rdz),
        z_mid("z_mid", {grid.get_local_total_points_z()}),
        z_up("z_up", {grid.get_local_total_points_z()}),
        flex_height_coef_mid("flex_height_coef_mid", {grid.get_local_total_points_z()}),
        flex_height_coef_up("flex_height_coef_up", {grid.get_local_total_points_z()}),
        dz_mid("dz_mid", {grid.get_local_total_points_z()}),
        dz_up("dz_up", {grid.get_local_total_points_z()})
    {

    }
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_PARAMETERS_HPP
