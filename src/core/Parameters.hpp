#ifndef VVM_CORE_PARAMETERS_HPP
#define VVM_CORE_PARAMETERS_HPP

#include "Grid.hpp"
#include "../utils/ConfigurationManager.hpp"

namespace VVM {
namespace Core {

// This struct holds all the constant parameters for the simulation.
// It is initialized once and then passed by const reference to where it's needed.
struct ModelParameters {
    // Physical constants
    const double gravity;
    const double Rd;

    // Grid-derived parameters
    const double rdx;
    const double rdy;
    const double rdz;
    const double rdx2 = rdx * rdx;
    const double rdy2 = rdy * rdy;
    const double rdz2 = rdz * rdz;

    // This is put on CPU side only
    ModelParameters(const Grid& grid, const Utils::ConfigurationManager& config) : 
        gravity(config.get_value<double>("constants.gravity")),
        Rd(config.get_value<double>("constants.Rd")),
        rdx(1.0 / config.get_value<double>("grid.dx")),
        rdy(1.0 / config.get_value<double>("grid.dy")),
        rdz(1.0 / config.get_value<double>("grid.dz")),
        rdx2(rdx * rdx),
        rdy2(rdy * rdy),
        rdz2(rdz * rdz)
    {

    }
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_PARAMETERS_HPP