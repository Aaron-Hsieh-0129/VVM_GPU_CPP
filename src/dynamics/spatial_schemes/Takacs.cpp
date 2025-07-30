#include "Takacs.hpp"

namespace VVM {
namespace Dynamics {

Core::Field<3> Takacs::calculate_flux_divergence_x(
    const Core::Field<3>& scalar, const Core::Field<3>& u, const Core::Field<3>& w,
    const Core::Grid& grid, const Core::Parameters& params) const {
    // Core::Field<3> flux("Flux", {params.nz, params.ny, params.nx});

    // std::cout << "Flux dim: " << flux.extent(0) << ", " << flux.extent(1) << ", " << flux.extent(2) << std::endl;
    
    return Core::Field<3>("empty_tendency", {1,1,1});
}

} // namespace Dynamics
} // namespace VVM
