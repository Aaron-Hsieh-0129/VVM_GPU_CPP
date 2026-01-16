#include "SurfaceProcess.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace VVM {
namespace Physics {

SurfaceProcess::SurfaceProcess(const Utils::ConfigurationManager& config, 
                                     const Core::Grid& grid, 
                                     const Core::Parameters& params,
                                     Core::HaloExchanger& halo_exchanger,
                                     Core::State& state)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger) {

    return;
}

void SurfaceProcess::initialize(Core::State& state) {
    return;
}


void SurfaceProcess::compute_coefficients(Core::State& state) {
    return;
}

} // namespace Physics
} // namespace VVM
