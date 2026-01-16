#ifndef VVM_PHYSICS_SURFACE_PROCESS_HPP
#define VVM_PHYSICS_SURFACE_PROCESS_HPP

#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Field.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Physics {

    SurfaceProcess(const Utils::ConfigurationManager& config, 
                      const Core::Grid& grid, 
                      const Core::Parameters& params,
                      Core::HaloExchanger& halo_exchanger,
                      Core::State& state);

    void initialize(Core::State& state);
    void compute_coefficients(Core::State& state);

    template<size_t Dim>
    void calculate_tendencies(Core::State& state, 
                              const std::string& var_name, 
                              Core::Field<Dim>& out_tendency);


} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_SURFACE_PROCESS_HPP
