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
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"

namespace VVM {
namespace Physics {

class SurfaceProcess {
public:

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

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    Core::HaloExchanger& halo_exchanger_;

    static KOKKOS_INLINE_FUNCTION
    void sflux_2d(double sigmau, double thvm, double thvsm, double speed1, 
                  double zr, double zrough, 
                  double& ustar, double ventfc[2], double& molen);

    static KOKKOS_INLINE_FUNCTION
    double compute_es(double t);
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_SURFACE_PROCESS_HPP
