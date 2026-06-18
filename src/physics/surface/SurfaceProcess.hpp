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
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"

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
    void sflux_2d(VVM::Real sigmau, VVM::Real thvm, VVM::Real thvsm, VVM::Real speed1, 
                  VVM::Real zr, VVM::Real zrough, VVM::Real speed1_filter,
                  VVM::Real& ustar, VVM::Real ventfc[2], VVM::Real& molen);

    static KOKKOS_INLINE_FUNCTION
    void sflux_tc_2d(VVM::Real sigmau, VVM::Real thvm, VVM::Real thvsm, VVM::Real speed1, 
                     VVM::Real zr, VVM::Real zrough, VVM::Real speed1_filter, 
                     VVM::Real& ustar, VVM::Real ventfc[2], VVM::Real& molen);

    static KOKKOS_INLINE_FUNCTION
    VVM::Real compute_es(VVM::Real t);

    std::string mode_;
    std::string land_scheme_;
    std::string v_coord_type_;
    VVM::Real speed1_filter_;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_SURFACE_PROCESS_HPP
