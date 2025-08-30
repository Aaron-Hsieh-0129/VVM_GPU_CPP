#ifndef VVM_DYNAMICS_WIND_SOLVER_HPP
#define VVM_DYNAMICS_WIND_SOLVER_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include "utils/ConfigurationManager.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/HaloExchanger.hpp"
#include <memory>

namespace VVM {
namespace Dynamics {

class WindSolver {
public:
    WindSolver(const Core::Grid& grid, const Utils::ConfigurationManager& config, const Core::Parameters& params);

    void solve_w(Core::State& state);
    void solve_uv(Core::State& state);
    void relax_2d(Core::Field<2>& A_field, Core::Field<2>& ANM1_field, Core::Field<2>& RHSV_field, Core::Field<2>& AOUT_field);

private:
    const Core::Grid& grid_;
    const Utils::ConfigurationManager& config_;
    mutable Core::HaloExchanger halo_exchanger_;
    const Core::Parameters& params_;

    mutable Core::Field<3> YTEM_field_;
    mutable Core::Field<3> W3DNP1_field_;
    mutable Core::Field<3> W3DN_field_;
    mutable Core::Field<3> RHSV_field_;
    mutable Core::Field<3> pm_temp_field_;
    mutable Core::Field<3> pm_field_;

    mutable Core::Field<2> RIP1_field_;
    mutable Core::Field<2> ROP1_field_;
    mutable Core::Field<2> RIP2_field_;
    mutable Core::Field<2> ROP2_field_;
    mutable Core::Field<2> ATEMP_field_;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_WIND_SOLVER_HPP
