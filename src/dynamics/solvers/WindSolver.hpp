#ifndef VVM_DYNAMICS_WIND_SOLVER_HPP
#define VVM_DYNAMICS_WIND_SOLVER_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include "utils/ConfigurationManager.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/HaloExchanger.hpp"
#include <memory>

namespace VVM {
namespace Dynamics {

class WindSolver {
public:
    WindSolver(const Core::Grid& grid, const Utils::ConfigurationManager& config);

    void solve_w(Core::State& state, const Core::Parameters& params) const;

private:
    const Core::Grid& grid_;
    const Utils::ConfigurationManager& config_;
    mutable Core::HaloExchanger halo_exchanger_;

    mutable Core::Field<3> YTEM_field_;
    mutable Core::Field<3> W3DNP1_field_;
    mutable Core::Field<3> W3DN_field_;
    mutable Core::Field<3> RHSV_field_;
    mutable Core::Field<3> pm_temp_field_;
    mutable Core::Field<3> pm_field_;

    void solve_poisson_2d(const Core::Field<3>& source, Core::Field<3>& result) const;
};

} // namespace Dynamics
} // namespace VVM

#endif // VVM_DYNAMICS_WIND_SOLVER_HPP
