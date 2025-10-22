#ifndef VVM_DYNAMICS_DYNAMICAL_CORE_HPP
#define VVM_DYNAMICS_DYNAMICAL_CORE_HPP

#include <map>
#include <string>
#include <memory>
#include <vector>
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include "temporal_schemes/TemporalScheme.hpp"
#include "tendency_processes/TendencyCalculator.hpp"
#include "solvers/WindSolver.hpp"

namespace VVM {
namespace Dynamics {

struct IntegrationStep {
    int step;
    std::string description;
    std::vector<std::string> vars_to_calculate_tendency;
    std::vector<std::string> vars_to_update;
};

class DynamicalCore {
public:
    DynamicalCore(const Utils::ConfigurationManager& config, 
                  const Core::Grid& grid, 
                  const Core::Parameters& params,
                  Core::State& state);
    ~DynamicalCore();

    void compute_diagnostic_fields() const;
    void compute_zeta_vertical_structure(Core::State& state) const;
    void compute_uvtopmn();
    void step(Core::State& state, double dt);

    mutable size_t time_step_count = 0;

private:
    const Utils::ConfigurationManager& config_;
    Core::State& state_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    
    std::map<std::string, std::unique_ptr<TendencyCalculator>> tendency_calculators_;
    std::map<std::string, std::unique_ptr<TemporalScheme>> time_integrators_;
    std::vector<IntegrationStep> integration_procedure_;

    std::unique_ptr<WindSolver> wind_solver_;
    void compute_wind_fields();
};

} // namespace Dynamics
} // namespace VVM
#endif
