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

namespace VVM {
namespace Dynamics {

class DynamicalCore {
public:
    DynamicalCore(const Utils::ConfigurationManager& config, 
                  const Core::Grid& grid, 
                  const Core::Parameters& params,
                  Core::State& state);
    ~DynamicalCore();

    void compute_diagnostic_fields() const;
    void step(Core::State& state, double dt);

private:
    const Utils::ConfigurationManager& config_;
    Core::State& state_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    
    std::map<std::string, std::unique_ptr<TemporalScheme>> variable_schemes_;
    std::vector<std::string> prognostic_variables_;


    std::unique_ptr<TemporalScheme> create_temporal_scheme(
        const std::string& var_name, 
        const nlohmann::json& var_config) const;
};

} // namespace Dynamics
} // namespace VVM
#endif
