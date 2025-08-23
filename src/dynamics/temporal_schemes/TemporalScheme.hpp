#ifndef VVM_DYNAMICS_TEMPORAL_SCHEME_HPP
#define VVM_DYNAMICS_TEMPORAL_SCHEME_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "dynamics/tendency_processes/TendencyTerm.hpp"
#include <vector>
#include <memory>

namespace VVM {
namespace Dynamics {

class TemporalScheme {
public:
    virtual ~TemporalScheme() = default;

    virtual void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        double dt
    ) const = 0;

    virtual std::vector<std::string> get_required_state_suffixes() const {
        return {};
    }
};

} // namespace Dynamics
} // namespace VVM
#endif
