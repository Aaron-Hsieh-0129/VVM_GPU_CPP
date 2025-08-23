#ifndef VVM_DYNAMICS_TIME_INTEGRATOR_HPP
#define VVM_DYNAMICS_TIME_INTEGRATOR_HPP

#include "TemporalScheme.hpp"
#include <string>
#include <vector>

namespace VVM {
namespace Dynamics {

class TimeIntegrator : public TemporalScheme {
public:
    explicit TimeIntegrator(std::string var_name, bool has_ab2, bool has_fe);
    ~TimeIntegrator() override;

    void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        double dt
    ) const override;

    std::vector<std::string> get_required_state_suffixes() const override {
        // Only AB2 needs a previous state (_m)
        return has_ab2_terms_ ? std::vector<std::string>{"_m"} : std::vector<std::string>{};
    }

private:
    std::string variable_name_;
    bool has_ab2_terms_;
    bool has_fe_terms_;
    mutable size_t time_step_count_ = 0;
};

} // namespace Dynamics
} // namespace VVM
#endif
