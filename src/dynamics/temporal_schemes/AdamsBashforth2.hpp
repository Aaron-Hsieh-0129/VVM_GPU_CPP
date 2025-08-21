#ifndef VVM_DYNAMICS_ADAMS_BASHFORTH_2_HPP
#define VVM_DYNAMICS_ADAMS_BASHFORTH_2_HPP

#include <string>
#include <vector>
#include <memory>
#include "TemporalScheme.hpp"

namespace VVM {
namespace Dynamics {

class AdamsBashforth2 : public TemporalScheme {
public:
    explicit AdamsBashforth2(std::string var_name, std::vector<std::unique_ptr<TendencyTerm>> terms);
    ~AdamsBashforth2() override;

    void calculate_tendency(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params
    ) const override;

    void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        double dt
    ) const override;

    // Add snapshot (can be previous and furture) for AB2 integration
    // Here, _m represents one previous step
    std::vector<std::string> get_required_state_suffixes() const override {
        return {"_m"};
    }


private:
    std::string variable_name_;
    std::vector<std::unique_ptr<TendencyTerm>> tendency_terms_;
    mutable size_t time_step_count_ = 0;
};

} // namespace Dynamics
} // namespace VVM
#endif
