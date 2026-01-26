#pragma once
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include <Kokkos_Random.hpp>

namespace VVM {
namespace Dynamics {

class RandomForcing {
public:
    RandomForcing(const Utils::ConfigurationManager& config, 
                  const Core::Grid& grid,
                  const Core::Parameters& params);

    void initialize(Core::State& state);
    void apply(Core::State& state);

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    bool enabled_;
    double end_time_;
    double amplitude_;
    int k_start_;
    int k_end_;
    int seed_;
};

} // namespace Dynamics
} // namespace VVM
