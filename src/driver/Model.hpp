#pragma once
#include "dynamics/DynamicalCore.hpp"
#include "physics/p3/VVM_p3_process_interface.hpp"
#include "physics/rrtmgp/VVM_rrtmgp_process_interface.hpp"
// #include "physics/turbulence/TurbulenceProcess.hpp"
#include "core/Initializer.hpp"

namespace VVM {
namespace Driver {

class Model {
public:
    Model(const Utils::ConfigurationManager& config,
          Core::Parameters& params,
          const Core::Grid& grid,
          Core::State& state,
          Core::HaloExchanger& halo_exchanger);

    void init();
    void run_step(double dt);
    void finalize();

private:
    const Utils::ConfigurationManager& config_;
    Core::Parameters& params_;
    const Core::Grid& grid_;
    Core::HaloExchanger& halo_exchanger_;

    Core::State& state_;

    std::unique_ptr<Dynamics::DynamicalCore> dycore_;
    std::unique_ptr<Physics::VVM_P3_Interface> microphysics_;
    // std::unique_ptr<Physics::TurbulenceProcess> turbulence_;
    std::unique_ptr<Physics::RRTMGP::RRTMGPRadiation> radiation_;

    int rad_freq_in_steps_;
};

}
}
