#pragma once
#include "dynamics/DynamicalCore.hpp"
#include "physics/p3/VVM_p3_process_interface.hpp"
#include "physics/rrtmgp/VVM_rrtmgp_process_interface.hpp"
#include "physics/turbulence/TurbulenceProcess.hpp"
#include "physics/surface/SurfaceProcess.hpp"
#include "physics/land/LandProcess.hpp"
#include "core/Initializer.hpp"
#include "core/BoundaryConditionManager.hpp"
#include "core/vvm_types.hpp"
#include "dynamics/temporal_schemes/TimeIntegrator.hpp"
#include "dynamics/forcings/SpongeLayer.hpp"
#include "dynamics/forcings/RandomForcing.hpp"
#include "dynamics/forcings/LateralBoundaryNudging.hpp"
#include "dynamics/forcings/AreaMeanNudging.hpp"
#include <set>

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
    void run_step(VVM::Real dt);
    void finalize();

private:
    const Utils::ConfigurationManager& config_;
    Core::Parameters& params_;
    const Core::Grid& grid_;
    Core::HaloExchanger& halo_exchanger_;
    Core::BoundaryConditionManager bc_manager_;
    std::vector<std::string> dynamics_vars_;
    std::vector<std::string> thermodynamics_vars_;
    std::vector<std::string> sfc_thermodynamics_vars_;
    std::vector<std::string> sfc_dynamics_vars_;

    Core::State& state_;

    std::unique_ptr<Dynamics::DynamicalCore> dycore_;
    std::unique_ptr<Physics::VVM_P3_Interface> microphysics_;
    std::unique_ptr<Physics::TurbulenceProcess> turbulence_;
    std::unique_ptr<Physics::SurfaceProcess> surface_;
    std::unique_ptr<Physics::RRTMGP::RRTMGPRadiation> radiation_;
    std::unique_ptr<Dynamics::SpongeLayer> sponge_layer_;
    std::unique_ptr<Dynamics::RandomForcing> random_forcing_;
    std::unique_ptr<Dynamics::LateralBoundaryNudging> lateral_boundary_nudging_;
    std::unique_ptr<Physics::LandProcess> land_;
    std::unique_ptr<Dynamics::AreaMeanNudging> area_mean_nudging_;

    int rad_freq_in_steps_;
    int surface_process_steps_;
    double surface_process_s_;
    // int surface_freq_in_steps_;
    // int land_freq_in_steps_;

    bool wind_solver_ = true;
    bool enable_surface_process_ = false;

    VVM::Real uvtau_;
    bool predict_uvtopmn_ = true;
};

}
}
