#include "Model.hpp"

namespace VVM {
namespace Driver {

Model::Model(const Utils::ConfigurationManager& config,
             Core::Parameters& params,
             const Core::Grid& grid,
             Core::State& state,
             Core::HaloExchanger& halo_exchanger)
    : config_(config),
      params_(params),
      grid_(grid),
      state_(state),
      halo_exchanger_(halo_exchanger), bc_manager_(grid)
{
    dycore_ = std::make_unique<Dynamics::DynamicalCore>(config_, grid_, params_, state_, halo_exchanger_);
    if (config_.get_value<bool>("physics.p3.enable_p3", false)) {
        microphysics_ = std::make_unique<Physics::VVM_P3_Interface>(config_, grid_, params_, halo_exchanger_);
    }

    if (config_.get_value<bool>("physics.turbulence.enable_turbulence", false)) {
        turbulence_ = std::make_unique<Physics::TurbulenceProcess>(config_, grid_, params_, halo_exchanger_, state_);
    }

    if (config_.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
        radiation_ = std::make_unique<Physics::RRTMGP::RRTMGPRadiation>(config_, grid_, params_);
        rad_freq_in_steps_ = config_.get_value<int>("physics.rrtmgp.rad_frequency_step", 1);
    }
}

void Model::init() {
    int rank = grid_.get_mpi_rank();
    if (rank == 0) std::cout << "\n=== Initializing VVM Model ===" << std::endl;

    if (rank == 0) std::cout << "Loading Initial Conditions..." << std::endl;
    Core::Initializer initializer(config_, grid_, params_, state_, halo_exchanger_);
    initializer.initialize_state();

    if (microphysics_) microphysics_->initialize(state_);
    if (turbulence_) turbulence_->initialize(state_);
    if (radiation_) radiation_->initialize(state_);
    
    halo_exchanger_.exchange_halos(state_);
    
    if (rank == 0) std::cout << "=== Model Initialization Complete ===\n" << std::endl;
}

void Model::run_step(double dt) {
    size_t current_step = state_.get_step();
    // Caculate tendencies of thermodynamics variables
    dycore_->calculate_thermo_tendencies();

    // Calculate radiation based on t
    if (radiation_) {
        // Update net heating used for calculating th tendency
        // FIXME: If the grid size (nx, ny) can't be divided by core number, it will cause kokkos copy errors here.
        if (state_.get_step() % rad_freq_in_steps_ == 0) {
            radiation_->run(state_, dt); 
        }
        
        // Update forward th tendency
        radiation_->apply_heating(state_);
    }

    // Update thermodynamics variables using tendencies above
    dycore_->update_thermodynamics(dt);

    // P3 Microphysics based on (t+1) thermodynamics variables
    if (microphysics_) {
        microphysics_->run(state_, dt);
    }

    // Turbulence diffusion on thermodynamics variables
    if (turbulence_) {
        turbulence_->compute_coefficients(state_, dt);
        for (const auto& var_name : turbulence_->get_thermodynamics_vars()) {
            std::string fe_name = "fe_tendency_" + var_name;

            if (!state_.has_field(fe_name)) {
                std::cout << "Error: fe_tendency_" << var_name << " doesn't exist" << std::endl;
                exit(1);
            }

            auto& fe_tend_field = state_.get_field<3>(fe_name);
            fe_tend_field.set_to_zero(); 
            turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);

            VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            halo_exchanger_.exchange_halos(state_.get_field<3>(var_name));
            if (var_name == "th" || var_name == "qv") {
                 bc_manager_.apply_zero_gradient(state_.get_field<3>(var_name));
            }
            else {
                bc_manager_.apply_zero_gradient_bottom_zero_top(state_.get_field<3>(var_name));
            }
        }
    }

    // Calculate buoyancy based on thermodynamics variables at t+1
    // dycore_->update_buoyancy_term(state_);
    // This is included in calculate vorticity tendencies 

    // Caulcate vorticity tendencies using variables at t 
    dycore_->calculate_vorticity_tendencies();
    // Update vorticity to t+1
    dycore_->update_vorticity(dt);

    // Vorticity diffusion
    if (turbulence_) {
        for (const auto& var_name : turbulence_->get_dynamics_vars()) {
            std::string fe_name = "fe_tendency_" + var_name;
            
            if (!state_.has_field(fe_name)) {
                std::cout << "Error: fe_tendency_" << var_name << " doesn't exist" << std::endl;
                exit(1);
            }

            if (var_name == "zeta") {
                auto& fe_tend_field = state_.get_field<2>(fe_name);
                fe_tend_field.set_to_zero(); 
                turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            }
            else {
                auto& fe_tend_field = state_.get_field<3>(fe_name);
                fe_tend_field.set_to_zero(); 
                turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            }
            halo_exchanger_.exchange_halos(state_.get_field<3>(var_name));
            bc_manager_.apply_vorticity_bc(state_.get_field<3>(var_name));
        }
        dycore_->compute_zeta_vertical_structure(state_);
    }
    dycore_->diagnose_wind_fields(state_);
}

void Model::finalize() {
    if (microphysics_) microphysics_->finalize();
    if (radiation_) radiation_->finalize();
}

}
}
