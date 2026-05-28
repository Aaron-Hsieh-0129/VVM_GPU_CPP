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
    std::string x_bc = config.get_value<std::string>("grid.boundary_condition.x", "periodic");
    std::string y_bc = config.get_value<std::string>("grid.boundary_condition.y", "periodic");
    bc_manager_.initialize_bc_types(x_bc, y_bc);
    VVM::Real dt_s = params_.get_value_host(params_.dt);
    VVM::Real epsilon = real(1e-6);

    std::string mode = config_.get_value<std::string>("simulation.idealized_test", "none");
    std::vector<std::string> no_solver_mode = {"advection_u", "advection_v", "advection_w", "stretching", "twisting"};
    auto it = std::find(no_solver_mode.begin(), no_solver_mode.end(), mode);
    if (it != no_solver_mode.end()) {
        wind_solver_ = false;
    }

    dycore_ = std::make_unique<Dynamics::DynamicalCore>(config_, grid_, params_, state_, halo_exchanger_, bc_manager_);
    if (config_.get_value<bool>("physics.p3.enable_p3", false)) {
        microphysics_ = std::make_unique<Physics::VVM_P3_Interface>(config_, grid_, params_, halo_exchanger_);
    }

    if (config_.get_value<bool>("physics.turbulence.enable_turbulence", false)) {
        turbulence_ = std::make_unique<Physics::TurbulenceProcess>(config_, grid_, params_, halo_exchanger_, state_);
    }

    if (config_.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
        radiation_ = std::make_unique<Physics::RRTMGP::RRTMGPRadiation>(config_, grid_, params_);

        VVM::Real rad_freq_s = config_.get_value<VVM::Real>("physics.rrtmgp.rad_frequency_s", 1.0);
        VVM::Real remainder = std::fmod(rad_freq_s, dt_s);

        if (remainder > epsilon && (dt_s - remainder) > epsilon) {
            throw std::runtime_error("Error: RRTMGP radiation calling frequency can't be evenly divided by dt.");
        }

        rad_freq_in_steps_ = static_cast<int>(std::round(rad_freq_s / dt_s));
    }

    if (config_.get_value<bool>("dynamics.forcings.sponge_layer.enable", false)) {
        sponge_layer_ = std::make_unique<Dynamics::SpongeLayer>(config_, grid_, params_, halo_exchanger_, state_);
    }

    if (config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.enable", false)) {
        lateral_boundary_nudging_ = std::make_unique<Dynamics::LateralBoundaryNudging>(config_, grid_, params_, state_);
    }

    uvtau_ = config.get_value<VVM::Real>("dynamics.forcings.areamn.uvtau", 0.0);
    if (config_.get_value<bool>("dynamics.forcings.areamn.enable", false)) {
        area_mean_nudging_ = std::make_unique<Dynamics::AreaMeanNudging>(config_, grid_, params_);
    }

    if (config_.get_value<bool>("dynamics.forcings.random_perturbation.enable", false)) {
        random_forcing_ = std::make_unique<Dynamics::RandomForcing>(config_, grid_, params_);
    }
    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv"};
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        thermodynamics_vars_.insert(thermodynamics_vars_.end(), {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"});
    }

    sfc_thermodynamics_vars_ = {"th", "qv"};
    sfc_dynamics_vars_ = {"xi", "eta"};
    enable_surface_process_ = config.get_value<bool>("physics.surface_process.enable", false);
    std::string land_scheme  = config.get_value<std::string>("physics.surface_process.land_scheme", "none");
    std::string ocean_scheme = config.get_value<std::string>("physics.surface_process.ocean_scheme", "none");
    if (enable_surface_process_) {
        surface_ = std::make_unique<Physics::SurfaceProcess>(config_, grid_, params_, halo_exchanger_, state_);

        if (land_scheme == "noahlsm") {
            land_ = std::make_unique<Physics::LandProcess>(config_, grid_, params_, halo_exchanger_, state_, ocean_scheme);
        }

        VVM::Real surface_process_s = config_.get_value<VVM::Real>("physics.surface_process.frequency_s", 1);

        VVM::Real remainder = std::fmod(surface_process_s, dt_s);

        if (remainder > epsilon && (dt_s - remainder) > epsilon) {
            throw std::runtime_error("Error: surface process calling frequency can't be evenly divided by dt.");
        }

        surface_process_steps_ = static_cast<int>(std::round(surface_process_s / dt_s));
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
    if (sponge_layer_) sponge_layer_->initialize(state_);
    if (lateral_boundary_nudging_) lateral_boundary_nudging_->initialize(state_);
    if (area_mean_nudging_) area_mean_nudging_->initialize(state_);
    if (surface_) surface_->initialize(state_);
    if (land_) land_->init();
    if (random_forcing_) random_forcing_->initialize(state_);
    
    if (rank == 0) std::cout << "=== Model Initialization Complete ===\n" << std::endl;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    if (!state_.has_field("th_perturb")) state_.add_field<3>("th_perturb", {nz, ny, nx});

    if (area_mean_nudging_ && uvtau_ == 0.0) {
        predict_uvtopmn_ = false;
    }

    dycore_->compute_diagnostic_fields();
}

void Model::run_step(VVM::Real dt) {
    size_t current_step = state_.get_step();
    VVM::Real current_time = state_.get_time();

    if (lateral_boundary_nudging_) {
        lateral_boundary_nudging_->update_large_scale_forcing(state_, current_time);
    }

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
        // The effects of radiation is updated in update_thermodynamics
        radiation_->calculate_tendencies(state_);
    }

    // Update thermodynamics variables using tendencies above
    dycore_->update_thermodynamics(dt);

    if (random_forcing_) {
        random_forcing_->apply(state_);
    }

    // P3 Microphysics based on (t+1) thermodynamics variables
    if (microphysics_) {
        microphysics_->run(state_, dt);
    }

    // Turbulence diffusion on thermodynamics variables
    if (turbulence_) {
        turbulence_->compute_coefficients(state_, dt);
        for (const auto& var_name : turbulence_->get_thermodynamics_vars()) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            fe_tend_field.set_to_zero(); 
            turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);
        }
    }

    // Surface process (sea/land/ice)
    if (enable_surface_process_) {
        bool is_compute_step = (state_.get_step()-1) % surface_process_steps_ == 0;
        if (is_compute_step) {
            // NOTE: Even the configuration specified tco_ocean model which is not from surface_, surface_ stil calculates surface friction for xi and eta. 
            if (land_) land_->run(dt);
            surface_->compute_coefficients(state_);
        }

        for (const auto& var_name : sfc_thermodynamics_vars_) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            if (!turbulence_) fe_tend_field.set_to_zero(); 
            if (surface_) {
                surface_->calculate_tendencies(state_, var_name, fe_tend_field);
            }
            if (land_) {
                land_->calculate_tendencies(var_name, fe_tend_field);
            }
        }
    }

    if (turbulence_ || enable_surface_process_) {
        for (const auto& var_name : (turbulence_ ? turbulence_->get_thermodynamics_vars() : sfc_thermodynamics_vars_) ) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
        }
    }

    // Apply sponge layer
    if (sponge_layer_) {
        for (const auto& var_name : sponge_layer_->get_thermodynamics_vars()) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            fe_tend_field.set_to_zero(); 
            sponge_layer_->calculate_tendencies(state_, var_name, fe_tend_field);

            VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
        }
    }
    
    // Apply lateral boundary nudge
    if (lateral_boundary_nudging_) {
        for (const auto& var_name : lateral_boundary_nudging_->get_target_vars()) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            
            fe_tend_field.set_to_zero(); 
            lateral_boundary_nudging_->calculate_tendencies(state_, var_name, fe_tend_field);
            VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
        }
    }

    if (turbulence_ || sponge_layer_ || enable_surface_process_ || lateral_boundary_nudging_) {
        halo_exchanger_.exchange_multiple_halos(thermodynamics_vars_, state_);
        for (const auto& var_name : thermodynamics_vars_) {
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
            
            if (var_name == "zeta") {
                auto& fe_tend_field = state_.get_field<2>(fe_name);
                fe_tend_field.set_to_zero(); 
                turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);
            }
            else {
                auto& fe_tend_field = state_.get_field<3>(fe_name);
                fe_tend_field.set_to_zero(); 
                turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);
            }
        }
    }

    if (enable_surface_process_) {
        for (const auto& var_name : sfc_dynamics_vars_) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            if (!turbulence_) fe_tend_field.set_to_zero(); 
            surface_->calculate_tendencies(state_, var_name, fe_tend_field);
        }
    }

    if (turbulence_ || enable_surface_process_) {
        for (const auto& var_name : (turbulence_ ? turbulence_->get_dynamics_vars() : sfc_dynamics_vars_)) {
            std::string fe_name = "fe_tendency_" + var_name;
            if (var_name == "zeta") {
                auto& fe_tend_field = state_.get_field<2>(fe_name);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            } else {
                auto& fe_tend_field = state_.get_field<3>(fe_name);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            }
        }
    }

    if (sponge_layer_) {
        for (const auto& var_name : sponge_layer_->get_dynamics_vars()) {
            if (var_name == "zeta") {
                std::string fe_name = "fe_tendency_" + var_name;
                auto& fe_tend_field = state_.get_field<2>(fe_name);
                fe_tend_field.set_to_zero(); 
                sponge_layer_->calculate_tendencies(state_, var_name, fe_tend_field);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            }
            else {
                std::string fe_name = "fe_tendency_" + var_name;
                auto& fe_tend_field = state_.get_field<3>(fe_name);
                fe_tend_field.set_to_zero(); 
                sponge_layer_->calculate_tendencies(state_, var_name, fe_tend_field);
                VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
            }
        }
    }

    if (area_mean_nudging_) {
        area_mean_nudging_->apply_vorticity(state_, dt);
    }

    if (turbulence_ || sponge_layer_ || enable_surface_process_ || area_mean_nudging_) {
        halo_exchanger_.exchange_multiple_halos(dynamics_vars_, state_);
        for (const auto& var_name : dynamics_vars_) {
            bc_manager_.apply_vorticity_bc(state_.get_field<3>(var_name));
        }
        dycore_->compute_zeta_vertical_structure(state_);
    }

    if (wind_solver_) {
        if (predict_uvtopmn_) dycore_->compute_uvtopmn();
        if (area_mean_nudging_) area_mean_nudging_->apply_uvtopmn(state_, dt);
        dycore_->compute_wind_fields();
    }
    dycore_->compute_diagnostic_fields();
}

void Model::finalize() {
    if (microphysics_) microphysics_->finalize();
    if (radiation_) radiation_->finalize();
    if (land_) land_->finalize();
}

}
}
