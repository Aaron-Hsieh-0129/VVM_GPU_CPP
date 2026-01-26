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

    if (config_.get_value<bool>("physics.surface.enable_surface", false)) {
        surface_ = std::make_unique<Physics::SurfaceProcess>(config_, grid_, params_, halo_exchanger_, state_);
        surface_freq_in_steps_ = config_.get_value<int>("physics.surface.frequency_step", 12);
    }

    if (config_.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
        radiation_ = std::make_unique<Physics::RRTMGP::RRTMGPRadiation>(config_, grid_, params_);
        rad_freq_in_steps_ = config_.get_value<int>("physics.rrtmgp.rad_frequency_step", 1);
    }

    if (config_.get_value<bool>("dynamics.filters.sponge_layer.enable", false)) {
        sponge_layer_ = std::make_unique<Dynamics::SpongeLayer>(config_, grid_, params_, halo_exchanger_, state_);
    }
    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv"};
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        thermodynamics_vars_.insert(thermodynamics_vars_.end(), {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"});
    }
    sfc_vars_ = {"th", "qv"};
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
    if (surface_) surface_->initialize(state_);
    
    halo_exchanger_.exchange_halos(state_);
    
    if (rank == 0) std::cout << "=== Model Initialization Complete ===\n" << std::endl;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    if (!state_.has_field("th_perturb")) state_.add_field<3>("th_perturb", {nz, ny, nx});
}

void Model::run_step(double dt) {
    if (state_.get_time() < 50) {
        int nz = grid_.get_local_total_points_z();
        int ny = grid_.get_local_total_points_y();
        int nx = grid_.get_local_total_points_x();
        int h = grid_.get_halo_cells();

        auto& th_perturb = state_.get_field<3>("th_perturb").get_mutable_device_data();
        auto seed = state_.get_step();
        using GeneratorPool = Kokkos::Random_XorShift64_Pool<>;
        GeneratorPool rand_pool(seed);
        using PolicyType = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        PolicyType policy({0, 0, 0}, {nz, ny, nx});
        Kokkos::parallel_for("fill_random", policy, 
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                auto gen = rand_pool.get_state();
                th_perturb(k, j, i) = gen.drand(-1.0, 1.0);
                rand_pool.free_state(gen);
            });
        Kokkos::fence();

        auto& th = state_.get_field<3>("th").get_mutable_device_data();
        Kokkos::parallel_for("Add_perturb",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{h+4, ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                th(k,j,i) += th_perturb(k,j,i);
            }
        );
    }

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
        // The effects of radiation is updated in update_thermodynamics
        radiation_->calculate_tendencies(state_);
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
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            fe_tend_field.set_to_zero(); 
            turbulence_->calculate_tendencies(state_, var_name, fe_tend_field);
        }
    }
    if (surface_) {
        surface_->compute_coefficients(state_);
        for (const auto& var_name : sfc_vars_) {
            if ((state_.get_step()-1) % surface_freq_in_steps_ == 0) {
                std::string fe_name = "fe_tendency_" + var_name;
                auto& fe_tend_field = state_.get_field<3>(fe_name);
                if (!turbulence_) fe_tend_field.set_to_zero(); 
                surface_->calculate_tendencies(state_, var_name, fe_tend_field);
            }
        }
    }
    if (turbulence_ || surface_) {
        for (const auto& var_name : (turbulence_ ? turbulence_->get_thermodynamics_vars() : sfc_vars_) ) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
        }
    } 


    // Apply sponge layer
    if (sponge_layer_) {
        for (const auto& var_name : turbulence_->get_thermodynamics_vars()) {
            std::string fe_name = "fe_tendency_" + var_name;
            auto& fe_tend_field = state_.get_field<3>(fe_name);
            fe_tend_field.set_to_zero(); 
            sponge_layer_->calculate_tendencies(state_, var_name, fe_tend_field);

            VVM::Dynamics::TimeIntegrator::apply_forward_update(state_, var_name, grid_, dt, fe_tend_field);
        }
    }

    if (turbulence_ || sponge_layer_ || surface_) {
        for (const auto& var_name : thermodynamics_vars_) {
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
        }
    }

    if (sponge_layer_) {
        for (const auto& var_name : turbulence_->get_dynamics_vars()) {
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

    if (turbulence_ || sponge_layer_) {
        for (const auto& var_name : dynamics_vars_) {
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
