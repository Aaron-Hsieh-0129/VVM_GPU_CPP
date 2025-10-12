#include "DynamicalCore.hpp"
#include "temporal_schemes/TimeIntegrator.hpp"
#include "tendency_processes/AdvectionTerm.hpp"
#include "tendency_processes/StretchingTerm.hpp"
#include "tendency_processes/TwistingTerm.hpp"
#include "tendency_processes/BuoyancyTerm.hpp"
#include "spatial_schemes/Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include <stdexcept>
#include <iostream>

namespace VVM {
namespace Dynamics {

DynamicalCore::DynamicalCore(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid, 
                             const Core::Parameters& params,
                             Core::State& state)
    : config_(config), grid_(grid), params_(params), state_(state), 
      wind_solver_(std::make_unique<WindSolver>(grid, config, params)) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "\n--- Initializing Dynamical Core ---" << std::endl;
    }
    
    auto prognostic_config = config_.get_value<nlohmann::json>("dynamics.prognostic_variables");
    
    for (auto& [var_name, var_conf] : prognostic_config.items()) {
        if (rank == 0) {
            std::cout << "  * Loading prognostic variable: " << var_name << std::endl;
        }
        std::vector<std::unique_ptr<TendencyTerm>> ab2_terms;
        std::vector<std::unique_ptr<TendencyTerm>> fe_terms;
        bool has_ab2 = false;
        bool has_fe = false;

        if (var_conf.contains("tendency_terms")) {
            for (auto& [term_name, term_conf] : var_conf.at("tendency_terms").items()) {
                std::string spatial_scheme_name = term_conf.at("spatial_scheme");
                std::string time_scheme_name = term_conf.value("temporal_scheme", "AdamsBashforth2");

                if (rank == 0) {
                    std::cout << "    - Tendency term: " << term_name 
                              << " | Temporal Scheme: " << time_scheme_name 
                              << " | Spatial Scheme: " << spatial_scheme_name << std::endl;
                }

                std::unique_ptr<SpatialScheme> spatial_scheme;
                if (spatial_scheme_name == "Takacs") {
                    spatial_scheme = std::make_unique<Takacs>(grid_);
                } 
                else {
                    throw std::runtime_error("Unknown spatial scheme: " + spatial_scheme_name);
                }
                
                std::unique_ptr<TendencyTerm> term;
                if (term_name == "advection") term = std::make_unique<AdvectionTerm>(std::move(spatial_scheme), var_name);
                else if (term_name == "stretching") term = std::make_unique<StretchingTerm>(std::move(spatial_scheme), var_name);
                else if (term_name == "twisting") term = std::make_unique<TwistingTerm>(std::move(spatial_scheme), var_name);
                else if (term_name == "buoyancy") term = std::make_unique<BuoyancyTerm>(std::move(spatial_scheme), var_name);

                if (time_scheme_name == "AdamsBashforth2") {
                    ab2_terms.push_back(std::move(term));
                    has_ab2 = true;
                } 
                else {
                    fe_terms.push_back(std::move(term));
                    has_fe = true;
                }
            }
        }
        
        tendency_calculators_[var_name] = std::make_unique<TendencyCalculator>(var_name, std::move(ab2_terms), std::move(fe_terms));
        time_integrators_[var_name] = std::make_unique<TimeIntegrator>(var_name, has_ab2, has_fe);
        
        int nz = grid_.get_local_total_points_z();
        int ny = grid_.get_local_total_points_y();
        int nx = grid_.get_local_total_points_x();

        if (has_ab2 || has_fe) {
             state_.add_field<3>(var_name + "_m", {nz, ny, nx});
        }
        if (has_ab2) {
             state_.add_field<4>("d_" + var_name, {2, nz, ny, nx});
        }
        if (has_fe) {
             state_.add_field<3>("fe_tendency_" + var_name, {nz, ny, nx});
        }
    }

    auto integration_config = config_.get_value<nlohmann::json>("dynamics.time_integration.procedure");
    for (const auto& step_conf : integration_config) {
        IntegrationStep step;
        step.step = step_conf.at("step");
        step.description = step_conf.value("description", "");
        if (step_conf.contains("calculate_tendencies")) {
            step.vars_to_calculate_tendency = step_conf.at("calculate_tendencies").get<std::vector<std::string>>();
        }
        if (step_conf.contains("update_states")) {
            step.vars_to_update = step_conf.at("update_states").get<std::vector<std::string>>();
        }
        integration_procedure_.push_back(step);
    }

    if (rank == 0) {
        std::cout << "\n--- Time Integration Procedure ---" << std::endl;
        for (const auto& step : integration_procedure_) {
            std::cout << "  Step " << step.step << ": " << step.description << std::endl;
            
            if (!step.vars_to_calculate_tendency.empty()) {
                std::cout << "    Calculate Tendencies for: ";
                for(const auto& v : step.vars_to_calculate_tendency) std::cout << v << " ";
                std::cout << std::endl;
            }
            if (!step.vars_to_update.empty()) {
                std::cout << "    Update States for: ";
                for(const auto& v : step.vars_to_update) std::cout << v << " ";
                std::cout << std::endl;
            }
        }
        std::cout << "------------------------------------" << std::endl;
    }

}

DynamicalCore::~DynamicalCore() = default;

void DynamicalCore::step(Core::State& state, double dt) {
    Core::HaloExchanger halo_exchanger(grid_);
    compute_diagnostic_fields();

    for (const auto& procedure_step : integration_procedure_) {
        for (const auto& var_name : procedure_step.vars_to_calculate_tendency) {
            if (tendency_calculators_.count(var_name)) {
                tendency_calculators_.at(var_name)->calculate_tendencies(state, grid_, params_, time_step_count);
            }
        }

        for (const auto& var_name : procedure_step.vars_to_update) {
            if (time_integrators_.count(var_name)) {
                time_integrators_.at(var_name)->step(state, grid_, params_, dt);
                
                VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, var_name);
                if (var_name == "zeta") halo_exchanger.exchange_halos_top_slice(state.get_field<3>(var_name));
                else halo_exchanger.exchange_halos(state.get_field<3>(var_name));
                
                if (var_name != "zeta") bc_manager.apply_z_bcs_to_field(state.get_field<3>(var_name));
            }
        }
    }

    compute_zeta_vertical_structure(state);
    compute_wind_fields();
    time_step_count++;
}

void DynamicalCore::compute_diagnostic_fields() const {
    auto scheme = std::make_unique<Takacs>(grid_);

    auto& R_xi_field = state_.get_field<3>("R_xi");
    auto& R_eta_field = state_.get_field<3>("R_eta");
    auto& R_zeta_field = state_.get_field<3>("R_zeta");

    scheme->calculate_R_xi(state_, grid_, params_, R_xi_field);
    scheme->calculate_R_eta(state_, grid_, params_, R_eta_field);
    scheme->calculate_R_zeta(state_, grid_, params_, R_zeta_field);
}

void DynamicalCore::compute_zeta_vertical_structure(Core::State& state) const {
    auto scheme = std::make_unique<Takacs>(grid_);
    auto& zeta_field = state.get_field<3>("zeta");
    auto zeta_data = zeta_field.get_mutable_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    
    const double dz = grid_.get_dz();
    const double dy = grid_.get_dy();
    const double dx = grid_.get_dx();

    Core::Field<3> rhs_field("rhs_zeta_diag", {nz, ny, nx});
    scheme->calculate_vorticity_divergence(state, grid_, params_, rhs_field);
    // const auto& rhs_data = rhs_field.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    Kokkos::parallel_for("zeta_downward_integration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // The for-loop inside is to prevent racing condition because lower layers depend on upper layers.
            for (int k = nz-h-2; k >= h-1; --k) {
                // zeta_data(k,j,i) = zeta_data(k+1,j,i) + rhs_data(k,j,i) * -dz / flex_height_coef_up(k);
                zeta_data(k,j,i) = zeta_data(k+1,j,i) 
                                 + ( xi(k,j,i+1) -  xi(k,j,i)) * dz / (dx * flex_height_coef_up(k))
                                 - (eta(k,j+1,i) - eta(k,j,i)) * dz / (dy * flex_height_coef_up(k));
            }
            // WARNING: NK3 has a upward integration in original VVM code.
            // zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) + rhs_data(nz-h-1,j,i) * dz / flex_height_coef_up(nz-h-1);
            zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) 
                             - ( xi(nz-h-1,j,i+1) -  xi(nz-h-1,j,i)) * dz / (dx * flex_height_coef_up(nz-h-1))
                             + (eta(nz-h-1,j+1,i) - eta(nz-h-1,j,i)) * dz / (dy * flex_height_coef_up(nz-h-1));
        }
    );
    Core::HaloExchanger halo_exchanger(grid_);
    halo_exchanger.exchange_halos(zeta_field);
    // VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, "zeta");
    // bc_manager.apply_z_bcs_to_field(zeta_field);
}

void DynamicalCore::compute_wind_fields() {
    wind_solver_->solve_w(state_);
    wind_solver_->solve_uv(state_);
}

} // namespace Dynamics
} // namespace VVM

