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
                             Core::State& state, 
                             Core::HaloExchanger& halo_exchanger)
    : config_(config), grid_(grid), params_(params), state_(state), 
      wind_solver_(std::make_unique<WindSolver>(grid, config, params, halo_exchanger)), 
      halo_exchanger_(halo_exchanger) {

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
                    spatial_scheme = std::make_unique<Takacs>(grid_, halo_exchanger_);
                } 
                else {
                    throw std::runtime_error("Unknown spatial scheme: " + spatial_scheme_name);
                }
                
                std::unique_ptr<TendencyTerm> term;
                if (term_name == "advection") term = std::make_unique<AdvectionTerm>(std::move(spatial_scheme), var_name, halo_exchanger_);
                else if (term_name == "stretching") term = std::make_unique<StretchingTerm>(std::move(spatial_scheme), var_name, halo_exchanger_);
                else if (term_name == "twisting") term = std::make_unique<TwistingTerm>(std::move(spatial_scheme), var_name, halo_exchanger_);
                else if (term_name == "buoyancy") term = std::make_unique<BuoyancyTerm>(std::move(spatial_scheme), var_name, halo_exchanger_);

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

        // This is for predict utopmn and vtopmn
        state_.add_field<1>("d_utopmn", {2});
        state_.add_field<1>("d_vtopmn", {2});
        state_.add_field<1>("utopmn_m", {1});
        state_.add_field<1>("vtopmn_m", {1});
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
    compute_diagnostic_fields();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();

    auto& xi = state.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state.get_field<3>("eta").get_mutable_device_data();
    auto& zeta = state.get_field<3>("zeta").get_mutable_device_data();
    Kokkos::parallel_for("divide_by_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            xi(k, j, i) /= rhobar_up(k);
            eta(k, j, i) /= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("divide_by_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            zeta(k, j, i) /= rhobar(k);
        }
    );


    for (const auto& procedure_step : integration_procedure_) {
        for (const auto& var_name : procedure_step.vars_to_calculate_tendency) {
            if (tendency_calculators_.count(var_name)) {
                tendency_calculators_.at(var_name)->calculate_tendencies(state, grid_, params_, time_step_count);
            }
        }

        for (const auto& var_name : procedure_step.vars_to_update) {
            if (var_name == "xi") {
                Kokkos::parallel_for("divide_by_density",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        xi(k, j, i) *= rhobar_up(k);
                    }
                );
            }
            else if (var_name == "eta") {
                Kokkos::parallel_for("divide_by_density",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        eta(k, j, i) *= rhobar_up(k);
                    }
                );
            }
            else if (var_name == "zeta") {
                Kokkos::parallel_for("divide_by_density",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz, ny, nx}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        zeta(k, j, i) *= rhobar(k);
                    }
                );
            }


            if (time_integrators_.count(var_name)) {
                time_integrators_.at(var_name)->step(state, grid_, params_, dt);
                if (var_name == "th") {
                    const auto& ITYPEW = state.get_field<3>("ITYPEW").get_device_data();
                    const auto& max_topo_idx = params_.max_topo_idx;
                    auto& th = state_.get_field<3>("th").get_mutable_device_data();
                    const auto& thbar = state_.get_field<1>("thbar").get_device_data();
                    Kokkos::parallel_for("topo",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
                        KOKKOS_LAMBDA(const int k, const int j, const int i) {
                            // Set tendency to 0 if ITYPEV = 0
                            if (ITYPEW(k,j,i) != 1) {
                                th(k,j,i) = thbar(k);
                            }
                        }
                    );
                }
                
                VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, var_name);
                // if (var_name == "zeta") halo_exchanger.exchange_halos_top_slice(state.get_field<3>(var_name));
                // else halo_exchanger.exchange_halos(state.get_field<3>(var_name));
                halo_exchanger_.exchange_halos(state.get_field<3>(var_name));
                
                // if (var_name != "zeta") bc_manager.apply_z_bcs_to_field(state.get_field<3>(var_name));
            }
        }
    }

    compute_zeta_vertical_structure(state);
    compute_uvtopmn();
    compute_wind_fields();
    time_step_count++;
}

void DynamicalCore::compute_diagnostic_fields() const {
    auto scheme = std::make_unique<Takacs>(grid_, halo_exchanger_);

    auto& R_xi_field = state_.get_field<3>("R_xi");
    auto& R_eta_field = state_.get_field<3>("R_eta");
    auto& R_zeta_field = state_.get_field<3>("R_zeta");

    scheme->calculate_R_xi(state_, grid_, params_, R_xi_field);
    scheme->calculate_R_eta(state_, grid_, params_, R_eta_field);
    scheme->calculate_R_zeta(state_, grid_, params_, R_zeta_field);
}

void DynamicalCore::compute_zeta_vertical_structure(Core::State& state) const {
    auto scheme = std::make_unique<Takacs>(grid_, halo_exchanger_);
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
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;

    // WARNING: If RHS needs different scheme, it can be put into scheme for calculate_vorticity_divergence 
    // Core::Field<3> rhs_field("rhs_zeta_diag", {nz, ny, nx});
    // scheme->calculate_vorticity_divergence(state, grid_, params_, rhs_field);
    // const auto& rhs_data = rhs_field.get_device_data();

    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    Kokkos::parallel_for("zeta_downward_integration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // The for-loop inside is to prevent racing condition because lower layers depend on upper layers.
            for (int k = nz-h-2; k >= h-1; --k) {
                // zeta_data(k,j,i) = zeta_data(k+1,j,i) + rhs_data(k,j,i) * -dz / flex_height_coef_up(k);
                zeta_data(k,j,i) = zeta_data(k+1,j,i) 
                                 + ( xi(k,j,i+1) -  xi(k,j,i)) * rdx() * dz / (flex_height_coef_up(k))
                                 - (eta(k,j+1,i) - eta(k,j,i)) * rdy() * dz / (flex_height_coef_up(k));
            }
            // WARNING: NK3 has a upward integration in original VVM code.
            // zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) + rhs_data(nz-h-1,j,i) * dz / flex_height_coef_up(nz-h-1);
            zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) 
                             - ( xi(nz-h-1,j,i+1) -  xi(nz-h-1,j,i)) * dz * rdx() / (flex_height_coef_up(nz-h-1))
                             + (eta(nz-h-1,j+1,i) - eta(nz-h-1,j,i)) * dz * rdy() / (flex_height_coef_up(nz-h-1));

            // for (int k = 0; k < nz; k++) {
            //     if (k != nz-h-1) zeta_data(k,j,i) = 0;
            // }
        }
    );
    halo_exchanger_.exchange_halos(zeta_field);
    // VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, "zeta");
    // bc_manager.apply_z_bcs_to_field(zeta_field);
}

void DynamicalCore::compute_wind_fields() {
    // Assign wind for topography 
    const auto& ITYPEU = state_.get_field<3>("ITYPEU").get_device_data();
    const auto& ITYPEV = state_.get_field<3>("ITYPEV").get_device_data();
    const auto& ITYPEW = state_.get_field<3>("ITYPEW").get_device_data();
    const auto& max_topo_idx = params_.max_topo_idx;

    auto& u_topo = state_.get_field<3>("u_topo").get_mutable_device_data();
    const auto& u = state_.get_field<3>("u").get_device_data();
    auto& v_topo = state_.get_field<3>("v_topo").get_mutable_device_data();
    const auto& v = state_.get_field<3>("v").get_device_data();
    auto& w_topo = state_.get_field<3>("w_topo").get_mutable_device_data();
    const auto& w = state_.get_field<3>("w").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    Kokkos::parallel_for("wind_topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {max_topo_idx+2, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEU(k,j,i) != 1) u_topo(k,j,i) = 0;
            else u_topo(k,j,i) = u(k,j,i);

            if (ITYPEV(k,j,i) != 1) v_topo(k,j,i) = 0;
            else v_topo(k,j,i) = v(k,j,i);

            if (ITYPEW(k,j,i) != 1) w_topo(k,j,i) = 0;
            else w_topo(k,j,i) = w(k,j,i);
        }
    );

    auto& xi_topo = state_.get_field<3>("xi_topo").get_mutable_device_data();
    const auto& xi = state_.get_field<3>("xi").get_device_data();
    auto& eta_topo = state_.get_field<3>("eta_topo").get_mutable_device_data();
    const auto& eta = state_.get_field<3>("eta").get_device_data();
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;
    const auto& rdz = params_.rdz;
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    Kokkos::deep_copy(xi_topo, xi);
    Kokkos::deep_copy(eta_topo, eta);
    // Assign vorticity for topography
    Kokkos::parallel_for("vorticity_topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEV(k,j,i) != 1) {
                xi_topo(k,j,i) = (w_topo(k,j+1,i) - w_topo(k,j,i)) * rdy()
                               - (v_topo(k+1,j,i) - v_topo(k,j,i)) * rdz() * flex_height_coef_up(k);
            }

            if (ITYPEU(k,j,i) != 1) {
                eta_topo(k,j,i) = (w_topo(k,j,i+1) - w_topo(k,j,i)) * rdx()
                                - (u_topo(k+1,j,i) - u_topo(k,j,i)) * rdz() * flex_height_coef_up(k);
            }
        }
    );
    halo_exchanger_.exchange_halos(state_.get_field<3>("xi_topo"));
    halo_exchanger_.exchange_halos(state_.get_field<3>("eta_topo"));


    wind_solver_->solve_w(state_);
    wind_solver_->solve_uv(state_);
}

void DynamicalCore::compute_uvtopmn() {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& dt = params_.dt;

    const auto& u = state_.get_field<3>("u").get_device_data();
    const auto& v = state_.get_field<3>("v").get_device_data();
    const auto& w = state_.get_field<3>("w").get_device_data();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();
    const auto& rdz = params_.rdz;

    auto& tempu_field = state_.get_field<2>("tempu");
    auto& tempv_field = state_.get_field<2>("tempv");
    auto& tempu = tempu_field.get_mutable_device_data();
    auto& tempv = tempv_field.get_mutable_device_data();

    auto &utopmn = state_.get_field<1>("utopmn");
    auto &vtopmn = state_.get_field<1>("vtopmn");

    Kokkos::parallel_for("calculate_utopmn",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            tempu(j,i) = (rhobar(nz-h-2)*u(nz-h-2,j,i) + rhobar(nz-h-1)*u(nz-h-1,j,i)) 
                       * (w(nz-h-2,j,i)+w(nz-h-2,j,i+1));
            tempv(j,i) = (rhobar(nz-h-2)*v(nz-h-2,j,i) + rhobar(nz-h-1)*v(nz-h-1,j,i)) 
                       * (w(nz-h-2,j,i)+w(nz-h-2,j+1,i));
        }
    );
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> tempumn("tempumn");
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> tempvmn("tempvmn");
    state_.calculate_horizontal_mean(tempu_field, tempumn);
    state_.calculate_horizontal_mean(tempv_field, tempvmn);
    // double tempumn = state_.calculate_horizontal_mean(tempu_field);
    // double tempvmn = state_.calculate_horizontal_mean(tempv_field);


    auto& utopmn_to_update = state_.get_field<1>("utopmn");
    auto& utopmn_new_view = utopmn_to_update.get_mutable_device_data();
    auto& utopmn_prev_step = state_.get_field<1>("utopmn_m");
    auto& vtopmn_to_update = state_.get_field<1>("vtopmn");
    auto& vtopmn_new_view = vtopmn_to_update.get_mutable_device_data();
    auto& vtopmn_prev_step = state_.get_field<1>("vtopmn_m");

    // update utopmn, vtopmn
    Kokkos::deep_copy(utopmn_prev_step.get_mutable_device_data(), utopmn_to_update.get_device_data());
    auto& utopmn_old_view = utopmn_prev_step.get_device_data();
    Kokkos::deep_copy(vtopmn_prev_step.get_mutable_device_data(), vtopmn_to_update.get_device_data());
    auto& vtopmn_old_view = vtopmn_prev_step.get_device_data();

    auto& d_utopmn = state_.get_field<1>("d_utopmn").get_mutable_device_data();
    auto& d_vtopmn = state_.get_field<1>("d_vtopmn").get_mutable_device_data();

    size_t now_idx = time_step_count % 2;
    size_t prev_idx = (time_step_count + 1) % 2;
    Kokkos::parallel_for("Cauculate_uvtopmn", 
        1, 
        KOKKOS_LAMBDA(const int i) {
            d_utopmn(now_idx) = 0.25 * flex_height_coef_mid(nz-h-1) * tempumn() * rdz() / rhobar(nz-h-1);
            d_vtopmn(now_idx) = 0.25 * flex_height_coef_mid(nz-h-1) * tempvmn() * rdz() / rhobar(nz-h-1);

            utopmn_new_view(0) = utopmn_old_view(0) 
                    + dt() * (1.5 * d_utopmn(now_idx) - 0.5 * d_utopmn(prev_idx));
            vtopmn_new_view(0) = vtopmn_old_view(0) 
                    + dt() * (1.5 * d_vtopmn(now_idx) - 0.5 * d_vtopmn(prev_idx));
        }
    );
    return;
}

} // namespace Dynamics
} // namespace VVM

