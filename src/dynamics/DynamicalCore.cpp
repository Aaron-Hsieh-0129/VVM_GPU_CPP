#include "DynamicalCore.hpp"
#include "temporal_schemes/TimeIntegrator.hpp"
#include "tendency_processes/AdvectionTerm.hpp"
#include "tendency_processes/StretchingTerm.hpp"
#include "tendency_processes/TwistingTerm.hpp"
#include "tendency_processes/BuoyancyTerm.hpp"
#include "tendency_processes/CoriolisTerm.hpp"
#include "spatial_schemes/Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include <stdexcept>
#include <iostream>
#include <unordered_set>

namespace VVM {
namespace Dynamics {

bool startsWith(const std::string& fullString, const std::string& prefix) {
    if (fullString.length() < prefix.length()) return false;
    return fullString.compare(0, prefix.length(), prefix) == 0;
}

DynamicalCore::DynamicalCore(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid, 
                             const Core::Parameters& params,
                             Core::State& state, 
                             Core::HaloExchanger& halo_exchanger)
    : config_(config), grid_(grid), params_(params), state_(state), 
      wind_solver_(std::make_unique<WindSolver>(grid, config, params, halo_exchanger)), 
      halo_exchanger_(halo_exchanger), bc_manager_(grid) {

    int rank = grid_.get_mpi_rank();
    if (rank == 0) std::cout << "\n--- Initializing Dynamical Core ---" << std::endl;
    std::vector<std::string> common_thermo = {"th", "qv"};
    
    auto prognostic_config = config_.get_value<nlohmann::json>("dynamics.prognostic_variables");
    if (!config.get_value<bool>("physics.p3.enable_p3", false)) {
        if (rank == 0) std::cout << "[WARNING] P3 is not turned on but the P3 variables are listed in prognostic variables so they are deleted!!" << std::endl;
        std::unordered_set<std::string> P3toRemove = {"qc", "qi", "qr", "qm", "nc", "ni", "nr", "bm"};
        std::vector<std::string> keysToDelete;
        for (auto& [key, value] : prognostic_config.items()) {
            for (const auto& prefix : P3toRemove) {
                if (startsWith(key, prefix)) {
                    keysToDelete.push_back(key);
                    break;
                }
            }
        }
        for (const auto& key : keysToDelete) {
            prognostic_config.erase(key);
        }
    }
    else common_thermo.insert(common_thermo.end(), {"qc", "qr", "qi", "nc", "nr", "ni", "qm", "bm"});

    bool coriolis_xi = config.get_value<bool>("dynamics.prognostic_variables.xi.tendency_terms.coriolis.enable", false);
    bool coriolis_eta = config.get_value<bool>("dynamics.prognostic_variables.eta.tendency_terms.coriolis.enable", false);
    bool coriolis_zeta = config.get_value<bool>("dynamics.prognostic_variables.zeta.tendency_terms.coriolis.enable", false);
    enable_coriolis_ = coriolis_xi && coriolis_eta && coriolis_zeta;
    
    for (auto& [var_name, var_conf] : prognostic_config.items()) {
        if (rank == 0) {
            std::cout << "  * Loading prognostic variable: " << var_name << std::endl;
        }
        bool has_ab2 = false;
        bool has_fe = false;

        if (var_name == "th" && config.get_value<bool>("physics.rrtmgp.enable_rrtmgp", false)) {
            has_fe = true;
            if (rank == 0) {
                std::cout << "    - Enabled radiation forcing integration. " << std::endl;
            }
        }

        bool is_thermo = std::find(common_thermo.begin(), common_thermo.end(), var_name) != common_thermo.end();

        if (is_thermo) thermo_vars_.push_back(var_name);
        else vorticity_vars_.push_back(var_name);

        std::vector<std::unique_ptr<TendencyTerm>> ab2_terms;
        std::vector<std::unique_ptr<TendencyTerm>> fe_terms;

        if (var_conf.contains("tendency_terms")) {
            for (auto& [term_name, term_conf] : var_conf.at("tendency_terms").items()) {
                bool is_enabled = term_conf.value("enable", true);
                if (!is_enabled) {
                    if (rank == 0) {
                        std::cout << "    - [Disabled] Tendency term: " << term_name << " is skipped." << std::endl;
                    }
                    continue;
                }

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
                else if (term_name == "coriolis") term = std::make_unique<CoriolisTerm>(std::move(spatial_scheme), var_name, halo_exchanger_);

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
    // This is for predict utopmn and vtopmn
    state_.add_field<1>("d_utopmn", {2});
    state_.add_field<1>("d_vtopmn", {2});
    state_.add_field<0>("utopmn_m", {});
    state_.add_field<0>("vtopmn_m", {});
    Kokkos::deep_copy(state_.get_field<0>("utopmn_m").get_mutable_device_data(), state_.get_field<0>("utopmn").get_mutable_device_data());
    Kokkos::deep_copy(state_.get_field<0>("vtopmn_m").get_mutable_device_data(), state_.get_field<0>("vtopmn").get_mutable_device_data());
}

DynamicalCore::~DynamicalCore() = default;

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
    Kokkos::deep_copy(u_topo, u);
    Kokkos::deep_copy(v_topo, v);
    Kokkos::deep_copy(w_topo, w);

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    Kokkos::parallel_for("wind_topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {max_topo_idx+2, ny, nx}),
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

    // Assign vorticity for topography
    Kokkos::parallel_for("vorticity_topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEV(k,j,i) != 1) {
                xi_topo(k,j,i) = (w_topo(k,j+1,i) - w_topo(k,j,i)) * rdy()
                               - (v_topo(k+1,j,i) - v_topo(k,j,i)) * rdz() * flex_height_coef_up(k);
            }
            else xi_topo(k,j,i) = xi(k,j,i);
            

            if (ITYPEU(k,j,i) != 1) {
                eta_topo(k,j,i) = (w_topo(k,j,i+1) - w_topo(k,j,i)) * rdx()
                                - (u_topo(k+1,j,i) - u_topo(k,j,i)) * rdz() * flex_height_coef_up(k);
            }
            else eta_topo(k,j,i) = eta(k,j,i);
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
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();
    const auto& rdz = params_.rdz;

    auto& tempu_field = state_.get_field<2>("tempu");
    auto& tempv_field = state_.get_field<2>("tempv");
    auto& tempu = tempu_field.get_mutable_device_data();
    auto& tempv = tempv_field.get_mutable_device_data();

    auto &utopmn = state_.get_field<0>("utopmn");
    auto &vtopmn = state_.get_field<0>("vtopmn");

    Kokkos::parallel_for("calculate_utopmn",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            tempu(j,i) = (rhobar(nz-h-2)*u(nz-h-2,j,i) + rhobar(nz-h-1)*u(nz-h-1,j,i)) 
                       * (w(nz-h-2,j,i)+w(nz-h-2,j,i+1));
            tempv(j,i) = (rhobar(nz-h-2)*v(nz-h-2,j,i) + rhobar(nz-h-1)*v(nz-h-1,j,i)) 
                       * (w(nz-h-2,j,i)+w(nz-h-2,j+1,i));
        }
    );
#if defined(ENABLE_NCCL)
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> tempumn("tempumn");
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> tempvmn("tempvmn");
    state_.calculate_horizontal_mean(tempu_field, tempumn);
    state_.calculate_horizontal_mean(tempv_field, tempvmn);
#else
    auto tempumn = state_.calculate_horizontal_mean(tempu_field);
    auto tempvmn = state_.calculate_horizontal_mean(tempv_field);
#endif

    int NK2 = nz-h-1;
    int NK1 = nz-h-2;

    const auto& RKM = state_.get_field<3>("RKM").get_device_data();
    const auto& RKH = state_.get_field<3>("RKH").get_device_data();
    const auto& R_xi = state_.get_field<3>("R_xi").get_device_data();
    const auto& R_eta = state_.get_field<3>("R_eta").get_device_data();
    const auto& f = state_.get_field<1>("f").get_device_data();

    // Diffusion
    Kokkos::parallel_for("calculate_utopmn_diffusion",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            tempu(j,i) = (RKM(NK1,j,i)+RKM(NK1,j,i+1)+RKM(NK2,j,i)+RKM(NK2,j,i+1))
                         *R_eta(NK1,j,i)*rhobar_up(NK1);
            tempv(j,i) = (RKM(NK1,j,i)+RKM(NK1,j+1,i)+RKM(NK2,j,i)+RKM(NK2,j+1,i))
                         *R_xi(NK1,j,i)*rhobar_up(NK1);
        }
    );
#if defined(ENABLE_NCCL)
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> mean_u_turb("mean_u_turb");
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> mean_v_turb("mean_v_turb");
    state_.calculate_horizontal_mean(tempu_field, mean_u_turb);
    state_.calculate_horizontal_mean(tempv_field, mean_v_turb);
#else
    auto mean_u_turb = state_.calculate_horizontal_mean(tempu_field);
    auto mean_v_turb = state_.calculate_horizontal_mean(tempv_field);
#endif

    Kokkos::View<double> mean_u_coriolis("mean_u_coriolis");
    Kokkos::View<double> mean_v_coriolis("mean_v_coriolis");
    Kokkos::deep_copy(mean_u_coriolis, 0.0);
    Kokkos::deep_copy(mean_v_coriolis, 0.0);

    if (enable_coriolis_) {
        // Coriolis force
        Kokkos::parallel_for("calculate_utopmn_coriolis",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h, nx-h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                tempu(j,i) = f(j) * v(NK2, j, i);
                tempv(j,i) = f(j) * u(NK2, j, i);
            }
        );
#if defined(ENABLE_NCCL)
        Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> mean_u_coriolis("mean_u_coriolis");
        Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> mean_v_coriolis("mean_v_coriolis");
        state_.calculate_horizontal_mean(tempu_field, mean_u_coriolis);
        state_.calculate_horizontal_mean(tempv_field, mean_v_coriolis);
#else
        auto mean_u_coriolis = state_.calculate_horizontal_mean(tempu_field);
        auto mean_v_coriolis = state_.calculate_horizontal_mean(tempv_field);
#endif
    }

    auto& utopmn_to_update = state_.get_field<0>("utopmn");
    auto& utopmn_new_view = utopmn_to_update.get_mutable_device_data();
    auto& utopmn_prev_step = state_.get_field<0>("utopmn_m");
    auto& vtopmn_to_update = state_.get_field<0>("vtopmn");
    auto& vtopmn_new_view = vtopmn_to_update.get_mutable_device_data();
    auto& vtopmn_prev_step = state_.get_field<0>("vtopmn_m");

    // update utopmn, vtopmn
    Kokkos::deep_copy(utopmn_prev_step.get_mutable_device_data(), utopmn_to_update.get_device_data());
    auto& utopmn_old_view = utopmn_prev_step.get_device_data();
    Kokkos::deep_copy(vtopmn_prev_step.get_mutable_device_data(), vtopmn_to_update.get_device_data());
    auto& vtopmn_old_view = vtopmn_prev_step.get_device_data();

    auto& d_utopmn = state_.get_field<1>("d_utopmn").get_mutable_device_data();
    auto& d_vtopmn = state_.get_field<1>("d_vtopmn").get_mutable_device_data();

    size_t now_idx = state_.get_step() % 2;
    size_t prev_idx = (state_.get_step() + 1) % 2;

    if (state_.get_step() == 0) {
        Kokkos::parallel_for("Cauculate_uvtopmn", 
            1, 
            KOKKOS_LAMBDA(const int i) {
                d_utopmn(now_idx) = 0.25 * flex_height_coef_mid(NK2) * tempumn() * rdz() / rhobar(NK2)
                                   -0.25 * flex_height_coef_mid(NK2) * mean_u_turb() * rdz() / rhobar(NK2)
                                   +mean_u_coriolis();
                d_vtopmn(now_idx) = 0.25 * flex_height_coef_mid(NK2) * tempvmn() * rdz() / rhobar(NK2)
                                   -0.25 * flex_height_coef_mid(NK2) * mean_v_turb() * rdz() / rhobar(NK2)
                                   -mean_v_coriolis();

                utopmn_new_view() = utopmn_old_view() + dt() * d_utopmn(now_idx);
                vtopmn_new_view() = vtopmn_old_view() + dt() * d_vtopmn(now_idx);
            }
        );
    }
    else {
        Kokkos::parallel_for("Cauculate_uvtopmn", 
            1, 
            KOKKOS_LAMBDA(const int i) {
                d_utopmn(now_idx) = 0.25 * flex_height_coef_mid(NK2) * tempumn() * rdz() / rhobar(NK2)
                                   -0.25 * flex_height_coef_mid(NK2) * mean_u_turb() * rdz() / rhobar(NK2)
                                   +mean_u_coriolis();
                d_vtopmn(now_idx) = 0.25 * flex_height_coef_mid(NK2) * tempvmn() * rdz() / rhobar(NK2)
                                   -0.25 * flex_height_coef_mid(NK2) * mean_v_turb() * rdz() / rhobar(NK2)
                                   -mean_v_coriolis();

                utopmn_new_view() = utopmn_old_view() 
                        + dt() * (1.5 * d_utopmn(now_idx) - 0.5 * d_utopmn(prev_idx));
                vtopmn_new_view() = vtopmn_old_view() 
                        + dt() * (1.5 * d_vtopmn(now_idx) - 0.5 * d_vtopmn(prev_idx));
            }
        );
    }
    return;
}


void DynamicalCore::calculate_thermo_tendencies() {
    compute_diagnostic_fields(); 

    for (const auto& var_name : thermo_vars_) {
        std::string fe_name = "fe_tendency_" + var_name;
        if (state_.has_field(fe_name)) {
            auto& field = state_.get_field<3>(fe_name);
            auto data = field.get_mutable_device_data();
            Kokkos::deep_copy(data, 0.0);
        }
    }

    for (const auto& var_name : thermo_vars_) {
        if (tendency_calculators_.count(var_name)) {
            tendency_calculators_.at(var_name)->calculate_tendencies(state_, grid_, params_);
        }
    }
}

void DynamicalCore::update_thermodynamics(double dt) {
    const int h = grid_.get_halo_cells();
    const auto& max_topo_idx = params_.max_topo_idx;
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    for (const auto& var_name : thermo_vars_) {
        if (time_integrators_.count(var_name)) {
            // var += dt * (AB2 + FE)
            time_integrators_.at(var_name)->step(state_, grid_, params_, dt);
            
            const auto& ITYPEW = state_.get_field<3>("ITYPEW").get_device_data();
            auto& var = state_.get_field<3>(var_name).get_mutable_device_data();
            // Topography for theta
            if (var_name == "th") {
                const auto& thbar = state_.get_field<1>("thbar").get_device_data();
                Kokkos::parallel_for("topo_bc_th",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        if (ITYPEW(k,j,i) != 1) {
                            var(k,j,i) = thbar(k);
                        }
                    }
                );
            }
            else {
                Kokkos::parallel_for("topo",
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        if (ITYPEW(k,j,i) != 1) {
                            var(k,j,i) = 0.;
                        }
                    }
                );
            }
            halo_exchanger_.exchange_halos(state_.get_field<3>(var_name));
            if (var_name == "th" || var_name == "qv") {
                bc_manager_.apply_zero_gradient(state_.get_field<3>(var_name));
            }
            else {
                bc_manager_.apply_zero_gradient_bottom_zero_top(state_.get_field<3>(var_name));
            }
        }
    }
}

void DynamicalCore::calculate_vorticity_tendencies() {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();

    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();
    auto& zeta = state_.get_field<3>("zeta").get_mutable_device_data();
    
    // Divide by density
    Kokkos::parallel_for("divide_by_density_xi_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            xi(k, j, i) /= rhobar_up(k);
            eta(k, j, i) /= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("divide_by_density_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h+1, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            zeta(k, j, i) /= rhobar(k);
        }
    );

    // Calculate vorticity tendency
    for (const auto& var_name : vorticity_vars_) {
        if (tendency_calculators_.count(var_name)) {
            tendency_calculators_.at(var_name)->calculate_tendencies(state_, grid_, params_);
        }
    }
}

void DynamicalCore::update_vorticity(double dt) {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();

    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();
    auto& zeta = state_.get_field<3>("zeta").get_mutable_device_data();

    Kokkos::parallel_for("multiply_density_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            xi(k, j, i) *= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("multiply_density_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            eta(k, j, i) *= rhobar_up(k);
        }
    );
    Kokkos::parallel_for("multiply_density_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h+1, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            zeta(k, j, i) *= rhobar(k);
        }
    );

    for (const auto& var_name : vorticity_vars_) {
        if (time_integrators_.count(var_name)) {
            time_integrators_.at(var_name)->step(state_, grid_, params_, dt);
            halo_exchanger_.exchange_halos(state_.get_field<3>(var_name));
        }
    }
    bc_manager_.apply_vorticity_bc(state_.get_field<3>("xi"));
    bc_manager_.apply_vorticity_bc(state_.get_field<3>("eta"));

    compute_zeta_vertical_structure(state_);
}

void DynamicalCore::diagnose_wind_fields(Core::State& state) {
    compute_uvtopmn(); 
    compute_wind_fields();
}


} // namespace Dynamics
} // namespace VVM

