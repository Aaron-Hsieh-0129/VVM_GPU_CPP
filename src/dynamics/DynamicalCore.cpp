#include "DynamicalCore.hpp"
#include "temporal_schemes/AdamsBashforth2.hpp"
#include "tendency_processes/AdvectionTerm.hpp"
#include "tendency_processes/StretchingTerm.hpp"
#include "tendency_processes/TwistingTerm.hpp"
#include "spatial_schemes/Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include <stdexcept>
#include <iostream> // for debugging output

namespace VVM {
namespace Dynamics {

std::unique_ptr<TemporalScheme> DynamicalCore::create_temporal_scheme(
    const std::string& var_name, 
    const nlohmann::json& var_config) const {
    
    std::string scheme_name = var_config.at("temporal_scheme");
    std::vector<std::unique_ptr<TendencyTerm>> terms;

    if (scheme_name == "AdamsBashforth2") {
        // Advection for scalar, xi, eta, zeta
        if (var_config.contains("tendency_terms") && var_config.at("tendency_terms").contains("advection")) {
            std::string advection_scheme_name = var_config.at("tendency_terms").at("advection").at("spatial_scheme");
            std::unique_ptr<SpatialScheme> spatial_scheme;
            if (advection_scheme_name == "Takacs") {
                spatial_scheme = std::make_unique<Takacs>(grid_, config_);
            } 
            else {
                throw std::runtime_error("Unknown spatial scheme: " + advection_scheme_name);
            }
            terms.push_back(std::make_unique<AdvectionTerm>(std::move(spatial_scheme), var_name));
        }
        // Stretching for xi, eta, zeta
        if (var_config.contains("tendency_terms") && var_config.at("tendency_terms").contains("stretching")) {
            std::string stretching_scheme_name = var_config.at("tendency_terms").at("stretching").at("spatial_scheme");
            std::unique_ptr<SpatialScheme> spatial_scheme;
            if (stretching_scheme_name == "Takacs") {
                spatial_scheme = std::make_unique<Takacs>(grid_, config_);
            }
            else {
                throw std::runtime_error("Unknown spatial scheme for stretching: " + stretching_scheme_name);
            }
            terms.push_back(std::make_unique<StretchingTerm>(std::move(spatial_scheme), var_name));
        }

        // Twisting for xi, eta, zeta
        if (var_config.contains("tendency_terms") && var_config.at("tendency_terms").contains("twisting")) {
            std::string twisting_scheme_name = var_config.at("tendency_terms").at("twisting").at("spatial_scheme");
            std::unique_ptr<SpatialScheme> spatial_scheme;
            if (twisting_scheme_name == "Takacs") {
                spatial_scheme = std::make_unique<Takacs>(grid_, config_);
            }
            else {
                throw std::runtime_error("Unknown spatial scheme for twisting: " + twisting_scheme_name);
            }
            terms.push_back(std::make_unique<TwistingTerm>(std::move(spatial_scheme), var_name));
        }


        return std::make_unique<AdamsBashforth2>(var_name, std::move(terms));
    }
    
    throw std::runtime_error("Unknown temporal scheme: " + scheme_name);
}


DynamicalCore::DynamicalCore(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid, 
                             const Core::Parameters& params,
                             Core::State& state)
    : config_(config), state_(state), grid_(grid), params_(params) {
    
    auto prognostic_config = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    
    for (auto& [var_name, var_conf] : prognostic_config.items()) {
        prognostic_variables_.push_back(var_name);
        
        // Create time integration instance
        variable_schemes_[var_name] = create_temporal_scheme(var_name, var_conf);
        
        int nz = grid_.get_local_total_points_z();
        int ny = grid_.get_local_total_points_y();
        int nx = grid_.get_local_total_points_x();

        // Create time integration shadow vairables in State (such as th_m)
        auto required_suffixes = variable_schemes_[var_name]->get_required_state_suffixes();
        for (const auto& suffix : required_suffixes) {
            std::string shadow_field_name = var_name + suffix;
            state_.add_field<3>(shadow_field_name, {nz, ny, nx});
            std::cout << "DynamicalCore: Automatically declared state variable '" << shadow_field_name << "' for prognostic variable '" << var_name << "'." << std::endl;
        }

        // Create tendency 4D variables to State for AB2 scheme
        if (var_conf.contains("temporal_scheme") && var_conf.at("temporal_scheme") == "AdamsBashforth2") {
            std::string tendency_field_name = "d_" + var_name; 
            state_.add_field<4>(tendency_field_name, {2, nz, ny, nx});
            std::cout << "DynamicalCore: Automatically declared 4D state variable '" << tendency_field_name << "' for AdamsBashforth2 scheme of '" << var_name << "'." << std::endl;
        }
    }
}

DynamicalCore::~DynamicalCore() = default;

void DynamicalCore::step(Core::State& state, double dt) {
    // Before tendency calculation, get the diagnostic fields such as rotation
    compute_diagnostic_fields();

    // Calculate all tendencies
    bool zeta_flag = false;
    for (const auto& var_name : prognostic_variables_) {
        variable_schemes_.at(var_name)->calculate_tendency(state, grid_, params_);
        if (var_name == "zeta") zeta_flag = true;
    }

    // Step to the next step
    // FIXME: The boundary should correspond to the variables
    Core::HaloExchanger halo_exchanger(grid_);
    for (const auto& var_name : prognostic_variables_) {
        variable_schemes_.at(var_name)->step(state, grid_, params_, dt);

        VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, var_name);
        if (var_name == "zeta") halo_exchanger.exchange_halos_top_slice(state.get_field<3>(var_name));
        else halo_exchanger.exchange_halos(state.get_field<3>(var_name));
        bc_manager.apply_z_bcs_to_field(state.get_field<3>(var_name));
    }

    if (zeta_flag) {
        compute_zeta_vertical_structure(state);
        halo_exchanger.exchange_halos_top_slice(state.get_field<3>("zeta"));
    }
}

void DynamicalCore::compute_diagnostic_fields() const {
    auto scheme = std::make_unique<Takacs>(grid_, config_);

    auto& R_xi_field = state_.get_field<3>("R_xi");
    auto& R_eta_field = state_.get_field<3>("R_eta");
    auto& R_zeta_field = state_.get_field<3>("R_zeta");

    scheme->calculate_R_xi(state_, grid_, params_, R_xi_field);
    scheme->calculate_R_eta(state_, grid_, params_, R_eta_field);
    scheme->calculate_R_zeta(state_, grid_, params_, R_zeta_field);
}

void DynamicalCore::compute_zeta_vertical_structure(Core::State& state) const {
    auto scheme = std::make_unique<Takacs>(grid_, config_);
    auto& zeta_field = state.get_field<3>("zeta");
    auto zeta_data = zeta_field.get_mutable_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    
    const double dz = grid_.get_dz();

    Core::Field<3> rhs_field("rhs_zeta_diag", {nz, ny, nx});
    scheme->calculate_vorticity_divergence(state, grid_, params_, rhs_field);
    auto rhs_data = rhs_field.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();

    Kokkos::parallel_for("zeta_downward_integration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // The for-loop inside is to prevent racing condition because lower layers depend on upper layers.
            for (int k = nz-h-2; k >= h; --k) {
                zeta_data(k,j,i) = zeta_data(k+1,j,i) + rhs_data(k,j,i) * -dz / flex_height_coef_up(k);
            }
            // WARNING: NK3 has a upward integration in original VVM code.
            zeta_data(nz-h,j,i) = zeta_data(nz-h-1,j,i) + rhs_data(nz-h-1,j,i) * dz / flex_height_coef_up(nz-h-1);
        }
    );
    Core::HaloExchanger halo_exchanger(grid_);
    halo_exchanger.exchange_halos(zeta_field);

    // FIXME: vertical boundary process
}

} // namespace Dynamics
} // namespace VVM

