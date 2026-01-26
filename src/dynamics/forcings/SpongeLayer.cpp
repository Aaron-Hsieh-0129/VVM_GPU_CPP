#include "SpongeLayer.hpp"
#include <iostream>
#include <cmath>

namespace VVM {
namespace Dynamics {

SpongeLayer::SpongeLayer(const Utils::ConfigurationManager& config, 
                         const Core::Grid& grid, 
                         const Core::Parameters& params,
                         Core::HaloExchanger& halo_exchanger,
                         Core::State& state)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger)
{
    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv"};

    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        std::vector<std::string> p3_vars = {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"};
        thermodynamics_vars_.insert(thermodynamics_vars_.end(), p3_vars.begin(), p3_vars.end());
    }
}

void SpongeLayer::initialize(Core::State& state) {
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    auto z_mid_host = params_.z_mid.get_host_data();
    auto z_up_host = params_.z_up.get_host_data();

    auto dims = std::array<int, 3>{nz, ny, nx};

    for (const auto& var_name : thermodynamics_vars_) {
        std::string fe_tendency_name = "fe_tendency_" + var_name;
        if (!state.has_field(fe_tendency_name)) {
            state.add_field<3>(fe_tendency_name, dims);
        }
    }
    for (const auto& var_name : dynamics_vars_) {
        std::string fe_tendency_name = "fe_tendency_" + var_name;
        if (!state.has_field(fe_tendency_name)) {
            if (var_name == "zeta") state.add_field<2>(fe_tendency_name, {ny, nx});
            else state.add_field<3>(fe_tendency_name, dims);
        }
    }
    if (!state.has_field("CGR_thermo")) state.add_field<1>("CGR_thermo", {nz});
    if (!state.has_field("CGR_vort")) state.add_field<1>("CGR_vort", {nz});

    auto& CRAD = CRAD_;
    CRAD = 1. / config_.get_value<double>("dynamics.forcings.sponge_layer.inv_CRAD", -1.);
    double sponge_layer_base = config_.get_value<double>("dynamics.forcings.sponge_layer.sponge_layer_base", -1);

    auto& k_start_thermo = k_start_thermo_;
    for (int k = nz-h; k > h; k--) {
        if (z_mid_host(k) < sponge_layer_base)
        if (sponge_layer_base >= z_mid_host(k)) {
            k_start_thermo = k+1;
            break;
        }
    }

    auto& k_start_vort = k_start_vort_;
    for (int k = nz-h; k > h; k--) {
        if (z_up_host(k) < sponge_layer_base)
        if (sponge_layer_base >= z_up_host(k)) {
            k_start_vort = k+1;
            break;
        }
    }

    std::cout << "--- Initializing Sponge Layer --- " << std::endl;
    std::cout << "The sponge layer for thermo variables will start at k (physical grid) = " << k_start_thermo - h << ", z = " << z_mid_host(k_start_thermo) << std::endl;
    std::cout << "The sponge layer for vort variables will start at k (physical grid) = " << k_start_vort - h << ", z = " << z_up_host(k_start_vort) << std::endl;

    auto& z_mid = params_.z_mid.get_device_data();
    auto& z_up = params_.z_up.get_device_data();
    auto& CGR_thermo = state.get_field<1>("CGR_thermo").get_mutable_device_data();
    Kokkos::parallel_for("assign_coefficient", Kokkos::RangePolicy<>(k_start_thermo, nz-h),
        KOKKOS_LAMBDA(const int k) {
            CGR_thermo(k) = CRAD*(z_mid(k)-z_mid(k_start_thermo-1))/(z_mid(nz-h-1)-z_mid(k_start_thermo-1));
        }
    );
    state.get_field<1>("CGR_thermo").print_profile(grid_, 0, 0, 0);
    
    auto& CGR_vort = state.get_field<1>("CGR_vort").get_mutable_device_data();
    Kokkos::parallel_for("assign_coefficient", Kokkos::RangePolicy<>(k_start_vort, nz-h-1),
        KOKKOS_LAMBDA(const int k) {
            // CGR_vort(k) = CRAD*(z_up(k)-z_mid(k_start_vort-1))/(z_mid(nz-h-1)-z_mid(k_start_vort-1));
            // FIXME: This follows original VVM now, but I think the ratio should be considered carefully.
            CGR_vort(k) = CRAD*(z_up(k)-z_mid(k_start_thermo-1))/(z_mid(nz-h-1)-z_mid(k_start_thermo-1));
        }
    );
    state.get_field<1>("CGR_vort").print_profile(grid_, 0, 0, 0);
}

template<size_t Dim>
void SpongeLayer::calculate_tendencies(Core::State& state, 
                                       const std::string& var_name, 
                                       Core::Field<Dim>& out_tendency) 
{
    const auto& CGR_thermo = state.get_field<1>("CGR_thermo").get_device_data();
    const auto& CGR_vort = state.get_field<1>("CGR_vort").get_device_data();
    const auto& var = state.get_field<3>(var_name).get_device_data();
    auto& tend = out_tendency.get_mutable_device_data();
    
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();

    auto& ref_profile = ref_profile_;
    if (var_name == "th") {
        ref_profile = state.get_field<1>("thbar").get_device_data();
    } 
    else if (var_name == "qv") {
        ref_profile = state.get_field<1>("qvbar").get_device_data();
    }

    const auto& k_start_thermo = k_start_thermo_;
    const auto& k_start_vort = k_start_vort_;
    int k_end = nz-h;
    if (var_name == "xi" || var_name == "eta") k_end = nz-h-1;

    if constexpr (Dim == 3) {
        if (var_name == "xi" || var_name == "eta") {
            Kokkos::parallel_for("Sponge_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{k_start_vort, h, h}}, {{k_end, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    tend(k, j, i) += -CGR_vort(k) * (var(k, j, i));
                }
            );
        }
        else if (var_name == "th" || var_name == "qv") {
            Kokkos::parallel_for("Sponge_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{k_start_thermo, h, h}}, {{k_end, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    tend(k, j, i) += -CGR_thermo(k) * (var(k, j, i) - ref_profile(k));
                }
            );
        }
        else {
            Kokkos::parallel_for("Sponge_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{k_start_thermo, h, h}}, {{k_end, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    tend(k, j, i) += -CGR_thermo(k) * var(k, j, i);
                }
            );
        }
    } 
    else if constexpr (Dim == 2) {
        int NK2 = nz-h-1;
        Kokkos::parallel_for("Sponge_Tendency_" + var_name,
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                tend(j, i) += -CGR_vort(NK2) * (var(NK2, j, i));
            }
        );
    }
}

template void SpongeLayer::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<2ul>& out_tendency);
template void SpongeLayer::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Dynamics
} // namespace VVM
