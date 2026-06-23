#include "Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"

namespace VVM {
namespace Dynamics {

Takacs::Takacs(const Utils::ConfigurationManager& config, const Core::Grid& grid, Core::HaloExchanger& halo_exchanger, const Core::BoundaryConditionManager& bc_manager)
    : config_(config), halo_exchanger_(halo_exchanger), bc_manager_(bc_manager) {}

void Takacs::calculate_flux_convergence_x(
    const Core::Field<3>& scalar, const Core::Field<3>& u_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& u = u_field.get_device_data();
    const auto& q = scalar.get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    int k_start = h;
    int k_end = nz-h;
    // zeta only needs to do the top prediction
    if (var_name == "zeta") {
        k_start = nz-h-1;
        k_end = nz-h;
    }
    else if (var_name == "eta" || var_name == "xi") {
        k_end = nz-h-1;
    }

    auto rdx_view = params.rdx;

    const int j_start = h;
    const int j_end = ny - h;
    const int i_start = h;
    const int i_end = nx - h;
    const int num_j = j_end - j_start;
    const int num_i = i_end - i_start;
    const int league_size = num_j * num_i;

    Kokkos::parallel_for("Takacs_flux_convergence_x_fused", 
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = j_start + league_rank / num_i;
            const int i = i_start + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k_start, k_end), 
                [&](const int k) {
                    
                    auto get_uplus = [&](int idx_i) {
                        return real(0.5)*(u(k,j,idx_i) + Kokkos::abs(u(k,j,idx_i)));
                    };
                    auto get_uminus = [&](int idx_i) {
                        return real(0.5)*(u(k,j,idx_i) - Kokkos::abs(u(k,j,idx_i)));
                    };

                    VVM::Real up_i   = get_uplus(i);
                    VVM::Real up_im1 = get_uplus(i-1);
                    VVM::Real um_i   = get_uminus(i);
                    VVM::Real um_ip1 = get_uminus(i+1);

                    VVM::Real flux_i = u(k,j,i)*(q(k,j,i+1)+q(k,j,i)) + 
                            -real(1.)/real(3.)*( 
                                 up_i*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(up_i)*Kokkos::sqrt(up_im1)*(q(k,j,i)-q(k,j,i-1)) - 
                                 um_i*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(um_i))*Kokkos::sqrt(Kokkos::abs(um_ip1))*(q(k,j,i+2)-q(k,j,i+1)) 
                            );

                    VVM::Real up_im2 = get_uplus(i-2);
                    VVM::Real um_im1 = get_uminus(i-1);

                    VVM::Real flux_im1 = u(k,j,i-1)*(q(k,j,i)+q(k,j,i-1)) + 
                            -real(1.)/real(3.)*( 
                                 up_im1*(q(k,j,i)-q(k,j,i-1)) - Kokkos::sqrt(up_im1)*Kokkos::sqrt(up_im2)*(q(k,j,i-1)-q(k,j,i-2)) - 
                                 um_im1*(q(k,j,i)-q(k,j,i-1)) - Kokkos::sqrt(Kokkos::abs(um_im1))*Kokkos::sqrt(Kokkos::abs(um_i))*(q(k,j,i+1)-q(k,j,i)) 
                            );

                    tendency(k,j,i) += -real(0.5)*(flux_i - flux_im1) * rdx_view();
                }
            );
        }
    );
    return;
}

void Takacs::calculate_flux_convergence_y(
    const Core::Field<3>& scalar, const Core::Field<3>& v_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();
    auto rdy_view = params.rdy;

    const auto& v = v_field.get_device_data();
    const auto& q = scalar.get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    int k_start = h;
    int k_end = nz-h;
    // zeta only needs to do the top prediction
    if (var_name == "zeta") {
        k_start = nz-h-1;
        k_end = nz-h;
    }
    else if (var_name == "xi" || var_name == "eta") {
        k_end = nz-h-1;
    }

    const int j_start = h;
    const int j_end = ny - h;
    const int i_start = h;
    const int i_end = nx - h;
    const int num_j = j_end - j_start;
    const int num_i = i_end - i_start;
    const int league_size = num_j * num_i;
    Kokkos::parallel_for("Takacs_flux_convergence_y_fused", 
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = j_start + league_rank / num_i;
            const int i = i_start + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, k_start, k_end), 
                [&](const int k) {
                    
                    auto get_vplus = [&](int idx_j) {
                        return real(0.5)*(v(k,idx_j,i) + Kokkos::abs(v(k,idx_j,i)));
                    };
                    auto get_vminus = [&](int idx_j) {
                        return real(0.5)*(v(k,idx_j,i) - Kokkos::abs(v(k,idx_j,i)));
                    };

                    VVM::Real vp_j   = get_vplus(j);
                    VVM::Real vp_jm1 = get_vplus(j-1);
                    VVM::Real vm_j   = get_vminus(j);
                    VVM::Real vm_jp1 = get_vminus(j+1);

                    VVM::Real flux_j = v(k,j,i)*(q(k,j+1,i)+q(k,j,i)) + 
                                  -real(1.)/real(3.)*( 
                                     vp_j*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(vp_j)*Kokkos::sqrt(vp_jm1)*(q(k,j,i)-q(k,j-1,i)) - 
                                     vm_j*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(vm_j))*Kokkos::sqrt(Kokkos::abs(vm_jp1))*(q(k,j+2,i)-q(k,j+1,i)) 
                                  );

                    VVM::Real vp_jm2 = get_vplus(j-2);
                    VVM::Real vm_jm1 = get_vminus(j-1);

                    VVM::Real flux_jm1 = v(k,j-1,i)*(q(k,j,i)+q(k,j-1,i)) + 
                                    -real(1.)/real(3.)*( 
                                       vp_jm1*(q(k,j,i)-q(k,j-1,i)) - Kokkos::sqrt(vp_jm1)*Kokkos::sqrt(vp_jm2)*(q(k,j-1,i)-q(k,j-2,i)) - 
                                       vm_jm1*(q(k,j,i)-q(k,j-1,i)) - Kokkos::sqrt(Kokkos::abs(vm_jm1))*Kokkos::sqrt(Kokkos::abs(vm_j))*(q(k,j+1,i)-q(k,j,i)) 
                                    );

                    tendency(k,j,i) += -real(0.5)*(flux_j - flux_jm1) * rdy_view();
                }
            );
        }
    );
    return;
}

void Takacs::calculate_flux_convergence_z(
    const Core::Field<3>& scalar, const Core::Field<3>& w_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& w = w_field.get_device_data();
    const auto& q = scalar.get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    auto rdz_view = params.rdz;
    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const bool is_idealized = (config_.get_value<std::string>("simulation.idealized_test", "none") == "advection_w");
    const bool is_zeta = (var_name == "zeta");
    const bool is_xi_or_eta = (var_name == "xi" || var_name == "eta");

    const int j_start = h;
    const int j_end = ny - h;
    const int i_start = h;
    const int i_end = nx - h;
    const int num_j = j_end - j_start;
    const int num_i = i_end - i_start;
    const int league_size = num_j * num_i;

    int tend_k_min = h;
    int tend_k_max = nz - h - 1;
    if (var_name == "zeta") {
        tend_k_min = nz - h - 1;
        tend_k_max = nz - h - 1;
    }
    else if (var_name == "xi" || var_name == "eta") {
        tend_k_max = nz - h - 2;
    }

    Kokkos::parallel_for("Takacs_flux_convergence_z_fused", 
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = j_start + league_rank / num_i;
            const int i = i_start + league_rank % num_i;

            auto get_wplus = [&](int idx_k) { return real(0.5)*(w(idx_k,j,i) + Kokkos::abs(w(idx_k,j,i))); };
            auto get_wminus = [&](int idx_k) { return real(0.5)*(w(idx_k,j,i) - Kokkos::abs(w(idx_k,j,i))); };

            auto compute_raw_f = [&](int k_raw) -> VVM::Real {
                VVM::Real raw_f = w(k_raw,j,i)*(q(k_raw+1,j,i)+q(k_raw,j,i));
                if (k_raw == h) {
                    if (w(k_raw,j,i) < 0.) {
                        VVM::Real wm_k = get_wminus(k_raw);
                        VVM::Real wm_kp1 = get_wminus(k_raw+1);
                        raw_f += -real(1.)/real(3.)*( -wm_k*(q(k_raw+1,j,i)-q(k_raw,j,i)) - Kokkos::sqrt(Kokkos::abs(wm_k))*Kokkos::sqrt(Kokkos::abs(wm_kp1))*(q(k_raw+2,j,i)-q(k_raw+1,j,i)) );
                    }
                } 
                else if (k_raw == nz-h-2) {
                    if (w(k_raw,j,i) >= 0.) {
                        VVM::Real wp_k = get_wplus(k_raw);
                        VVM::Real wp_km1 = get_wplus(k_raw-1);
                        raw_f += -real(1.)/real(3.)*( wp_k*(q(k_raw+1,j,i)-q(k_raw,j,i)) - Kokkos::sqrt(wp_k)*Kokkos::sqrt(wp_km1)*(q(k_raw,j,i)-q(k_raw-1,j,i)) );
                    }
                } 
                else {
                    VVM::Real wp_k = get_wplus(k_raw);
                    VVM::Real wp_km1 = get_wplus(k_raw-1);
                    VVM::Real wm_k = get_wminus(k_raw);
                    VVM::Real wm_kp1 = get_wminus(k_raw+1);
                    raw_f += -real(1.)/real(3.)*( 
                         wp_k*(q(k_raw+1,j,i)-q(k_raw,j,i)) - Kokkos::sqrt(wp_k)*Kokkos::sqrt(wp_km1)*(q(k_raw,j,i)-q(k_raw-1,j,i)) - 
                         wm_k*(q(k_raw+1,j,i)-q(k_raw,j,i)) - Kokkos::sqrt(Kokkos::abs(wm_k))*Kokkos::sqrt(Kokkos::abs(wm_kp1))*(q(k_raw+2,j,i)-q(k_raw+1,j,i)) 
                    );
                }
                return raw_f;
            };

            auto get_flux_z = [&](int k_idx) -> VVM::Real {
                if (is_zeta) {
                    if (k_idx == nz-h-2) {
                        VVM::Real f = w(k_idx,j,i)*(q(k_idx+1,j,i)+q(k_idx,j,i));
                        if (w(k_idx,j,i) >= real(0.)) {
                            VVM::Real wp_k = get_wplus(k_idx);
                            VVM::Real wp_km1 = get_wplus(k_idx-1);
                            f += -real(1.)/real(3.)*( wp_k*(q(k_idx+1,j,i)-q(k_idx,j,i)) - Kokkos::sqrt(wp_k)*Kokkos::sqrt(wp_km1)*(q(k_idx,j,i)-q(k_idx-1,j,i)) );
                        }
                        return f;
                    }
                    return real(0.);
                }
                else if (is_xi_or_eta) {
                    if (k_idx >= h-1 && k_idx <= nz-h-2) {
                        VVM::Real f = w(k_idx,j,i)*(q(k_idx+1,j,i)+q(k_idx,j,i));
                        if (k_idx == h-1) {
                            if (w(k_idx,j,i) < real(0.)) {
                                VVM::Real wm_k = get_wminus(k_idx);
                                VVM::Real wm_kp1 = get_wminus(k_idx+1);
                                f += -real(1.)/real(3.)*( -wm_k*(q(k_idx+1,j,i)-q(k_idx,j,i)) - Kokkos::sqrt(Kokkos::abs(wm_k))*Kokkos::sqrt(Kokkos::abs(wm_kp1))*(q(k_idx+2,j,i)-q(k_idx+1,j,i)) );
                            }
                        }
                        else if (k_idx == nz-h-2) {
                            if (w(k_idx,j,i) >= 0.) {
                                VVM::Real wp_k = get_wplus(k_idx);
                                VVM::Real wp_km1 = get_wplus(k_idx-1);
                                f += -real(1.)/real(3.)*( wp_k*(q(k_idx+1,j,i)-q(k_idx,j,i)) - Kokkos::sqrt(wp_k)*Kokkos::sqrt(wp_km1)*(q(k_idx,j,i)-q(k_idx-1,j,i)) );
                            }
                        } 
                        else {
                            VVM::Real wp_k = get_wplus(k_idx);
                            VVM::Real wp_km1 = get_wplus(k_idx-1);
                            VVM::Real wm_k = get_wminus(k_idx);
                            VVM::Real wm_kp1 = get_wminus(k_idx+1);
                            f += -real(1.)/real(3.)*( 
                                 wp_k*(q(k_idx+1,j,i)-q(k_idx,j,i)) - Kokkos::sqrt(wp_k)*Kokkos::sqrt(wp_km1)*(q(k_idx,j,i)-q(k_idx-1,j,i)) - 
                                 wm_k*(q(k_idx+1,j,i)-q(k_idx,j,i)) - Kokkos::sqrt(Kokkos::abs(wm_k))*Kokkos::sqrt(Kokkos::abs(wm_kp1))*(q(k_idx+2,j,i)-q(k_idx+1,j,i)) 
                            );
                        }
                        return f;
                    }
                    return real(0.);
                }
                else { 
                    if (k_idx == h-1) return is_idealized ? compute_raw_f(nz-h-2) : real(0.);
                    if (k_idx == nz-h-1) return is_idealized ? compute_raw_f(h) : real(0.);
                    if (k_idx >= h && k_idx <= nz-h-2) return compute_raw_f(k_idx);
                    return real(0.);
                }
            };

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tend_k_min, tend_k_max + 1), 
                [&](const int k) {
                    VVM::Real flux_k = get_flux_z(k);
                    VVM::Real flux_km1 = get_flux_z(k-1);

                    if (is_zeta) {
                        tendency(k,j,i) += 0.5 * flux_km1 * rdz_view() * flex_height_coef_mid(k);
                    }
                    else if (is_xi_or_eta) {
                        tendency(k,j,i) += -0.5 * (flux_k - flux_km1) * rdz_view() * flex_height_coef_up(k);
                    }
                    else {
                        tendency(k,j,i) += -0.5 * (flux_k - flux_km1) * rdz_view() * flex_height_coef_mid(k);
                    }
                }
            );
        }
    );
    return;
}

void Takacs::calculate_stretching_tendency_x(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();
    
    const auto& rdx = params.rdx;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    const auto& fact2_xi_eta = params.fact2_xi_eta.get_device_data();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;
    
    // Implements Eq. (3.25) for [ρ₀ξ(∂u/∂x)] at (i, j+1/2, k+1/2)
    Kokkos::parallel_for("stretching_term_xi_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, h, nz-h-1),
                [&](const int k) {
                    const VVM::Real term_at_j_plus_1 = 
                        (xi(k,j,i)+xi(k,j+1,i)) * (fact1_xi_eta(k) * rhobar(k+1) * (u(k+1, j+1, i) - u(k+1, j+1, i-1)) +
                         fact2_xi_eta(k) * rhobar(k)   * (u(k,   j+1, i) - u(k,   j+1, i-1)) );

                    const VVM::Real term_at_j = 
                        (xi(k,j,i)+xi(k,j-1,i)) * (fact1_xi_eta(k) * rhobar(k+1) * (u(k+1, j, i) - u(k+1, j, i-1)) +
                         fact2_xi_eta(k) * rhobar(k)   * (u(k,   j, i) - u(k,   j, i-1)) );

                    tendency(k, j, i) += real(0.125) * rdx() * (term_at_j_plus_1 + term_at_j);
                }
            );
        }
    );
    return;
}

void Takacs::calculate_stretching_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& v = state.get_field<3>("v").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();
    
    const auto& rdy = params.rdy;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    const auto& fact2_xi_eta = params.fact2_xi_eta.get_device_data();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    // Implements Eq. (3.26) for [ρ₀η(∂v/∂y)] at (i+1/2, j, k+1/2)
    Kokkos::parallel_for("stretching_term_eta_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, h, nz-h-1),
                [&](const int k) {
                    const VVM::Real term_at_i_plus_1 = 
                        (eta(k,j,i)+eta(k,j,i+1)) * (fact1_xi_eta(k) * rhobar(k+1) * (v(k+1, j, i+1) - v(k+1, j-1, i+1)) +
                         fact2_xi_eta(k) * rhobar(k)   * (v(k,   j, i+1) - v(k,   j-1, i+1)) );

                    const VVM::Real term_at_i = 
                        (eta(k,j,i)+eta(k,j,i-1)) * (fact1_xi_eta(k) * rhobar(k+1) * (v(k+1, j, i) - v(k+1, j-1, i)) +
                         fact2_xi_eta(k) * rhobar(k)   * (v(k,   j, i) - v(k,   j-1, i)) );

                    tendency(k, j, i) += real(0.125) * rdy() * (term_at_i_plus_1 + term_at_i);
                }
            );
        }
    );
    return;
}

void Takacs::calculate_stretching_tendency_z(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
    
    const auto& w = state.get_field<3>("w").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    const auto& rdz = params.rdz;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    // Implements Eq. (3.27) for [ρ₀ζ(∂w/∂z)] at (i+1/2, j+1/2, k)
    Kokkos::parallel_for("stretching_term_zeta_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nz-h-1, nz-h),
                [&](const int k) {
                    const VVM::Real term_at_i = (zeta(k,j,i-1)+zeta(k,j,i)) * (w(k-1, j, i) + w(k-1, j+1, i));
                    const VVM::Real term_at_i_plus_1 = (zeta(k,j,i)+zeta(k,j,i+1)) * (w(k-1, j, i+1) + w(k-1, j+1, i+1));

                    tendency(k, j, i) += -real(0.125) * flex_height_coef_mid(k) * rdz() * rhobar(k) * (term_at_i + term_at_i_plus_1);
                }
            );
        }
    );
    return;
}

// Equation (3.32)
// This is the deformation of strain
void Takacs::calculate_R_xi(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_R_xi) const {
    
    const auto& v = state.get_field<3>("v").get_device_data();
    const auto& w = state.get_field<3>("w").get_device_data();
    auto& R_xi = out_R_xi.get_mutable_device_data();

    const auto& rdy = params.rdy;
    const auto& rdz = params.rdz;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    Kokkos::parallel_for("compute_R_xi_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            // R_xi at i, j+1/2, k+1/2
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, nz - h),
                [&](const int k) {
                    R_xi(k, j, i) = (w(k, j+1, i) - w(k, j, i)) * rdy() +
                                    (v(k+1, j, i) - v(k, j, i)) * rdz() * flex_height_coef_up(k);
                }
            );
        }
    );

    halo_exchanger_.exchange_halos(out_R_xi);
    bc_manager_.apply_horizontal_bcs(out_R_xi);
    return;
}

void Takacs::calculate_R_eta(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_R_eta) const {

    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& w = state.get_field<3>("w").get_device_data();
    auto& R_eta = out_R_eta.get_mutable_device_data();

    const auto& rdx = params.rdx;
    const auto& rdz = params.rdz;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    Kokkos::parallel_for("compute_R_eta_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            // R_eta at i+1/2, j, k+1/2
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, nz - h),
                [&](const int k) {
                    R_eta(k, j, i) = (w(k, j, i+1) - w(k, j, i)) * rdx() +
                                     (u(k+1, j, i) - u(k, j, i)) * rdz() * flex_height_coef_up(k);
                }
            );
        }
    );

    halo_exchanger_.exchange_halos(out_R_eta);
    bc_manager_.apply_horizontal_bcs(out_R_eta);
    return;
}

void Takacs::calculate_R_zeta(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_R_zeta) const {

    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& v = state.get_field<3>("v").get_device_data();
    auto& R_zeta = out_R_zeta.get_mutable_device_data();

    auto rdx = params.rdx;
    auto rdy = params.rdy;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    Kokkos::parallel_for("compute_R_zeta_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            // R_zeta at i+1/2, j+1/2, k
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, nz),
                [&](const int k) {
                    R_zeta(k, j, i) = (v(k, j, i+1) - v(k, j, i)) * rdx() +
                                      (u(k, j+1, i) - u(k, j, i)) * rdy();
                }
            );
        }
    );

    halo_exchanger_.exchange_halos(out_R_zeta);
    bc_manager_.apply_horizontal_bcs(out_R_zeta);
    return;
}


void Takacs::calculate_twisting_tendency_x(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& R_eta_field = state.get_field<3>("R_eta");
    const auto& R_zeta_field = state.get_field<3>("R_zeta");
    auto& R_eta = R_eta_field.get_device_data();
    auto& R_zeta = R_zeta_field.get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    const auto& fact2_xi_eta = params.fact2_xi_eta.get_device_data();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    // Implements Eq. (3.28) for [0.5ρ₀(eta*Rzeta+zeta*Reta)] at (i+1/2, j+1/2, k)
    Kokkos::parallel_for("twisting_term_xi_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, h, nz-h),
                [&](const int k) {
                    // WARNING: The documentation has rho weighted but the VVM code doesn't have. This code follows the VVM code.
                    const VVM::Real term_etaRzeta = (
                        (eta(k,j+1,i  )+eta(k,j,i  )) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j,i  ) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j,i  ))
                      + (eta(k,j+1,i-1)+eta(k,j,i-1)) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j,i-1) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j,i-1))
                    );

                    const VVM::Real term_zetaReta = (
                        rhobar_up(k)*(zeta(k,j,i  )+zeta(k+1,j,i  ))*(R_eta(k,j+1,i  )+R_eta(k,j,i  ))
                      + rhobar_up(k)*(zeta(k,j,i-1)+zeta(k+1,j,i-1))*(R_eta(k,j+1,i-1)+R_eta(k,j,i-1))
                    );

                    // WARNING: term_etaRzeta has a negative sign in original VVM because the definition of eta in that VVM is negative from this one.
                    tendency(k, j, i) += real(0.0625) * (-term_etaRzeta + term_zetaReta);
                }
            );
        }
    );
    return;
}

void Takacs::calculate_twisting_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& R_xi_field = state.get_field<3>("R_xi");
    const auto& R_zeta_field = state.get_field<3>("R_zeta");
    auto& R_xi = R_xi_field.get_device_data();
    auto& R_zeta = R_zeta_field.get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    const auto& fact2_xi_eta = params.fact2_xi_eta.get_device_data();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    // Implements Eq. (3.29) for [0.5ρ₀(xi*Rzeta+zeta*Rxi)]
    Kokkos::parallel_for("twisting_term_eta_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, h, nz-h),
                [&](const int k) {
                    const VVM::Real term_xiRzeta = (
                        (xi(k,j  ,i+1)+xi(k,j  ,i)) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j  ,i) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j  ,i))
                      + (xi(k,j-1,i+1)+xi(k,j-1,i)) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j-1,i) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j-1,i))
                    );

                    const VVM::Real term_zetaRxi = (
                        rhobar_up(k)*(zeta(k,j  ,i)+zeta(k+1,j  ,i))*(R_xi(k,j  ,i)+R_xi(k,j  ,i+1))
                      + rhobar_up(k)*(zeta(k,j-1,i)+zeta(k+1,j-1,i))*(R_xi(k,j-1,i)+R_xi(k,j-1,i+1))
                    );

                    // WARNING: term_xiRzeta and eter_zetaRxi have negative signs in original VVM because the definition of eta in that VVM is negative from this one.
                    tendency(k, j, i) += real(0.0625) * -(term_xiRzeta + term_zetaRxi);
                }
            );
        }
    );
    return;
}

void Takacs::calculate_twisting_tendency_z(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& R_xi = state.get_field<3>("R_xi").get_device_data();
    const auto& R_eta = state.get_field<3>("R_eta").get_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& fact1 = params.fact1_zeta.get_device_data();
    const auto& fact2 = params.fact2_zeta.get_device_data();

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    // Implements Eq. (3.30) for [0.5ρ₀(xi*Reta+eta*Rxi)]
    Kokkos::parallel_for("twisting_term_zeta_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nz-h-1, nz-h),
                [&](const int k) {
                    // WARNING: The documentation has rho weighted but the VVM code doesn't have. This code follows the VVM code.
                    const VVM::Real term_xiReta = (
                        fact1()*(xi(k,j,i+1)+xi(k  ,j,i  )) * (R_eta(k  ,j+1,i)+R_eta(k  ,j,i))
                      + fact2()*(xi(k-1,j,i)+xi(k-1,j,i+1)) * (R_eta(k-1,j+1,i)+R_eta(k-1,j,i))
                    );

                    const VVM::Real term_etaRxi = (
                        fact1()*(eta(k  ,j+1,i)+eta(k  ,j,i)) * (R_xi(k  ,j,i)+R_xi(k  ,j,i+1))
                      + fact2()*(eta(k-1,j+1,i)+eta(k-1,j,i)) * (R_xi(k-1,j,i)+R_xi(k-1,j,i+1))
                    );

                    // WARNING: term_etaRxi has a negative sign in original VVM because the definition of eta in that VVM is negative from this one.
                    tendency(k, j, i) += real(0.0625) * (term_xiReta - term_etaRxi);
                }
            );
        }
    );
    return;
}


void Takacs::calculate_vorticity_divergence(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_field) const {

    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    auto& out_data = out_field.get_mutable_device_data();

    const auto& rdx = params.get_value_host(params.rdx);
    const auto& rdy = params.get_value_host(params.rdy);

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("vorticity_divergence",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const VVM::Real d_xi_dx = (xi(k, j, i+1) - xi(k, j, i)) * rdx;
            const VVM::Real d_eta_dy = (eta(k, j+1, i) - eta(k, j, i)) * rdy;
            // WARNING: Original VVM has a negative sign for eta due to different definition
            out_data(k, j, i) = -(d_xi_dx - d_eta_dy);
        }
    );
    halo_exchanger_.exchange_halos(out_field);
    bc_manager_.apply_horizontal_bcs(out_field);
}

void Takacs::calculate_buoyancy_tendency_x(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {
    
    const auto& thbar = state.get_field<1>("thbar").get_device_data();
    const auto& th = state.get_field<3>("th").get_device_data();
    const auto& qv = state.get_field<3>("qv").get_device_data();

    // qp is used when P3 is turned on. 
    // To prevent 'if' at each step or #define flags, qp points to a random variables and times a 0 coefficient if P3 is not defined.
    bool has_qp = state.has_field("qp");
    auto qp = has_qp ? state.get_field<3>("qp").get_device_data() : qv;
    const VVM::Real qp_coeff = has_qp ? real(1.0) : real(0.0);

    auto& tendency = out_tendency.get_mutable_device_data();
    const auto& rdy = params.rdy;
    const auto& gravity = params.gravity;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& ITYPEV = state.get_field<3>("ITYPEV").get_device_data();
    const auto& max_topo_idx = params.max_topo_idx;

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    Kokkos::parallel_for("buoyancy_tendency_x_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, h, nz-h-1),
                [&](const int k) {
                    const VVM::Real dB_dy = (th(k  ,j+1,i)-th(k  ,j,i)) / thbar(k)
                                       + (th(k+1,j+1,i)-th(k+1,j,i)) / thbar(k+1)
                                       + real(0.608)*(qv(k,j+1,i)-qv(k,j,i)+qv(k+1,j+1,i)-qv(k+1,j,i))
                                       - qp_coeff * (qp(k,j+1,i)-qp(k,j,i)+qp(k+1,j+1,i)-qp(k+1,j,i));

                    tendency(k, j, i) += gravity() * real(0.5) * dB_dy * rdy();

                    if (k <= max_topo_idx && ITYPEV(k,j,i) == 0) {
                        tendency(k, j, i) = real(0.);
                    }
                }
            );
        }
    );
    return;
}

void Takacs::calculate_buoyancy_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {
    
    const auto& thbar = state.get_field<1>("thbar").get_device_data();
    const auto& th = state.get_field<3>("th").get_device_data();
    const auto& qv = state.get_field<3>("qv").get_device_data();

    // qp is used when P3 is turned on. 
    // To prevent 'if' at each step or #define flags, qp points to a random variables and times a 0 coefficient if P3 is not defined.
    bool has_qp = state.has_field("qp");
    auto qp = has_qp ? state.get_field<3>("qp").get_device_data() : qv;
    const VVM::Real qp_coeff = has_qp ? real(1.0) : real(0.0);

    auto& tendency = out_tendency.get_mutable_device_data();
    auto& rdx = params.rdx;
    auto& gravity = params.gravity;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& ITYPEU = state.get_field<3>("ITYPEU").get_device_data();
    const int max_topo_idx = params.max_topo_idx;

    const int num_j = ny - 2 * h;
    const int num_i = nx - 2 * h;
    const int league_size = num_j * num_i;

    Kokkos::parallel_for("buoyancy_tendency_y_team",
        TeamPolicy(league_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MemberType& team) {
            
            const int league_rank = team.league_rank();
            const int j = h + league_rank / num_i;
            const int i = h + league_rank % num_i;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, h, nz-h-1),
                [&](const int k) {
                    const VVM::Real dB_dx = (th(k  ,j,i+1)-th(k  ,j,i)) / thbar(k)
                                       + (th(k+1,j,i+1)-th(k+1,j,i)) / thbar(k+1)
                                       + (real(0.608)*(qv(k,j,i+1)-qv(k,j,i)+qv(k+1,j,i+1)-qv(k+1,j,i)))
                                       - qp_coeff * (qp(k,j,i+1)-qp(k,j,i)+qp(k+1,j,i+1)-qp(k+1,j,i));

                    // WARNING: dB_dy has a negative sign in original VVM because the definition of eta in that VVM is negative from this one.
                    // Fix the comparison negative sign
                    tendency(k, j, i) += gravity() * real(0.5) * dB_dx * rdx();

                    if (k <= max_topo_idx && ITYPEU(k,j,i) == 0) {
                        tendency(k, j, i) = real(0.);
                    }
                }
            );
        }
    );
}

void Takacs::calculate_coriolis_tendency_x(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {
    
    auto& tendency = out_tendency.get_mutable_device_data();
    const auto& u = state.get_field<3>("u").get_device_data();
    // const auto& f = state.get_field<1>("f").get_device_data();
    const auto& f_2d = state.get_field<2>("f_2d").get_device_data();
    const auto& rdz = params.rdz;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("coriolis_tendency_x",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h-1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k, j, i) += real(0.25) * f_2d(j,i) * flex_height_coef_up(k) * rdz() *
                                 (u(k+1,j,i-1)-u(k,j,i-1)                                           
                                 +u(k+1,j,i  )-u(k,j,i)                                           
                                 +u(k+1,j+1,i  )-u(k,j+1,i  )                                           
                                 +u(k+1,j+1,i-1)-u(k,j+1,i-1));
        }
    );
    return;
}


void Takacs::calculate_coriolis_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {

    auto& tendency = out_tendency.get_mutable_device_data();
    const auto& v = state.get_field<3>("v").get_device_data();
    // const auto& f = state.get_field<1>("f").get_device_data();
    const auto& f_2d = state.get_field<2>("f_2d").get_device_data();
    const auto& rdz = params.rdz;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("coriolis_tendency_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h-1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k, j, i) += -real(0.25) * f_2d(j,i) * flex_height_coef_up(k) * rdz() *
                                 (v(k+1,j,i+1)-v(k,j,i+1)
                                 +v(k+1,j,i  )-v(k,j,i)
                                 +v(k+1,j-1,i)-v(k,j-1,i)
                                 +v(k+1,j-1,i+1)-v(k,j-1,i+1));
        }
    );
    return;
}

void Takacs::calculate_coriolis_tendency_z(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {

    auto& tendency = out_tendency.get_mutable_device_data();
    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& v = state.get_field<3>("v").get_device_data();
    // const auto& f = state.get_field<1>("f").get_device_data();
    const auto& f_2d = state.get_field<2>("f_2d").get_device_data();
    const auto& rdx = params.rdx;
    const auto& rdy = params.rdy;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();
    int NK2 = nz-h-1;

    Kokkos::parallel_for("coriolis_tendency_z",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            tendency(NK2, j, i) +=
                -real(0.25)*f_2d(j,i)*(u(NK2,j,i+1)-u(NK2,j,i-1)
                           +u(NK2,j+1,i+1)-u(NK2,j+1,i-1)) * rdx()                                         
                -real(0.25)*f_2d(j,i)*(v(NK2,j+1,i)-v(NK2,j-1,i)
                           +v(NK2,j+1,i+1)-v(NK2,j-1,i+1)) * rdy();
        }
    );
    return;
}

} // namespace Dynamics
} // namespace VVM
