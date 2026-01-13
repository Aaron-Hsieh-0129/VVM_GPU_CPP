#include "Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"

namespace VVM {
namespace Dynamics {

Takacs::Takacs(const Core::Grid& grid, Core::HaloExchanger& halo_exchanger)
    : halo_exchanger_(halo_exchanger) {}

void Takacs::calculate_flux_convergence_x(
    const Core::Field<3>& scalar, const Core::Field<3>& u_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    VVM::Utils::Timer advection_x_timer("ADVECTION_X");

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& u = u_field.get_device_data();
    const auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Core::Field<3> uplus_field("uplus", {nz, ny, nx});
    Core::Field<3> uminus_field("uminus", {nz, ny, nx});
    auto& uplus = uplus_field.get_mutable_device_data();
    auto& uminus = uminus_field.get_mutable_device_data();

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

    Kokkos::parallel_for("uplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            uplus(k,j,i)  = 0.5*(u(k,j,i)+Kokkos::abs(u(k,j,i)));
            uminus(k,j,i) = 0.5*(u(k,j,i)-Kokkos::abs(u(k,j,i)));
        }
    );
    halo_exchanger_.exchange_halos(uplus_field);
    halo_exchanger_.exchange_halos(uminus_field);

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            flux(k,j,i) = u(k,j,i)*(q(k,j,i+1)+q(k,j,i)) + 
                      -1./3.*( 
                         uplus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(uplus(k,j,i))*Kokkos::sqrt(uplus(k,j,i-1))*(q(k,j,i)-q(k,j,i-1)) - 
                        uminus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(uminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(uminus(k,j,i+1)))*(q(k,j,i+2)-q(k,j,i+1)) 
                      );
            // flux(k,j,i) = u(k,j,i)*(q(k,j,i+1)+q(k,j,i)); 
        }
    );

    // if (var_name == "zeta") halo_exchanger_.exchange_halos_top_slice(flux_field);
    // else halo_exchanger_.exchange_halos(flux_field);
    halo_exchanger_.exchange_halos(flux_field);

    auto rdx_view = params.rdx;
    Kokkos::parallel_for("flux_convergence_tendency", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k,j,i-1)) * rdx_view();
        }
    );
    return;
}

void Takacs::calculate_flux_convergence_y(
    const Core::Field<3>& scalar, const Core::Field<3>& v_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    VVM::Utils::Timer advection_x_timer("ADVECTION_Y");

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& v = v_field.get_device_data();
    const auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Core::Field<3> vplus_field("vplus", {nz, ny, nx});
    Core::Field<3> vminus_field("vminus", {nz, ny, nx});
    auto& vplus  = vplus_field.get_mutable_device_data();
    auto& vminus = vminus_field.get_mutable_device_data();
    // DEBUG print

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

    Kokkos::parallel_for("vplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            vplus(k,j,i)  = 0.5*(v(k,j,i)+Kokkos::abs(v(k,j,i)));
            vminus(k,j,i) = 0.5*(v(k,j,i)-Kokkos::abs(v(k,j,i)));
        }
    );
    halo_exchanger_.exchange_halos(vplus_field);
    halo_exchanger_.exchange_halos(vminus_field);

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            flux(k,j,i) = v(k,j,i)*(q(k,j+1,i)+q(k,j,i)) + 
                          -1./3.*( 
                             vplus(k,j,i)*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(vplus(k,j,i))*Kokkos::sqrt(vplus(k,j-1,i))*(q(k,j,i)-q(k,j-1,i)) - 
                            vminus(k,j,i)*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(vminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(vminus(k,j+1,i)))*(q(k,j+2,i)-q(k,j+1,i)) 
                          );
            // flux(k,j,i) = v(k,j,i)*(q(k,j+1,i)+q(k,j,i)); 
        }
    );

    // if (var_name == "zeta") halo_exchanger_.exchange_halos_top_slice(flux_field);
    // else halo_exchanger_.exchange_halos(flux_field);
    halo_exchanger_.exchange_halos(flux_field);

    auto rdy_view = params.rdy;
    Kokkos::parallel_for("flux_convergence_tendency", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k,j-1,i)) * rdy_view();
        }
    );
    return;
}

void Takacs::calculate_flux_convergence_z(
    const Core::Field<3>& scalar, const Core::Field<3>& w_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    VVM::Utils::Timer advection_x_timer("ADVECTION_Z");

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& w = w_field.get_device_data();
    const auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Core::Field<3> wplus_field("wplus", {nz, ny, nx});
    Core::Field<3> wminus_field("wminus", {nz, ny, nx});
    auto& wplus  = wplus_field.get_mutable_device_data();
    auto& wminus = wminus_field.get_mutable_device_data();

    Kokkos::parallel_for("wplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            wplus(k,j,i)  = 0.5*(w(k,j,i)+Kokkos::abs(w(k,j,i)));
            wminus(k,j,i) = 0.5*(w(k,j,i)-Kokkos::abs(w(k,j,i)));
        }
    );
    halo_exchanger_.exchange_halos(wplus_field);
    halo_exchanger_.exchange_halos(wminus_field);

    // zeta only needs to do the top prediction
    // It has two layer of w.
    if (var_name == "zeta") {
        Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
            KOKKOS_LAMBDA(int j, int i) {
                // It's supposed to be rho*w*q, the w here is rho*w from the input
                flux(nz-h-2,j,i) = w(nz-h-2,j,i)*(q(nz-h-1,j,i)+q(nz-h-2,j,i));
                // flux(nz-h-2,j,i) = w(nz-h-2,j,i);
                // flux(nz-h-2,j,i) = q(nz-h-1,j,i)+q(nz-h-2,j,i);
                if (w(nz-h-2,j,i) >= 0.) {
                    flux(nz-h-2,j,i) += -1./3.*(
                            wplus(nz-h-2,j,i)*(q(nz-h-1,j,i)-q(nz-h-2,j,i)) - Kokkos::sqrt(wplus(nz-h-2,j,i))*Kokkos::sqrt(wplus(nz-h-3,j,i))*(q(nz-h-2,j,i)-q(nz-h-3,j,i))
                        );
                }
            }
        );
    }
    else if (var_name == "xi" || var_name == "eta") {
        Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1,h,h}, {nz-h-1,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                flux(k,j,i) = w(k,j,i)*(q(k+1,j,i)+q(k,j,i));
                if (k == h-1)  {
                    if (w(k,j,i) < 0.) {
                        flux(k,j,i) += -1./3.*( 
                                -wminus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(wminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(wminus(k+1,j,i)))*(q(k+2,j,i)-q(k+1,j,i)) 
                              );
                    }
                }
                else if (k == nz-h-2) {
                    if (w(k,j,i) >= 0.) {
                        flux(k,j,i) += -1./3.*( 
                                 wplus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i)) 
                              );
                    }
                }
                else {
                    flux(k,j,i) += -1./3.*( 
                             wplus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i)) - 
                            wminus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(wminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(wminus(k+1,j,i)))*(q(k+2,j,i)-q(k+1,j,i)) 
                          );
                }
            }
        );
    }
    else {
        Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                flux(k,j,i) = w(k,j,i)*(q(k+1,j,i)+q(k,j,i));
                if (k == h) {
                    if (w(k,j,i) < 0.) {
                        flux(k,j,i) += -1./3.*(
                                -wminus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(wminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(wminus(k+1,j,i)))*(q(k+2,j,i)-q(k+1,j,i)) 
                              );
                    }
                }
                else if (k == nz-h-2) {
                    if (w(k,j,i) >= 0.) {
                        flux(k,j,i) += -1./3.*( 
                                 wplus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i)) 
                              );
                    }
                }
                else {
                    flux(k,j,i) += -1./3.*( 
                             wplus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i)) - 
                            wminus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(wminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(wminus(k+1,j,i)))*(q(k+2,j,i)-q(k+1,j,i)) 
                          );
                }
            }
        );
        // This is for vertical periodic boundary
        Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
            KOKKOS_LAMBDA(int j, int i) {
                // flux(h-1,j,i) = flux(nz-h-2,j,i);
                flux(h-1,j,i) = 0.;

                // flux(nz-h-1,j,i) = flux(h,j,i);
                flux(nz-h-1,j,i) = 0.;
            }
        );
    }
    // No need of x-y halo exchanges because this is z direction tendency
    // No need of vertical boundary process because it's supposed to be 0 in ghost points and it's been processed during initialization

    auto rdz_view = params.rdz;
    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    if (var_name == "zeta") {
        Kokkos::parallel_for("flux_convergence_tendency", 
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // w(nz-h-1) = 0, top b.c.
                tendency(nz-h-1,j,i) += 0.5*flux(nz-h-2,j,i) * rdz_view() * flex_height_coef_mid(nz-h-1);
                // tendency(nz-h-1,j,i) = 0.5*flux(nz-h-2,j,i) * rdz_view() * flex_height_coef_mid(nz-h-1);
            }
        );
    }
    else if (var_name == "xi" || var_name == "eta") {
        Kokkos::parallel_for("flux_convergence_tendency", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k-1,j,i)) * rdz_view() * flex_height_coef_up(k);
            }
        );
    }
    else {
        Kokkos::parallel_for("flux_convergence_tendency", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k-1,j,i)) * rdz_view() * flex_height_coef_mid(k);
            }
        );
    }
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

    auto fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    auto fact2_xi_eta = params.fact2_xi_eta.get_device_data();
    
    // Implements Eq. (3.25) for [ρ₀ξ(∂u/∂x)] at (i, j+1/2, k+1/2)
    Kokkos::parallel_for("stretching_term_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h-1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double term_at_j_plus_1 = 
                (xi(k,j,i)+xi(k,j+1,i)) * 
                (fact1_xi_eta(k) * rhobar(k+1) * (u(k+1, j+1, i) - u(k+1, j+1, i-1)) +
                 fact2_xi_eta(k) * rhobar(k)   * (u(k,   j+1, i) - u(k,   j+1, i-1)) );

            const double term_at_j = 
                (xi(k,j,i)+xi(k,j-1,i)) * 
                (fact1_xi_eta(k) * rhobar(k+1) * (u(k+1, j, i) - u(k+1, j, i-1)) +
                 fact2_xi_eta(k) * rhobar(k)   * (u(k,   j, i) - u(k,   j, i-1)) );

            tendency(k, j, i) += 0.125 * rdx() * (term_at_j_plus_1 + term_at_j);
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
    
    const auto& rdy = params.get_value_host(params.rdy);

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    auto fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    auto fact2_xi_eta = params.fact2_xi_eta.get_device_data();

    // Implements Eq. (3.26) for [ρ₀η(∂v/∂y)] at (i+1/2, j, k+1/2)
    Kokkos::parallel_for("stretching_term_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h-1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double term_at_i_plus_1 = 
                (eta(k,j,i)+eta(k,j,i+1)) * 
                (fact1_xi_eta(k) * rhobar(k+1) * (v(k+1, j, i+1) - v(k+1, j-1, i+1)) +
                 fact2_xi_eta(k) * rhobar(k)   * (v(k,   j, i+1) - v(k,   j-1, i+1)) );

            const double term_at_i = 
                (eta(k,j,i)+eta(k,j,i-1)) * 
                (fact1_xi_eta(k) * rhobar(k+1) * (v(k+1, j, i) - v(k+1, j-1, i)) +
                 fact2_xi_eta(k) * rhobar(k)   * (v(k,   j, i) - v(k,   j-1, i)) );

            tendency(k, j, i) += 0.125 * rdy * (term_at_i_plus_1 + term_at_i);
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

    const auto& rdz = params.get_value_host(params.rdz);

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    Kokkos::View<double> fact1("fact1");
    Kokkos::View<double> fact2("fact2");

    // Implements Eq. (3.27) for [ρ₀ζ(∂w/∂z)] at (i+1/2, j+1/2, k)
    Kokkos::parallel_for("stretching_term_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nz-h-1, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // w(nz-h-1,:,:) = 0.
            const double term_at_i = (zeta(k,j,i-1)+zeta(k,j,i)) * (w(k-1, j, i) + w(k-1, j+1, i));
            const double term_at_i_plus_1 = (zeta(k,j,i)+zeta(k,j,i+1)) * (w(k-1, j, i+1) + w(k-1, j+1, i+1));

            tendency(k, j, i) += -0.125 * flex_height_coef_mid(k) * rdz * rhobar(k) * (term_at_i + term_at_i_plus_1);
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

    Kokkos::parallel_for("compute_R_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // R_xi at i, j+1/2, k+1/2
            R_xi(k, j, i) = (w(k, j+1, i) - w(k, j, i)) * rdy() +
                            (v(k+1, j, i) - v(k, j, i)) * rdz() * flex_height_coef_up(k);
        }
    );

    halo_exchanger_.exchange_halos(out_R_xi);
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

    Kokkos::parallel_for("compute_R_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // R_eta at i+1/2, j, k+1/2
            R_eta(k, j, i) = (w(k, j, i+1) - w(k, j, i)) * rdx() +
                             (u(k+1, j, i) - u(k, j, i)) * rdz() * flex_height_coef_up(k);
        }
    );

    halo_exchanger_.exchange_halos(out_R_eta);
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

    Kokkos::parallel_for("compute_R_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // R_zeta at i+1/2, j+1/2, k
            R_zeta(k, j, i) = (v(k, j, i+1) - v(k, j, i)) * rdx() +
                              (u(k, j+1, i) - u(k, j, i)) * rdy();
        }
    );

    halo_exchanger_.exchange_halos(out_R_zeta);
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

    auto fact1_xi_eta = params.fact1_xi_eta.get_device_data();
    auto fact2_xi_eta = params.fact2_xi_eta.get_device_data();

    // Implements Eq. (3.28) for [0.5ρ₀(eta*Rzeta+zeta*Reta)] at (i+1/2, j+1/2, k)
    Kokkos::parallel_for("twisting_term_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // WARNING: The documentation has rho weighted but the VVM code doesn't have. This code follows the VVM code.
            const double term_etaRzeta = (
                (eta(k,j+1,i  )+eta(k,j,i  )) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j,i  ) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j,i  ))
              + (eta(k,j+1,i-1)+eta(k,j,i-1)) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j,i-1) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j,i-1))
            );

            const double term_zetaReta = (
                rhobar_up(k)*(zeta(k,j,i  )+zeta(k+1,j,i  ))*(R_eta(k,j+1,i  )+R_eta(k,j,i  ))
              + rhobar_up(k)*(zeta(k,j,i-1)+zeta(k+1,j,i-1))*(R_eta(k,j+1,i-1)+R_eta(k,j,i-1))
            );

            // WARNING: term_etaRzeta has a negative sign in original VVM because the definition of eta in that VVM is negative from this one.
            // FIXME: Fix the comparison negative sign
            tendency(k, j, i) += 0.0625 * (-term_etaRzeta + term_zetaReta);
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

    // Implements Eq. (3.29) for [0.5ρ₀(xi*Rzeta+zeta*Rxi)]
    Kokkos::parallel_for("twisting_term_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // WARNING: The documentation has rho weighted but the VVM code doesn't have. This code follows the VVM code.
            const double term_xiRzeta = (
                (xi(k,j  ,i+1)+xi(k,j  ,i)) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j  ,i) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j  ,i))
              + (xi(k,j-1,i+1)+xi(k,j-1,i)) * (fact2_xi_eta(k)*rhobar(k)*R_zeta(k,j-1,i) + fact1_xi_eta(k)*rhobar(k+1)*R_zeta(k+1,j-1,i))
            );

            const double term_zetaRxi = (
                rhobar_up(k)*(zeta(k,j  ,i)+zeta(k+1,j  ,i))*(R_xi(k,j  ,i)+R_xi(k,j  ,i+1))
              + rhobar_up(k)*(zeta(k,j-1,i)+zeta(k+1,j-1,i))*(R_xi(k,j-1,i)+R_xi(k,j-1,i+1))
            );

            // WARNING: term_xiRzeta and eter_zetaRxi have negative signs in original VVM because the definition of eta in that VVM is negative from this one.
            // FIXME: Fix the comparison negative sign
            tendency(k, j, i) += 0.0625 * -(term_xiRzeta + term_zetaRxi);
        }
    );
    return;
}

void Takacs::calculate_twisting_tendency_z(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& R_xi_field = state.get_field<3>("R_xi");
    const auto& R_eta_field = state.get_field<3>("R_eta");
    auto& R_xi = R_xi_field.get_device_data();
    auto& R_eta = R_eta_field.get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    Kokkos::View<double> fact1("fact1");
    Kokkos::View<double> fact2("fact2");
    Kokkos::parallel_for("AssignFactor", 1, KOKKOS_LAMBDA(const int) {
        fact1() = flex_height_coef_mid(nz-h-1) * rhobar_up(nz-h-1) / flex_height_coef_up(nz-h-1);
        fact2() = flex_height_coef_mid(nz-h-1) * rhobar_up(nz-h-2) / flex_height_coef_up(nz-h-2);
    });


    // Implements Eq. (3.30) for [0.5ρ₀(xi*Reta+eta*Rxi)]
    Kokkos::parallel_for("twisting_term_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nz-h-1, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // WARNING: The documentation has rho weighted but the VVM code doesn't have. This code follows the VVM code.
            const double term_xiReta = (
                fact1()*(xi(k,j,i+1)+xi(k  ,j,i  )) * (R_eta(k  ,j+1,i)+R_eta(k  ,j,i))
              + fact2()*(xi(k-1,j,i)+xi(k-1,j,i+1)) * (R_eta(k-1,j+1,i)+R_eta(k-1,j,i))
            );

            const double term_etaRxi = (
                fact1()*(eta(k  ,j+1,i)+eta(k  ,j,i)) * (R_xi(k  ,j,i)+R_xi(k  ,j,i+1))
              + fact2()*(eta(k-1,j+1,i)+eta(k-1,j,i)) * (R_xi(k-1,j,i)+R_xi(k-1,j,i+1))
            );

            // WARNING: term_etaRxi has a negative sign in original VVM because the definition of eta in that VVM is negative from this one.
            // FIXME: Fix the comparison negative sign
            tendency(k, j, i) += 0.0625 * (term_xiReta - term_etaRxi);
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
            const double d_xi_dx = (xi(k, j, i+1) - xi(k, j, i)) * rdx;
            const double d_eta_dy = (eta(k, j+1, i) - eta(k, j, i)) * rdy;
            // WARNING: Original VVM has a negative sign for eta due to different definition
            // FIXME: Fix the comparison negative sign
            out_data(k, j, i) = -(d_xi_dx - d_eta_dy);
        }
    );
    halo_exchanger_.exchange_halos(out_field);
}

void Takacs::calculate_buoyancy_tendency_x(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {
    
    const auto& thbar = state.get_field<1>("thbar").get_device_data();
    const auto& th = state.get_field<3>("th").get_device_data();
    const auto& qv = state.get_field<3>("qv").get_device_data();
    const auto& qp = state.get_field<3>("qp").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();
    const auto& rdy = params.rdy;
    const auto& gravity = params.gravity;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("buoyancy_tendency_x",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h-1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double dB_dy = (th(k  ,j+1,i)-th(k  ,j,i)) / thbar(k)
                               + (th(k+1,j+1,i)-th(k+1,j,i)) / thbar(k+1)
                               + 0.608*(qv(k,j+1,i)-qv(k,j,i)+qv(k+1,j+1,i)-qv(k+1,j,i))
                               - (qp(k,j+1,i)-qp(k,j,i)+qp(k+1,j+1,i)-qp(k+1,j,i));
            // const double dB_dy = (th(k  ,j+1,i)-th(k  ,j,i)) / thbar(k)
            //                    + (th(k+1,j+1,i)-th(k+1,j,i)) / thbar(k+1);

            tendency(k, j, i) += gravity() * 0.5 * dB_dy * rdy();
        }
    );

    const auto& ITYPEV = state.get_field<3>("ITYPEV").get_device_data();
    const auto& max_topo_idx = params.max_topo_idx;
    Kokkos::parallel_for("topo",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // Set tendency to 0 if it's topo for v-loc point (ITYPEV = 0)
            if (ITYPEV(k,j,i) == 0) tendency(k,j,i) = 0.;
        }
    );
}

void Takacs::calculate_buoyancy_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency) const {
    
    const auto& thbar = state.get_field<1>("thbar").get_device_data();
    const auto& th = state.get_field<3>("th").get_device_data();
    const auto& qv = state.get_field<3>("qv").get_device_data();
    const auto& qp = state.get_field<3>("qp").get_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();
    auto& rdx = params.rdx;
    auto& gravity = params.gravity;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    Kokkos::parallel_for("buoyancy_tendency_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h-1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double dB_dx = (th(k  ,j,i+1)-th(k  ,j,i)) / thbar(k)
                               + (th(k+1,j,i+1)-th(k+1,j,i)) / thbar(k+1)
                               + (0.608*(qv(k,j,i+1)-qv(k,j,i)+qv(k+1,j,i+1)-qv(k+1,j,i)))
                               - (qp(k,j,i+1)-qp(k,j,i)+qp(k+1,j,i+1)-qp(k+1,j,i));

            // const double dB_dx = (th(k  ,j,i+1)-th(k  ,j,i)) / thbar(k)
            //                    + (th(k+1,j,i+1)-th(k+1,j,i)) / thbar(k+1);
            // WARNING: dB_dy has a negative sign in original VVM because the definition of eta in that VVM is negative from this one.
            // Fix the comparison negative sign
            tendency(k, j, i) += gravity() * 0.5 * dB_dx * rdx();
        }
    );

    const auto& ITYPEU = state.get_field<3>("ITYPEU").get_device_data();
    const auto& max_topo_idx = params.max_topo_idx;
    Kokkos::parallel_for("buoyancy_tendency_y",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {max_topo_idx+1, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // Set tendency to 0 if it's topo for u-loc point (ITYPEU = 0)
            if (ITYPEU(k,j,i) == 0) tendency(k,j,i) = 0.;
        }
    );
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) out_tendency.print_slice_z_at_k(grid, 0, h+15);
}

} // namespace Dynamics
} // namespace VVM
