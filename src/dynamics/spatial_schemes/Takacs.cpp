#include "Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"

namespace VVM {
namespace Dynamics {

Takacs::Takacs(const Core::Grid& grid, const Utils::ConfigurationManager& config)
    : halo_exchanger_(grid),
      flux_bc_manager_(grid, config, "flux") {}

void Takacs::calculate_flux_convergence_x(
    const Core::Field<3>& scalar, const Core::Field<3>& u_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    auto& u = u_field.get_device_data();
    auto& q = scalar.get_device_data();
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

    Kokkos::parallel_for("uplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,0,0}, {k_end,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            uplus(k,j,i)  = 0.5*(u(k,j,i)+Kokkos::abs(u(k,j,i)));
            uminus(k,j,i) = 0.5*(u(k,j,i)-Kokkos::abs(u(k,j,i)));
        }
    );

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            flux(k,j,i) = u(k,j,i)*(q(k,j,i+1)+q(k,j,i)) + 
                      -1./3.*( 
                         uplus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(uplus(k,j,i))*Kokkos::sqrt(uplus(k,j,i-1))*(q(k,j,i)-q(k,j,i-1)) - 
                        uminus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(uminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(uminus(k,j,i+1)))*(q(k,j,i+2)-q(k,j,i+1)) 
                      );
        }
    );

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

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    auto& v = v_field.get_device_data();
    auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Core::Field<3> vplus_field("vplus", {nz, ny, nx});
    Core::Field<3> vminus_field("vminus", {nz, ny, nx});
    auto& vplus  = vplus_field.get_mutable_device_data();
    auto& vminus = vminus_field.get_mutable_device_data();

    int k_start = h;
    int k_end = nz-h;
    // zeta only needs to do the top prediction
    if (var_name == "zeta") {
        k_start = nz-h-1;
        k_end = nz-h;
    }

    Kokkos::parallel_for("vplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,0,0}, {k_end,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            vplus(k,j,i)  = 0.5*(v(k,j,i)+Kokkos::abs(v(k,j,i)));
            vminus(k,j,i) = 0.5*(v(k,j,i)-Kokkos::abs(v(k,j,i)));
        }
    );

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            flux(k,j,i) = v(k,j,i)*(q(k,j+1,i)+q(k,j,i)) + 
                          -1./3.*( 
                             vplus(k,j,i)*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(vplus(k,j,i))*Kokkos::sqrt(vplus(k,j-1,i))*(q(k,j,i)-q(k,j-1,i)) - 
                            vminus(k,j,i)*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(vminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(vminus(k,j+1,i)))*(q(k,j+2,i)-q(k,j+1,i)) 
                          );
        }
    );

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
    const Core::Field<3>& scalar, const Core::Field<1>& rhobar_divide_field, const Core::Field<3>& w_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    auto rhobar_divide = rhobar_divide_field.get_device_data();

    const int nz_phys = grid.get_local_physical_points_z();
    const int ny_phys = grid.get_local_physical_points_y();
    const int nx_phys = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();

    int k_start = h;
    int k_end = nz-h;

    // zeta only needs to do the top prediction
    // It has two layer of w.
    if (var_name == "zeta") {
        k_start = nz-h-2;
        k_end = nz-h;
    }

    auto& w = w_field.get_device_data();
    auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Core::Field<3> wplus_field("wplus", {nz, ny, nx});
    Core::Field<3> wminus_field("wminus", {nz, ny, nx});
    auto& wplus  = wplus_field.get_mutable_device_data();
    auto& wminus = wminus_field.get_mutable_device_data();

    Kokkos::parallel_for("wplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            wplus(k,j,i)  = 0.5*(w(k,j,i)+Kokkos::abs(w(k,j,i)));
            wminus(k,j,i) = 0.5*(w(k,j,i)-Kokkos::abs(w(k,j,i)));
        }
    );
    // Only 1 layer is needed to calculate flux
    if (var_name == "zeta") k_start = nz-h-1;

    if (var_name == "zeta") {
        Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // It's supposed to be rho*w*q, the w here is rho*w from the input
                flux(k,j,i) = w(k,j,i)*(q(k+1,j,i)+q(k,j,i));
                if (w(k,j,i) >= 0.) {
                    flux(k,j,i) += -1./3.*(
                            wplus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i))
                        );
                }
            }
        );
    }
    else {
        Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                flux(k,j,i) = w(k,j,i)*(q(k+1,j,i)+q(k,j,i));
                if (k == h && w(k,j,i) < 0.) {
                    flux(k,j,i) += -1./3.*(
                            -wminus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(wminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(wminus(k+1,j,i)))*(q(k+2,j,i)-q(k+1,j,i)) 
                          );
                }
                else if (k == nz-h-1 && w(k,j,i) >= 0.) {
                    flux(k,j,i) += -1./3.*( 
                             wplus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i)) 
                          );
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

    flux_bc_manager_.apply_z_bcs_to_field(flux_field);

    // DEBUG print
    // Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny,nx}),
    //     KOKKOS_LAMBDA(int j, int i) {
    //         // flux(0,j,i) = flux(nz-2,j,i);
    //         // flux(nz-1,j,i) = flux(1,j,i);
    //         flux(2,j,i) = 100.;
    //         flux(3,j,i) = 200.;
    //         flux(nz-4,j,i) = 300.;
    //         flux(nz-3,j,i) = 400.;
    //     }
    // );
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0 && var_name == "th") w_field.print_xz_cross_at_j(grid, 0, 3);

    auto rdz_view = params.rdz;
    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    if (var_name == "zeta") {
        Kokkos::parallel_for("flux_convergence_tendency", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                // w(nz-h) = 0, top b.c.
                tendency(k,j,i) += 0.5*flux(k,j,i) * rdz_view() * flex_height_coef_mid(k);
            }
        );
    }
    else if (var_name == "xi" || var_name == "eta") {
        Kokkos::parallel_for("flux_convergence_tendency", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k-1,j,i)) * rdz_view() * flex_height_coef_up(k);
            }
        );
    }
    else {
        Kokkos::parallel_for("flux_convergence_tendency", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start,h,h}, {k_end, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k-1,j,i)) * rdz_view() * flex_height_coef_mid(k) / rhobar_divide(k);
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
    auto tendency = out_tendency.get_mutable_device_data();
    
    const double rdx = params.get_value_host(params.rdx);

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();
    
    // Implements Eq. (3.25) for [ρ₀ξ(∂u/∂x)] at (i, j+1/2, k+1/2)
    Kokkos::parallel_for("stretching_term_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz - 1, h + ny - 1, h + nx - 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double term_at_j_plus_1 = 
                (xi(k,j,i)+xi(k,j+1,i)) * 
                (rhobar(k+1) * (u(k+1, j+1, i) - u(k+1, j+1, i-1)) +
                 rhobar(k)   * (u(k,   j+1, i) - u(k,   j+1, i-1)) );

            const double term_at_j = 
                (xi(k,j,i)+xi(k,j-1,i)) * 
                (rhobar(k+1) * (u(k+1, j, i)   - u(k+1, j, i-1)) +
                 rhobar(k)   * (u(k,   j, i-1) - u(k,   j, i-1)) );

            tendency(k, j, i) += 0.125 * rdx * (term_at_j + term_at_j_plus_1);
        }
    );
}

void Takacs::calculate_stretching_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& v = state.get_field<3>("v").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    auto tendency = out_tendency.get_mutable_device_data();
    
    const double rdy = params.get_value_host(params.rdy);

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();

    // Implements Eq. (3.26) for [ρ₀η(∂v/∂y)] at (i+1/2, j, k+1/2)
    Kokkos::parallel_for("stretching_term_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz - 1, h + ny - 1, h + nx - 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double term_at_i_plus_1 = 
                (eta(k,j,i-1)+eta(k,j,i)) * 
                (rhobar(k+1) * (v(k+1, j, i) - v(k+1, j-1, i)) +
                 rhobar(k)   * (v(k,   j, i) - v(k,   j-1, i)) );

            const double term_at_i = 
                (eta(k,j,i+1)+eta(k,j,i)) * 
                (rhobar(k+1) * (v(k+1, j, i+1) - v(k+1, j-1, i+1)) +
                 rhobar(k)   * (v(k,   j, i+1) - v(k,   j-1, i+1)) );

            const double eta_avg = eta(k, j, i) + eta(k, j, i+1);

            tendency(k, j, i) += 0.125 * rdy * (term_at_i + term_at_i_plus_1);
        }
    );
}

void Takacs::calculate_stretching_tendency_z(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
    
    const auto& w = state.get_field<3>("w").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    auto tendency = out_tendency.get_mutable_device_data();

    const double rdz = params.get_value_host(params.rdz);

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();

    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();

    // Implements Eq. (3.27) for [ρ₀ζ(∂w/∂z)] at (i+1/2, j+1/2, k)
    Kokkos::parallel_for("stretching_term_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz - 1, h + ny - 1, h + nx - 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double term_at_j_plus_1 =
                (zeta(k,j,i-1)+zeta(k,j,i)) * 
                (w(k, j,   i) - w(k-1, j,   i) +
                 w(k, j+1, i) - w(k-1, j+1, i));

            const double term_at_j = 
                (zeta(k,j,i)+zeta(k,j,i+1)) * 
                (w(k, j,   i)   - w(k-1, j,   i) +
                 w(k, j+1, i+1) - w(k-1, j+1, i+1));

            tendency(k, j, i) += 0.125 * flex_height_coef_mid(k) * rdz * rhobar(k) * (term_at_j + term_at_j_plus_1);
        }
    );
}

// Equation (3.32)
void Takacs::calculate_R_xi(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_R_xi) const {
    
    const auto& v = state.get_field<3>("v").get_device_data();
    const auto& w = state.get_field<3>("w").get_device_data();
    auto R_xi = out_R_xi.get_mutable_device_data();

    auto rdy = params.rdy;
    auto rdz = params.rdz;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Kokkos::parallel_for("compute_R_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // R_xi at i, j+1/2, k+1/2
            R_xi(k, j, i) = (w(k, j + 1, i) - w(k, j, i)) * rdy() +
                            (v(k + 1, j, i) - v(k, j, i)) * rdz() * flex_height_coef_up(k);
        }
    );
}

void Takacs::calculate_R_eta(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_R_eta) const {

    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& w = state.get_field<3>("w").get_device_data();
    auto R_eta = out_R_eta.get_mutable_device_data();

    auto rdx = params.rdx;
    auto rdz = params.rdz;
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Kokkos::parallel_for("compute_R_eta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // R_eta at i+1/2, j, k+1/2
            R_eta(k, j, i) = (w(k, j, i + 1) - w(k, j, i)) * rdx() +
                             (u(k + 1, j, i) - u(k, j, i)) * rdz() * flex_height_coef_up(k);
        }
    );
}

void Takacs::calculate_R_zeta(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_R_zeta) const {

    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& v = state.get_field<3>("v").get_device_data();
    auto R_zeta = out_R_zeta.get_mutable_device_data();

    auto rdx = params.rdx;
    auto rdy = params.rdy;

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    Kokkos::parallel_for("compute_R_zeta",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // R_zeta at i+1/2, j+1/2, k
            R_zeta(k, j, i) = (v(k, j, i + 1) - v(k, j, i)) * rdx() +
                              (u(k, j + 1, i) - u(k, j, i)) * rdy();
        }
    );
}


void Takacs::calculate_twisting_tendency_x(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    const auto& R_eta_field = state.get_field<3>("R_eta");
    const auto& R_zeta_field = state.get_field<3>("R_zeta");
    auto R_eta = R_eta_field.get_device_data();
    auto R_zeta = R_zeta_field.get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    auto tendency = out_tendency.get_mutable_device_data();

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();

    // Implements Eq. (3.33) for [0.5ρ₀(eta*Rzeta+zeta*Reta)] at (i+1/2, j+1/2, k)
    Kokkos::parallel_for("twisting_term_xi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz - 1, h + ny - 1, h + nx - 1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            const double term_etaRzeta = 0.0625 * rhobar_up(k) * (
                        (eta(k,j+1,i)  +eta(k,j,i)  ) * (rhobar(k)/rhobar_up(k)*R_zeta(k,j,i)   + rhobar(k+1)/rhobar_up(k)*R_zeta(k+1,j,i  ))
                      + (eta(k,j+1,i-1)+eta(k,j,i-1)) * (rhobar(k)/rhobar_up(k)*R_zeta(k,j,i-1) + rhobar(k+1)/rhobar_up(k)*R_zeta(k+1,j,i-1))
                    );

            const double term_zetaReta = 0.0625 * (
                        (rhobar(k)*zeta(k,j,i  ) + rhobar(k+1)*zeta(k+1,j,i)  ) * (R_eta(k,j+1,i  ) + R_eta(k,j,i)  )
                      + (rhobar(k)*zeta(k,j,i-1) + rhobar(k+1)*zeta(k+1,j,i-1)) * (R_eta(k,j+1,i-1) + R_eta(k,j,i-1))
                    );

            tendency(k, j, i) += (term_etaRzeta + term_zetaReta);
        }
    );
}

void Takacs::calculate_twisting_tendency_y(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {

    // const auto& R_xi_field = state.get_field<3>("R_xi");
    // const auto& R_zeta_field = state.get_field<3>("R_zeta");
    // auto R_xi = R_xi_field.get_device_data();
    // auto R_zeta = R_zeta_field.get_device_data();
    // const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    // const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    // const auto& xi = state.get_field<3>("xi").get_device_data();
    // const auto& zeta = state.get_field<3>("zeta").get_device_data();
    //
    // const int nz = grid.get_local_physical_points_z();
    // const int ny = grid.get_local_physical_points_y();
    // const int nx = grid.get_local_physical_points_x();
    // const int h = grid.get_halo_cells();
    //
    // // TODO: Starts from here
    // // Implements for [0.5ρ₀(xi*Rzeta+zeta*Rxi)]
    // Kokkos::parallel_for("twisting_term_zeta",
    //     Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz - 1, h + ny - 1, h + nx - 1}),
    //     KOKKOS_LAMBDA(const int k, const int j, const int i) {
    //         const double term_xiRzeta = 0.0625 * rhobar_up(k) * (
    //                     (eta(k,j+1,i)  +eta(k,j,i)  ) * (rhobar(k)/rhobar_up(k)*R_zeta(k,j,i)   + rhobar(k+1)/rhobar_up(k)*R_zeta(k+1,j,i  ))
    //                   + (eta(k,j+1,i-1)+eta(k,j,i-1)) * (rhobar(k)/rhobar_up(k)*R_zeta(k,j,i-1) + rhobar(k+1)/rhobar_up(k)*R_zeta(k+1,j,i-1))
    //                 );
    //
    //         const double term_zetaReta = 0.0625 * (
    //                     (rhobar(k)*zeta(k,j,i  ) + rhobar(k+1)*zeta(k+1,j,i)  ) * (R_eta(k,j+1,i  ) + R_eta(k,j,i)  )
    //                   + (rhobar(k)*zeta(k,j,i-1) + rhobar(k+1)*zeta(k+1,j,i-1)) * (R_eta(k,j+1,i-1) + R_eta(k,j,i-1))
    //                 );
    //
    //         tendency(k, j, i) += (term_etaRzeta + term_zetaReta);
    //     }
    // );
}

void Takacs::calculate_twisting_tendency_z(
    const Core::State& state, const Core::Grid& grid,
    const Core::Parameters& params, Core::Field<3>& out_tendency, const std::string& var_name) const {
    // TODO: 在這裡實作 twisting term z-方向的計算
}

} // namespace Dynamics
} // namespace VVM
