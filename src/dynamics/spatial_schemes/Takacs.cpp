#include "Takacs.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"

namespace VVM {
namespace Dynamics {

void Takacs::calculate_flux_convergence_x(
    const Core::Field<3>& scalar, const Core::Field<3>& u_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const {

    VVM::Core::HaloExchanger haloexchanger(grid);
    // VVM::Core::BoundaryConditionManager bc_manager_flux(grid, VVM::Core::ZBoundaryType::ZERO, VVM::Core::ZBoundaryType::ZERO);
    VVM::Core::BoundaryConditionManager bc_manager_flux(grid, VVM::Core::ZBoundaryType::PERIODIC, VVM::Core::ZBoundaryType::PERIODIC);
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    auto& u = u_field.get_device_data();
    auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Kokkos::View<double***> uplus("uplus", nz, ny, nx);
    Kokkos::View<double***> uminus("uminus", nz, ny, nx);

    Kokkos::parallel_for("uplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            uplus(k,j,i) = 0.5*(u(k,j,i)+Kokkos::abs(u(k,j,i)));
            uminus(k,j,i) = 0.5*(u(k,j,i)-Kokkos::abs(u(k,j,i)));
        }
    );

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1,ny-1,nx-1}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            flux(k,j,i) = u(k,j,i)*(q(k,j,i+1)+q(k,j,i));
            if (i >= 2 && i <= nx-3) {
                flux(k,j,i) += -1./3.*( 
                            uplus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(uplus(k,j,i))*Kokkos::sqrt(uplus(k,j,i-1))*(q(k,j,i)-q(k,j,i-1)) - 
                            uminus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(uminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(uminus(k,j,i+1)))*(q(k,j,i+2)-q(k,j,i+1)) 
                          );
            }
        }
    );

    haloexchanger.exchange_halos(flux_field);
    bc_manager_flux.apply_z_bcs_to_field(flux_field);

    auto rdx_view = params.rdx;
    Kokkos::parallel_for("flux_convergence_tendency", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k,j,i-1)) * rdx_view();
        }
    );
    return;
}

void Takacs::calculate_flux_convergence_y(
    const Core::Field<3>& scalar, const Core::Field<3>& v_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const {

    VVM::Core::HaloExchanger haloexchanger(grid);
    // VVM::Core::BoundaryConditionManager bc_manager_flux(grid, VVM::Core::ZBoundaryType::ZERO, VVM::Core::ZBoundaryType::ZERO);
    VVM::Core::BoundaryConditionManager bc_manager_flux(grid, VVM::Core::ZBoundaryType::PERIODIC, VVM::Core::ZBoundaryType::PERIODIC);
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();

    auto& v = v_field.get_device_data();
    auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Kokkos::View<double***> vplus("vplus", nz, ny, nx);
    Kokkos::View<double***> vminus("vminus", nz, ny, nx);

    Kokkos::parallel_for("vplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            vplus(k,j,i) = 0.5*(v(k,j,i)+Kokkos::abs(v(k,j,i)));
            vminus(k,j,i) = 0.5*(v(k,j,i)-Kokkos::abs(v(k,j,i)));
        }
    );

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1,ny-1,nx-1}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            flux(k,j,i) = v(k,j,i)*(q(k,j+1,i)+q(k,j,i));
            if (j >= 2 && j <= ny-3) {
                flux(k,j,i) += -1./3.*( 
                            vplus(k,j,i) *(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(vplus(k,j,i))*Kokkos::sqrt(vplus(k,j-1,i))*(q(k,j,i)-q(k,j-1,i)) - 
                            vminus(k,j,i)*(q(k,j+1,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(vminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(vminus(k,j+1,i)))*(q(k,j+2,i)-q(k,j+1,i)) 
                          );
            }
        }
    );

    haloexchanger.exchange_halos(flux_field);
    bc_manager_flux.apply_z_bcs_to_field(flux_field);

    auto rdy_view = params.rdy;
    Kokkos::parallel_for("flux_convergence_tendency", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k,j-1,i)) * rdy_view();
        }
    );
    return;
}

void Takacs::calculate_flux_convergence_z(
    const Core::Field<3>& scalar, const Core::Field<1>& rhobar_divide_field, const Core::Field<3>& w_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const {

    VVM::Core::HaloExchanger haloexchanger(grid);
    // VVM::Core::BoundaryConditionManager bc_manager_flux(grid, VVM::Core::ZBoundaryType::ZERO, VVM::Core::ZBoundaryType::ZERO);
    VVM::Core::BoundaryConditionManager bc_manager_flux(grid, VVM::Core::ZBoundaryType::PERIODIC, VVM::Core::ZBoundaryType::PERIODIC);
    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    auto rhobar_divide = rhobar_divide_field.get_device_data();

    auto& w = w_field.get_device_data();
    auto& q = scalar.get_device_data();
    Core::Field<3> flux_field("flux", {nz, ny, nx});
    auto& flux = flux_field.get_mutable_device_data();
    auto& tendency = out_tendency.get_mutable_device_data();

    Kokkos::View<double***> wplus("wplus", nz, ny, nx);
    Kokkos::View<double***> wminus("wminus", nz, ny, nx);

    Kokkos::parallel_for("wplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            wplus(k,j,i) = 0.5*(w(k,j,i)+Kokkos::abs(w(k,j,i)));
            wminus(k,j,i) = 0.5*(w(k,j,i)-Kokkos::abs(w(k,j,i)));
        }
    );

    Kokkos::parallel_for("flux_convergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1,ny-1,nx-1}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // It's supposed to be rho*w*q, the w here is rho*w from the input
            flux(k,j,i) = w(k,j,i)*(q(k+1,j,i)+q(k,j,i));
            if (k >= 2 && k <= nz-3) {
                flux(k,j,i) += -1./3.*( 
                            wplus(k,j,i) *(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(wplus(k,j,i))*Kokkos::sqrt(wplus(k-1,j,i))*(q(k,j,i)-q(k-1,j,i)) - 
                            wminus(k,j,i)*(q(k+1,j,i)-q(k,j,i)) - Kokkos::sqrt(Kokkos::abs(wminus(k,j,i)))*Kokkos::sqrt(Kokkos::abs(wminus(k+1,j,i)))*(q(k+2,j,i)-q(k+1,j,i)) 
                          );
            }
        }
    );

    haloexchanger.exchange_halos(flux_field);
    bc_manager_flux.apply_z_bcs_to_field(flux_field);

    // flux_field.print_xz_cross_at_j(grid, 0, 3);

    auto rdz_view = params.rdz;
    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    Kokkos::parallel_for("flux_convergence_tendency", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k,j,i) += -0.5*(flux(k,j,i) - flux(k-1,j,i)) * rdz_view() * flex_height_coef_mid(k) / rhobar_divide(k);
        }
    );
    return;
}

} // namespace Dynamics
} // namespace VVM
