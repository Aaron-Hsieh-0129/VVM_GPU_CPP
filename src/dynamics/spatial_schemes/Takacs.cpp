#include "Takacs.hpp"
#include "core/HaloExchanger.hpp"

namespace VVM {
namespace Dynamics {

void Takacs::calculate_flux_divergence_x(
    const Core::Field<3>& scalar, const Core::Field<3>& u_field,
    const Core::Grid& grid, const Core::Parameters& params, Core::Field<3>& out_tendency) const {

    VVM::Core::HaloExchanger haloexchanger(grid);
    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();

    auto& u = u_field.get_device_data();
    auto& q = scalar.get_device_data();
    auto& flux = out_tendency.get_mutable_device_data();

    Kokkos::View<double***> uplus("uplus", nz, ny, nx);
    Kokkos::View<double***> uminus("uminus", nz, ny, nx);

    Kokkos::parallel_for("uplus_minus_cal", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            uplus(k,j,i) = 0.5*(u(k,j,i)+Kokkos::abs(u(k,j,i)));
            uminus(k,j,i) = 0.5*(u(k,j,i)-Kokkos::abs(u(k,j,i)));
    });

    Kokkos::parallel_for("flux_divergence", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1,ny-1,nx-1}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            if (i >= 2 && i <= nx-3) {
                flux(k,j,i) = u(k,j,i)*(q(k,j,i+1)+q(k,j,i)) - 
                          1./3.*( 
                            uplus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(uplus(k,j,i))*Kokkos::sqrt(uplus(k,j,i-1))*(q(k,j,i)-q(k,j,i-1)) - 
                            uminus(k,j,i)*(q(k,j,i+1)-q(k,j,i)) - Kokkos::sqrt(uminus(k,j,i))*Kokkos::sqrt(uminus(k,j,i+1))*(q(k,j,i+2)-q(k,j,i+1)) 
                          );
            }
            else {
                flux(k,j,i) = u(k,j,i)*(q(k,j,i+1)+q(k,j,i));
            }
    });

    haloexchanger.exchange_halos(out_tendency);


    return;
}

} // namespace Dynamics
} // namespace VVM
