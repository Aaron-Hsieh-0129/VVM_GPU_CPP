#include "AdvectionTerm.hpp"
#include "core/HaloExchanger.hpp"
#include "core/BoundaryConditionManager.hpp"

namespace VVM {
namespace Dynamics {

AdvectionTerm::AdvectionTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name)
    : scheme_(std::move(scheme)), variable_name_(std::move(var_name)) {}

AdvectionTerm::~AdvectionTerm() = default;


void AdvectionTerm::compute_tendency(
    const Core::State& state, 
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency) const {
    // Get scalar field that needs to be advected
    const auto& advected_field = state.get_field<3>(variable_name_);
    const auto& u_field = state.get_field<3>("u");
    const auto& v_field = state.get_field<3>("v");
    const auto& w_field = state.get_field<3>("w");
    auto u = u_field.get_device_data();
    auto v = v_field.get_device_data();
    auto w = w_field.get_device_data();
    const auto& rhobar_field = state.get_field<1>("rhobar");
    auto rhobar = rhobar_field.get_device_data();
    const auto& rhobar_up_field = state.get_field<1>("rhobar_up");
    auto rhobar_up = rhobar_up_field.get_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    VVM::Core::HaloExchanger haloexchanger(grid);
    VVM::Core::BoundaryConditionManager bc_manager_zerograd(grid, VVM::Core::ZBoundaryType::PERIODIC, VVM::Core::ZBoundaryType::PERIODIC);
    VVM::Core::BoundaryConditionManager bc_manager_zero(grid, VVM::Core::ZBoundaryType::ZERO, VVM::Core::ZBoundaryType::ZERO);
    VVM::Core::BoundaryConditionManager bc_manager_periodic(grid, VVM::Core::ZBoundaryType::PERIODIC, VVM::Core::ZBoundaryType::PERIODIC);

    VVM::Core::Field<3> u_mean_field("u_mean", {nz, ny, nx});
    auto u_mean_data = u_mean_field.get_mutable_device_data();
    VVM::Core::Field<3> v_mean_field("v_mean", {nz, ny, nx});
    auto v_mean_data = v_mean_field.get_mutable_device_data();
    VVM::Core::Field<3> w_mean_field("w_mean", {nz, ny, nx});
    auto w_mean_data = w_mean_field.get_mutable_device_data();
    if (variable_name_ == "xi") {
        auto fact1_xi_eta = params.fact1_xi_eta.get_device_data();
        auto fact2_xi_eta = params.fact1_xi_eta.get_device_data();
        Kokkos::parallel_for("calculate_rhou_for_xi",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                u_mean_data(k,j,i) = 0.25*(fact1_xi_eta(k) * rhobar(k+1) * ( u(k+1,j,i) + u(k+1,j+1,i))
                                         + fact2_xi_eta(k) * rhobar(k)   * ( u(k,j,i)   + u(k,j+1,i)  )  );
            }
        );

        Kokkos::parallel_for("calculate_rhov_for_xi",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                v_mean_data(k,j,i) = 0.25*(fact1_xi_eta(k) * rhobar(k+1) * ( v(k+1,j,i) + v(k+1,j+1,i))
                                         + fact2_xi_eta(k) * rhobar(k)   * ( v(k,j,i)   + v(k,j+1,i)  )  );
            }
        );

        // FIXME: I think the w needs to have fact but it turns out the source code doesn't have this. The code follows it for now.
        Kokkos::parallel_for("calculate_rhow_for_xi",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                w_mean_data(k,j,i) = 0.25*(rhobar_up(k+1) * ( w(k+1,j,i) + w(k+1,j+1,i))
                                         + rhobar_up(k)   * ( w(k,j,i)   + w(k,j+1,i)  )  );
            }
        );
    }
    else if (variable_name_ == "eta") {
        auto fact1_xi_eta = params.fact1_xi_eta.get_device_data();
        auto fact2_xi_eta = params.fact1_xi_eta.get_device_data();
        Kokkos::parallel_for("calculate_rhou_for_eta",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                u_mean_data(k,j,i) = 0.25*(fact1_xi_eta(k) * rhobar(k+1) * ( u(k+1,j,i) + u(k+1,j,i+1))
                                         + fact2_xi_eta(k) * rhobar(k)   * ( u(k,j,i)   + u(k,j,i+1)  )  );
            }
        );

        Kokkos::parallel_for("calculate_rhov_for_eta",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                v_mean_data(k,j,i) = 0.25*(fact1_xi_eta(k) * rhobar(k+1) * ( v(k+1,j,i) + v(k+1,j,i+1))
                                         + fact2_xi_eta(k) * rhobar(k)   * ( v(k,j,i)   + v(k,j,i+1)  )  );
            }
        );

        // FIXME: I think the w needs to have fact but it turns out the source code doesn't have this. The code follows it for now.
        Kokkos::parallel_for("calculate_rhow_for_eta",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                w_mean_data(k,j,i) = 0.25*(rhobar_up(k+1) * ( w(k+1,j,i) + w(k+1,j,i+1))
                                         + rhobar_up(k)   * ( w(k,j,i)   + w(k,j,i+1)  )  );
            }
        );
    }
    else if (variable_name_ == "zeta") {
        Kokkos::parallel_for("calculate_rhou_for_zeta",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nz-h-1,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                u_mean_data(k,j,i) = 0.25*rhobar(k)*(u(k,j,i)   + u(k,j,i+1)
                                                   + u(k,j+1,i) + u(k,j+1,i+1)   );
            }
        );

        Kokkos::parallel_for("calculate_rhov_for_zeta",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nz-h-1,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                v_mean_data(k,j,i) = 0.25*rhobar(k)*(v(k,j,i)   + v(k,j,i+1)
                                                   + v(k,j+1,i) + v(k,j+1,i+1)   );
            }
        );

        Kokkos::parallel_for("calculate_rhow_for_zeta",
            // The original code adopts Tackas 3rd order difference for boundary zeta, so it needs two w.
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nz-h-2,h,h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                w_mean_data(k,j,i) = 0.25*rhobar_up(k)*(w(k,j,i)   + w(k,j,i+1) 
                                                      + w(k,j+1,i) + w(k,j+1,i+1));
            }
        );
    }
    else {
        Kokkos::parallel_for("calculate_rhow_for_scalar",
            // Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h, ny-h, nx-h}),
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                w_mean_data(k,j,i) = rhobar_up(k) * w(k,j,i);
            }
        );

        // u_mean_field.print_slice_z_at_k(grid, 0, 16);
        
    }

    // FIXME: The haloexchanger should be processed for zeta
    if (variable_name_ != "zeta") {
        haloexchanger.exchange_halos(u_mean_field);
        haloexchanger.exchange_halos(v_mean_field);
        haloexchanger.exchange_halos(w_mean_field);
        bc_manager_zerograd.apply_z_bcs_to_field(u_mean_field);
        bc_manager_zerograd.apply_z_bcs_to_field(v_mean_field);
        // bc_manager_zero.apply_z_bcs_to_field(w_mean_field);
        // bc_manager_periodic.apply_z_bcs_to_field(w_mean_field);
    }

    if (variable_name_ == "xi" || variable_name_ == "eta" || variable_name_ == "zeta") {
        scheme_->calculate_flux_convergence_x(advected_field, u_mean_field, grid, params, out_tendency, variable_name_);
        scheme_->calculate_flux_convergence_y(advected_field, v_mean_field, grid, params, out_tendency, variable_name_);
    }
    else {
        scheme_->calculate_flux_convergence_x(advected_field, u_field, grid, params, out_tendency, variable_name_);
        scheme_->calculate_flux_convergence_y(advected_field, v_field, grid, params, out_tendency, variable_name_);
    }
    scheme_->calculate_flux_convergence_z(advected_field, rhobar_field, w_mean_field, grid, params, out_tendency, variable_name_);
}

} // namespace Dynamics
} // namespace VVM
