#include "WindSolver.hpp"
#include "core/HaloExchanger.hpp"

namespace VVM {
namespace Dynamics {

WindSolver::WindSolver(const Core::Grid& grid, const Utils::ConfigurationManager& config, const Core::Parameters& params)
    : grid_(grid), config_(config), halo_exchanger_(grid), params_(params),
      YTEM_field_("YTEM", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      W3DNP1_field_("W3DNP1", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      W3DN_field_("W3DN", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      RHSV_field_("RHSV", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      pm_temp_field_("pm_temp", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      pm_field_("pm", {grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      RIP1_field_("RIP1", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      ROP1_field_("ROP1", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      RIP2_field_("RIP2", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      ROP2_field_("ROP2", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      ATEMP_field_("ATEMP", {grid.get_local_total_points_y(), grid.get_local_total_points_x()}) {

    std::string solver_method_str = config.get_value<std::string>("dynamics.solver.w_solver_method");
    if (solver_method_str == "tridiagonal") {
        w_solver_method_ = WSolverMethod::TRIDIAGONAL;
    } 
    else {
        w_solver_method_ = WSolverMethod::JACOBI;
    }

}

void WindSolver::solve_w(Core::State& state) {
    VVM::Utils::Timer solve_w_timer("SOLVE_W");

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto& iter_num = params_.solver_iteration;
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;
    const auto& rdx2 = params_.rdx2;
    const auto& rdy2 = params_.rdy2;
    // const auto& rdz2 = params.rdz2;
    const auto& WRXMU = params_.WRXMU;
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& AGAU = params_.AGAU.get_device_data();
    const auto& BGAU = params_.BGAU.get_device_data();
    const auto& CGAU = params_.CGAU.get_device_data();

    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();

    auto& w = state.get_field<3>("w").get_mutable_device_data();

    auto& YTEM = YTEM_field_.get_mutable_device_data();
    auto& W3DNP1 = W3DNP1_field_.get_mutable_device_data();
    auto& W3DN = W3DN_field_.get_mutable_device_data();
    auto& RHSV = RHSV_field_.get_mutable_device_data();
    auto& pm_temp = pm_temp_field_.get_mutable_device_data();
    auto& pm = pm_field_.get_mutable_device_data();

    Kokkos::parallel_for("Poisson", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
             YTEM(k,j,i)=-(eta(k,j,i) - eta(k,j,i-1))*rdx()
                         -( xi(k,j,i) -  xi(k,j-1,i))*rdy();
        }
    );
    halo_exchanger_.exchange_halos(YTEM_field_);
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) state.get_field<3>("w").print_slice_z_at_k(grid_, 0, 17);
    // exit(1);

    // Linear extrapolation of initial guess
    auto& W3DNM1 = state.get_field<3>("W3DNM1").get_mutable_device_data();
    Kokkos::parallel_for("W3DNP1", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,0,0}, {nz-h-1,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            W3DNP1(k,j,i) = 2.*w(k,j,i) - W3DNM1(k,j,i);
        }
    );

    // Store w to previous step
    Kokkos::deep_copy(W3DNM1, w);

    // Assign interpolated w to w
    Kokkos::deep_copy(w, W3DNP1);

    const auto& bn_new = params_.bn_new.get_device_data();
    const auto& cn_new = params_.cn_new.get_device_data();

    VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, "w");
    if (w_solver_method_ == WSolverMethod::TRIDIAGONAL) {
        for (int iter = 0; iter < iter_num; iter++) {
            // Copy w to w3dn
            Kokkos::deep_copy(W3DN, w);

            Kokkos::parallel_for("calculate_RHSV", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    RHSV(k,j,i) = WRXMU() * W3DN(k,j,i)
                               + (W3DN(k,j,i+1)+W3DN(k,j,i-1))*rdx2()
                               + (W3DN(k,j+1,i)+W3DN(k,j-1,i))*rdy2()
                               + YTEM(k,j,i);
                }
            );

            // Gauss elimination
            Kokkos::parallel_for("fused_tridiagonal_solver", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
                KOKKOS_LAMBDA(int j, int i) {

                    // Forward elimination)
                    pm_temp(h,j,i) = RHSV(h,j,i) / BGAU(h);
                    for (int k = h+1; k <= nz-h-2; k++) {
                        pm_temp(k,j,i) = (RHSV(k,j,i) - AGAU(k) * pm_temp(k-1,j,i)) / bn_new(k);
                    }

                    // Backward substitution
                    pm(nz-h-2,j,i) = pm_temp(nz-h-2,j,i);
                    for (int k = nz-h-3; k >= h; k--) {
                        pm(k,j,i) = pm_temp(k,j,i) - cn_new(k) * pm(k+1,j,i);
                    }
                }
            );
            // set boundary to be 0
            Kokkos::parallel_for("set_boundary", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {h,ny,nx}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    pm(k,j,i) = 0.;
                }
            );

            // Get w
            Kokkos::parallel_for("copy_w_to_w3dn", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,h,h}, {nz-h,ny-h,nx-h}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    if (k == h-1) w(k,j,i) = 0.;
                    else if (k == nz-h-1) w(k,j,i) = 0.;
                    else w(k,j,i) = pm(k,j,i) / rhobar_up(k);
                }
            );
            halo_exchanger_.exchange_halos(state.get_field<3>("w"));
        }
    }
    else {
        for (int iter = 0; iter < iter_num; iter++) {
            Kokkos::deep_copy(W3DN, w);

            Kokkos::parallel_for("jacobi_w_solver", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    const double horizontal_terms = (W3DN(k,j,i+1)+W3DN(k,j,i-1))*rdx2()
                                                  + (W3DN(k,j+1,i)+W3DN(k,j-1,i))*rdy2();

                    const double vertical_terms = -AGAU(k)*W3DN(k-1,j,i)*rhobar_up(k-1)
                                                  -CGAU(k)*W3DN(k+1,j,i)*rhobar_up(k+1);

                    const double diagonal_term = BGAU(k)*rhobar_up(k) - WRXMU();
                    
                    if (diagonal_term != 0.0) {
                        w(k,j,i) = (YTEM(k,j,i) + horizontal_terms + vertical_terms) / diagonal_term;
                    }
                }
            );
            halo_exchanger_.exchange_halos(state.get_field<3>("w"));
        }
    }
    // bc_manager.apply_z_bcs_to_field(state.get_field<3>("w"));
    return;
}


void WindSolver::solve_uv(Core::State& state) {
    VVM::Utils::Timer solve_uv_timer("SOLVE_UV");

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& rdz = params_.rdz;

    auto& psi_field = state.get_field<2>("psi");
    auto& psinm1_field = state.get_field<2>("psinm1");
    auto& psi = psi_field.get_mutable_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    const auto& zeta_slice = Kokkos::subview(zeta, nz-h-1, Kokkos::ALL(), Kokkos::ALL());
    auto& RIP1 = RIP1_field_.get_mutable_device_data();

    auto& w = state.get_field<3>("w").get_mutable_device_data();
    auto& chi_field = state.get_field<2>("chi");
    auto& chi = chi_field.get_mutable_device_data();
    auto& chinm1_field = state.get_field<2>("chinm1");
    auto& RIP2 = RIP2_field_.get_mutable_device_data();

    // Solve psi
    Kokkos::deep_copy(RIP1, zeta_slice);
    relax_2d(psi_field, psinm1_field, RIP1_field_, ROP1_field_);

    // Copy psi data to previous step and step
    Kokkos::deep_copy(psinm1_field.get_mutable_device_data(), psi);
    Kokkos::deep_copy(psi, ROP1_field_.get_mutable_device_data());

    // Solve chi
    Kokkos::parallel_for("interpolation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(int j, int i) {
            RIP2(j,i) = flex_height_coef_mid(nz-h-1)*rhobar_up(nz-h-2)*w(nz-h-2,j,i)*rdz() / rhobar(nz-h-1);
        }
    );
    relax_2d(chi_field, chinm1_field, RIP2_field_, ROP2_field_);

    // Copy psi data to previous step and step
    Kokkos::deep_copy(chinm1_field.get_mutable_device_data(), chi);
    Kokkos::deep_copy(chi, ROP2_field_.get_mutable_device_data());

    // Calculate utop, vtop
    auto& utop_field = state.get_field<2>("utop");
    auto& vtop_field = state.get_field<2>("vtop");
    auto& utop = utop_field.get_mutable_device_data();
    auto& vtop = vtop_field.get_mutable_device_data();
    const auto& rdx = params_.rdx;
    const auto& rdy = params_.rdy;
    Kokkos::parallel_for("calculate_uvtop", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(int j, int i) {
            utop(j,i) = -(psi(j,i) - psi(j-1,i)) * rdy() + (chi(j,i+1) - chi(j,i)) * rdx();
            vtop(j,i) = (psi(j,i) - psi(j,i-1)) * rdx() + (chi(j+1,i) - chi(j,i)) * rdy();
        }
    );
    
    // calculate u
    auto& u_field = state.get_field<3>("u");
    auto& u = u_field.get_mutable_device_data();
    double utopm = state.calculate_horizontal_mean(utop_field);
    auto& v_field = state.get_field<3>("v");
    auto& v = v_field.get_mutable_device_data();
    double vtopm = state.calculate_horizontal_mean(vtop_field);
    Kokkos::parallel_for("uvtop_process", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(int j, int i) {
            // TODO: uvtop predict
            // u(nz-h-1,j,i) = utop(j,i) - utopm;
            // v(nz-h-1,j,i) = vtop(j,i) - vtopm;
            u(nz-h-1,j,i) = utop(j,i);
            v(nz-h-1,j,i) = vtop(j,i);
        }
    );

    const auto& xi = state.get_field<3>("xi").get_mutable_device_data();
    const auto& eta = state.get_field<3>("eta").get_mutable_device_data();
    const auto& dz = params_.dz;
    Kokkos::parallel_for("u_downward_integration",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // The for-loop inside is to prevent racing condition because lower layers depend on upper layers.
            for (int k = nz-h-2; k >= h-1; --k) {
                // WARNING: The eta is with negative because of the eta definition in original VVM
                // FIXME: Need to fix it if the definition is reversed.
                u(k,j,i) = u(k+1,j,i) 
                         - ((w(k,j,i+1) - w(k,j,i))*rdx() - eta(k,j,i)) * dz() / flex_height_coef_up(k); 
                v(k,j,i) = v(k+1,j,i) 
                         - ((w(k,j+1,i) - w(k,j,i))*rdy() - xi(k,j,i)) * dz() / flex_height_coef_up(k); 
            }
            // WARNING: NK3 has a upward integration in original VVM code.
            u(nz-h,j,i) = u(nz-h-1,j,i)
                      + ((w(nz-h-1,j,i+1) - w(nz-h-1,j,i))*rdx() - eta(nz-h-1,j,i)) * dz() / flex_height_coef_up(nz-h-1); 
            v(nz-h,j,i) = v(nz-h-1,j,i)
                      + ((w(nz-h-1,j+1,i) - w(nz-h-1,j,i))*rdy() -  xi(nz-h-1,j,i)) * dz() / flex_height_coef_up(nz-h-1); 
        }
    );
    VVM::Core::BoundaryConditionManager bc_manager(grid_);
    halo_exchanger_.exchange_halos(u_field);
    halo_exchanger_.exchange_halos(v_field);
    // bc_manager.apply_z_bcs_to_field(u_field);
    // bc_manager.apply_z_bcs_to_field(v_field);
    return;
}

void WindSolver::relax_2d(Core::Field<2>& A_field, Core::Field<2>& ANM1_field, Core::Field<2>& RHSV_field, Core::Field<2>& AOUT_field) {
    const auto& WRXMU = params_.WRXMU;
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    auto& ATEMP = ATEMP_field_.get_mutable_device_data();
    auto& A = A_field.get_mutable_device_data();
    auto& ANM1 = ANM1_field.get_mutable_device_data();
    auto& RHSV = RHSV_field.get_mutable_device_data();
    auto& AOUT = AOUT_field.get_mutable_device_data();

    const auto& iter_num = params_.solver_iteration;
    const auto& rdx2 = params_.rdx2;
    const auto& rdy2 = params_.rdy2;
    // const auto C0 = WRXMU() + 2.*rdx2() + 2.*rdy2();

    Kokkos::parallel_for("interpolation", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny,nx}),
        KOKKOS_LAMBDA(int j, int i) {
            AOUT(j,i) = 2.*A(j,i) - ANM1(j,i);
        }
    );

    for (int iter = 0; iter < iter_num; iter++) {
        Kokkos::deep_copy(ATEMP, AOUT);

        Kokkos::parallel_for("AOUT", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
            KOKKOS_LAMBDA(int j, int i) {
                AOUT(j,i) = (WRXMU()*ATEMP(j,i) + rdx2()*(ATEMP(j,i-1)+ATEMP(j,i+1)) 
                          + rdy2()*(ATEMP(j-1,i)+ATEMP(j+1,i)) - RHSV(j,i)) / (WRXMU() + 2.*rdx2() + 2.*rdy2());
            }
        );

        halo_exchanger_.exchange_halos(AOUT_field);
    }
    return;
}

} // namespace Dynamics
} // namespace VVM
