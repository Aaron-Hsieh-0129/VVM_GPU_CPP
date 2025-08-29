#include "WindSolver.hpp"
#include "core/HaloExchanger.hpp"

namespace VVM {
namespace Dynamics {

WindSolver::WindSolver(const Core::Grid& grid, const Utils::ConfigurationManager& config)
    : grid_(grid), config_(config), halo_exchanger_(grid) {}

void WindSolver::solve_w(Core::State& state, const Core::Parameters& params) const {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto& rdx = params.rdx;
    const auto& rdy = params.rdy;
    const auto& rdx2 = params.rdx2;
    const auto& rdy2 = params.rdy2;
    // const auto& rdz2 = params.rdz2;
    const auto& WRXMU = params.WRXMU;
    const auto& flex_height_coef_mid = params.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params.flex_height_coef_up.get_device_data();
    const auto& AGAU = params.AGAU.get_device_data();
    const auto& BGAU = params.BGAU.get_device_data();
    const auto& CGAU = params.CGAU.get_device_data();

    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& xi = state.get_field<3>("xi").get_device_data();
    const auto& eta = state.get_field<3>("eta").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();

    auto& w = state.get_field<3>("w").get_mutable_device_data();

    Core::Field<3> YTEM_field("YTEM", {nz, ny, nx});
    auto& YTEM = YTEM_field.get_mutable_device_data();
    Kokkos::parallel_for("Poisson", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
             YTEM(k,j,i)=-(eta(k,j,i) - eta(k,j,i-1))*rdx()
                         -( xi(k,j,i) -  xi(k,j-1,i))*rdy();
        }
    );
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) state.get_field<3>("w").print_slice_z_at_k(grid_, 0, 17);
    // exit(1);

    // Linear extrapolation of initial guess
    Core::Field<3> W3DNP1_field("W3DNP1", {nz, ny, nx});
    auto& W3DNP1 = W3DNP1_field.get_mutable_device_data();
    auto& W3DNM1 = state.get_field<3>("W3DNM1").get_mutable_device_data();
    Kokkos::parallel_for("W3DNP1", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,0,0}, {nz-h,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            W3DNP1(k,j,i) = 2.*w(k,j,i) - W3DNM1(k,j,i);
        }
    );

    Kokkos::parallel_for("W3DNM1", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            W3DNM1(k,j,i) = w(k,j,i);
        }
    );

    Kokkos::parallel_for("W", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,0,0}, {nz-h,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            w(k,j,i) = W3DNP1(k,j,i);
        }
    );

    const auto& bn_new = params.bn_new.get_device_data();
    const auto& cn_new = params.cn_new.get_device_data();
    Core::Field<3> W3DN_field("W3DN", {nz, ny, nx});
    auto& W3DN = W3DN_field.get_mutable_device_data();
    Core::Field<3> RHSV_field("RHSV", {nz, ny, nx});
    auto& RHSV = RHSV_field.get_mutable_device_data();

    Core::Field<3> pm_temp_field("pm_temp", {nz, ny, nx});
    auto& pm_temp = pm_temp_field.get_mutable_device_data();
    Core::Field<3> pm_field("pm", {nz, ny, nx});
    auto& pm = pm_field.get_mutable_device_data();
    VVM::Core::BoundaryConditionManager bc_manager(grid_, config_, "w");
    for (int iter = 0; iter < 200; iter++) {
        Kokkos::parallel_for("copy_w_to_w3dn", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,0,0}, {nz-h-1,ny,nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                W3DN(k,j,i) = w(k,j,i);
            }
        );

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

        // Get w
        Kokkos::parallel_for("copy_w_to_w3dn", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                w(k,j,i) = pm(k,j,i) / rhobar_up(k);
            }
        );

        // FIXME: This is a strong bottleneck. Check whether it's using GPU communication.
        halo_exchanger_.exchange_halos(state.get_field<3>("w"));
    }
    bc_manager.apply_z_bcs_to_field(state.get_field<3>("w"));
    return;
}

void WindSolver::solve_poisson_2d(const Core::Field<3>& source, Core::Field<3>& result) const {
    // 實作 2D Poisson Solver
    // 這裡可以使用迭代法，例如 Jacobi 或 Successive-Over-Relaxation (SOR)。
    // 迭代的每一步都需要進行 halo 交換。
    // 這是一個簡化的 Jacobi 迭代範例：
    //
    // for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
    //     // 使用 source 和舊的 result 來計算新的 result
    //     scheme_->apply_laplacian(result, temp_field); // laplacian of current result
    //     // new_result = (source - temp_field) / ...
    //
    //     // 交換 result 的 halo 區
    //     Core::HaloExchanger halo_exchanger(grid_);
    //     halo_exchanger.exchange_halos(result);
    //
    //     // 檢查收斂
    // }
    //
    // 提醒：一個功能完整的 Poisson solver 需要謹慎地處理邊界條件和 MPI 通訊。
    // 由於其複雜性，這裡暫不提供完整程式碼。
}

} // namespace Dynamics
} // namespace VVM
