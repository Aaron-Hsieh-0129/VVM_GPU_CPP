#include "AreaMeanNudging.hpp"
#include <iostream>

namespace VVM {
namespace Dynamics {

AreaMeanNudging::AreaMeanNudging(const Utils::ConfigurationManager& config, 
                                 const Core::Grid& grid, 
                                 const Core::Parameters& params)
    : config_(config), grid_(grid), params_(params) 
{
    enable_ = config_.get_value<bool>("dynamics.forcings.areamn.enable", false);
    
    if (enable_) {
        uvtau_ = config_.get_value<VVM::Real>("dynamics.forcings.areamn.uvtau", 0.0);
        nudgelim_ = config_.get_value<VVM::Real>("dynamics.forcings.areamn.nudge_start_m", 0.0);
        
        VVM::Real total_pts = static_cast<VVM::Real>(grid_.get_global_points_x() * grid_.get_global_points_y());
        inv_total_xy_pts_ = 1.0 / total_pts;

        if (grid_.get_mpi_rank() == 0) {
            std::cout << "--- Initializing Area Mean Nudging (AREAMN) ---" << std::endl;
            std::cout << "  * UVTAU: " << uvtau_ << " s, Nudge Limit: " << nudgelim_ << " m" << std::endl;
        }
    }
}

void AreaMeanNudging::initialize(Core::State& state) {
    if (!enable_) return;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h  = grid_.get_halo_cells();
    int top_k = nz - h - 1;

    if (!state.has_field("areamn_xi0")) state.add_field<1>("areamn_xi0", {nz});
    if (!state.has_field("areamn_eta0")) state.add_field<1>("areamn_eta0", {nz});
    if (!state.has_field("areamn_zeta0_top")) state.add_field<0>("areamn_zeta0_top", {});
    if (!state.has_field("areamn_local_sum_xi")) state.add_field<1>("areamn_local_sum_xi", {nz});
    if (!state.has_field("areamn_global_sum_xi")) state.add_field<1>("areamn_global_sum_xi", {nz});
    if (!state.has_field("areamn_local_sum_eta")) state.add_field<1>("areamn_local_sum_eta", {nz});
    if (!state.has_field("areamn_global_sum_eta")) state.add_field<1>("areamn_global_sum_eta", {nz});
    if (!state.has_field("areamn_local_sum_zeta_top")) state.add_field<0>("areamn_local_sum_zeta_top", {});
    if (!state.has_field("areamn_global_sum_zeta_top")) state.add_field<0>("areamn_global_sum_zeta_top", {});
    if (!state.has_field("areamn_utopmn0")) state.add_field<0>("areamn_utopmn0", {});
    if (!state.has_field("areamn_vtopmn0")) state.add_field<0>("areamn_vtopmn0", {});

    const auto& xi   = state.get_field<3>("xi").get_device_data();
    const auto& eta  = state.get_field<3>("eta").get_device_data();
    const auto& zeta = state.get_field<3>("zeta").get_device_data();
    const auto& u    = state.get_field<3>("u").get_device_data();
    const auto& v    = state.get_field<3>("v").get_device_data();

    auto& l_sum_xi = state.get_field<1>("areamn_local_sum_xi").get_mutable_device_data();
    auto& l_sum_eta = state.get_field<1>("areamn_local_sum_eta").get_mutable_device_data();
    auto& g_sum_xi = state.get_field<1>("areamn_global_sum_xi").get_mutable_device_data();
    auto& g_sum_eta = state.get_field<1>("areamn_global_sum_eta").get_mutable_device_data();
    auto& l_sum_zeta = state.get_field<0>("areamn_local_sum_zeta_top").get_mutable_device_data();
    auto& g_sum_zeta = state.get_field<0>("areamn_global_sum_zeta_top").get_mutable_device_data();

    Kokkos::View<VVM::Real> l_sum_u("l_sum_u");
    Kokkos::View<VVM::Real> g_sum_u("g_sum_u");
    Kokkos::View<VVM::Real> l_sum_v("l_sum_v");
    Kokkos::View<VVM::Real> g_sum_v("g_sum_v");

    Kokkos::deep_copy(l_sum_xi, 0.0);
    Kokkos::deep_copy(l_sum_eta, 0.0);
    Kokkos::deep_copy(l_sum_zeta, 0.0);
    Kokkos::deep_copy(l_sum_u, 0.0);
    Kokkos::deep_copy(l_sum_v, 0.0);

    Kokkos::parallel_for("AREAMN_Init_Local_Sum",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            Kokkos::atomic_add(&l_sum_xi(k), xi(k, j, i));
            Kokkos::atomic_add(&l_sum_eta(k), eta(k, j, i));
            
            if (k == top_k) {
                Kokkos::atomic_add(&l_sum_zeta(), zeta(top_k, j, i));
                Kokkos::atomic_add(&l_sum_u(), u(top_k, j, i));
                Kokkos::atomic_add(&l_sum_v(), v(top_k, j, i));
            }
        }
    );

#if defined(ENABLE_NCCL)
    ncclComm_t comm = state.get_nccl_comm();
    cudaStream_t stream = state.get_cuda_stream();

    ncclGroupStart();
    ncclAllReduce(l_sum_xi.data(), g_sum_xi.data(), nz, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclAllReduce(l_sum_eta.data(), g_sum_eta.data(), nz, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclAllReduce(l_sum_zeta.data(), g_sum_zeta.data(), 1, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclAllReduce(l_sum_u.data(), g_sum_u.data(), 1, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclAllReduce(l_sum_v.data(), g_sum_v.data(), 1, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclGroupEnd();
#else
    MPI_Allreduce(l_sum_xi.data(), g_sum_xi.data(), nz, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
    MPI_Allreduce(l_sum_eta.data(), g_sum_eta.data(), nz, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
    MPI_Allreduce(l_sum_zeta.data(), g_sum_zeta.data(), 1, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
    MPI_Allreduce(l_sum_u.data(), g_sum_u.data(), 1, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
    MPI_Allreduce(l_sum_v.data(), g_sum_v.data(), 1, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
#endif

    auto& xi0 = state.get_field<1>("areamn_xi0").get_mutable_device_data();
    auto& eta0 = state.get_field<1>("areamn_eta0").get_mutable_device_data();
    auto& zeta0_top = state.get_field<0>("areamn_zeta0_top").get_mutable_device_data();
    auto& utopmn0 = state.get_field<0>("areamn_utopmn0").get_mutable_device_data();
    auto& vtopmn0 = state.get_field<0>("areamn_vtopmn0").get_mutable_device_data();

    auto& utopmn = state.get_field<0>("utopmn").get_mutable_device_data();
    auto& vtopmn = state.get_field<0>("vtopmn").get_mutable_device_data();

    VVM::Real inv_pts = inv_total_xy_pts_;

    Kokkos::parallel_for("AREAMN_Init_Save", nz, KOKKOS_LAMBDA(const int k) {
        xi0(k) = g_sum_xi(k) * inv_pts;
        eta0(k) = g_sum_eta(k) * inv_pts;

        if (k == top_k) {
            zeta0_top() = g_sum_zeta() * inv_pts;
            utopmn0() = g_sum_u() * inv_pts;
            vtopmn0() = g_sum_v() * inv_pts;
            
            utopmn() = utopmn0();
            vtopmn() = vtopmn0();
        }
    });
}

void AreaMeanNudging::apply_vorticity(Core::State& state, VVM::Real dt) {
    if (!enable_) return;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h  = grid_.get_halo_cells();
    int top_k = nz - h - 1;

    auto& xi = state.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state.get_field<3>("eta").get_mutable_device_data();
    auto& zeta = state.get_field<3>("zeta").get_mutable_device_data();

    const auto& itypeu = state.get_field<3>("ITYPEU").get_device_data();
    const auto& itypev = state.get_field<3>("ITYPEV").get_device_data();

    auto& l_sum_xi = state.get_field<1>("areamn_local_sum_xi").get_mutable_device_data();
    auto& l_sum_eta = state.get_field<1>("areamn_local_sum_eta").get_mutable_device_data();
    auto& g_sum_xi = state.get_field<1>("areamn_global_sum_xi").get_mutable_device_data();
    auto& g_sum_eta = state.get_field<1>("areamn_global_sum_eta").get_mutable_device_data();
    auto& l_sum_zeta = state.get_field<0>("areamn_local_sum_zeta_top").get_mutable_device_data();
    auto& g_sum_zeta = state.get_field<0>("areamn_global_sum_zeta_top").get_mutable_device_data();

    Kokkos::deep_copy(l_sum_xi, 0.0);
    Kokkos::deep_copy(l_sum_eta, 0.0);
    Kokkos::deep_copy(l_sum_zeta, 0.0);

    // NOTE: This might need to consider topography to do the average but now just follow Fortran VVM
    Kokkos::parallel_for("AREAMN_Sum",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            Kokkos::atomic_add(&l_sum_xi(k), xi(k, j, i));
            Kokkos::atomic_add(&l_sum_eta(k), eta(k, j, i));
            if (k == top_k) Kokkos::atomic_add(&l_sum_zeta(), zeta(top_k, j, i));
        }
    );

#if defined(ENABLE_NCCL)
    ncclComm_t comm = state.get_nccl_comm();
    cudaStream_t stream = state.get_cuda_stream();

    ncclGroupStart();
    ncclAllReduce(l_sum_xi.data(), g_sum_xi.data(), nz, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclAllReduce(l_sum_eta.data(), g_sum_eta.data(), nz, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclAllReduce(l_sum_zeta.data(), g_sum_zeta.data(), 1, VVM_NCCL_REAL, ncclSum, comm, stream);
    ncclGroupEnd();
#else
    MPI_Allreduce(l_sum_xi.data(), g_sum_xi.data(), nz, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
    MPI_Allreduce(l_sum_eta.data(), g_sum_eta.data(), nz, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
    MPI_Allreduce(l_sum_zeta.data(), g_sum_zeta.data(), 1, VVM_MPI_REAL, MPI_SUM, grid_.get_cart_comm());
#endif

    const auto& xi0 = state.get_field<1>("areamn_xi0").get_device_data();
    const auto& eta0 = state.get_field<1>("areamn_eta0").get_device_data();
    const auto& z0_top = state.get_field<0>("areamn_zeta0_top").get_device_data();
    const auto& z_coords = params_.z_up.get_device_data();

    VVM::Real uvtau = uvtau_;
    VVM::Real nlim = nudgelim_;
    VVM::Real inv_pts = inv_total_xy_pts_;

    Kokkos::parallel_for("AREAMN_Apply",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz - h, ny - h, nx - h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {

            if (z_coords(k) >= nlim) {
                VVM::Real sum_xi = g_sum_xi(k) * inv_pts;
                VVM::Real sum_eta = g_sum_eta(k) * inv_pts;
                
                VVM::Real sumxn = (uvtau == 0.0) ? xi0(k) : (1.0 - dt/uvtau)*sum_xi + xi0(k)*(dt/uvtau);
                VVM::Real sumyn = (uvtau == 0.0) ? eta0(k) : (1.0 - dt/uvtau)*sum_eta + eta0(k)*(dt/uvtau);

                if (itypev(k,j,i) == 1) {
                    xi(k,j,i) = xi(k,j,i) - sum_xi + sumxn;
                }
                if (itypeu(k,j,i) == 1) {
                    eta(k,j,i) = eta(k,j,i) - sum_eta + sumyn;
                }
            }

            if (k == top_k) {
                VVM::Real sum_zeta = g_sum_zeta() * inv_pts;
                zeta(top_k,j,i) = zeta(top_k,j,i) - sum_zeta + z0_top();
            }
        }
    );
}

void AreaMeanNudging::apply_uvtopmn(Core::State& state, VVM::Real dt) {
    if (!enable_) return;

    auto& utopmn = state.get_field<0>("utopmn").get_mutable_device_data();
    auto& vtopmn = state.get_field<0>("vtopmn").get_mutable_device_data();

    const auto& utopmn0 = state.get_field<0>("areamn_utopmn0").get_device_data();
    const auto& vtopmn0 = state.get_field<0>("areamn_vtopmn0").get_device_data();

    VVM::Real uvtau = uvtau_;

    Kokkos::parallel_for("AREAMN_Apply_UVTOPMN", 1, KOKKOS_LAMBDA(const int i) {
        if (uvtau == 0.0) {
            utopmn() = utopmn0();
            vtopmn() = vtopmn0();
        } 
        else {
            utopmn() = (1.0 - dt / uvtau) * utopmn() + utopmn0() * (dt / uvtau);
            vtopmn() = (1.0 - dt / uvtau) * vtopmn() + vtopmn0() * (dt / uvtau);
        }
    });
}

} // namespace Dynamics
} // namespace VVM
