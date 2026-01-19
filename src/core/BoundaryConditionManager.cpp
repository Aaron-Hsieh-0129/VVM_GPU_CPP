#include "core/BoundaryConditionManager.hpp"
#include <Kokkos_Core.hpp>

namespace VVM {
namespace Core {

BoundaryConditionManager::BoundaryConditionManager(const Grid& grid)
    : grid_(grid) {}

template<size_t Dim>
void BoundaryConditionManager::apply_dirichlet_zero(Field<Dim>& field) const {
    const int h = grid_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_.get_local_total_points_z();

    if constexpr (Dim == 1) {
        Kokkos::parallel_for("bc_dirichlet_0_1d", Kokkos::RangePolicy<>(0, h),
            KOKKOS_LAMBDA(const int k) {
                // Bottom Halo
                data(k) = 0.0;
                // Top Halo
                data(nz-1-k) = 0.0;
            }
        );
    }
    else if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);
        
        Kokkos::parallel_for("bc_dirichlet_0_3d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Bottom Halo
                data(k, j, i) = 0.0;
                // Top Halo
                data(nz-1-k, j, i) = 0.0;
            }
        );
    }
    else if constexpr (Dim == 4) {
        const int N = data.extent(0);
        const int ny = data.extent(2);
        const int nx = data.extent(3);

        Kokkos::parallel_for("bc_dirichlet_0_4d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {N, h, ny, nx}),
            KOKKOS_LAMBDA(int n, int k, int j, int i) {
                // Bottom Halo
                data(n, k, j, i) = 0.0;
                // Top Halo
                data(n, nz-1-k, j, i) = 0.0;
            }
        );
    }
}

template<size_t Dim>
void BoundaryConditionManager::apply_vorticity_bc(Field<Dim>& field) const {
    const int h = grid_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_.get_local_total_points_z();

    apply_dirichlet_zero(field);

    // NOTE: VVM requires physical top (nz-h-1) to be zero 
    if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);

        Kokkos::parallel_for("bc_vorticity_special_3d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
            KOKKOS_LAMBDA(int j, int i) {
                data(h-1, j, i) = 0.0;
                data(nz-h-1, j, i) = 0.0;
            }
        );
    }
}

template<size_t Dim>
void BoundaryConditionManager::apply_zero_gradient(Field<Dim>& field) const {
    const int h = grid_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_.get_local_total_points_z();

    if constexpr (Dim == 1) {
        Kokkos::parallel_for("bc_zerograd_1d", Kokkos::RangePolicy<>(0, h),
            KOKKOS_LAMBDA(const int k) {
                // Bottom: copy from h
                data(k) = data(h);
                // Top: copy from nz-1-h
                data(nz-1-k) = data(nz-1-h);
            }
        );
    }
    else if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);
        
        Kokkos::parallel_for("bc_zerograd_3d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Bottom
                data(k, j, i) = data(h, j, i);
                // Top
                data(nz-1-k, j, i) = data(nz-1-h, j, i);
            }
        );
    }
    else if constexpr (Dim == 4) {
        const int N = data.extent(0);
        const int ny = data.extent(2);
        const int nx = data.extent(3);

        Kokkos::parallel_for("bc_zerograd_4d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {N, h, ny, nx}),
            KOKKOS_LAMBDA(int n, int k, int j, int i) {
                // Bottom
                data(n, k, j, i) = data(n, h, j, i);
                // Top
                data(n, nz-1-k, j, i) = data(n, nz-1-h, j, i);
            }
        );
    }
    Kokkos::fence();
}

template<size_t Dim>
void BoundaryConditionManager::apply_fixed_profile_z(Field<Dim>& field, const Field<1>& profile) const {
    const int h = grid_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    auto p_data = profile.get_device_data();
    const int nz = grid_.get_local_total_points_z();

    if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);

        Kokkos::parallel_for("bc_fixed_profile_3d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Bottom
                data(k, j, i) = p_data(k);
                // Top
                data(nz-1-k, j, i) = p_data(nz-1-k);
            }
        );
    }
    Kokkos::fence();
}

template<size_t Dim>
void BoundaryConditionManager::apply_periodic(Field<Dim>& field) const {
    const int h = grid_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_.get_local_total_points_z();

    if constexpr (Dim == 1) {
        Kokkos::parallel_for("bc_periodic_1d", Kokkos::RangePolicy<>(0, h),
            KOKKOS_LAMBDA(const int k) {
                // Bottom: data(nz - 2h + k)
                data(k) = data(nz-2*h+k);
                // Top: data(h + k)
                data(nz-h+k) = data(h+k);
            }
        );
    }
    else if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);

        Kokkos::parallel_for("bc_periodic_3d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Bottom
                data(k, j, i) = data(nz-2*h + k, j, i);
                // Top
                data(nz-h+k, j, i) = data(h+k, j, i);
            }
        );
    }
    else if constexpr (Dim == 4) {
        const int N = data.extent(0);
        const int ny = data.extent(2);
        const int nx = data.extent(3);

        Kokkos::parallel_for("bc_periodic_4d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {N, h, ny, nx}),
            KOKKOS_LAMBDA(int n, int k, int j, int i) {
                // Bottom
                data(n, k, j, i) = data(n, nz-2*h + k, j, i);
                // Top
                data(n, nz-h+k, j, i) = data(n, h+k, j, i);
            }
        );
    }
    Kokkos::fence();
}

template<size_t Dim>
void BoundaryConditionManager::apply_zero_gradient_bottom_zero_top(Field<Dim>& field) const {
    const int h = grid_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_.get_local_total_points_z();

    if constexpr (Dim == 1) {
        Kokkos::parallel_for("bc_mixed_bot_grad_top_zero_1d", Kokkos::RangePolicy<>(0, h),
            KOKKOS_LAMBDA(const int k) {
                // Bottom: Copy from h (Zero Gradient)
                data(k) = data(h);
                // Top: Set to 0 (Zero Value)
                data(nz - 1 - k) = 0.0;
            }
        );
    }
    else if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);
        Kokkos::parallel_for("bc_mixed_bot_grad_top_zero_3d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Bottom: Copy from h (Zero Gradient)
                data(k, j, i) = data(h, j, i);
                // Top: Set to 0 (Zero Value)
                data(nz - 1 - k, j, i) = 0.0;
            }
        );
    }
    else if constexpr (Dim == 4) {
        const int N = data.extent(0);
        const int ny = data.extent(2);
        const int nx = data.extent(3);
        Kokkos::parallel_for("bc_mixed_bot_grad_top_zero_4d", 
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {N, h, ny, nx}),
            KOKKOS_LAMBDA(int n, int k, int j, int i) {
                // Bottom
                data(n, k, j, i) = data(n, h, j, i);
                // Top
                data(n, nz-1-k, j, i) = 0.0;
            }
        );
    }
    Kokkos::fence();
}

// Explicit Instantiation
// Dim = 1
template void BoundaryConditionManager::apply_dirichlet_zero<1>(Field<1>&) const;
template void BoundaryConditionManager::apply_vorticity_bc<1>(Field<1>&) const;
template void BoundaryConditionManager::apply_zero_gradient<1>(Field<1>&) const;
template void BoundaryConditionManager::apply_periodic<1>(Field<1>&) const;
template void BoundaryConditionManager::apply_zero_gradient_bottom_zero_top<1>(Field<1>&) const;

// Dim = 3
template void BoundaryConditionManager::apply_dirichlet_zero<3>(Field<3>&) const;
template void BoundaryConditionManager::apply_vorticity_bc<3>(Field<3>&) const;
template void BoundaryConditionManager::apply_zero_gradient<3>(Field<3>&) const;
template void BoundaryConditionManager::apply_fixed_profile_z<3>(Field<3>&, const Field<1>&) const;
template void BoundaryConditionManager::apply_periodic<3>(Field<3>&) const;
template void BoundaryConditionManager::apply_zero_gradient_bottom_zero_top<3>(Field<3>&) const;

// Dim = 4
template void BoundaryConditionManager::apply_dirichlet_zero<4>(Field<4>&) const;
template void BoundaryConditionManager::apply_vorticity_bc<4>(Field<4>&) const;
template void BoundaryConditionManager::apply_zero_gradient<4>(Field<4>&) const;
template void BoundaryConditionManager::apply_periodic<4>(Field<4>&) const;
template void BoundaryConditionManager::apply_zero_gradient_bottom_zero_top<4>(Field<4>&) const;

} // namespace Core
} // namespace VVM
