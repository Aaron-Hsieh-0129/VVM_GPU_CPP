#include "core/BoundaryConditionManager.hpp"
#include <Kokkos_Core.hpp>

namespace VVM {
namespace Core {

BoundaryConditionManager::BoundaryConditionManager(const Grid& grid)
    : grid_ref_(grid) {
    top_bc_ = ZBoundaryType::ZERO_GRADIENT;
    bottom_bc_ = ZBoundaryType::ZERO_GRADIENT;
}

void BoundaryConditionManager::apply_z_bcs(State& state) const {
    for (auto& field_pair : state) {
        std::visit([this](auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                this->apply_z_bcs_to_field(field);
            }
        }, field_pair.second);
    }
}

template<size_t Dim>
void BoundaryConditionManager::apply_z_bcs_to_field(Field<Dim>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_ref_.get_local_total_points_z();
    
    if constexpr (Dim == 1) {
        Kokkos::parallel_for("apply_bc_1d", Kokkos::RangePolicy<>(0, h),
            KOKKOS_LAMBDA(const int k_h) {
                // Zero gradiednt
                // Bottom
                data(k_h) = data(h);

                // Top
                data(nz-1-k_h) = data(nz-1-h);
            }
        );
    }
    else if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);
        
        // Bottom B.C.
        Kokkos::parallel_for("apply_bc_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k_h, int j, int i) {
                // Zero gradient
                // Bottom
                data(k_h, j, i) = data(h, j, i);
            
                // Top
                data(nz-1-k_h, j, i) = data(nz-1-h, j, i);
            }
        );
    }
    else if constexpr (Dim == 4) {
        const int N = data.extent(0);
        const int ny = data.extent(2);
        const int nx = data.extent(3);
        
        // Bottom B.C.
        Kokkos::parallel_for("apply_bottom_bc_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {N, h, ny, nx}),
            KOKKOS_LAMBDA(int N_h, int k_h, int j, int i) {
                // Zero gradient
                // Bottom
                data(N_h, k_h, j, i) = data(N_h, h, j, i);
            
                // Top
                data(N_h, nz-1-k_h, j, i) = data(N_h, nz-1-h, j, i);
            }
        );
    }
    
    Kokkos::fence();
}

} // namespace Core
} // namespace VVM
