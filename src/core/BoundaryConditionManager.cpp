#include "core/BoundaryConditionManager.hpp"
#include <Kokkos_Core.hpp>

namespace VVM {
namespace Core {

BoundaryConditionManager::BoundaryConditionManager(const Grid& grid, ZBoundaryType top_bc, ZBoundaryType bottom_bc)
    : grid_ref_(grid), top_bc_(top_bc), bottom_bc_(bottom_bc) {}

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
    const int nz_phys = grid_ref_.get_local_physical_points_z();

    const ZBoundaryType top_bc = top_bc_;
    const ZBoundaryType bottom_bc = bottom_bc_;
    
    if constexpr (Dim == 1) {
        Kokkos::parallel_for("apply_bc_1d", Kokkos::RangePolicy<>(0, h),
            KOKKOS_LAMBDA(const int k_h) {
                // Bottom Boundary Condition
                if (bottom_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(k_h) = data(h);
                } 
                else if (bottom_bc == ZBoundaryType::ZERO) {
                    data(k_h) = 0.0;
                }
                else if (bottom_bc == ZBoundaryType::PERIODIC) {
                    data(k_h) = data(nz_phys - h + k_h);
                }

                // Top Boundary Condition
                if (top_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(nz-1-k_h) = data(nz-1-h);
                } 
                else if (top_bc == ZBoundaryType::ZERO) {
                    data(nz-1-k_h) = 0.0;
                }
                else if (top_bc == ZBoundaryType::PERIODIC) {
                    data(nz - h + k_h) = data(h + k_h);
                }
            }
        );
    }
    else if constexpr (Dim == 3) {
        const int ny = data.extent(1);
        const int nx = data.extent(2);
        
        // Bottom B.C.
        Kokkos::parallel_for("apply_bc_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {h, ny, nx}),
            KOKKOS_LAMBDA(int k_h, int j, int i) {
                // Bottom Boundary Condition
                if (bottom_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(k_h, j, i) = data(h, j, i);
                } 
                else if (bottom_bc == ZBoundaryType::ZERO) {
                    data(k_h, j, i) = 0.0;
                }
                else if (bottom_bc == ZBoundaryType::PERIODIC) {
                    data(k_h, j, i) = data(nz_phys - h + k_h, j, i);
                }
            
                // Top Boundary Condition
                if (top_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(nz-1-k_h, j, i) = data(nz-1-h, j, i);
                } 
                else if (top_bc == ZBoundaryType::ZERO) {
                    data(nz-1-k_h, j, i) = 0.0;
                }
                else if (top_bc == ZBoundaryType::PERIODIC) {
                    data(nz - h + k_h, j, i) = data(h + k_h, j, i);
                }
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
                // Bottom Boundary Condition
                if (bottom_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(N_h, k_h, j, i) = data(N_h, h, j, i);
                } 
                else if (bottom_bc == ZBoundaryType::ZERO) {
                    data(N_h, k_h, j, i) = 0.0;
                }
                else if (bottom_bc == ZBoundaryType::PERIODIC) {
                    data(N_h, k_h, j, i) = data(N_h, nz_phys - h + k_h, j, i);
                }
            
                // Top Boundary Condition
                if (top_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(N_h, nz-1-k_h, j, i) = data(N_h, nz-1-h, j, i);
                } 
                else if (top_bc == ZBoundaryType::ZERO) {
                    data(N_h, nz-1-k_h, j, i) = 0.0;
                }
                else if (top_bc == ZBoundaryType::PERIODIC) {
                    data(N_h, nz - h + k_h, j, i) = data(N_h, h + k_h, j, i);
                }
            }
        );
    }
    
    Kokkos::fence();
}

} // namespace Core
} // namespace VVM
