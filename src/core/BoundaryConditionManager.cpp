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

BoundaryConditionManager::BoundaryConditionManager(const Grid& grid, const Utils::ConfigurationManager& config, const std::string& var_name)
    : grid_ref_(grid), var_name_(var_name) {
    ZBoundaryType default_top_bc = ZBoundaryType::ZERO_GRADIENT;
    ZBoundaryType default_bottom_bc = ZBoundaryType::ZERO_GRADIENT;

    if (config.has_key("boundary_conditions.default.top")) {
        default_top_bc = string_to_bc_type(config.get_value<std::string>("boundary_conditions.default.top"));
    }
    if (config.has_key("boundary_conditions.default.bottom")) {
        default_bottom_bc = string_to_bc_type(config.get_value<std::string>("boundary_conditions.default.bottom"));
    }

    top_bc_ = default_top_bc;
    bottom_bc_ = default_bottom_bc;

    std::string var_top_key = "boundary_conditions." + var_name + ".top";
    if (config.has_key(var_top_key)) {
        top_bc_ = string_to_bc_type(config.get_value<std::string>(var_top_key));
    }

    std::string var_bottom_key = "boundary_conditions." + var_name + ".bottom";
    if (config.has_key(var_bottom_key)) {
        bottom_bc_ = string_to_bc_type(config.get_value<std::string>(var_bottom_key));
    }
}

ZBoundaryType BoundaryConditionManager::string_to_bc_type(const std::string& bc_string) const {
    if (bc_string == "ZERO") return ZBoundaryType::ZERO;
    if (bc_string == "ZERO_GRADIENT") return ZBoundaryType::ZERO_GRADIENT;
    if (bc_string == "PERIODIC") return ZBoundaryType::PERIODIC;
    throw std::runtime_error("Unknown boundary condition type: " + bc_string);
}

template<size_t Dim>
void BoundaryConditionManager::apply_z_bcs_to_field(Field<Dim>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nz = grid_ref_.get_local_total_points_z();

    const ZBoundaryType top_bc = top_bc_;
    const ZBoundaryType bottom_bc = bottom_bc_;
    
    const bool is_special_zero_bc = (top_bc == ZBoundaryType::ZERO && (var_name_ == "xi" || var_name_ == "eta" || var_name_ == "w" || var_name_ == "flux"));
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
                    data(k_h) = data(nz-2*h+k_h);
                }

                // Top Boundary Condition
                if (top_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(nz-1-k_h) = data(nz-1-h);
                } 
                else if (top_bc == ZBoundaryType::ZERO) {
                    data(nz-1-k_h) = 0.0;
                }
                else if (top_bc == ZBoundaryType::PERIODIC) {
                    data(nz-h+k_h) = data(h+k_h);
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
                    data(k_h, j, i) = data(nz-2*h+k_h, j, i);
                }
            
                // Top Boundary Condition
                if (top_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(nz-1-k_h, j, i) = data(nz-1-h, j, i);
                } 
                else if (top_bc == ZBoundaryType::ZERO) {
                    data(nz-1-k_h, j, i) = 0.0;
                }
                else if (top_bc == ZBoundaryType::PERIODIC) {
                    data(nz-h+k_h, j, i) = data(h+k_h, j, i);
                }
            }
        );
        Kokkos::parallel_for("apply_bc_3d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
            KOKKOS_LAMBDA(int j, int i) {
                // This is for top (xi, eta, w) 
                if (is_special_zero_bc) {
                    data(nz-h-1, j, i) = 0.0;
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
                    data(N_h, k_h, j, i) = data(N_h, nz-2*h+k_h, j, i);
                }
            
                // Top Boundary Condition
                if (top_bc == ZBoundaryType::ZERO_GRADIENT) {
                    data(N_h, nz-1-k_h, j, i) = data(N_h, nz-1-h, j, i);
                } 
                else if (top_bc == ZBoundaryType::ZERO) {
                    data(N_h, nz-1-k_h, j, i) = 0.0;
                }
                else if (top_bc == ZBoundaryType::PERIODIC) {
                    data(N_h, nz-h+k_h, j, i) = data(N_h, h+k_h, j, i);
                }
            }
        );
    }
    
    Kokkos::fence();
}

} // namespace Core
} // namespace VVM
