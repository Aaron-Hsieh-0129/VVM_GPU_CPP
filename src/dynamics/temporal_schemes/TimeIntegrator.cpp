#include "TimeIntegrator.hpp"
#include "core/Field.hpp"

namespace VVM {
namespace Dynamics {

TimeIntegrator::TimeIntegrator(std::string var_name, bool has_ab2, bool has_fe)
    : variable_name_(std::move(var_name)),
      has_ab2_terms_(has_ab2),
      has_fe_terms_(has_fe) {}

TimeIntegrator::~TimeIntegrator() = default;

void TimeIntegrator::step(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    double dt) const {

    auto& field_to_update = state.get_field<3>(variable_name_);
    auto field_new_view = field_to_update.get_mutable_device_data();

    const int nz = grid.get_local_total_points_z();
    const int ny = grid.get_local_total_points_y();
    const int nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();

    int k_start = h;
    int k_end = nz-h;
    if (variable_name_ == "xi" || variable_name_ == "eta") {
        k_end = nz-h-1;
    }

    if (has_ab2_terms_) {
        // --- Case 1: Variable uses Adams-Bashforth (and possibly also Forward Euler) ---
        auto& field_prev_step = state.get_field<3>(variable_name_ + "_m");
        Kokkos::deep_copy(field_prev_step.get_mutable_device_data(), field_to_update.get_device_data());
        auto& field_old_view = field_prev_step.get_device_data();
        
        size_t now_idx = time_step_count_ % 2;
        size_t prev_idx = (time_step_count_ + 1) % 2;
        
        auto& tendency_history = state.get_field<4>("d_" + variable_name_);
        auto tendency_now_view = Kokkos::subview(tendency_history.get_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto tendency_prev_view = Kokkos::subview(tendency_history.get_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

        if (time_step_count_ == 0) {
            if (variable_name_ == "zeta") {
                Kokkos::parallel_for("AB2_Forward_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int j, const int i) {
                        field_new_view(nz-h-1, j, i) = field_old_view(nz-h-1, j, i) + dt * tendency_now_view(nz-h-1, j, i);
                    }
                );
            }
            else {
                Kokkos::parallel_for("AB2_Forward_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny - h, nx - h}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        field_new_view(k, j, i) = field_old_view(k, j, i) + dt * tendency_now_view(k, j, i);
                    }
                );
            }

        } 
        else {
            if (variable_name_ == "zeta") {
                Kokkos::parallel_for("AdamsBashforth2_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny - h, nx - h}),
                    KOKKOS_LAMBDA(const int j, const int i) {
                        field_new_view(nz-h-1, j, i) = field_old_view(nz-h-1, j, i) 
                                                + dt * (1.5 * tendency_now_view(nz-h-1, j, i) - 0.5 * tendency_prev_view(nz-h-1, j, i));
                    }
                );
            }
            else {
                Kokkos::parallel_for("AdamsBashforth2_Step", 
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int k, const int j, const int i) {
                        field_new_view(k, j, i) = field_old_view(k, j, i) 
                                                + dt * (1.5 * tendency_now_view(k, j, i) - 0.5 * tendency_prev_view(k, j, i));
                    }
                );
            }
        }
    } 
    else if (has_fe_terms_) {
        // --- Case 2: Variable *only* uses Forward Euler ---
        // We need a copy of the original field state before updating
        Core::Field<3> field_old_state("temp_old_state", {nz, ny, nx});
        Kokkos::deep_copy(field_old_state.get_mutable_device_data(), field_new_view);
        auto field_old_view = field_old_state.get_device_data();
        
        auto& fe_tendency_field = state.get_field<3>("fe_tendency_" + variable_name_);
        auto fe_tendency_data = fe_tendency_field.get_device_data();
        
        Kokkos::parallel_for("Pure_Forward_Euler_Step",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_new_view(k, j, i) = field_old_view(k, j, i) + dt * fe_tendency_data(k, j, i);
            }
        );
    }

    // --- Add Forward Euler tendencies on top of AB2 update if applicable ---
    if (has_ab2_terms_ && has_fe_terms_) {
        auto& fe_tendency_field = state.get_field<3>("fe_tendency_" + variable_name_);
        auto fe_tendency_data = fe_tendency_field.get_device_data();

        Kokkos::parallel_for("Forward_Euler_FE_Terms_Additive",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_new_view(k, j, i) += dt * fe_tendency_data(k, j, i);
            }
        );
    }

    time_step_count_++;
}

} // namespace Dynamics
} // namespace VVM
