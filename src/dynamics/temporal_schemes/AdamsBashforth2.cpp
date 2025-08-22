#include "AdamsBashforth2.hpp"
#include <iostream>

namespace VVM {
namespace Dynamics {

AdamsBashforth2::AdamsBashforth2(std::string var_name, 
                                 std::vector<std::unique_ptr<TendencyTerm>> ab2_terms,
                                 std::vector<std::unique_ptr<TendencyTerm>> fe_terms)
    : variable_name_(std::move(var_name)), 
      ab2_tendency_terms_(std::move(ab2_terms)),
      fe_tendency_terms_(std::move(fe_terms)) {
}

AdamsBashforth2::~AdamsBashforth2() = default;

void AdamsBashforth2::calculate_tendency(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params) const {

    if (ab2_tendency_terms_.empty()) {
        return;
    }
    
    size_t now_idx = time_step_count_ % 2;
    auto& tendency_history = state.get_field<4>("d_" + variable_name_);
    auto total_current_tendency_view = Kokkos::subview(tendency_history.get_mutable_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    auto& field_to_update = state.get_field<3>(variable_name_);
    auto& field_current_view = field_to_update.get_mutable_device_data();

    const int nztot = grid.get_local_total_points_z();
    const int nytot = grid.get_local_total_points_y();
    const int nxtot = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();

    // Divide rho for xi, eta, zeta
    if (variable_name_ == "xi" || variable_name_ == "eta") {
        Kokkos::parallel_for("divide_by_density",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nztot, nytot, nxtot}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_current_view(k, j, i) /= rhobar_up(k);
            }
        );
    }
    else if (variable_name_ == "zeta") {
        Kokkos::parallel_for("divide_by_density",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nztot-h-2, 0, 0}, {nztot-h, nytot, nxtot}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_current_view(k, j, i) /= rhobar(k);
            }
        );
    }
    
    // Temp field
    Core::Field<3> current_tendency_field("temp_tendency_wrapper", 
        { (int)total_current_tendency_view.extent(0), 
          (int)total_current_tendency_view.extent(1), 
          (int)total_current_tendency_view.extent(2) 
        });
    current_tendency_field.initialize_to_zero();
    // Calculate all tendency terms
    for (const auto& term : ab2_tendency_terms_) {
        term->compute_tendency(state, grid, params, current_tendency_field);
    }


    // Divide rho for xi, eta, zeta
    if (variable_name_ == "xi" || variable_name_ == "eta") {
        Kokkos::parallel_for("divide_by_density",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nztot, nytot, nxtot}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_current_view(k, j, i) *= rhobar_up(k);
            }
        );
    }
    else if (variable_name_ == "zeta") {
        Kokkos::parallel_for("divide_by_density",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nztot-h-2, 0, 0}, {nztot-h, nytot, nxtot}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_current_view(k, j, i) *= rhobar(k);
            }
        );
    }

    // Save tendency to history
    Kokkos::deep_copy(total_current_tendency_view, current_tendency_field.get_device_data());
    return;
}


void AdamsBashforth2::step(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    double dt) const {

    if (!ab2_tendency_terms_.empty()) {
        // This index can decide which tendency is for now and which is for the previous step.
        // If step_count % 2 == 0, the now idx will be the first one of the 4D tendency field.
        size_t now_idx = time_step_count_ % 2;
        size_t prev_idx = (time_step_count_ + 1) % 2;
        
        auto& field_to_update = state.get_field<3>(variable_name_);
        auto& field_prev_step = state.get_field<3>(variable_name_ + "_m");
        auto& tendency_history = state.get_field<4>("d_" + variable_name_);

        // Copy now step to previous step to prepare for next step
        Kokkos::deep_copy(field_prev_step.get_mutable_device_data(), field_to_update.get_device_data());

        // Prepare the tendency and field to integrate
        auto& field_new_view = field_to_update.get_mutable_device_data();
        auto& field_old_view = field_prev_step.get_device_data();
        auto tendency_now_view = Kokkos::subview(tendency_history.get_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        auto tendency_prev_view = Kokkos::subview(tendency_history.get_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int h = grid.get_halo_cells();

        int k_start = h;
        int k_end = nz-h;
        if (variable_name_ == "zeta") {
            k_start = nz-h-1;
            k_end = nz-h;
        }

        // Use Forward Euler for the first step integration
        if (time_step_count_ == 0) {
            // new_x = old_x + dt * d_x_current
            Kokkos::parallel_for("Forward_step", 
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    field_new_view(k, j, i) = field_old_view(k, j, i) + dt * tendency_now_view(k, j, i);
                }
            );
        }
        else {
            // new_x = old_x + dt * (1.5 * d_x_current - 0.5 * d_x_prev)
            Kokkos::parallel_for("AdamsBashforth2_step", 
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    field_new_view(k, j, i) = field_old_view(k, j, i) 
                                            + dt * (1.5 * tendency_now_view(k, j, i) - 0.5 * tendency_prev_view(k, j, i));
                }
            );
        }

    }

    if (!fe_tendency_terms_.empty()) {
        auto& field_to_update = state.get_field<3>(variable_name_);
        auto field_new_view = field_to_update.get_mutable_device_data();

        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();

        Core::Field<3> fe_tendency_field("fe_tendency", {nz, ny, nx});
        fe_tendency_field.initialize_to_zero();

        for (const auto& term : fe_tendency_terms_) {
            term->compute_tendency(state, grid, params, fe_tendency_field);
        }
        
        auto fe_tendency_data = fe_tendency_field.get_device_data();

        Kokkos::parallel_for("Forward_Euler_step_FE",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                field_new_view(k, j, i) += dt * fe_tendency_data(k, j, i);
            }
        );
    }

    // Update time count
    time_step_count_++;
}

} // namespace Dynamics
} // namespace VVM
