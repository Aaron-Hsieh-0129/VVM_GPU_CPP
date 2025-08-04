#include "AdamsBashforth2.hpp"
#include <iostream>

namespace VVM {
namespace Dynamics {

AdamsBashforth2::AdamsBashforth2(std::string var_name, std::vector<std::unique_ptr<TendencyTerm>> terms)
    : variable_name_(std::move(var_name)), tendency_terms_(std::move(terms)) {

}

AdamsBashforth2::~AdamsBashforth2() = default;

void AdamsBashforth2::step(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    double dt) const {

    // This index can decide which tendency is for now and which is for the previous step.
    // If step_count % 2 == 0, the now idx will be the first one of the 4D tendency field.
    size_t now_idx = time_step_count_ % 2;
    size_t prev_idx = (time_step_count_ + 1) % 2;
    
    auto& field_to_update = state.get_field<3>(variable_name_);
    auto& field_prev_step = state.get_field<3>(variable_name_ + "_m");
    auto& tendency_history = state.get_field<4>("d_" + variable_name_);

    // Copy now step to previous step to prepare for next step
    auto field_current_view = field_to_update.get_device_data();
    auto field_prev_view_mutable = field_prev_step.get_mutable_device_data();
    Kokkos::deep_copy(field_prev_view_mutable, field_current_view);

    // Get the total tendency for now
    auto total_current_tendency_view = Kokkos::subview(tendency_history.get_mutable_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    // Temporary tendency
    Core::Field<3> current_tendency_field("temp_tendency_wrapper", 
        { (int)total_current_tendency_view.extent(0), 
          (int)total_current_tendency_view.extent(1), 
          (int)total_current_tendency_view.extent(2) 
        });
    current_tendency_field.initialize_to_zero();
    
    // Iterate and calculate all tendency terms and add them to `current_tendency_field`
    for (const auto& term : tendency_terms_) {
        term->compute_tendency(state, grid, params, current_tendency_field);
    }
    Kokkos::deep_copy(total_current_tendency_view, current_tendency_field.get_device_data());
    
    // Prepare the flux and field to integrate
    auto field_new_view = field_to_update.get_mutable_device_data();
    auto field_old_view = field_prev_step.get_device_data();
    auto flux_now_view = Kokkos::subview(tendency_history.get_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto flux_prev_view = Kokkos::subview(tendency_history.get_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    // Use Forward Euler for the first step integration
    if (time_step_count_ == 0) {
        std::cout << "First time step: Forcing Forward Euler by setting flux_prev = flux_now." << std::endl;
        auto flux_prev_view_mutable = Kokkos::subview(tendency_history.get_mutable_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(flux_prev_view_mutable, flux_now_view);
    }

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();


    // new_x = old_x + dt * (1.5 * d_x_current - 0.5 * d_x_prev)
    Kokkos::parallel_for("AdamsBashforth2_step", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz, h + ny, h + nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            field_new_view(k, j, i) = field_old_view(k, j, i) 
                                    + dt * (1.5 * flux_now_view(k, j, i) - 0.5 * flux_prev_view(k, j, i));
        }
    );

    // Update time count
    time_step_count_++;
}

} // namespace Dynamics
} // namespace VVM
