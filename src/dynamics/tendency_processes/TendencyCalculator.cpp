#include "TendencyCalculator.hpp"
#include "core/Field.hpp"

namespace VVM {
namespace Dynamics {

TendencyCalculator::TendencyCalculator(std::string var_name,
                                       std::vector<std::unique_ptr<TendencyTerm>> ab2_terms,
                                       std::vector<std::unique_ptr<TendencyTerm>> fe_terms)
    : variable_name_(std::move(var_name)),
      ab2_tendency_terms_(std::move(ab2_terms)),
      fe_tendency_terms_(std::move(fe_terms)) {}

void TendencyCalculator::calculate_tendencies(Core::State& state, const Core::Grid& grid, const Core::Parameters& params, size_t time_step_count) const {
    if (ab2_tendency_terms_.empty() && fe_tendency_terms_.empty()) {
        return;
    }

    const int& nz = grid.get_local_total_points_z();
    const int& ny = grid.get_local_total_points_y();
    const int& nx = grid.get_local_total_points_x();
    const int h = grid.get_halo_cells();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();

    auto& field_to_update = state.get_field<3>(variable_name_);
    auto& field_current_view = field_to_update.get_mutable_device_data();

    // if (variable_name_ == "xi" || variable_name_ == "eta") {
    //     Kokkos::parallel_for("divide_by_density",
    //         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
    //         KOKKOS_LAMBDA(const int k, const int j, const int i) {
    //             field_current_view(k, j, i) /= rhobar_up(k);
    //         }
    //     );
    // }
    // else if (variable_name_ == "zeta") {
    //     Kokkos::parallel_for("divide_by_density",
    //         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz, ny, nx}),
    //         KOKKOS_LAMBDA(const int k, const int j, const int i) {
    //             field_current_view(k, j, i) /= rhobar(k);
    //         }
    //     );
    // }

    // Calculate AB2 tendencies
    if (!ab2_tendency_terms_.empty()) {
        size_t now_idx = time_step_count % 2;
        auto& tendency_history = state.get_field<4>("d_" + variable_name_);
        auto total_current_tendency_view = Kokkos::subview(tendency_history.get_mutable_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        
        Core::Field<3> current_tendency_field("temp_ab2_tendency", {nz, ny, nx});
        current_tendency_field.initialize_to_zero();
        for (const auto& term : ab2_tendency_terms_) {
            term->compute_tendency(state, grid, params, current_tendency_field);
        }
        Kokkos::deep_copy(total_current_tendency_view, current_tendency_field.get_device_data());
    }

    // Calculate Forward Euler tendencies
    if (!fe_tendency_terms_.empty()) {
        auto& fe_tendency_field = state.get_field<3>("fe_tendency_" + variable_name_);
        fe_tendency_field.initialize_to_zero();
        for (const auto& term : fe_tendency_terms_) {
            term->compute_tendency(state, grid, params, fe_tendency_field);
        }
    }

     // Divide rho for xi, eta, zeta
    // if (variable_name_ == "xi" || variable_name_ == "eta") {
    //     Kokkos::parallel_for("divide_by_density",
    //         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz-h, ny, nx}),
    //         KOKKOS_LAMBDA(const int k, const int j, const int i) {
    //             field_current_view(k, j, i) *= rhobar_up(k);
    //         }
    //     );
    // }
    // else if (variable_name_ == "zeta") {
    //     Kokkos::parallel_for("divide_by_density",
    //         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h-1, 0, 0}, {nz, ny, nx}),
    //         KOKKOS_LAMBDA(const int k, const int j, const int i) {
    //             field_current_view(k, j, i) *= rhobar(k);
    //         }
    //     );
    // }

}

} // namespace Dynamics
} // namespace VVM
