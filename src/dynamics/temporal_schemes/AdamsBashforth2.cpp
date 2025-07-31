#include "AdamsBashforth2.hpp"

namespace VVM {
namespace Dynamics {

AdamsBashforth2::AdamsBashforth2(std::vector<std::unique_ptr<TendencyTerm>> terms)
    : tendency_terms_(std::move(terms)) {
    // 初始化 previous_tendency_
}

AdamsBashforth2::~AdamsBashforth2() = default;

void AdamsBashforth2::step(
    Core::State& state,
    const Core::Grid& grid,
    const Core::Parameters& params,
    double dt) const {

    size_t now_idx = time_step_count_ % 2;
    size_t prev_idx = (time_step_count_ + 1) % 2;
    // 從 state 中取得預報變數的名稱，這裡我們假設所有 tendency term 都作用於同一個變數
    // 在一個更完整的實作中，這個變數名稱應該被傳遞或儲存
    const std::string variable_name = "th"; // 暫時寫死為 "th"
    auto& tendency_history = state.get_field<4>("d_" + variable_name);

    // 2. 取得指向 "當前" 總傾向的 3D subview，並將其清零
    auto total_current_tendency_view = Kokkos::subview(tendency_history.get_mutable_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Core::Field<3> current_tendency_field("temp_tendency_wrapper", 
        { (int)total_current_tendency_view.extent(0), 
          (int)total_current_tendency_view.extent(1), 
          (int)total_current_tendency_view.extent(2) 
        });

    // 3. 遍歷 tendency_terms_，直接將計算結果累加到 `current_tendency_field`
    for (const auto& term : tendency_terms_) {
        term->compute_tendency(state, grid, params, current_tendency_field);
    }
    
    std::cout << "Finished tendency calculation." << std::endl;

    // 取得預報變數和上一步的狀態
    auto& field_to_update = state.get_field<3>(variable_name);
    const auto& field_prev_step = state.get_field<3>(variable_name + "_m");
    
    // 4. 根據 AB2 演算法更新狀態
    // th_new = th_old + dt * (1.5 * d_th_current - 0.5 * d_th_prev)
    // (這裡需要一個 Kokkos kernel 來完成)
    std::cout << "Updating state using AB2 scheme..." << std::endl;

    auto field_new_view = field_to_update.get_mutable_device_data();
    auto field_old_view = field_prev_step.get_device_data();
    auto flux_now_view = Kokkos::subview(tendency_history.get_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto flux_prev_view = Kokkos::subview(tendency_history.get_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();


    if (time_step_count_ == 0) {
        std::cout << "First time step: Forcing Forward Euler by setting flux_prev = flux_now." << std::endl;
        // flux_prev = flux_now
        auto flux_prev_view = Kokkos::subview(tendency_history.get_mutable_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(flux_prev_view, flux_now_view);
    }

    Kokkos::parallel_for("AdamsBashforth2_step", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz, h + ny, h + nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            field_new_view(k, j, i) = field_new_view(k, j, i) 
                                    + 1.5 * dt * flux_now_view(k, j, i)
                                    - 0.5 * dt * flux_prev_view(k, j, i);
        }
    );

    // 5. 更新時間步計數器
    time_step_count_++;
}

} // namespace Dynamics
} // namespace VVM
