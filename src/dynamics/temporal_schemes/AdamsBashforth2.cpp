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

    // 從 state 中取得預報變數的名稱，這裡我們假設所有 tendency term 都作用於同一個變數
    // 在一個更完整的實作中，這個變數名稱應該被傳遞或儲存
    const std::string variable_name = "th"; // 暫時寫死為 "th"

    // 1. 建立一個 3D Field 來累加 "當前" 的總傾向
    Core::Field<3> total_current_tendency("total_tendency", {
        grid.get_local_total_points_z(),
        grid.get_local_total_points_y(),
        grid.get_local_total_points_x()
    });
    total_current_tendency.initialize_to_zero();
    Core::State tendencies(state.config_ref_, state.parameters_);

    // 2. 遍歷 tendency_terms_，呼叫 compute_tendency，並將結果累加
    for (const auto& term : tendency_terms_) {
        term->compute_tendency(state, tendencies, grid, params);
        // 這裡需要一個 Kokkos parallel_for 來完成累加
        // Kokkos::parallel_for(...) { total_current_tendency += tendency; }
    }
    
    std::cout << "Finished tendency calculation." << std::endl;

    // 3. 使用環形緩衝區邏輯
    size_t now_idx = time_step_count_ % 2;
    size_t prev_idx = (time_step_count_ + 1) % 2;

    // 從 State 中取得 4D 的傾向歷史記錄 Field
    auto& tendency_history = state.get_field<4>("d_" + variable_name);
    
    // 取得預報變數和上一步的狀態
    auto& field_to_update = state.get_field<3>(variable_name);
    const auto& field_prev_step = state.get_field<3>(variable_name + "_m");

    // 4. 根據 AB2 演算法更新狀態
    // th_new = th_old + dt * (1.5 * d_th_current - 0.5 * d_th_prev)
    // (這裡需要一個 Kokkos kernel 來完成)
    std::cout << "Updating state using AB2 scheme..." << std::endl;
    // Kokkos::parallel_for(...) {
    //    auto prev_tendency = Kokkos::subview(tendency_history.get_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    //    ...
    // }

    // 5. 將 "當前" 的總傾向儲存到 4D 歷史記錄中，供下一步使用
    auto tendency_history_now_slice = Kokkos::subview(tendency_history.get_mutable_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(tendency_history_now_slice, total_current_tendency.get_device_data());
    
    // 6. 更新時間步計數器
    time_step_count_++;
}

} // namespace Dynamics
} // namespace VVM
