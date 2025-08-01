#include "AdamsBashforth2.hpp"
#include <iostream>

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
    
    // 從傾向項中獲取變數名稱 (假設所有傾向項都作用於同一個變數)
    // 這樣可以避免將變數名稱寫死
    const std::string variable_name = "th"; 
    auto& field_to_update = state.get_field<3>(variable_name);
    auto& field_prev_step = state.get_field<3>(variable_name + "_m");
    auto& tendency_history = state.get_field<4>("d_" + variable_name);

    // 1. **(重要)** 複製目前的 th 到 th_m，為下一個時間步做準備
    auto field_current_view = field_to_update.get_device_data();
    auto field_prev_view_mutable = field_prev_step.get_mutable_device_data();
    Kokkos::deep_copy(field_prev_view_mutable, field_current_view);

    // 2. 取得指向 "當前" 總傾向的 3D subview，並將其清零
    auto total_current_tendency_view = Kokkos::subview(tendency_history.get_mutable_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    
    // 建立一個臨時的 Field 來儲存當前的總傾向
    Core::Field<3> current_tendency_field("temp_tendency_wrapper", 
        { (int)total_current_tendency_view.extent(0), 
          (int)total_current_tendency_view.extent(1), 
          (int)total_current_tendency_view.extent(2) 
        });
    current_tendency_field.initialize_to_zero(); // 確保傾向是從 0 開始累加

    // 3. 遍歷 tendency_terms_，將計算結果累加到 `current_tendency_field`
    for (const auto& term : tendency_terms_) {
        term->compute_tendency(state, grid, params, current_tendency_field);
    }
    
    // 將計算好的傾向複製到 history view
    Kokkos::deep_copy(total_current_tendency_view, current_tendency_field.get_device_data());

    auto field_new_view = field_to_update.get_mutable_device_data();
    auto field_old_view = field_prev_step.get_device_data();
    auto flux_now_view = Kokkos::subview(tendency_history.get_device_data(), now_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto flux_prev_view = Kokkos::subview(tendency_history.get_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    // 第一次時間步，使用 Forward Euler
    if (time_step_count_ == 0) {
        std::cout << "First time step: Forcing Forward Euler by setting flux_prev = flux_now." << std::endl;
        auto flux_prev_view_mutable = Kokkos::subview(tendency_history.get_mutable_device_data(), prev_idx, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(flux_prev_view_mutable, flux_now_view);
    }

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    const int h = grid.get_halo_cells();


    // 4. **(重要)** 根據 AB2 演算法更新狀態
    //    th_new = th_old + dt * (1.5 * d_th_current - 0.5 * d_th_prev)
    //    注意右邊是使用 field_old_view (th_m)
    Kokkos::parallel_for("AdamsBashforth2_step", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {h + nz, h + ny, h + nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            field_new_view(k, j, i) = field_old_view(k, j, i) 
                                    + dt * (1.5 * flux_now_view(k, j, i) - 0.5 * flux_prev_view(k, j, i));
        }
    );

    // 5. 更新時間步計數器
    time_step_count_++;
}

} // namespace Dynamics
} // namespace VVM
