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

    // 這是 AB2 的演算法邏輯：
    // 1. 建立一個臨時的 State 物件來儲存當前的總傾向
    // 2. 遍歷 tendency_terms_，呼叫 compute_tendency，將結果累加
    // 3. 使用當前傾向和儲存的前一步傾向，更新 state
    // 4. 將當前傾向儲存起來，供下一個時間步使用
}

} // namespace Dynamics
} // namespace VVM
