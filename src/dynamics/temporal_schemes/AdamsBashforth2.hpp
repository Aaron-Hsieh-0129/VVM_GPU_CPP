#ifndef VVM_DYNAMICS_ADAMS_BASHFORTH_2_HPP
#define VVM_DYNAMICS_ADAMS_BASHFORTH_2_HPP

#include "TemporalScheme.hpp"

namespace VVM {
namespace Dynamics {

class AdamsBashforth2 : public TemporalScheme {
public:
    // 建構時接收所有它需要負責的傾向項
    explicit AdamsBashforth2(std::vector<std::unique_ptr<TendencyTerm>> terms);
    ~AdamsBashforth2() override;

    void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        double dt
    ) const override;

    std::vector<std::string> get_required_state_suffixes() const override {
        return {"_m"}; // 這裡可以擴充，例如 {"_m", "_p"} 代表需要 th_m 和 th_p
    }


private:
    std::vector<std::unique_ptr<TendencyTerm>> tendency_terms_;
    // AB2 需要儲存上一個時間步的傾向，這可以用 mutable 或其他方式管理
    // mutable Core::Field<3> previous_tendency_; 
};

} // namespace Dynamics
} // namespace VVM
#endif
