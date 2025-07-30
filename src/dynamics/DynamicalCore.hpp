#ifndef VVM_DYNAMICS_DYNAMICAL_CORE_HPP
#define VVM_DYNAMICS_DYNAMICAL_CORE_HPP

#include <map>
#include <string>
#include <memory>
#include <vector>
#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include "temporal_schemes/TemporalScheme.hpp"

namespace VVM {
namespace Dynamics {

class DynamicalCore {
public:
    // 根據設定檔建立所有需要的物件
    DynamicalCore(const Utils::ConfigurationManager& config, 
                  const Core::Grid& grid, 
                  const Core::Parameters& params,
                  Core::State& state); // 接受 State 參考以進行修改
    ~DynamicalCore();

    // 執行一個完整的時間步
    void step(Core::State& state, double dt);

private:
    Core::State& state_; // 儲存對 State 物件的參考
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    
    // 每個預報變數都對應一個專屬的時間積分器
    std::map<std::string, std::unique_ptr<TemporalScheme>> variable_schemes_;
    std::vector<std::string> prognostic_variables_;

    // 輔助函式，用於根據設定創建 TemporalScheme 和 TendencyTerm
    // 這個函式不再負責宣告 State 變數
    std::unique_ptr<TemporalScheme> create_temporal_scheme(
        const std::string& var_name, 
        const nlohmann::json& var_config) const;
};

} // namespace Dynamics
} // namespace VVM
#endif
