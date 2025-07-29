#include "DynamicalCore.hpp"
#include "temporal_schemes/AdamsBashforth2.hpp"
#include "tendency_processes/AdvectionTerm.hpp"
#include "spatial_schemes/Takacs.hpp"
#include <stdexcept>

namespace VVM {
namespace Dynamics {

// --- 工廠函式 ---
// 根據設定檔建立一個完整的時間積分方案 (包含其下的傾向項和空間格式)
std::unique_ptr<TemporalScheme> create_temporal_scheme_for_variable(
    const std::string& var_name, 
    const nlohmann::json& config) {
    
    std::string scheme_name = config.at("time_scheme");

    if (scheme_name == "AdamsBashforth2") {
        std::vector<std::unique_ptr<TendencyTerm>> terms;

        // 根據設定建立 AdvectionTerm
        if (config.contains("advection")) {
            std::string advection_scheme_name = config.at("advection").at("spatial_scheme");
            std::unique_ptr<SpatialScheme> spatial_scheme;
            if (advection_scheme_name == "Takacs") {
                spatial_scheme = std::make_unique<Takacs>();
            } else {
                throw std::runtime_error("Unknown spatial scheme: " + advection_scheme_name);
            }
            terms.push_back(std::make_unique<AdvectionTerm>(std::move(spatial_scheme), var_name));
        }

        // ... 在這裡根據設定建立其他 TendencyTerm (stretching, tilting) ...

        return std::make_unique<AdamsBashforth2>(std::move(terms));
    }
    
    throw std::runtime_error("Unknown temporal scheme: " + scheme_name);
}


DynamicalCore::DynamicalCore(const Utils::ConfigurationManager& config, const Core::Grid& grid, const Core::Parameters& params)
    : grid_(grid), params_(params) {
    
    auto prognostic_config = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    
    for (auto& [var, conf] : prognostic_config.items()) {
        prognostic_variables_.push_back(var);
        variable_schemes_[var] = create_temporal_scheme_for_variable(var, conf);
    }
}

DynamicalCore::~DynamicalCore() = default;

void DynamicalCore::step(Core::State& state, double dt) {
    for (const auto& var_name : prognostic_variables_) {
        variable_schemes_.at(var_name)->step(state, grid_, params_, dt);
    }
}

} // namespace Dynamics
} // namespace VVM
