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

    auto prognostic_config = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    
    for (auto& [var_name, conf] : prognostic_config.items()) {
        prognostic_variables_.push_back(var_name);
        
        // 1. 創建時間積分方案的實例
        variable_schemes_[var_name] = create_temporal_scheme_for_variable(var_name, conf);
        
        // 2. 詢問此方案需要哪些額外的狀態變數
        auto required_suffixes = variable_schemes_[var_name]->get_required_state_suffixes();
        
        // 3. 自動在 State 中宣告這些變數
        for (const auto& suffix : required_suffixes) {
            std::string shadow_field_name = var_name + suffix;
            
            // 假設所有預報變數都是 3D 的 (您可以根據需求擴充)
            int nz = grid.get_local_total_points_z();
            int ny = grid.get_local_total_points_y();
            int nx = grid.get_local_total_points_x();
            
            // 檢查變數是否已存在，避免重複宣告
            // (這需要為 State 類別增加一個 has_field 的方法)
            // if (!state.has_field(shadow_field_name)) {
                state.add_field<3>(shadow_field_name, {nz, ny, nx});
                std::cout << "DynamicalCore: Automatically declared state variable '" << shadow_field_name << "' for prognostic variable '" << var_name << "'." << std::endl;
            // }
        }

        // ... (自動宣告 4D tendency 變數的邏輯也可以整合進來)
    }

    if (scheme_name == "AdamsBashforth2") {
        // ...就自動宣告一個對應的 4D 變數
        std::string tendency_field_name = "d_" + var;

        // Adams-Bashforth 2 階方案需要儲存前一個時間步的傾向，
        // 所以 4D 變數的第一維大小是 2
        int nz = grid.get_local_total_points_z();
        int ny = grid.get_local_total_points_y();
        int nx = grid.get_local_total_points_x();

        state.add_field<4>(tendency_field_name, {2, nz, ny, nx});

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
