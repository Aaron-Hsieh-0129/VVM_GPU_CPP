#include "DynamicalCore.hpp"
#include "temporal_schemes/AdamsBashforth2.hpp"
#include "tendency_processes/AdvectionTerm.hpp"
#include "spatial_schemes/Takacs.hpp"
#include <stdexcept>
#include <iostream> // for debugging output

namespace VVM {
namespace Dynamics {

// 輔助函式：根據設定創建 TemporalScheme (及其TendencyTerms)
// 這個函式不應再自動宣告 State 變數
std::unique_ptr<TemporalScheme> DynamicalCore::create_temporal_scheme(
    const std::string& var_name, 
    const nlohmann::json& var_config) const {
    
    std::string scheme_name = var_config.at("temporal_scheme");
    std::vector<std::unique_ptr<TendencyTerm>> terms;

    if (scheme_name == "AdamsBashforth2") {
        // 根據設定建立 AdvectionTerm
        if (var_config.contains("tendency_terms") && var_config.at("tendency_terms").contains("advection")) {
            std::string advection_scheme_name = var_config.at("tendency_terms").at("advection").at("spatial_scheme");
            std::unique_ptr<SpatialScheme> spatial_scheme;
            if (advection_scheme_name == "Takacs") {
                spatial_scheme = std::make_unique<Takacs>();
            } else {
                throw std::runtime_error("Unknown spatial scheme: " + advection_scheme_name);
            }
            terms.push_back(std::make_unique<AdvectionTerm>(std::move(spatial_scheme), var_name));
        }
        // TODO: 在這裡根據設定建立其他 TendencyTerm (diffusion, stretching, tilting 等)

        return std::make_unique<AdamsBashforth2>(std::move(terms));
    }
    
    throw std::runtime_error("Unknown temporal scheme: " + scheme_name);
}


DynamicalCore::DynamicalCore(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid, 
                             const Core::Parameters& params,
                             Core::State& state)
    : state_(state), grid_(grid), params_(params) {
    
    auto prognostic_config = config.get_value<nlohmann::json>("dynamics.prognostic_variables");
    
    for (auto& [var_name, var_conf] : prognostic_config.items()) {
        prognostic_variables_.push_back(var_name);
        
        // 1. 創建時間積分方案的實例
        // 這裡直接呼叫成員函式 create_temporal_scheme
        variable_schemes_[var_name] = create_temporal_scheme(var_name, var_conf);
        
        // 2. 詢問此方案需要哪些額外的狀態變數後，自動在 State 中宣告這些變數
        // 這裡需要確保 prognostic_variables_ 是 3D 的。
        // 未來如果需要支援不同維度，可能需要從設定檔讀取維度資訊，或者在 TemporalScheme 介面中加入獲取場維度的方法。
        int nz = grid_.get_local_total_points_z();
        int ny = grid_.get_local_total_points_y();
        int nx = grid_.get_local_total_points_x();

        // 宣告預報變數本身 (例如 'th')
        // 由於 State 建構子已經預設宣告，這裡僅為自動宣告額外變數做示範
        // 實際應用中，確保 prognostic_variables_ 在 State 中被正確初始化
        // 例如：state_.add_field<3>(var_name, {nz, ny, nx}); // 如果 State 初始沒有宣告

        // 宣告時間積分方案所需的影子變數 (例如 'th_m')
        auto required_suffixes = variable_schemes_[var_name]->get_required_state_suffixes();
        for (const auto& suffix : required_suffixes) {
            std::string shadow_field_name = var_name + suffix;
            
            // 檢查變數是否已存在，避免重複宣告
            // 這裡需要 State 類別有一個 has_field 的方法來實現
            // if (!state_.has_field(shadow_field_name)) {
                // 假設所有這些額外變數也都是 3D 的
                state_.add_field<3>(shadow_field_name, {nz, ny, nx});
                std::cout << "DynamicalCore: Automatically declared state variable '" << shadow_field_name << "' for prognostic variable '" << var_name << "'." << std::endl;
            // }
        }

        // 針對 AdamsBashforth2，如果它需要一個 4D 的 tendency 變數，也在此宣告
        // 檢查 var_conf 是否有 "temporal_scheme" 且其值為 "AdamsBashforth2"
        if (var_conf.contains("temporal_scheme") && var_conf.at("temporal_scheme") == "AdamsBashforth2") {
            // Adams-Bashforth 2 階方案需要儲存前一個時間步的傾向，
            // 所以 4D 變數的第一維大小是 2
            std::string tendency_field_name_prev = "d_" + var_name + "_prev"; 
            // 這裡假設所有 Tendency 的結果是 3D，儲存歷史 Tendency 時增加一個時間步維度變成 4D
            state_.add_field<4>(tendency_field_name_prev, {2, nz, ny, nx});
            std::cout << "DynamicalCore: Automatically declared 4D state variable '" << tendency_field_name_prev << "' for AdamsBashforth2 scheme of '" << var_name << "'." << std::endl;
        }
    }
}

DynamicalCore::~DynamicalCore() = default;

void DynamicalCore::step(Core::State& state, double dt) {
    for (const auto& var_name : prognostic_variables_) {
        // 使用 this->state_ 確保操作的是正確的 State 物件
        variable_schemes_.at(var_name)->step(state, grid_, params_, dt);
    }
}

} // namespace Dynamics
} // namespace VVM
