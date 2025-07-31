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
            } 
            else {
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
        variable_schemes_[var_name] = create_temporal_scheme(var_name, var_conf);
        
        // 2. 自動在 State 中宣告方案所需的額外變數
        int nz = grid_.get_local_total_points_z();
        int ny = grid_.get_local_total_points_y();
        int nx = grid_.get_local_total_points_x();

        // 宣告時間積分方案所需的影子變數 (例如 'th_m')
        auto required_suffixes = variable_schemes_[var_name]->get_required_state_suffixes();
        for (const auto& suffix : required_suffixes) {
            std::string shadow_field_name = var_name + suffix;
            state_.add_field<3>(shadow_field_name, {nz, ny, nx});
            std::cout << "DynamicalCore: Automatically declared state variable '" << shadow_field_name << "' for prognostic variable '" << var_name << "'." << std::endl;
        }

        // 針對 AdamsBashforth2，宣告 4D 的 tendency 變數
        if (var_conf.contains("temporal_scheme") && var_conf.at("temporal_scheme") == "AdamsBashforth2") {
            std::string tendency_field_name = "d_" + var_name; 
            state_.add_field<4>(tendency_field_name, {2, nz, ny, nx});
            std::cout << "DynamicalCore: Automatically declared 4D state variable '" << tendency_field_name << "' for AdamsBashforth2 scheme of '" << var_name << "'." << std::endl;
        }
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
