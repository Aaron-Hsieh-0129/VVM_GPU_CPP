#include "State.hpp"
#include <utility>

namespace VVM {
namespace Core {

State::State(const Grid& grid, const Utils::ConfigurationManager& config, const ModelParameters& params)
    : grid_ref_(grid), config_ref_(config), model_parameters_(params) {
    initialize_fields();
}

void State::initialize_fields() {
    fields_.emplace(std::piecewise_construct,
                    std::forward_as_tuple("th"),
                    std::forward_as_tuple(grid_ref_, "Potential Temperature"));
                    
    fields_.emplace(std::piecewise_construct,
                    std::forward_as_tuple("eta"),
                    std::forward_as_tuple(grid_ref_, "Vorticity in y-direction"));



}

Field& State::get_field(const std::string& name) {
    try {
        return fields_.at(name);
    } 
    catch (const std::out_of_range& e) {
        throw std::runtime_error("Field '" + name + "' not found in State.");
    }
}

const Field& State::get_field(const std::string& name) const {
    try {
        return fields_.at(name);
    } 
    catch (const std::out_of_range& e) {
        throw std::runtime_error("Field '" + name + "' not found in State.");
    }
}

} // namespace Core
} // namespace VVM