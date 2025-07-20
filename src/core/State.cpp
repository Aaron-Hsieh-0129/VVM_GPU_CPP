#include "State.hpp"

namespace VVM {
namespace Core {

State::State(const Utils::ConfigurationManager& config, const ModelParameters& params)
    : config_ref_(config), model_parameters_(params) {
    // Initialization of fields is now typically done from the main application logic
    // based on the configuration.
}

// get_field is now a template in the header file.

} // namespace Core
} // namespace VVM