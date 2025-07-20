// State class integrates various fields in the simulation.

#ifndef VVM_CORE_VVMSTATE_HPP
#define VVM_CORE_VVMSTATE_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "../utils/ConfigurationManager.hpp"
#include "Parameters.hpp"
#include <map>
#include <string>
#include <memory>

namespace VVM {
namespace Core {

class State {
public:
    // Constructor
    State(const Grid& grid, const Utils::ConfigurationManager& config, const ModelParameters& params);

    // Get a field by name
    Field& get_field(const std::string& name); // The data can be modified through this method
    const Field& get_field(const std::string& name) const; // The data cannot be modified through this method

    // Provide iterators to loop over all fields
    auto begin() { return fields_.begin(); } // First value
    auto end() { return fields_.end(); } // Last value
    auto begin() const { return fields_.cbegin(); } // First key
    auto end() const { return fields_.cend(); } // Last key

private:
    const Grid& grid_ref_;
    const Utils::ConfigurationManager& config_ref_;
    const ModelParameters model_parameters_;

    // A map to store all the fields by name
    std::map<std::string, Field> fields_;

    void initialize_fields();
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_VVMSTATE_HPP