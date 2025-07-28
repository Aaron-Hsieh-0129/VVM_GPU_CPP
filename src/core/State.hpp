// State class integrates various fields in the simulation.

#ifndef VVM_CORE_VVMSTATE_HPP
#define VVM_CORE_VVMSTATE_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "utils/ConfigurationManager.hpp"
#include "Parameters.hpp"
#include <map>
#include <string>
#include <memory>
#include <variant>

namespace VVM {
namespace Core {

// A variant that can hold a field of any supported dimension
using AnyField = std::variant<
    std::monostate, // default state
    Field<1>,
    Field<2>,
    Field<3>,
    Field<4>
>;

class State {
public:
    // Constructor
    State(const Utils::ConfigurationManager& config, const Parameters& params);

    template<size_t Dim>
    void add_field(const std::string& name, std::initializer_list<int> dims_list) {
        if (dims_list.size() != Dim) {
            throw std::runtime_error("Dimension mismatch for field '" + name + "'");
        }
        std::array<int, Dim> dims;
        std::copy(dims_list.begin(), dims_list.end(), dims.begin());
        fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims);
    }

    // Get a field by name
    template<size_t Dim>
    Field<Dim>& get_field(const std::string& name) {
        try { 
            return std::get<Field<Dim>>(fields_.at(name));
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Field '" + name + "' not found in State.");
        }
        catch (const std::bad_variant_access& e) {
            throw std::runtime_error("Field '" + name + "' has incorrect dimension.");
        }
    }
    
    template<size_t Dim>
    const Field<Dim>& get_field(const std::string& name) const {
        try { 
            return std::get<Field<Dim>>(fields_.at(name));
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Field '" + name + "' not found in State.");
        }
        catch (const std::bad_variant_access& e) {
            throw std::runtime_error("Field '" + name + "' has incorrect dimension.");
        }
    }

    // Provide iterators to loop over all fields
    auto begin() { return fields_.begin(); } // First value
    auto end() { return fields_.end(); } // Last value
    auto begin() const { return fields_.cbegin(); } // First key
    auto end() const { return fields_.cend(); } // Last key

private:
    const Utils::ConfigurationManager& config_ref_;
    const Parameters& parameters_;
    std::map<std::string, AnyField> fields_;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_VVMSTATE_HPP
