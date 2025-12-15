#ifndef VVM_UTILS_CONFIGURATION_MANAGER_HPP
#define VVM_UTILS_CONFIGURATION_MANAGER_HPP

#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <mpi.h>

#include "../../externals/json/json.hpp"

namespace VVM {
namespace Utils {

class ConfigurationManager {
public:
    // Constructor: Load the configuration file path
    explicit ConfigurationManager(const std::string& config_file_path);

    // Get configuration value of any type
    template<typename T>
    T get_value(const std::string& key_path) const;

    // Default value mode
    template<typename T>
    T get_value(const std::string& key_path, const T& default_value) const;

    // Check if a key exists
    bool has_key(const std::string& key_path) const;

    // Print all loaded configurations (for debugging)
    void print_config() const;

private:
    nlohmann::json m_config_data; // Store parsed JSON data

    // Helper function: Find JSON node by key_path (e.g. "grid.nx")
    const nlohmann::json* find_node(const std::string& key_path) const;
};

// ====================================================================
// Template method implementations
// ====================================================================

template<typename T>
T ConfigurationManager::get_value(const std::string& key_path) const {
    const nlohmann::json* node = find_node(key_path);
    if (!node) {
        throw std::runtime_error("Configuration error: Key '" + key_path + "' not found.");
    }
    try {
        return node->get<T>();
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Configuration error: Type mismatch for key '" + key_path + "'. " + e.what());
    }
}

template<typename T>
T ConfigurationManager::get_value(const std::string& key_path, const T& default_value) const {
    const nlohmann::json* node = find_node(key_path);
    
    if (!node) {
        return default_value;
    }

    try {
        return node->get<T>();
    } 
    catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Configuration error: Type mismatch for key '" + key_path + "'. " + e.what());
    }
}

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_CONFIGURATION_MANAGER_HPP
