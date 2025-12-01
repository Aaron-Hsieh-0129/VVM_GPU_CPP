#include "ConfigurationManager.hpp"

namespace VVM {
namespace Utils {

ConfigurationManager::ConfigurationManager(const std::string &config_file_path) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + config_file_path);
    }
    try {
        file >> m_config_data;
        if (rank == 0) std::cout << "Configuration loaded from: " << config_file_path << std::endl;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse configuration file '" + config_file_path + "': " + e.what());
    }
}

// Helper function: Find JSON node by key_path (e.g. "grid.nx")
const nlohmann::json* ConfigurationManager::find_node(const std::string &key_path) const {
    const nlohmann::json* current_node = &m_config_data;
    size_t start = 0;
    size_t end = key_path.find('.');

    while (end != std::string::npos) {
        std::string key = key_path.substr(start, end - start);
        if (!current_node->is_object() || !current_node->contains(key)) {
            return nullptr; // A key in the path is not an object or does not exist
        }
        current_node = &((*current_node)[key]);
        start = end + 1;
        end = key_path.find('.', start);
    }

    std::string last_key = key_path.substr(start);
    if (!current_node->is_object() || !current_node->contains(last_key)) {
        return nullptr; // The last key is not an object or does not exist
    }
    return &((*current_node)[last_key]);
}

bool ConfigurationManager::has_key(const std::string& key_path) const {
    return find_node(key_path) != nullptr;
}

void ConfigurationManager::print_config() const {
    std::cout << "--- Loaded Configuration ---" << std::endl;
    std::cout << m_config_data.dump(4) << std::endl;
    std::cout << "----------------------------" << std::endl;
}

} // namespace Utils
} // namespace VVM
