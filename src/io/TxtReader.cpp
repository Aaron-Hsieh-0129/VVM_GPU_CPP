#include "TxtReader.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace VVM {
namespace IO {

TxtReader::TxtReader(const std::string& filepath, const VVM::Core::Grid& grid) : grid_(grid), source_file_(filepath) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    std::ifstream infile(source_file_);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open initial conditions file: " + source_file_);
    }

    std::string line;
    // Skip comment lines and separator lines
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#' || line.find("===") != std::string::npos) {
            continue;
        }
        break; // First non-comment/separator line is the header
    }

    // Parse header to find column indices
    std::stringstream ss_header(line);
    std::string header_token;
    std::vector<std::string> column_names;
    std::map<std::string, int> header_indices; // Store original column name to index mapping
    int col_idx = 0;
    while (ss_header >> header_token) {
        column_names.push_back(header_token);
        header_indices[header_token] = col_idx++;
        // Initialize an empty vector for each profile in all_profiles_
        all_profiles_[header_token] = std::vector<std::pair<double, double>>();
    }

    if (header_indices.find("ZT") == header_indices.end()) {
        throw std::runtime_error("Missing 'ZT' column in profile file header.");
    }

    // Read data rows
    int zt_col_idx = header_indices["ZT"];
    while (std::getline(infile, line)) {
        if (line.empty()) continue; // Skip empty lines

        std::stringstream ss_data(line);
        std::vector<double> row_data;
        double val;
        while (ss_data >> val) {
            row_data.push_back(val);
        }

        if (row_data.size() <= static_cast<size_t>(zt_col_idx)) {
            if (rank == 0) std::cerr << "Warning: Skipping malformed row (missing ZT) in profile file: " << line << std::endl;
            continue;
        }

        double zt_val = row_data[zt_col_idx];

        // Populate all_profiles_ for each column found in the header
        for (const auto& col_name : column_names) {
            int current_col_idx = header_indices[col_name];
            if (row_data.size() > static_cast<size_t>(current_col_idx)) {
                all_profiles_[col_name].emplace_back(zt_val, row_data[current_col_idx]);
            } 
            else {
                if (rank == 0) std::cerr << "Warning: Missing data for column '" << col_name << "' in row: " << line << std::endl;
            }
        }
    }

    if (rank == 0) {
        if (all_profiles_.empty() || all_profiles_.at("ZT").empty()) {
            std::cerr << "Warning: No profile data loaded from " << source_file_ << std::endl;
        } 
        else {
            std::cout << "Profiles loaded from: " << source_file_ << std::endl;
            std::cout << "--- Loaded Profile Data Summary ---" << std::endl;
            for (const auto& pair : all_profiles_) {
                std::cout << "- " << pair.first << ": " << pair.second.size() << " data points" << std::endl;
            }
            std::cout << "-----------------------------------" << std::endl;
        }
    }
}

double TxtReader::linear_interpolate(const std::string& field_name, double target_z) const {
    // Ensure the ZT profile exists and has data before proceeding
    if (all_profiles_.find("ZT") == all_profiles_.end() || all_profiles_.at("ZT").empty()) {
        throw std::runtime_error("ZT profile data is not available for interpolation.");
    }

    const auto& profile_data = all_profiles_.at(field_name);
    
    static std::map<std::string, bool> warned_below;
    static std::map<std::string, bool> warned_above;

    if (profile_data.empty()) {
        throw std::runtime_error("Profile data for field '" + field_name + "' is empty for interpolation.");
    }

    if (target_z < profile_data.front().first) {
        if (!warned_below[field_name]) {
            std::cerr << "Warning for field '" << field_name << "': Target Z (" << target_z << "m) is below profile range (" << profile_data.front().first << "m). Extrapolating..." << std::endl;
            warned_below[field_name] = true;
        }
        return profile_data.front().second;
    }
    if (target_z > profile_data.back().first) {
        if (!warned_above[field_name]) {
             std::cerr << "Warning for field '" << field_name << "': Target Z (" << target_z << "m) is above profile range (" << profile_data.back().first << "m). Extrapolating..." << std::endl;
            warned_above[field_name] = true;
        }
        return profile_data.back().second;
    }

    for (size_t i = 0; i < profile_data.size() - 1; ++i) {
        const auto& p1 = profile_data[i];
        const auto& p2 = profile_data[i + 1];
        if (target_z >= p1.first && target_z <= p2.first) {
            const double factor = (p2.first == p1.first) ? 0.0 : (target_z - p1.first) / (p2.first - p1.first);
            return p1.second + factor * (p2.second - p1.second);
        }
    }
    return profile_data.back().second; // Should not be reached if range checks pass
}

void TxtReader::read_and_initialize(VVM::Core::State& state) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (all_profiles_.empty() || all_profiles_.at("ZT").empty()) {
        std::cout << "No profile data loaded. Skipping file-based initialization." << std::endl;
        return;
    }
    
    // Iterate through all loaded profiles (e.g., "ZT", "RHO", "THBAR", etc.)
    for (const auto& profile_pair : all_profiles_) {
        const std::string& profile_name = profile_pair.first;
        // Skip 'ZT' profile itself, as it's used for coordinates, not a field to initialize directly
        if (profile_name == "ZT") {
            continue;
        }

        // Try to get the 1D field from the State using the profile name
        try {
            auto& field_1d = state.get_field<1>(profile_name);
            auto host_mirror = Kokkos::create_mirror_view(field_1d.get_mutable_device_data());
            
            const int nz_phys = grid_.get_local_physical_points_z();
            const int h = grid_.get_halo_cells(); 

            // Initialize only physical points of the 1D field
            for (int k_phys = 0; k_phys < nz_phys; ++k_phys) {
                double target_z = (grid_.get_local_physical_start_z() + k_phys + 0.5) * grid_.get_dz();
                host_mirror(k_phys + h) = linear_interpolate(profile_name, target_z);
            }

            Kokkos::deep_copy(field_1d.get_mutable_device_data(), host_mirror);
            if (rank == 0) std::cout << "Successfully initialized field: " << profile_name << std::endl;
        } 
        catch (const std::runtime_error& e) {
            // Catch if the field is not a 1D field or doesn't exist in the State
            if (rank == 0) std::cerr << "Warning: Could not initialize 1D field for profile '" << profile_name << "'. Reason: " << e.what() << std::endl;
        }
    }
    Kokkos::fence();
}

} // namespace IO
} // namespace VVM
