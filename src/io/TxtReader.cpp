#include "TxtReader.hpp"
#include "core/BoundaryConditionManager.hpp"
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

TxtReader::TxtReader(const std::string& filepath, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, const VVM::Utils::ConfigurationManager& config) 
    : grid_(grid), params_(params), source_file_(filepath), config_(config) {
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

    if (header_indices.find("pbar") == header_indices.end()) {
        throw std::runtime_error("Missing 'pbar' column in profile file header.");
    }

    // Read data rows
    int pbar_col_idx = header_indices["pbar"];
    while (std::getline(infile, line)) {
        if (line.empty()) continue; // Skip empty lines

        std::stringstream ss_data(line);
        std::vector<double> row_data;
        double val;
        while (ss_data >> val) {
            row_data.push_back(val);
        }

        if (row_data.size() <= static_cast<size_t>(pbar_col_idx)) {
            if (rank == 0) std::cerr << "Warning: Skipping malformed row (missing pbar) in profile file: " << line << std::endl;
            continue;
        }

        double pbar_val = row_data[pbar_col_idx];

        // pair.first --> name, pair.second --> [[pbar], [data]]
        // Populate all_profiles_ for each column found in the header
        for (const auto& col_name : column_names) {
            int current_col_idx = header_indices[col_name];
            if (row_data.size() > static_cast<size_t>(current_col_idx)) {
                all_profiles_[col_name].emplace_back(pbar_val, row_data[current_col_idx]);
            } 
            else {
                if (rank == 0) std::cerr << "Warning: Missing data for column '" << col_name << "' in row: " << line << std::endl;
            }
        }
    }

    if (rank == 0) {
        if (all_profiles_.empty() || all_profiles_.at("pbar").empty()) {
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
    int rank = grid_.get_mpi_rank();

    const auto& profile_data = all_profiles_.at(field_name);
    
    if (profile_data.empty()) {
        throw std::runtime_error("Profile data for field '" + field_name + "' is empty for interpolation.");
    }

    static std::map<std::string, bool> warned_below;
    static std::map<std::string, bool> warned_above;

    // Handle extrapolation
    if (target_z < profile_data.front().first) {
        if (!warned_below[field_name]) {
            if (rank == 0) std::cerr << "Warning for field '" << field_name << "': Target Z (" << target_z << "m) is below profile range (" << profile_data.front().first << "m). Extrapolating..." << std::endl;
            warned_below[field_name] = true;
        }
        return profile_data.front().second;
    }
    if (target_z > profile_data.back().first) {
        if (!warned_above[field_name]) {
             if (rank == 0) std::cerr << "Warning for field '" << field_name << "': Target Z (" << target_z << "m) is above profile range (" << profile_data.back().first << "m). Extrapolating..." << std::endl;
            warned_above[field_name] = true;
        }
        return profile_data.back().second;
    }

    auto it = std::lower_bound(profile_data.begin(), profile_data.end(), target_z,
        [](const std::pair<double, double>& elem, double val) {
            return elem.first < val;
        });
    const auto& p2 = *it;
    const auto& p1 = *(it - 1);

    if (target_z == p2.first) {
        return p2.second;
    }

    if (target_z == p1.first) {
        return p1.second;
    }

    if (p2.first == p1.first) {
        return p1.second;
    }
    const double factor = (target_z - p1.first) / (p2.first - p1.first);
    return p1.second + factor * (p2.second - p1.second);
}

void TxtReader::read_and_initialize(VVM::Core::State& state) {
    int rank = grid_.get_mpi_rank();

    if (all_profiles_.empty() || all_profiles_.at("pbar").empty()) {
        std::cout << "No profile data loaded. Skipping file-based initialization." << std::endl;
        return;
    }

    calculate_input_z();
    auto z_mid_host = params_.z_mid.get_host_data();
    
    const auto& nz = grid_.get_local_total_points_z();
    const auto& h = grid_.get_halo_cells(); 
    // Interpolate the data (qv, T) from calculated z_input and calculate 
    for (const auto& profile_pair : all_profiles_) {
        const std::string& profile_name = profile_pair.first;
        try {
            auto& field_1d = state.get_field<1>(profile_name);
            auto host_mirror = field_1d.get_host_data();

            // Initialize only physical points of the 1D field
            for (int k = h-1; k < nz; k++) {
                double target_z = z_mid_host(k);
                host_mirror(k) = linear_interpolate(profile_name, target_z);
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

    auto& Tvbar = state.get_field<1>("Tvbar").get_mutable_device_data();
    auto Tvbar_h = state.get_field<1>("Tvbar").get_host_data();
    auto& qvbar = state.get_field<1>("qvbar").get_mutable_device_data();
    auto qvbar_h = state.get_field<1>("qvbar").get_host_data();
    auto& pbar = state.get_field<1>("pbar").get_mutable_device_data();
    auto pbar_h = state.get_field<1>("pbar").get_host_data();
    auto& pibar = state.get_field<1>("pibar").get_mutable_device_data();
    auto pibar_h = state.get_field<1>("pibar").get_host_data();
    const auto& Tbar = state.get_field<1>("Tbar").get_mutable_device_data();
    auto Tbar_h = state.get_field<1>("Tbar").get_host_data();
    auto& thbar = state.get_field<1>("thbar").get_mutable_device_data();
    auto thbar_h = state.get_field<1>("thbar").get_host_data();

    const auto& PSFC = config_.get_value<double>("constants.PSFC");
    const auto& P0 = config_.get_value<double>("constants.P0");
    const auto& Rd = config_.get_value<double>("constants.Rd");
    const auto& Cp = config_.get_value<double>("constants.Cp");
    const auto& gravity = config_.get_value<double>("constants.gravity");
    const auto& dz = config_.get_value<double>("grid.dz");

    const auto& z_up = params_.z_up.get_device_data();
    const auto z_up_h = params_.z_up.get_host_data();
    const auto& z_mid = params_.z_mid.get_device_data();
    const auto z_mid_h = params_.z_mid.get_host_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto flex_height_coef_up_h = params_.flex_height_coef_up.get_host_data();

    for (int k = 0; k < nz; k++) {
        Tvbar_h(k) = Tbar_h(k) * (1+0.608*qvbar_h(k));
    }
    Kokkos::deep_copy(Tvbar, Tvbar_h);

    VVM::Core::Field<1> pibarlog_field("pibarlog", {nz});
    auto& pibarlog = pibarlog_field.get_mutable_device_data();
    auto pibarlog_h = pibarlog_field.get_host_data();

    for (int k = 1; k < nz; k++) {
        if (k == 1) pibarlog_h(k) = std::log(std::pow(PSFC/P0, Rd/Cp));
        else if (k == 2) pibarlog_h(k) = pibarlog_h(k-1)-2.*gravity*dz/Cp / (Tvbar_h(k-1)+Tvbar_h(k)) * (z_mid_h(k)-z_up_h(k-1)) / dz;
        else pibarlog_h(k) = pibarlog_h(k-1) - 2.*gravity*dz/Cp / (Tvbar_h(k-1)+Tvbar_h(k)) / flex_height_coef_up_h(k-1); 
    }
    Kokkos::deep_copy(pibarlog, pibarlog_h);

    for (int k = 1; k < nz; k++) {
        pibar_h(k) = std::exp(pibarlog_h(k));
        pbar_h(k) = P0 * std::pow(pibar_h(k), Cp/Rd);
    }
    
    Kokkos::deep_copy(Tvbar, Tvbar_h);
    Kokkos::deep_copy(pibar, pibar_h);
    Kokkos::deep_copy(pbar, pbar_h);

    for (int k = 1; k < nz; k++) {
        thbar_h(k) = Tbar_h(k) / pibar_h(k);
    }
    Kokkos::deep_copy(thbar, thbar_h);

    // Calculate rhobar, rhobar_up
    std::vector<double> ALPHA, ALPHAW;
    auto& rhobar = state.get_field<1>("rhobar").get_mutable_device_data();
    auto rhobar_h = state.get_field<1>("rhobar").get_host_data();
    auto& rhobar_up = state.get_field<1>("rhobar_up").get_mutable_device_data();
    auto rhobar_up_h = state.get_field<1>("rhobar_up").get_host_data();
    for (int k = 0; k < nz; k++) {
        ALPHA.push_back(Rd * Tvbar_h(k) / pbar_h(k));
        rhobar_h(k) = 1. / ALPHA[k];
    }
    for (int k = 0; k < nz-1; k++) {
        ALPHAW.push_back((ALPHA[k] + ALPHA[k+1]) * 0.5);
        rhobar_up_h(k) = 1. / ALPHAW[k];
    }
    rhobar_up_h(h-1) = PSFC / (Rd * Tvbar_h(h-1));

    Kokkos::deep_copy(rhobar, rhobar_h);
    Kokkos::deep_copy(rhobar_up, rhobar_up_h);


    auto U_h = state.get_field<1>("U").get_host_data();
    auto& U = state.get_field<1>("U").get_mutable_device_data();
    auto V_h = state.get_field<1>("V").get_host_data();
    auto& V = state.get_field<1>("V").get_mutable_device_data();
    U_h(h-1) = U_h(h);
    U_h(nz-h) = U_h(nz-h-1);
    V_h(h-1) = V_h(h);
    V_h(nz-h) = V_h(nz-h-1);
    Kokkos::deep_copy(U, U_h);
    Kokkos::deep_copy(V, V_h);
    return;
}

void TxtReader::calculate_input_z() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int nz = grid_.get_local_total_points_z();
    const int h = grid_.get_halo_cells(); 
    const auto& pbar = all_profiles_.at("pbar");
    const auto& Tbar = all_profiles_.at("Tbar");
    const auto& qvbar = all_profiles_.at("qvbar");

    std::vector<double> pibar_input(nz, 0), tv_input(nz, 0);
    std::vector<double> zt_input(nz, 0);

    const auto& P0 = config_.get_value<double>("constants.P0");
    const auto& Rd = config_.get_value<double>("constants.Rd");
    const auto& Cp = config_.get_value<double>("constants.Cp");
    const auto& gravity = config_.get_value<double>("constants.gravity");
    auto z_mid_host = params_.z_mid.get_host_data();
    auto z_up_host = params_.z_up.get_host_data();
    
    for (int k = h-1; k < nz; k++) {
        pibar_input[k] = std::pow(pbar[k-1].second/P0, Rd/Cp);
        tv_input[k] = Tbar[k-1].second * (1+0.608*qvbar[k-1].second);

        if (k < h) {
            zt_input[k] = z_mid_host(k);
        }
        else {
            zt_input[k] = zt_input[k-1]-Cp/(2.*gravity)*((std::log(pibar_input[k])-std::log(pibar_input[k-1]))*(tv_input[k]+tv_input[k-1]));
        }
    }

    // Modify the pressure to height in order to do interpolation
    for (auto& profile_pair : all_profiles_) {
        for (int k = 0; k < nz-1; k++) {
            profile_pair.second[k].first = zt_input[k+1];
        }

    }

}

} // namespace IO
} // namespace VVM
