#include "TxtReader.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace VVM {
namespace IO {

TxtReader::TxtReader(const std::string& filepath, const VVM::Core::Grid& grid) : grid_(grid), source_file_(filepath) {

}

double TxtReader::linear_interpolate(const std::string& field_name, double target_z) const {
    const auto& profile_data = all_profiles_.at(field_name);
    
    static std::map<std::string, bool> warned_below;
    static std::map<std::string, bool> warned_above;

    if (target_z < profile_data.front().first) {
        if (!warned_below[field_name]) {
            std::cerr << "Warning for field '" << field_name << "': Target Z (" << target_z << "m) is below profile range. Extrapolating..." << std::endl;
            warned_below[field_name] = true;
        }
        return profile_data.front().second;
    }
    if (target_z > profile_data.back().first) {
        if (!warned_above[field_name]) {
             std::cerr << "Warning for field '" << field_name << "': Target Z (" << target_z << "m) is above profile range. Extrapolating..." << std::endl;
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
    return profile_data.back().second;
}

void TxtReader::read_and_initialize(VVM::Core::State& state) {
    std::ifstream infile(source_file_);
    if (!infile.is_open()) {
        std::cout << "No reader configured. Skipping file-based initialization." << std::endl;
        return;
    }
    
    for (const auto& profile_pair : all_profiles_) {
        const std::string& header_name = profile_pair.first;
        const std::string profile_field_name = header_name + "_profile";

        std::cout << header_name << std::endl;

        try {
            auto& field_1d = state.get_field<1>(profile_field_name);
            auto host_mirror = Kokkos::create_mirror_view(field_1d.get_mutable_device_data());
            
            const int nz_phys = grid_.get_local_physical_points_z();
            if (host_mirror.extent(0) != nz_phys) {
                throw std::runtime_error("1D profile field size mismatch.");
            }

            // 遍歷模式的每一個垂直物理網格點
            for (int k = 0; k < nz_phys; ++k) {
                double target_z = (grid_.get_local_physical_start_z() + k + 0.5) * grid_.get_dz();
                host_mirror(k) = linear_interpolate(header_name, target_z);
            }

            Kokkos::deep_copy(field_1d.get_mutable_device_data(), host_mirror);
        } 
        catch (const std::runtime_error&) {
            // 如果 State 中沒有對應的 _profile 欄位，就忽略
            continue;
        }
    }
    Kokkos::fence();
}

} // namespace IO
} // namespace VVM
