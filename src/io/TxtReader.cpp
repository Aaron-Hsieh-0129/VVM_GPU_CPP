#include "TxtReader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <mpi.h>

namespace VVM {
namespace IO {

TxtReader::TxtReader(const std::string& filepath, const VVM::Core::Grid& grid, 
                     const VVM::Core::Parameters& params, const VVM::Utils::ConfigurationManager& config)
    : grid_(grid), params_(params), config_(config), source_file_(filepath) {
    read_file();
}

void TxtReader::read_and_initialize(VVM::Core::State& state) {
    int rank = grid_.get_mpi_rank();
    
    if (raw_data_.empty()) {
        if (rank == 0) std::cerr << "[TxtReader] Warning: No data loaded. Skipping initialization." << std::endl;
        return;
    }

    if (rank == 0) std::cout << "[TxtReader] Starting Profile Initialization..." << std::endl;
    calculate_input_heights();
    initialize_thermodynamics(state);
    initialize_forcing(state);

    if (rank == 0) std::cout << "[TxtReader] Initialization Complete." << std::endl;
}

void TxtReader::read_file() {
    std::ifstream infile(source_file_);
    if (!infile.is_open()) throw std::runtime_error("Failed to open file: " + source_file_);

    std::string line;
    // Skip comments
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#' || line.find("===") != std::string::npos) continue;
        break; 
    }

    // Parse Header
    std::stringstream ss_header(line);
    std::string col_name;
    std::vector<std::string> headers;
    while (ss_header >> col_name) {
        headers.push_back(col_name);
        raw_data_[col_name] = std::vector<double>();
    }

    // Parse Data
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss_data(line);
        double val;
        for (const auto& h : headers) {
            if (ss_data >> val) raw_data_[h].push_back(val);
        }
    }
    
    // Check essential columns
    if (raw_data_.find("pbar") == raw_data_.end()) 
        throw std::runtime_error("Missing required column 'pbar' in input file.");
}

void TxtReader::calculate_input_heights() {
    const auto& P_in = raw_data_.at("pbar");
    const auto& T_in = raw_data_.at("Tbar"); 
    const auto& Qv_in = raw_data_.at("qvbar"); 
    
    size_t n = P_in.size();
    input_z_.resize(n);

    double Rd = config_.get_value<double>("constants.Rd");
    double Cp = config_.get_value<double>("constants.Cp");
    double g = config_.get_value<double>("constants.gravity");
    double P0 = config_.get_value<double>("constants.P0");
    double RbCp = Rd / Cp;

    std::vector<double> pilog1(n);
    std::vector<double> tv1(n);

    for (size_t i = 0; i < n; ++i) {
        double pi = std::pow(P_in[i] / P0, RbCp);
        pilog1[i] = std::log(pi);
        tv1[i] = T_in[i] * (1.0 + 0.608 * Qv_in[i]);
    }

    input_z_[0] = 0.0; 
    
    for (size_t k = 1; k < n; ++k) {
        double d_pilog = pilog1[k] - pilog1[k-1];
        double tv_sum = tv1[k] + tv1[k-1];
        double dz = - (Cp / (2.0 * g)) * d_pilog * tv_sum;
        
        input_z_[k] = input_z_[k-1] + dz;
    }
}

void TxtReader::initialize_thermodynamics(VVM::Core::State& state) {
    const int nz = grid_.get_local_total_points_z();
    const int h = grid_.get_halo_cells();
    
    auto z_mid = params_.z_mid.get_host_data(); 
    auto z_up = params_.z_up.get_host_data(); 
    auto flex_height_coef_up = params_.flex_height_coef_up.get_host_data();

    auto Tbar = state.get_field<1>("Tbar").get_host_data();
    auto qvbar = state.get_field<1>("qvbar").get_host_data();
    auto pbar = state.get_field<1>("pbar").get_host_data();
    auto pibar = state.get_field<1>("pibar").get_host_data();
    auto thbar = state.get_field<1>("thbar").get_host_data();
    auto Tvbar = state.get_field<1>("Tvbar").get_host_data();
    auto rhobar = state.get_field<1>("rhobar").get_host_data();
    auto rhobar_up = state.get_field<1>("rhobar_up").get_host_data();
    
    double P0 = config_.get_value<double>("constants.P0");
    double Rd = config_.get_value<double>("constants.Rd");
    double Cp = config_.get_value<double>("constants.Cp");
    double g = config_.get_value<double>("constants.gravity");
    double dz = config_.get_value<double>("grid.dz");
    double RbCp = Rd / Cp;
    double GDZBCP = 2. * g * dz / Cp;
    
    double P_sfc_val = raw_data_.at("pbar")[0];
    double T_sfc_val = raw_data_.at("Tbar")[0];
    double Qv_sfc_val = raw_data_.at("qvbar")[0];

    pbar(h-1) = P_sfc_val;
    Tbar(h-1) = T_sfc_val;
    qvbar(h-1) = Qv_sfc_val;
    
    const auto& T_in = raw_data_.at("Tbar");
    const auto& Qv_in = raw_data_.at("qvbar");

    for (int k = h; k < nz; ++k) {
        double z = z_mid(k);
        Tbar(k) = interpolate(z, input_z_, T_in);
        qvbar(k) = interpolate(z, input_z_, Qv_in);
    }

    std::vector<double> pilog(nz);
    
    double pi_sfc = std::pow(pbar(h-1) / P0, RbCp);
    pibar(h-1) = pi_sfc;
    pilog[h-1] = std::log(pi_sfc);

    for (int iter = 0; iter < 3; ++iter) {
        for (int k = h-1; k < nz; ++k) {
            Tvbar(k) = Tbar(k) * (1.0 + 0.608 * qvbar(k));
        }

        pilog[h] = pilog[h-1] - GDZBCP / (Tvbar(h-1) + Tvbar(h)) * (z_mid(h)-z_up(h-1)) / dz;
        for (int k = h + 1; k < nz; ++k) {
            pilog[k] = pilog[k-1] - GDZBCP/(Tvbar(k-1)+Tvbar(k)) / flex_height_coef_up(k-1); 
        }

        for (int k = h; k < nz; ++k) {
            pibar(k) = std::exp(pilog[k]);
            pbar(k) = P0 * std::pow(pibar(k), Cp/Rd);
            thbar(k) = Tbar(k) / pibar(k);
        }
        thbar(h-1) = Tbar(h-1) / pibar(h-1);
    }

    for (int k = h-1; k < nz; ++k) {
        rhobar(k) = pbar(k) / (Rd * Tvbar(k));
    }
    
    for (int k = h-1; k < nz - 1; ++k) {
        double alpha_k = 1.0 / rhobar(k);
        double alpha_kp1 = 1.0 / rhobar(k+1);
        double alpha_w = 0.5 * (alpha_k + alpha_kp1);
        rhobar_up(k) = 1.0 / alpha_w;
    }
    rhobar_up(h-1) = rhobar(h-1); 

    for (int k = 0; k < h-1; ++k) {
        Tbar(k) = Tbar(h-1);
        qvbar(k) = qvbar(h-1);
        pbar(k) = pbar(h-1);
        pibar(k) = pibar(h-1);
        thbar(k) = thbar(h-1);
        Tvbar(k) = Tvbar(h-1);
        rhobar(k) = rhobar(h-1);
        rhobar_up(k) = rhobar_up(h-1);
    }

    // Sync
    Kokkos::deep_copy(state.get_field<1>("Tbar").get_mutable_device_data(), Tbar);
    Kokkos::deep_copy(state.get_field<1>("qvbar").get_mutable_device_data(), qvbar);
    Kokkos::deep_copy(state.get_field<1>("pbar").get_mutable_device_data(), pbar);
    Kokkos::deep_copy(state.get_field<1>("pibar").get_mutable_device_data(), pibar);
    Kokkos::deep_copy(state.get_field<1>("thbar").get_mutable_device_data(), thbar);
    Kokkos::deep_copy(state.get_field<1>("Tvbar").get_mutable_device_data(), Tvbar);
    Kokkos::deep_copy(state.get_field<1>("rhobar").get_mutable_device_data(), rhobar);
    Kokkos::deep_copy(state.get_field<1>("rhobar_up").get_mutable_device_data(), rhobar_up);
}

void TxtReader::initialize_forcing(VVM::Core::State& state) {
    const int nz = grid_.get_local_total_points_z();
    const int h = grid_.get_halo_cells();
    bool has_q1 = raw_data_.count("Q1");
    bool has_q2 = raw_data_.count("Q2");
    bool has_u = raw_data_.count("U");
    bool has_v = raw_data_.count("V");

    Kokkos::View<double*> Q1LS, Q2LS, U, V;
    Kokkos::View<double*>::HostMirror Q1LS_h, Q2LS_h, U_h, V_h;

    if (has_q1) {
        if (!state.has_field("Q1")) state.add_field<1>("Q1", {nz});
        Q1LS = state.get_field<1>("Q1").get_mutable_device_data(); 
        Q1LS_h = Kokkos::create_mirror_view(Q1LS);
    }
    if (has_q2) {
        if (!state.has_field("Q2")) state.add_field<1>("Q2", {nz});
        Q2LS = state.get_field<1>("Q2").get_mutable_device_data(); 
        Q2LS_h = Kokkos::create_mirror_view(Q2LS);
    }

    if (has_u) {
        U = state.get_field<1>("U").get_mutable_device_data();
        U_h = state.get_field<1>("U").get_host_data();
    }
    if (has_v) {
        V = state.get_field<1>("V").get_mutable_device_data();
        V_h = state.get_field<1>("V").get_host_data();
    }
    
    auto pbar = state.get_field<1>("pbar").get_host_data();
    auto pibar = state.get_field<1>("pibar").get_host_data();
    const auto& P_in = raw_data_.at("pbar");

    double Cp = config_.get_value<double>("constants.Cp");
    double Lv = config_.get_value<double>("constants.Lv");
    double secday = 86400.0;
    double gamfac = Lv / Cp;

    for (int k = h-1; k < nz; ++k) {
        double p_target = pbar(k);

        if (has_u) U_h(k) = interpolate(p_target, P_in, raw_data_.at("U"), true);
        if (has_v) V_h(k) = interpolate(p_target, P_in, raw_data_.at("V"), true);

        // Interpolate Q1, Q2
        if (has_q1) {
            double q1_val = interpolate(p_target, P_in, raw_data_.at("Q1"), true);
            // Unit Conv: K/day -> K/s (divided by Exner) -> Advective Form
            Q1LS_h(k) = -1.0 * q1_val / pibar(k) / secday;
        }
        if (has_q2) {
            double q2_val = interpolate(p_target, P_in, raw_data_.at("Q2"), true);
            // Unit Conv: K/day -> kg/kg/s
            Q2LS_h(k) = q2_val / (gamfac * secday);
        }
    }

    if (has_u) {
        U_h(h-1) = U_h(h); U_h(nz-h) = U_h(nz-h-1);
        Kokkos::deep_copy(U, U_h);
    }
    if (has_v) {
        V_h(h-1) = V_h(h); V_h(nz-h) = V_h(nz-h-1);
        Kokkos::deep_copy(V, V_h);
    }

    if (has_q1) {
        Q1LS_h(h-1) = Q1LS_h(h); Q1LS_h(nz-h) = Q1LS_h(nz-h-1);
        Kokkos::deep_copy(Q1LS, Q1LS_h);
    }
    if (has_q2) {
        Q2LS_h(h-1) = Q2LS_h(h); Q2LS_h(nz-h) = Q2LS_h(nz-h-1);
        Kokkos::deep_copy(Q2LS, Q2LS_h);
    }
}

double TxtReader::interpolate(double target_x, const std::vector<double>& x_vec, 
                              const std::vector<double>& y_vec, bool is_pressure_coord) const {
    size_t n = x_vec.size();
    if (n == 0) return 0.0;
    if (n == 1) return y_vec[0];

    size_t k1 = 0;
    size_t k2 = 1;

    if (!is_pressure_coord) {
        // Ascending Order (Height)
        // Extrapolation Check (Lower Bound)
        if (target_x <= x_vec[0]) {
            k1 = 0; k2 = 1;
        } 
        // Extrapolation Check (Upper Bound)
        else if (target_x >= x_vec[n-1]) {
            k1 = n-2; k2 = n-1;
        } 
        else {
            // Find interval
            for (size_t i = 0; i < n - 1; ++i) {
                if (target_x >= x_vec[i] && target_x < x_vec[i+1]) {
                    k1 = i;
                    k2 = i + 1;
                    break;
                }
            }
        }
    } 
    else {
        // Descending Order (Pressure)
        // Extrapolation Check (Higher Pressure / Lower Index)
        if (target_x >= x_vec[0]) {
            k1 = 0; k2 = 1;
        }
        // Extrapolation Check (Lower Pressure / Higher Index)
        else if (target_x <= x_vec[n-1]) {
            k1 = n-2; k2 = n-1;
        }
        else {
            // Find interval (target_x is smaller than current, bigger than next)
            for (size_t i = 0; i < n - 1; ++i) {
                if (target_x <= x_vec[i] && target_x > x_vec[i+1]) {
                    k1 = i;
                    k2 = i + 1;
                    break;
                }
            }
        }
    }
    double X  = target_x;
    double X1 = x_vec[k1];
    double X2 = x_vec[k2];
    double F1 = y_vec[k1];
    double F2 = y_vec[k2];

    return F1 + (F1 - F2) * (X - X1) / (X1 - X2);
}


} // namespace IO
} // namespace VVM
