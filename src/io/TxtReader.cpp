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
        raw_data_[col_name] = std::vector<VVM::Real>();
    }

    // Parse Data
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss_data(line);
        VVM::Real val;
        for (const auto& h : headers) {
            if (ss_data >> val) raw_data_[h].push_back(val);
        }
    }
    
    // Check essential columns
    if (raw_data_.find("pbar") == raw_data_.end()) 
        throw std::runtime_error("Missing required column 'pbar' in input file.");

    if (raw_data_.find("Tbar") == raw_data_.end() && raw_data_.find("thbar") == raw_data_.end()) 
        throw std::runtime_error("Missing required column 'Tbar' or 'thbar' in input file.");

    bool enable_constant_wind = config_.get_value<bool>("initial_conditions.constant_upper_wind.enable", false);
    if (enable_constant_wind) {
        VVM::Real p_threshold = config_.get_value<VVM::Real>("initial_conditions.constant_upper_wind.pressure_threshold_Pa", 25000.0);
        
        auto& P_in = raw_data_["pbar"];
        bool has_u = raw_data_.find("U") != raw_data_.end();
        bool has_v = raw_data_.find("V") != raw_data_.end();
        
        if (has_u && has_v) {
            auto& U_in = raw_data_["U"];
            auto& V_in = raw_data_["V"];
            
            int ik1 = -1;
            
            for (size_t k = 0; k < P_in.size(); ++k) {
                if (P_in[k] <= p_threshold && ik1 == -1) {
                    ik1 = k;
                }
                
                if (ik1 != -1) {
                    U_in[k] = U_in[ik1];
                    V_in[k] = V_in[ik1];
                }
            }

            int rank = grid_.get_mpi_rank();
            if (rank == 0 && ik1 != -1) {
                std::cout << "[TxtReader] Applied constant upper wind (U, V) above " 
                          << p_threshold << " Pa (Data Index: " << ik1 << ")" << std::endl;
            }
        }
    }
}

void TxtReader::calculate_input_heights() {
    const auto& P_in  = raw_data_.at("pbar");
    const auto& Qv_in = raw_data_.at("qvbar"); 

    bool has_tbar = raw_data_.find("Tbar") != raw_data_.end();
    
    size_t n = P_in.size();
    input_z_.resize(n);

    VVM::Real Rd   = config_.get_value<VVM::Real>("constants.Rd");
    VVM::Real Cp   = config_.get_value<VVM::Real>("constants.Cp");
    VVM::Real g    = config_.get_value<VVM::Real>("constants.gravity");
    VVM::Real P0   = config_.get_value<VVM::Real>("constants.P0");
    VVM::Real RbCp = Rd / Cp;

    std::vector<VVM::Real> pilog1(n);
    std::vector<VVM::Real> tv1(n);

    for (size_t i = 0; i < n; ++i) {
        VVM::Real pi = std::pow(P_in[i] / P0, RbCp);
        pilog1[i] = std::log(pi);

        VVM::Real T_current;
        if (has_tbar) {
            T_current = raw_data_.at("Tbar")[i];
        } 
        else {
            T_current = raw_data_.at("thbar")[i] * pi;
        }

        tv1[i] = T_current * (real(1.0) + real(0.608) * Qv_in[i]);
    }

    input_z_[0] = real(0.0); 
    
    for (size_t k = 1; k < n; ++k) {
        VVM::Real d_pilog = pilog1[k] - pilog1[k-1];
        VVM::Real tv_sum = tv1[k] + tv1[k-1];
        VVM::Real dz = - (Cp / (real(2.0) * g)) * d_pilog * tv_sum;
        
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
    auto pibar_up = state.get_field<1>("pibar_up").get_host_data();
    auto thbar = state.get_field<1>("thbar").get_host_data();
    auto Tvbar = state.get_field<1>("Tvbar").get_host_data();
    auto rhobar = state.get_field<1>("rhobar").get_host_data();
    auto rhobar_up = state.get_field<1>("rhobar_up").get_host_data();
    
    VVM::Real P0 = config_.get_value<VVM::Real>("constants.P0");
    VVM::Real Rd = config_.get_value<VVM::Real>("constants.Rd");
    VVM::Real Cp = config_.get_value<VVM::Real>("constants.Cp");
    VVM::Real g = config_.get_value<VVM::Real>("constants.gravity");
    VVM::Real dz = config_.get_value<VVM::Real>("grid.dz");
    VVM::Real RbCp = Rd / Cp;
    VVM::Real GDZBCP = real(2.) * g * dz / Cp;

    bool has_tbar = raw_data_.find("Tbar") != raw_data_.end();
    size_t n_in = raw_data_.at("pbar").size();
    std::vector<VVM::Real> T_in(n_in);
    std::vector<VVM::Real> th_in(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        VVM::Real pi = std::pow(raw_data_.at("pbar")[i] / P0, RbCp);
        if (has_tbar) {
            T_in[i] = raw_data_.at("Tbar")[i];
            th_in[i] = T_in[i] / pi;
        } 
        else {
            th_in[i] = raw_data_.at("thbar")[i];
            T_in[i] = th_in[i] * pi;
        }
    }
    
    VVM::Real P_sfc_val = raw_data_.at("pbar")[0];
    VVM::Real Qv_sfc_val = raw_data_.at("qvbar")[0];
    VVM::Real T_sfc_val = T_in[0];

    pbar(h-1) = P_sfc_val;
    Tbar(h-1) = T_sfc_val;
    qvbar(h-1) = Qv_sfc_val;
    if (!has_tbar) thbar(h-1) = th_in[0];

    std::string v_coord_type = config_.get_value<std::string>("grid.vertical_coordinate_type", "default");

    if (v_coord_type == "rcemip") {
        for (int k = h - 1; k < nz; ++k) {
            int idx = k - (h - 1); 
            
            if (idx < n_in) {
                pbar(k)  = raw_data_.at("pbar")[idx];
                thbar(k) = th_in[idx];
                qvbar(k) = raw_data_.at("qvbar")[idx]; 
            } 
            else {
                if (k >= 2) {
                    pbar(k) = pbar(k-1) * (pbar(k-1) / pbar(k-2));
                } else {
                    pbar(k) = pbar(k-1);
                }
                thbar(k) = thbar(k-1);
                qvbar(k) = qvbar(k-1);
            }
            
            pibar(k) = std::pow(pbar(k) / P0, RbCp);
            
            Tbar(k)  = thbar(k) * pibar(k);
            Tvbar(k) = Tbar(k) * (real(1.0) + real(0.608) * qvbar(k));
            
            rhobar(k) = pbar(k) / (Rd * thbar(k) * pibar(k));
        }

        for (int k = h - 1; k < nz - 1; ++k) {
            rhobar_up(k) = rhobar(k) + (rhobar(k+1) - rhobar(k)) / (z_mid(k+1) - z_mid(k)) * (z_up(k) - z_mid(k));
            pibar_up(k) = real(0.5) * (pibar(k) + pibar(k+1));
        }

        rhobar_up(nz - 1) = rhobar_up(nz - 2) + 
            (rhobar_up(nz - 2) - rhobar_up(nz - 3)) / (z_up(nz - 2) - z_up(nz - 3)) * (z_up(nz - 1) - z_up(nz - 2));
        pibar_up(nz - 1) = pibar(nz - 1);
    }
    else {
        for (int k = h; k < nz; ++k) {
            VVM::Real z = z_mid(k);
            qvbar(k) = interpolate(z, input_z_, raw_data_.at("qvbar"));
            
            if (has_tbar) {
                Tbar(k) = interpolate(z, input_z_, T_in);
            } 
            else {
                thbar(k) = interpolate(z, input_z_, th_in);
                Tbar(k) = interpolate(z, input_z_, T_in);
            }
        }

        std::vector<VVM::Real> pilog(nz);
        VVM::Real pi_sfc = std::pow(pbar(h-1) / P0, RbCp);
        pibar(h-1) = pi_sfc;
        pilog[h-1] = std::log(pi_sfc);
        pibar_up(h-1) = pi_sfc;

        for (int iter = 0; iter < 3; ++iter) {
            for (int k = h-1; k < nz; ++k) {
                Tvbar(k) = Tbar(k) * (real(1.0) + real(0.608) * qvbar(k));
            }

            pilog[h] = pilog[h-1] - GDZBCP / (Tvbar(h-1) + Tvbar(h)) * (z_mid(h)-z_up(h-1)) / dz;
            for (int k = h + 1; k < nz; ++k) {
                pilog[k] = pilog[k-1] - GDZBCP/(Tvbar(k-1)+Tvbar(k)) / flex_height_coef_up(k-1); 
            }

            for (int k = h; k < nz; ++k) {
                pibar(k) = std::exp(pilog[k]);
                pbar(k) = P0 * std::pow(pibar(k), Cp/Rd);
                if (has_tbar) {
                    thbar(k) = Tbar(k) / pibar(k);
                } 
                else {
                    Tbar(k) = thbar(k) * pibar(k);
                }
            }
            if (has_tbar) {
                thbar(h-1) = Tbar(h-1) / pibar(h-1);
            } 
            else {
                Tbar(h-1) = thbar(h-1) * pibar(h-1);
            }
        }
    }
    
    for (int k = h; k < nz; k++) {
        pibar_up(k) = real(0.5) * (pibar(k) + pibar(k+1));
    }

    for (int k = h-1; k < nz; ++k) {
        rhobar(k) = pbar(k) / (Rd * Tvbar(k));
    }
    
    for (int k = h-1; k < nz - 1; ++k) {
        VVM::Real alpha_k = real(1.0) / rhobar(k);
        VVM::Real alpha_kp1 = real(1.0) / rhobar(k+1);
        VVM::Real alpha_w = real(0.5) * (alpha_k + alpha_kp1);
        rhobar_up(k) = real(1.0) / alpha_w;
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
    Kokkos::deep_copy(state.get_field<1>("pibar_up").get_mutable_device_data(), pibar_up);
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

    Kokkos::View<VVM::Real*> Q1LS, Q2LS, U, V;
    Kokkos::View<VVM::Real*>::HostMirror Q1LS_h, Q2LS_h, U_h, V_h;

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

    VVM::Real Cp = config_.get_value<VVM::Real>("constants.Cp");
    VVM::Real Lv = config_.get_value<VVM::Real>("constants.Lv");
    VVM::Real secday = real(86400.0);
    VVM::Real gamfac = Lv / Cp;

    std::string v_coord_type = config_.get_value<std::string>("grid.vertical_coordinate_type", "default");
    size_t n_in = P_in.size();

    for (int k = h-1; k < nz; ++k) {
        if (v_coord_type == "rcemip") {
            int idx = k - (h - 1);

            VVM::Real u_val = 0, v_val = 0, q1_val = 0, q2_val = 0;

            if (idx < n_in) {
                if (has_u) u_val = raw_data_.at("U")[idx];
                if (has_v) v_val = raw_data_.at("V")[idx];
                if (has_q1) q1_val = raw_data_.at("Q1")[idx];
                if (has_q2) q2_val = raw_data_.at("Q2")[idx];
            } 
            else {
                if (n_in >= 2) {
                    int last = n_in - 1;
                    int prev = n_in - 2;
                    int diff = idx - last;
                    
                    if (has_u) u_val = raw_data_.at("U")[last] + (raw_data_.at("U")[last] - raw_data_.at("U")[prev]) * diff;
                    if (has_v) v_val = raw_data_.at("V")[last] + (raw_data_.at("V")[last] - raw_data_.at("V")[prev]) * diff;
                    if (has_q1) q1_val = raw_data_.at("Q1")[last] + (raw_data_.at("Q1")[last] - raw_data_.at("Q1")[prev]) * diff;
                    if (has_q2) q2_val = raw_data_.at("Q2")[last] + (raw_data_.at("Q2")[last] - raw_data_.at("Q2")[prev]) * diff;
                } else {
                    if (has_u) u_val = raw_data_.at("U")[0];
                    if (has_v) v_val = raw_data_.at("V")[0];
                    if (has_q1) q1_val = raw_data_.at("Q1")[0];
                    if (has_q2) q2_val = raw_data_.at("Q2")[0];
                }
            }

            if (has_u) U_h(k) = u_val;
            if (has_v) V_h(k) = v_val;
            if (has_q1) Q1LS_h(k) = -real(1.0) * q1_val / pibar(k) / secday;
            if (has_q2) Q2LS_h(k) = q2_val / (gamfac * secday);
        }
        else {
            VVM::Real p_target = pbar(k);

            if (has_u) U_h(k) = interpolate(p_target, P_in, raw_data_.at("U"), true);
            if (has_v) V_h(k) = interpolate(p_target, P_in, raw_data_.at("V"), true);

            if (has_q1) {
                VVM::Real q1_val = interpolate(p_target, P_in, raw_data_.at("Q1"), true);
                Q1LS_h(k) = -real(1.0) * q1_val / pibar(k) / secday;
            }
            if (has_q2) {
                VVM::Real q2_val = interpolate(p_target, P_in, raw_data_.at("Q2"), true);
                Q2LS_h(k) = q2_val / (gamfac * secday);
            }
        }
    }

    if (v_coord_type != "rcemip") {
        if (has_u) { U_h(h-1) = U_h(h); U_h(nz-1) = U_h(nz-2); }
        if (has_v) { V_h(h-1) = V_h(h); V_h(nz-1) = V_h(nz-2); }
        if (has_q1) { Q1LS_h(h-1) = Q1LS_h(h); Q1LS_h(nz-1) = Q1LS_h(nz-2); }
        if (has_q2) { Q2LS_h(h-1) = Q2LS_h(h); Q2LS_h(nz-1) = Q2LS_h(nz-2); }
    }

    if (has_u) Kokkos::deep_copy(U, U_h);
    if (has_v) Kokkos::deep_copy(V, V_h);
    if (has_q1) Kokkos::deep_copy(Q1LS, Q1LS_h);
    if (has_q2) Kokkos::deep_copy(Q2LS, Q2LS_h);
}

VVM::Real TxtReader::interpolate(VVM::Real target_x, const std::vector<VVM::Real>& x_vec, 
                              const std::vector<VVM::Real>& y_vec, bool is_pressure_coord) const {
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
    VVM::Real X  = target_x;
    VVM::Real X1 = x_vec[k1];
    VVM::Real X2 = x_vec[k2];
    VVM::Real F1 = y_vec[k1];
    VVM::Real F2 = y_vec[k2];

    return F1 + (F1 - F2) * (X - X1) / (X1 - X2);
}


} // namespace IO
} // namespace VVM
