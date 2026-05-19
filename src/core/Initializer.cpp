#include "Initializer.hpp"
#include "io/TxtReader.hpp"
#include "io/PnetcdfReader.hpp"
#include "vvm_types.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace VVM {
namespace Core {

Initializer::Initializer(const Utils::ConfigurationManager& config, const Grid& grid, Parameters& parameters, State &state, HaloExchanger& halo_exchanger) 
    : config_(config), grid_(grid), parameters_(parameters), state_(state), halo_exchanger_(halo_exchanger) {
    initialize_grid();

    if (!config.has_key("initial_conditions") && !config.has_key("netcdf_reader")) {
        return;
    }

    std::string format = config.get_value<std::string>("initial_conditions.format");
    std::string source_file = config.get_value<std::string>("initial_conditions.source_file");
    std::string pnetcdf_source_file = config.get_value<std::string>("netcdf_reader.source_file");

    if (format == "txt") {
        reader_ = std::make_unique<VVM::IO::TxtReader>(source_file, grid, parameters_, config_);
    } 
    pnetcdf_reader_ = std::make_unique<VVM::IO::PnetcdfReader>(pnetcdf_source_file, grid, parameters_, config_, halo_exchanger_);
    // else if (format == "netcdf") {
    //     // TODO: Netcdf input
    //     // reader_ = std::make_unique<Initializers::NetCDFReader>(source_file, grid);
    //     std::cerr << "Warning: NetCDF reader is not implemented yet. Skipping initialization." << std::endl;
    // } else {
    //     std::cerr << "Warning: Unsupported input format '" << format << "'. Skipping initialization." << std::endl;
    // }
    

    // TODO: Initialize vorticity after loading velocity field

}

void Initializer::initialize_state() const {
    if (reader_) {
        reader_->read_and_initialize(state_);
    }
    if (pnetcdf_reader_) {
        pnetcdf_reader_->read_and_initialize(state_);
    }
    initialize_topo();
    assign_vars();
    initialize_perturbation();
    // init poisson should be placed after assign variables 
    // because the density would affect height factors.
    initialize_poisson();
    initialize_zeta_factor_for_twisting();
}

void Initializer::initialize_grid() const {
    VVM::Real DOMAIN = real(15000.);
    VVM::Real dz = config_.get_value<VVM::Real>("grid.dz");
    VVM::Real dz1 = config_.get_value<VVM::Real>("grid.dz1");
    VVM::Real CZ2 = (dz-dz1) / (dz * (DOMAIN-dz));
    VVM::Real CZ1 = real(1.) - CZ2 * DOMAIN;

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();

    auto& z_mid_mutable = parameters_.z_mid.get_mutable_device_data();
    auto& z_up_mutable = parameters_.z_up.get_mutable_device_data();
    auto& flex_height_coef_mid_mutable = parameters_.flex_height_coef_mid.get_mutable_device_data();
    auto& flex_height_coef_up_mutable = parameters_.flex_height_coef_up.get_mutable_device_data();

    auto z_mid_mutable_h = parameters_.z_mid.get_host_data();
    auto z_up_mutable_h = parameters_.z_up.get_host_data();
    auto flex_height_coef_mid_h = parameters_.flex_height_coef_mid.get_host_data();
    auto flex_height_coef_up_h = parameters_.flex_height_coef_up.get_host_data();

    VVM::Real ZB = real(0.);
    z_up_mutable_h(h-1) = ZB;

    std::string v_coord_type = config_.get_value<std::string>("grid.vertical_coordinate_type", "default");

    if (v_coord_type == "taiwanvvm") {
        // TaiwanVVM Vertical Coordinate Logic
        for (int k = h; k < nz; k++) {
            z_up_mutable_h(k) = z_up_mutable_h(k-1) + dz;
        }

        z_mid_mutable_h(h-1) = z_up_mutable_h(h-1);
        z_mid_mutable_h(h) = z_up_mutable_h(h-1) + real(0.5) * dz;
        for (int k = h+1; k < nz; k++) {
            z_mid_mutable_h(k) = z_mid_mutable_h(k-1) + dz;
        }

        // Fortran: DO 40 (ZZ & FNZ transformation)
        for (int k = h-1; k < nz; k++) {
            flex_height_coef_up_h(k) = real(1.) / (CZ1 + real(2.) * CZ2 * z_up_mutable_h(k));
            z_up_mutable_h(k) = z_up_mutable_h(k) * (CZ1 + CZ2 * z_up_mutable_h(k));
        }

        int KT = static_cast<int>((real(1.) - CZ1) / CZ2 / real(2.) / dz);
        int kt_idx = (KT + h - 2 >= nz) ? nz - 1 : KT + h - 2; 
        if (kt_idx < 0) kt_idx = 0;
        int KT1 = static_cast<int>(z_up_mutable_h(kt_idx) / dz1 + real(0.999));

        z_up_mutable_h(h-1) = ZB;
        flex_height_coef_up_h(h-1) = dz / dz1;

        for (int K = 2; K <= KT1 - 1; K++) {
            int kc = K + h - 2;
            for (int KK = nz - 1; KK >= kc + 1; KK--) {
                flex_height_coef_up_h(KK) = flex_height_coef_up_h(KK-1);
                z_up_mutable_h(KK) = z_up_mutable_h(KK-1) + dz1;
            }
            if (kc < nz) {
                flex_height_coef_up_h(kc) = dz / dz1;
                z_up_mutable_h(kc) = z_up_mutable_h(kc-1) + dz1;
            }
        }

        for (int k = h-1; k < nz; k++) {
            flex_height_coef_mid_h(k) = real(1.) / (CZ1 + real(2.) * CZ2 * z_mid_mutable_h(k));
            z_mid_mutable_h(k) = z_mid_mutable_h(k) * (CZ1 + CZ2 * z_mid_mutable_h(k));
        }

        z_mid_mutable_h(h-1) = ZB;
        flex_height_coef_mid_h(h-1) = dz / dz1;
        z_mid_mutable_h(h) = z_up_mutable_h(h-1) + dz1 / real(2.);
        flex_height_coef_mid_h(h) = dz / dz1;

        for (int K = 3; K <= KT1; K++) {
            int kc = K + h - 2;
            for (int KK = nz - 1; KK >= kc + 1; KK--) {
                flex_height_coef_mid_h(KK) = flex_height_coef_mid_h(KK-1);
                z_mid_mutable_h(KK) = z_mid_mutable_h(KK-1) + dz1;
            }
            if (kc < nz) {
                flex_height_coef_mid_h(kc) = dz / dz1;
                z_mid_mutable_h(kc) = z_mid_mutable_h(kc-1) + dz1;
            }
        }

        int kt1_idx = KT1 + 1 + h - 2;
        if (kt1_idx < nz && (KT1 + 2 + h - 2) < nz) {
            flex_height_coef_up_h(kt1_idx) = dz / (z_mid_mutable_h(KT1 + 2 + h - 2) - z_mid_mutable_h(KT1 + 1 + h - 2));
        }

        Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
        Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);
        Kokkos::deep_copy(flex_height_coef_up_mutable, flex_height_coef_up_h);
        Kokkos::deep_copy(flex_height_coef_mid_mutable, flex_height_coef_mid_h);
    } 
    else if (v_coord_type == "rcemip") {
        std::string source_file = config_.get_value<std::string>("grid.rcemip_grid_data_path", "../rundata/initial_conditions/profiles/snd_rcemip_anal300_v3.txt");
        std::ifstream infile(source_file);
        
        if (!infile.is_open()) {
            throw std::runtime_error("[Initializer] Cannot open rcemip source file: " + source_file);
        }

        std::string line;
        std::getline(infile, line);

        z_up_mutable_h(h-1) = ZB;
        z_mid_mutable_h(h-1) = ZB;

        for (int k = h; k < nz - h; k++) {
            if (!std::getline(infile, line)) {
                throw std::runtime_error("[Initializer] Unexpected end of file in rcemip source file.");
            }
            std::istringstream iss(line);
            VVM::Real zz_val, zt_val;
            
            if (!(iss >> zz_val >> zt_val)) {
                throw std::runtime_error("[Initializer] Error reading data from rcemip source file.");
            }
            
            z_up_mutable_h(k) = zz_val;
            z_mid_mutable_h(k) = zt_val;
        }
        infile.close();

        for (int k = nz - h; k < nz; k++) {
            z_up_mutable_h(k) = real(2.) * z_up_mutable_h(k-1) - z_up_mutable_h(k-2);
            z_mid_mutable_h(k) = real(2.) * z_mid_mutable_h(k-1) - z_mid_mutable_h(k-2);
        }

        for (int k = h - 2; k >= 0; k--) {
            z_up_mutable_h(k) = z_up_mutable_h(k+1) - (z_up_mutable_h(k+2) - z_up_mutable_h(k+1));
            z_mid_mutable_h(k) = z_mid_mutable_h(k+1) - (z_mid_mutable_h(k+2) - z_mid_mutable_h(k+1));
        }

        for (int k = 0; k < nz - 1; k++) {
            flex_height_coef_up_h(k) = dz / (z_mid_mutable_h(k+1) - z_mid_mutable_h(k));
        }
        flex_height_coef_up_h(nz - 1) = flex_height_coef_up_h(nz - 2);

        for (int k = 1; k < nz; k++) {
            flex_height_coef_mid_h(k) = dz / (z_up_mutable_h(k) - z_up_mutable_h(k-1));
        }
        flex_height_coef_mid_h(0) = flex_height_coef_mid_h(1);

        Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
        Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);
        Kokkos::deep_copy(flex_height_coef_up_mutable, flex_height_coef_up_h);
        Kokkos::deep_copy(flex_height_coef_mid_mutable, flex_height_coef_mid_h);
        
        int rank = grid_.get_mpi_rank();
        if (rank == 0) {
            std::cout << "  [Initializer] Initialized RCEMIP vertical coordinate from " << source_file << std::endl;
        }
    }
    else {
        for (int k = h; k < nz; k++) {
            z_up_mutable_h(k) = z_up_mutable_h(k-1) + dz;
        }
        z_mid_mutable_h(h-1) = z_up_mutable_h(h-1);
        z_mid_mutable_h(h) = z_up_mutable_h(h-1) + real(0.5) * dz;
        for (int k = h+1; k < nz; k++) {
            z_mid_mutable_h(k) = z_mid_mutable_h(k-1) + dz;
        }
        
        Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
        Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);

        Kokkos::parallel_for("Init_Z_flexZCoef", Kokkos::RangePolicy<>(h-1, nz),
            KOKKOS_LAMBDA(const int k) {
                flex_height_coef_mid_mutable(k) = real(1.) / (CZ1 + real(2.) * CZ2 * z_mid_mutable(k));
                flex_height_coef_up_mutable(k) = real(1.) / (CZ1 + real(2.) * CZ2 * z_up_mutable(k));
                z_mid_mutable(k) = z_mid_mutable(k) * (CZ1 + CZ2 * z_mid_mutable(k));
                z_up_mutable(k) = z_up_mutable(k) * (CZ1 + CZ2 * z_up_mutable(k));
            }
        );
    }

    auto& dz_mid_mutable = parameters_.dz_mid.get_mutable_device_data();
    auto& dz_up_mutable = parameters_.dz_up.get_mutable_device_data();
    Kokkos::parallel_for("Init_dz", Kokkos::RangePolicy<>(h, nz-h),
        KOKKOS_LAMBDA(const int k) {
            dz_mid_mutable(k) = z_up_mutable(k) - z_up_mutable(k-1);
            dz_up_mutable(k) = z_mid_mutable(k+1) - z_mid_mutable(k);
        }
    );

    auto& fact1_xi_eta_mutable = parameters_.fact1_xi_eta.get_mutable_device_data();
    auto& fact2_xi_eta_mutable = parameters_.fact2_xi_eta.get_mutable_device_data();
    Kokkos::parallel_for("Init_zflex_fact", Kokkos::RangePolicy<>(h, nz-h-1),
        KOKKOS_LAMBDA(const int k) {
            fact1_xi_eta_mutable(k) = flex_height_coef_up_mutable(k) / flex_height_coef_mid_mutable(k+1);
            fact2_xi_eta_mutable(k) = flex_height_coef_up_mutable(k) / flex_height_coef_mid_mutable(k);
        }
    );
    return;
}

void Initializer::initialize_topo() const {
    auto dims = std::array<int, 2>{
        grid_.get_local_total_points_y(), 
        grid_.get_local_total_points_x()
    };
    if (!state_.has_field("topou")) state_.add_field<2>("topou", dims);
    if (!state_.has_field("topov")) state_.add_field<2>("topov", dims);

    const auto& topo = state_.get_field<2>("topo").get_device_data();
    auto& ITYPEU = state_.get_field<3>("ITYPEU").get_mutable_device_data();
    auto& ITYPEV = state_.get_field<3>("ITYPEV").get_mutable_device_data();
    auto& ITYPEW = state_.get_field<3>("ITYPEW").get_mutable_device_data();
    auto topo_h = state_.get_field<2>("topo").get_host_data();

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    VVM::Real local_maxtopo_h, maxtopo_h;
    Kokkos::parallel_reduce("FindMax", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i, VVM::Real& local_max) {
            if (topo(j, i) > local_max) {
                local_max = topo(j, i);
            }
        },
        Kokkos::Max<VVM::Real>(local_maxtopo_h)
    );

    MPI_Allreduce(
        &local_maxtopo_h,
        &maxtopo_h,
        1,
        VVM_MPI_REAL,
        MPI_MAX,
        grid_.get_comm()
    );
    maxtopo_h += h;
    parameters_.max_topo_idx = maxtopo_h;

    // Assign ITYPE
    Kokkos::deep_copy(ITYPEU, real(1.));
    Kokkos::deep_copy(ITYPEV, real(1.));
    Kokkos::deep_copy(ITYPEW, real(1.));
    Kokkos::parallel_for("assign_ITYPE", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j, i) != 0) {
                for (int k = 0; k <= topo(j,i); k++) {
                    ITYPEU(k,j,i) = real(0.);
                    ITYPEV(k,j,i) = real(0);
                    ITYPEW(k,j,i) = real(0);
                } 
            }
        }
    );
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEW"));

    Kokkos::parallel_for("assign_ITYPE", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k,j,i) == 0) {
                ITYPEU(k,j,i-1) = real(0);
                ITYPEV(k,j-1,i) = real(0);
            }
        }
    );
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEU"));
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEV"));
      

    // Assign topou, topov
    auto& topou = state_.get_field<2>("topou").get_mutable_device_data();
    auto& topov = state_.get_field<2>("topov").get_mutable_device_data();
    Kokkos::parallel_for("assign_topou_topov", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j,i+1)-topo(j,i) > 0) topou(j,i) = topo(j,i+1);
            if (topo(j+1,i)-topo(j,i) > 0) topov(j,i) = topo(j+1,i);
        }
    );

    Kokkos::parallel_for("modifyTopo", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j,i) == 0) topo(j,i) = h-1;
        }
    );
    return;
}

void Initializer::initialize_poisson() const {
    const auto& rdx2 = parameters_.rdx2;
    const auto& rdy2 = parameters_.rdy2;
    const auto& rdz2 = parameters_.rdz2;
    const auto& WRXMU = parameters_.WRXMU;
    const auto& flex_height_coef_mid = parameters_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = parameters_.flex_height_coef_up.get_device_data();
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();

    // Poisson iteration coefficient
    auto& AGAU = parameters_.AGAU.get_mutable_device_data();
    auto& BGAU = parameters_.BGAU.get_mutable_device_data();
    auto& CGAU = parameters_.CGAU.get_mutable_device_data();
    Kokkos::parallel_for("Init_Poisson_Coef", Kokkos::RangePolicy<>(0, nz),
        KOKKOS_LAMBDA(const int k) {
            if (k >= h && k <= nz-h-2) {
                AGAU(k) = -flex_height_coef_up(k) * flex_height_coef_mid(k) * rdz2() / rhobar(k);
                BGAU(k) = (WRXMU() + real(2.)*rdx2() + real(2.)*rdy2()) / rhobar_up(k) + 
                          flex_height_coef_up(k) * (flex_height_coef_mid(k+1)/rhobar(k+1)+flex_height_coef_mid(k)/rhobar(k))*rdz2();
                CGAU(k) = -flex_height_coef_up(k) * flex_height_coef_mid(k+1) * rdz2() / rhobar(k+1);
            }
            else AGAU(k) = BGAU(k) = CGAU(k) = real(-9e16);
        }
    );

    auto& bn_new = parameters_.bn_new.get_mutable_device_data();
    auto& cn_new = parameters_.cn_new.get_mutable_device_data();

    auto h_AGAU = parameters_.AGAU.get_host_data();
    auto h_BGAU = parameters_.BGAU.get_host_data();
    auto h_CGAU = parameters_.CGAU.get_host_data();
    auto h_bn_new = parameters_.bn_new.get_host_data();
    auto h_cn_new = parameters_.cn_new.get_host_data();

    for (int k = 0; k <= nz; k++) {
        if (k == h) h_cn_new(h) = h_CGAU(h) / h_BGAU(h);
        else if (k >= h+1 && k <= nz-h-2) {
            h_bn_new(k) = h_BGAU(k) - h_AGAU(k) * h_cn_new(k-1);
            h_cn_new(k) = h_CGAU(k) / h_bn_new(k);
        }
        else h_bn_new(k) = h_cn_new(k) = real(-9e16);
    }
    Kokkos::deep_copy(bn_new, h_bn_new);
    Kokkos::deep_copy(cn_new, h_cn_new);
    return;
}


void Initializer::initialize_zeta_factor_for_twisting() const {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();

    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state_.get_field<1>("rhobar_up").get_device_data();
    const auto& flex_height_coef_mid = parameters_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = parameters_.flex_height_coef_up.get_device_data();

    auto& fact1_zeta_mutable = parameters_.fact1_zeta.get_mutable_device_data();
    auto& fact2_zeta_mutable = parameters_.fact2_zeta.get_mutable_device_data();
    Kokkos::parallel_for("AssignFactor", 1, KOKKOS_LAMBDA(const int) {
        fact1_zeta_mutable() = flex_height_coef_mid(nz-h-1) * rhobar_up(nz-h-1) / flex_height_coef_up(nz-h-1);
        fact2_zeta_mutable() = flex_height_coef_mid(nz-h-1) * rhobar_up(nz-h-2) / flex_height_coef_up(nz-h-2);
    });
}

void Initializer::assign_vars() const {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    // Vertical B.C. process
    // WARNING: This causes errors in P3
    // VVM::Core::BoundaryConditionManager bc_manager(grid_);
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("thbar")); bc_manager.apply_z_bcs_to_field(state_.get_field<1>("qvbar")); bc_manager.apply_z_bcs_to_field(state_.get_field<1>("Tbar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("Tvbar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("rhobar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("rhobar_up"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("pbar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("pibar"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("U"));
    // bc_manager.apply_z_bcs_to_field(state_.get_field<1>("V"));


    int rank = grid_.get_mpi_rank();
    if (rank == 0) state_.get_field<1>("qvbar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("rhobar_up").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("rhobar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("thbar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("Tbar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("Tvbar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("pibar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("pibar_up").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("pbar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.z_mid.print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.z_up.print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.flex_height_coef_mid.print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.flex_height_coef_up.print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("U").print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("V").print_profile(grid_, 0, 0, 0);
    if (rank == 0 && state_.has_field("Q1")) state_.get_field<1>("Q1").print_profile(grid_, 0, 0, 0);

    const auto& rdx = parameters_.rdx;
    const auto& rdy = parameters_.rdy;
    const auto& rdz = parameters_.rdz;
    const auto& flex_height_coef_mid = parameters_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = parameters_.flex_height_coef_up.get_device_data();

    auto& u = state_.get_field<3>("u").get_mutable_device_data();
    auto& v = state_.get_field<3>("v").get_mutable_device_data();
    const auto& w = state_.get_field<3>("w").get_device_data();
    const auto& U = state_.get_field<1>("U").get_device_data();
    const auto& V = state_.get_field<1>("V").get_device_data();
    Kokkos::parallel_for("assign_initial_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            u(k,j,i) = U(k);
            v(k,j,i) = V(k);
        }
    );
    Kokkos::deep_copy(w, real(0.));
// utop predict
#if defined(ENABLE_NCCL)
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> utopmn("utopmn");
    Kokkos::View<VVM::Real, Kokkos::DefaultExecutionSpace::memory_space> vtopmn("vtopmn");
    state_.calculate_horizontal_mean(state_.get_field<3>("u"), utopmn);
    state_.calculate_horizontal_mean(state_.get_field<3>("v"), vtopmn);
    auto utopmn_view = state_.get_field<0>("utopmn").get_mutable_device_data();
    auto vtopmn_view = state_.get_field<0>("vtopmn").get_mutable_device_data();
    Kokkos::parallel_for("assign_uvtopmn", Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int k) {
            utopmn_view() = utopmn();
            vtopmn_view() = vtopmn();
        }
    );
#else
    const auto utopmn = state_.calculate_horizontal_mean(state_.get_field<3>("u"), nz-h-1);
    const auto vtopmn = state_.calculate_horizontal_mean(state_.get_field<3>("v"), nz-h-1);
    Kokkos::deep_copy(state_.get_field<0>("utopmn").get_mutable_device_data(), utopmn);
    Kokkos::deep_copy(state_.get_field<0>("vtopmn").get_mutable_device_data(), vtopmn);
#endif

    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();
    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    Kokkos::parallel_for("assign_vorticity", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h-1,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            eta(k,j,i) = (w(k,j,i+1)-w(k,j,i))*rdx() - (u(k+1,j,i)-u(k,j,i))*rdz()*flex_height_coef_up(k);
            xi(k,j,i)  = (w(k,j+1,i)-w(k,j,i))*rdy() - (v(k+1,j,i)-v(k,j,i))*rdz()*flex_height_coef_up(k);
        }
    );


    // Assign pbar_up
    const auto& pbar = state_.get_field<1>("pbar").get_device_data();
    auto& pbar_up = state_.get_field<1>("pbar_up").get_mutable_device_data();
    auto& dpbar_mid = state_.get_field<1>("dpbar_mid").get_mutable_device_data();
    Kokkos::parallel_for("assign_pbar_up", Kokkos::RangePolicy<>(1, nz),
        KOKKOS_LAMBDA(const int k) {
            pbar_up(k) = real(0.5)*(pbar(k) + pbar(k+1));
        }
    );
    Kokkos::parallel_for("assign_pbar_up", Kokkos::RangePolicy<>(2, nz),
        KOKKOS_LAMBDA(const int k) {
            dpbar_mid(k) = -(pbar_up(k) - pbar_up(k-1)); // make it positive
        }
    );

    // Assign qv
    const auto& qvbar = state_.get_field<1>("qvbar").get_device_data();
    auto& qv = state_.get_field<3>("qv").get_mutable_device_data();
    Kokkos::parallel_for("assign_qv", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            qv(k,j,i) = qvbar(k);
        }
    );


    // Assign th
    auto& thbar = state_.get_field<1>("thbar").get_mutable_device_data();
    auto& th = state_.get_field<3>("th").get_mutable_device_data();
    auto& rhobar = state_.get_field<1>("rhobar").get_mutable_device_data();
    auto& rhobar_up = state_.get_field<1>("rhobar_up").get_mutable_device_data();
    Kokkos::parallel_for("assign_th", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            th(k,j,i) = thbar(k);
        }
    );

    auto& lon = state_.get_field<2>("lon").get_mutable_device_data();
    auto& lat = state_.get_field<2>("lat").get_mutable_device_data();
    if (config_.get_value<bool>("grid.fix_lonlat", false)) {
        Kokkos::deep_copy(lon, real(120.95));
        Kokkos::deep_copy(lat, real(23.458));
    }

    // TODO: This is tcvvm setting. It needs to be user friendly.
    VVM::Real OMEGA = config_.get_value<VVM::Real>("constants.OMEGA", real(7.292e-5));
    VVM::Real PI = config_.get_value<VVM::Real>("constants.PI", real(3.14159265));
    auto& f = state_.get_field<1>("f").get_mutable_device_data();
    auto& f_2d = state_.get_field<2>("f_2d").get_mutable_device_data();
    const int global_start_j = grid_.get_local_physical_start_y();
    const auto& dy = parameters_.dy;
    Kokkos::parallel_for("Init_Coriolis", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, nx}),
        KOKKOS_LAMBDA(const int j, const int i) {
            const int global_j = global_start_j + j;
            f(j) = real(2.) * OMEGA * Kokkos::sin((global_j - real(1540.)/real(2.) - real(0.5))*dy()/real(6.37e6));
            f_2d(j,i) = real(2.) * OMEGA * Kokkos::sin((global_j - real(1540.)/real(2.) - real(0.5))*dy()/real(6.37e6));
        }
    );
    halo_exchanger_.exchange_halos(state_.get_field<2>("f_2d"));

    // Assign Tg
    const auto& topo = state_.get_field<2>("topo").get_device_data();
    const auto& pibar = state_.get_field<1>("pibar").get_device_data();
    auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data();

    std::string Tg_source = config_.get_value<std::string>("netcdf_reader.Tg_source", "atmosphere");

    if (Tg_source == "atmosphere") {
        const auto& topo = state_.get_field<2>("topo").get_device_data();
        const auto& pibar = state_.get_field<1>("pibar").get_device_data();
        auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data();

        Kokkos::parallel_for("Init_Tg", 
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, nx}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // NOTE: Fortran VVM uses hx rather than hxp here
                int hx = topo(j, i);
                Tg(j, i) = th(hx, j, i) * pibar(hx);
            }
        );
        halo_exchanger_.exchange_halos(state_.get_field<2>("Tg"));
        
        int rank = grid_.get_mpi_rank();
        if (rank == 0) std::cout << "  [Initializer] Initialized Tg from the lowest atmosphere level." << std::endl;

    } 
    else if (Tg_source == "netcdf") {
        int rank = grid_.get_mpi_rank();
        if (rank == 0) std::cout << "  [Initializer] Skipped Tg initialization (using values read from NetCDF)." << std::endl;
    } 
    else {
        int rank = grid_.get_mpi_rank();
        if (rank == 0) std::cerr << "  [Initializer] Warning: Unknown Tg_source '" << Tg_source << "'. Keeping existing Tg values." << std::endl;
    }
    halo_exchanger_.exchange_halos(state_.get_field<2>("Tg"));
    return;
}

void Initializer::initialize_perturbation() const {
    std::string perturbation = config_.get_value<std::string>("initial_conditions.perturbation", "none");
    const int global_start_j = grid_.get_local_physical_start_y();
    const int global_start_i = grid_.get_local_physical_start_x();
    const auto& dx = parameters_.dx;
    const auto& dy = parameters_.dy;
    const auto& z_mid = parameters_.z_mid.get_device_data();

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    VVM::Real PI = config_.get_value<VVM::Real>("constants.PI");

    auto& th = state_.get_field<3>("th").get_mutable_device_data();
    auto& xi = state_.get_field<3>("xi").get_mutable_device_data();
    auto& eta = state_.get_field<3>("eta").get_mutable_device_data();
    auto& zeta = state_.get_field<3>("zeta").get_mutable_device_data();
    auto& u = state_.get_field<3>("u").get_mutable_device_data();
    auto& v = state_.get_field<3>("v").get_mutable_device_data();
    auto& w = state_.get_field<3>("w").get_mutable_device_data();

    std::string test_mode = config_.get_value<std::string>("simulation.idealized_test", "none");
    if (test_mode == "advection_u" || test_mode == "advection_v" || test_mode == "advection_w" || 
        test_mode == "stretching"  || test_mode == "twisting") {
        auto& rhobar = state_.get_field<1>("rhobar").get_mutable_device_data();
        auto& rhobar_up = state_.get_field<1>("rhobar_up").get_mutable_device_data();

        if (test_mode == "advection_w") {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), th, real(300.));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), rhobar, real(1.));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), rhobar_up, real(1.));
        }

        Kokkos::parallel_for("test_init", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (k == h+16 && (h+3 <= j && h+11 >= j) && (h+3 <= i && h+11 >= i)) {
                    th(k,j,i) += real(50.);
                    xi(k,j,i) += real(50.);
                    eta(k,j,i) += real(50.);
                    zeta(nz-h-1,j,i) += real(50.);
                }
        });
        if (test_mode == "advection_u") Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), u, real(10.));
        else if (test_mode == "advection_v") Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), v, real(10.));
        else if (test_mode == "advection_w") Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), w, real(10.));
        else if (test_mode == "stretching") {
            Kokkos::parallel_for("test_wind_init", 
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, nx}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    const int global_j = global_start_j + j;
                    const int global_i = global_start_i + i;

                    u(k,j,i) = real(32.)/real(2.) - global_i - real(1.);
                    v(k,j,i) = real(32.)/real(2.) - global_j - real(1.);
                    w(k,j,i) = -(real(32.)/real(2.) - k - real(1.));
            });
        }
        else if (test_mode == "twisting") {
            Kokkos::parallel_for("test_wind_init_cross_derivatives", 
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, nx}), 
                KOKKOS_LAMBDA(int k, int j, int i) { 
                    const int global_j = global_start_j + j; 
                    const int global_i = global_start_i + i; 

                    if (k == nz-h-1 && (h+3 <= j && h+11 >= j) && (h+3 <= i && h+11 >= i)) {
                        xi(k,j,i) += real(50.);
                        eta(k,j,i) += real(50.);
                    }

                    u(k,j,i) = -real(0.2)*(real(32.)/real(2.) - global_j - real(1.));
                    v(k,j,i) = -real(0.2)*(real(32.)/real(2.) - global_i - real(1.));
                    w(k,j,i) = real(0.2)*(real(32.)/real(2.) - global_i - real(1.));
                }
            );
        }
    } 
    else if (test_mode == "2dbubble") {
        Kokkos::parallel_for("init_perturbation", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_i = global_start_i + i;

                VVM::Real radius_norm = Kokkos::sqrt(
                                      Kokkos::pow(((global_i + 1) - (int) (nx/2)) * dx() / real(1000.), 2) +
                                      Kokkos::pow((z_mid(k) - real(5000.)) / real(1000.), 2)
                                     );
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(10.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            }
        );
    }
    else if (test_mode == "3dbubble") {
        Kokkos::parallel_for("init_perturbation", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_i = global_start_i + i;
                const int global_j = global_start_j + j;

                VVM::Real radius_norm = Kokkos::sqrt(
                                      Kokkos::pow(((global_i + 1) - (int) (nx/2)) * dx() / real(1000.), 2) +
                                      Kokkos::pow(((global_j + 1) - (int) (ny/2)) * dy() / real(1000.), 2) +
                                      Kokkos::pow((z_mid(k) - real(5000.)) / real(1000.), 2)
                                     );
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            }
        );
    }
    else if (test_mode == "topo") {
        auto& utopmn = state_.get_field<0>("utopmn").get_mutable_device_data();
        Kokkos::deep_copy(u, real(10.));
        Kokkos::deep_copy(utopmn, real(10.));
        
    }

    if (perturbation == "none") return;
    else if (perturbation == "2dbubble") {
        Kokkos::parallel_for("init_perturbation", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_i = global_start_i + i;

                VVM::Real radius_norm = Kokkos::sqrt(
                                      Kokkos::pow(((global_i + 1) - (int) (nx/2)) * dx() / real(2000.), 2) +
                                      Kokkos::pow((z_mid(k) - real(3000.)) / real(2000.), 2)
                                     );
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            }
        );
    }
    else if (perturbation == "3dbubble") {
        Kokkos::parallel_for("init_perturbation", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int global_j = global_start_j + j;
                const int global_i = global_start_i + i;

                VVM::Real radius_norm = Kokkos::sqrt(
                                      Kokkos::pow(((global_i + 1) - (int) (nx/2)) * dx() / real(2000.), 2) +
                                      Kokkos::pow(((global_j + 1) - (int) (ny/2)) * dy() / real(2000.), 2) +
                                      Kokkos::pow((z_mid(k) - real(3000.)) / real(2000.), 2)
                                     );
                if (radius_norm <= real(1.)) {
                    th(k, j, i) += real(5.) * (Kokkos::cos(PI * real(0.5) * radius_norm));
                }
            }
        );
    }

    halo_exchanger_.exchange_multiple_halos({"th", "xi", "eta", "zeta"}, state_);
    if (test_mode == "none") halo_exchanger_.exchange_multiple_halos({"u", "v", "w"}, state_);
    return;
}

} // namespace Core
} // namespace VVM
