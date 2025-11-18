#include "Initializer.hpp"
#include "BoundaryConditionManager.hpp"
#include "io/TxtReader.hpp"
#include "io/PnetcdfReader.hpp"
#include <Kokkos_Core.hpp>

namespace VVM {
namespace Core {

Initializer::Initializer(const Utils::ConfigurationManager& config, const Grid& grid, Parameters& parameters, State &state) 
    : config_(config), grid_(grid), parameters_(parameters), state_(state) {
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
    pnetcdf_reader_ = std::make_unique<VVM::IO::PnetcdfReader>(pnetcdf_source_file, grid, parameters_, config_);
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
    initialize_poisson();
    assign_vars();
}

void Initializer::initialize_grid() const {
    double DOMAIN = 15000.;
    double dz = config_.get_value<double>("grid.dz");
    double dz1 = config_.get_value<double>("grid.dz1");
    double CZ2 = (dz-dz1) / (dz * (DOMAIN-dz));
    double CZ1 = 1. - CZ2 * DOMAIN;

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();

    auto& z_mid_mutable = parameters_.z_mid.get_mutable_device_data();
    auto& z_up_mutable = parameters_.z_up.get_mutable_device_data();
    auto& flex_height_coef_mid_mutable = parameters_.flex_height_coef_mid.get_mutable_device_data();
    auto& flex_height_coef_up_mutable = parameters_.flex_height_coef_up.get_mutable_device_data();
    auto z_mid_mutable_h = parameters_.z_mid.get_host_data();
    auto z_up_mutable_h = parameters_.z_up.get_host_data();

    double ZB = 0.;
    z_up_mutable_h(h-1) = ZB;
    for (int k = h; k < nz; k++) {
        z_up_mutable_h(k) = z_up_mutable_h(k-1) + dz;
    }
    z_mid_mutable_h(h-1) = z_up_mutable_h(h-1);
    z_mid_mutable_h(h) = z_up_mutable_h(h-1) + 0.5 * dz;
    for (int k = h+1; k < nz; k++) {
        z_mid_mutable_h(k) = z_mid_mutable_h(k-1) + dz;
    }
    Kokkos::deep_copy(z_up_mutable, z_up_mutable_h);
    Kokkos::deep_copy(z_mid_mutable, z_mid_mutable_h);

    Kokkos::parallel_for("Init_Z_flexZCoef", Kokkos::RangePolicy<>(h-1, nz),
        KOKKOS_LAMBDA(const int k) {
            flex_height_coef_mid_mutable(k) = 1. / (CZ1 + 2 * CZ2 * z_mid_mutable(k));
            flex_height_coef_up_mutable(k) = 1. / (CZ1 + 2 * CZ2 * z_up_mutable(k));
            z_mid_mutable(k) = z_mid_mutable(k) * (CZ1 + CZ2 * z_mid_mutable(k));
            z_up_mutable(k) = z_up_mutable(k) * (CZ1 + CZ2 * z_up_mutable(k));
        }
    );
    
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

    // DEBUG output
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) parameters_.z_mid.print_profile(grid_, 0, 3, 3);
    // if (rank == 0) parameters_.z_up.print_profile(grid_, 0, 3, 3);
    // if (rank == 0) parameters_.flex_height_coef_mid.print_profile(grid_, 0, 3, 3);
    // if (rank == 0) parameters_.flex_height_coef_up.print_profile(grid_, 0, 3, 3);
    // if (rank == 0) parameters_.fact1_xi_eta.print_profile(grid_, 0, 3, 3);
    // if (rank == 0) parameters_.fact2_xi_eta.print_profile(grid_, 0, 3, 3);

    // auto fact1_data = parameters_.fact1_xi_eta.get_host_data();
    // auto fact2_data = parameters_.fact2_xi_eta.get_host_data();
    //
    // if (rank == 0) {
    //     for (int k = 0; k < grid_.get_local_total_points_z(); k++) {
    //         std::cout << fact1_data(k) + fact2_data(k) << " ";
    //     }
    // }
    // std::cout << std::endl;
    return;
}

void Initializer::initialize_topo() const {
    const auto& topo = state_.get_field<2>("topo").get_device_data();
    auto& ITYPEU = state_.get_field<3>("ITYPEU").get_mutable_device_data();
    auto& ITYPEV = state_.get_field<3>("ITYPEV").get_mutable_device_data();
    auto& ITYPEW = state_.get_field<3>("ITYPEW").get_mutable_device_data();
    auto topo_h = state_.get_field<2>("topo").get_host_data();

    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    double local_maxtopo_h, maxtopo_h;
    Kokkos::parallel_reduce("FindMax", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i, double& local_max) {
            if (topo(j, i) > local_max) {
                local_max = topo(i, j);
            }
        },
        Kokkos::Max<double>(local_maxtopo_h)
    );

    MPI_Allreduce(
        &local_maxtopo_h,
        &maxtopo_h,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD
    );
    maxtopo_h += h;
    parameters_.max_topo_idx = maxtopo_h;

    // Assign ITYPE
    Kokkos::deep_copy(ITYPEU, 1.);
    Kokkos::deep_copy(ITYPEV, 1.);
    Kokkos::deep_copy(ITYPEW, 1.);
    Kokkos::parallel_for("assign_ITYPE", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            if (topo(j, i) != 0) {
                for (int k = 0; k <= topo(j,i); k++) {
                    ITYPEU(k,j,i) = 0;
                    ITYPEV(k,j,i) = 0;
                    ITYPEW(k,j,i) = 0;
                } 
            }
        }
    );
    VVM::Core::HaloExchanger halo_exchanger(grid_);
    halo_exchanger.exchange_halos(state_.get_field<3>("ITYPEW"));

    Kokkos::parallel_for("assign_ITYPE", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k,j,i) == 0) {
                ITYPEU(k,j,i-1) = 0;
                ITYPEV(k,j-1,i) = 0;
            }
        }
    );
    halo_exchanger.exchange_halos(state_.get_field<3>("ITYPEU"));
    halo_exchanger.exchange_halos(state_.get_field<3>("ITYPEU"));
      
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
                BGAU(k) = (WRXMU() + 2.*rdx2() + 2.*rdy2()) / rhobar_up(k) + 
                          flex_height_coef_up(k) * (flex_height_coef_mid(k+1)/rhobar(k+1)+flex_height_coef_mid(k)/rhobar(k))*rdz2();
                CGAU(k) = -flex_height_coef_up(k) * flex_height_coef_mid(k+1) * rdz2() / rhobar(k+1);
            }
            else AGAU(k) = BGAU(k) = CGAU(k) = -9e16;
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
        else h_bn_new(k) = h_cn_new(k) = -9e16;
    }
    Kokkos::deep_copy(bn_new, h_bn_new);
    Kokkos::deep_copy(cn_new, h_cn_new);

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) {
    //     for (int k = 0; k <= nz; k++) {
    //         std::cout << "K = " << k << ", AGAU: " << h_AGAU(k) << ", BGAU: " << h_BGAU(k) << ", CGAU: " << h_CGAU(k) << ", bn_new: " << h_bn_new(k) << ", cn_new: " << h_cn_new(k) << std::endl; 
    //     }
    // }

    return;
}

void Initializer::assign_vars() const {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

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

    // utop predict
    double utopmn_h = state_.calculate_horizontal_mean(state_.get_field<3>("u"), nz-h-1);
    double vtopmn_h = state_.calculate_horizontal_mean(state_.get_field<3>("v"), nz-h-1);
    Kokkos::deep_copy(state_.get_field<1>("utopmn").get_mutable_device_data(), utopmn_h);
    Kokkos::deep_copy(state_.get_field<1>("vtopmn").get_mutable_device_data(), vtopmn_h);

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
            pbar_up(k) = 0.5*(pbar(k) + pbar(k+1));
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
    return;
}

} // namespace Core
} // namespace VVM
