#include "Initializer.hpp"
#include "io/TxtReader.hpp"
#include "io/PnetcdfReader.hpp"
#include <Kokkos_Core.hpp>

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

    double local_maxtopo_h, maxtopo_h;
    Kokkos::parallel_reduce("FindMax", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h,h}, {ny-h,nx-h}),
        KOKKOS_LAMBDA(const int j, const int i, double& local_max) {
            if (topo(j, i) > local_max) {
                local_max = topo(j, i);
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
    // VVM::Core::HaloExchanger halo_exchanger(grid_);
    Kokkos::fence();
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEW"));
    cudaDeviceSynchronize();

    Kokkos::parallel_for("assign_ITYPE", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,h,h}, {nz-h,ny-h,nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k,j,i) == 0) {
                ITYPEU(k,j,i-1) = 0;
                ITYPEV(k,j-1,i) = 0;
            }
        }
    );
    Kokkos::fence();
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEU"));
    halo_exchanger_.exchange_halos(state_.get_field<3>("ITYPEV"));
      
    cudaDeviceSynchronize();


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
            if (topo(j,i) == 0) topo(j,i) = h;
        }
    );
      
    cudaDeviceSynchronize();
    return;

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
    return;
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
    if (rank == 0) state_.get_field<1>("pbar").print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.z_mid.print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.z_up.print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.flex_height_coef_mid.print_profile(grid_, 0, 0, 0);
    if (rank == 0) parameters_.flex_height_coef_up.print_profile(grid_, 0, 0, 0);
    if (rank == 0) state_.get_field<1>("U").print_profile(grid_, 0, 0, 0);
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
    Kokkos::deep_copy(w, 0.);
// utop predict
#if defined(ENABLE_NCCL)
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> utopmn("utopmn");
    Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> vtopmn("vtopmn");
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

    auto& lon = state_.get_field<1>("lon").get_mutable_device_data();
    Kokkos::deep_copy(lon, 121.);

    auto& lat = state_.get_field<1>("lat").get_mutable_device_data();
    Kokkos::deep_copy(lat, 23.5);

    double OMEGA = config_.get_value<double>("constants.OMEGA", 7.292e-5);
    double PI = config_.get_value<double>("constants.PI", 3.14159265);
    auto& f = state_.get_field<1>("f").get_mutable_device_data();
    Kokkos::parallel_for("Init_Coriolis", Kokkos::RangePolicy<>(0, ny),
        KOKKOS_LAMBDA(const int j) {
            f(j) = 2. * OMEGA * Kokkos::sin(lat(j) * PI / 180.);
        }
    );
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
    double PI = config_.get_value<double>("constants.PI");

    if (perturbation == "none") return;
    else if (perturbation == "bubble") {
        auto& th = state_.get_field<3>("th").get_mutable_device_data();
        Kokkos::parallel_for("init_perturbation", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                const int local_j = j;
                const int local_i = i;

                const int global_j = global_start_j + local_j;
                const int global_i = global_start_i + local_i;

                double radius_norm = std::sqrt(
                                      std::pow(((global_i + 1) - (int) (nx/2)) * dx() / 2000., 2) +
                                      std::pow(((global_j + 1) - (int) (ny/2)) * dy() / 2000., 2) +
                                      std::pow((z_mid(k) - 3000.) / 2000., 2)
                                     );
                if (radius_norm <= 1) {
                    th(k, j, i) += 5. * (std::cos(PI * 0.5 * radius_norm));
                }
            }
        );
    }
    return;
}

} // namespace Core
} // namespace VVM
