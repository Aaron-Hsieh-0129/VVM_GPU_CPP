// Turbulence scheme for Shuts & Grey (1994)

#include "TurbulenceProcess.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace VVM {
namespace Physics {

TurbulenceProcess::TurbulenceProcess(const Utils::ConfigurationManager& config, 
                                     const Core::Grid& grid, 
                                     const Core::Parameters& params,
                                     Core::HaloExchanger& halo_exchanger,
                                     Core::State& state)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger),
        temp3d_tendency_("temp3d", std::array<int, 3>{
              grid.get_local_total_points_z(),
              grid.get_local_total_points_y(),
              grid.get_local_total_points_x()
        }),
        temp2d_tendency_("temp2d", std::array<int, 2>{
            grid.get_local_total_points_y(),
            grid.get_local_total_points_x()
        }),
      DHUU1_("DHUU1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHUU2_("DHUU2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHUV1_("DHUV1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHUV2_("DHUV2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHUW1_("DHUW1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHUW2_("DHUW2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHVU1_("DHVU1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHVU2_("DHVU2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHVV1_("DHVV1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHVV2_("DHVV2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHVW1_("DHVW1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHVW2_("DHVW2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHWU1_("DHWU1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHWU2_("DHWU2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHWV1_("DHWV1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHWV2_("DHWV2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHWW1_("DHWW1", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      DHWW2_("DHWW2", std::array<int, 3>{grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()}),
      masks_(grid.get_local_total_points_z(), grid.get_local_total_points_y(), grid.get_local_total_points_x()) 
{
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    auto dims = std::array<int, 3>{
          grid.get_local_total_points_z(),
          grid.get_local_total_points_y(),
          grid.get_local_total_points_x()
    };
    if (!state.has_field("RKM")) state.add_field<3>("RKM", dims);
    if (!state.has_field("RKH")) state.add_field<3>("RKH", dims);

    dynamics_vars_ = {"xi", "eta", "zeta"};
    thermodynamics_vars_ = {"th", "qv", "qc", "qr", "qi", "nc", "nr", "ni"};

    Kokkos::deep_copy(dx_, params_.dx);
    Kokkos::deep_copy(dy_, params_.dy);
    Kokkos::deep_copy(dz_, params_.dz);
    Kokkos::deep_copy(rdx_, params_.rdx);
    Kokkos::deep_copy(rdy_, params_.rdy);
    Kokkos::deep_copy(rdz_, params_.rdz);
    Kokkos::deep_copy(rdx2_, params_.rdx2);
    Kokkos::deep_copy(rdy2_, params_.rdy2);
    Kokkos::deep_copy(rdz2_, params_.rdz2);
    Kokkos::deep_copy(grav_, params_.gravity);
    
    vk_ = 0.4;
    deld_ = std::pow(dx_ * dy_ * dz_, 1.0/3.0);
    ramd0s_ = std::pow(0.23 * deld_, 2.0);
    critmn_ = 1.0;

    int rank = grid_.get_mpi_rank();
    if (rank == 0) {
        std::cout << "--- Initializing Turbulence Process ---" << std::endl;
        std::cout << "    Grid Scale (DELD): " << deld_ << " m" << std::endl;
        std::cout << "    Mixing Length Sq (RAMD0S): " << ramd0s_ << " m^2" << std::endl;
    }
}


void TurbulenceProcess::initialize(Core::State& state) {
    // init_boundary_masks(state);
    init_dh_coefficients(state);
    return;
}

void TurbulenceProcess::init_dh_coefficients(Core::State& state) {
    const auto& ITYPEU = state.get_field<3>("ITYPEU").get_device_data();
    const auto& ITYPEV = state.get_field<3>("ITYPEV").get_device_data();
    const auto& ITYPEW = state.get_field<3>("ITYPEW").get_device_data();
    const auto& hx     = state.get_field<2>("topo").get_device_data();

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    int max_topo = params_.max_topo_idx; 

    auto dhuu1 = DHUU1_.get_mutable_device_data(); auto dhuu2 = DHUU2_.get_mutable_device_data();
    auto dhuv1 = DHUV1_.get_mutable_device_data(); auto dhuv2 = DHUV2_.get_mutable_device_data();
    auto dhuw1 = DHUW1_.get_mutable_device_data(); auto dhuw2 = DHUW2_.get_mutable_device_data();
    auto dhvu1 = DHVU1_.get_mutable_device_data(); auto dhvu2 = DHVU2_.get_mutable_device_data();
    auto dhvv1 = DHVV1_.get_mutable_device_data(); auto dhvv2 = DHVV2_.get_mutable_device_data();
    auto dhvw1 = DHVW1_.get_mutable_device_data(); auto dhvw2 = DHVW2_.get_mutable_device_data();
    auto dhwu1 = DHWU1_.get_mutable_device_data(); auto dhwu2 = DHWU2_.get_mutable_device_data();
    auto dhwv1 = DHWV1_.get_mutable_device_data(); auto dhwv2 = DHWV2_.get_mutable_device_data();
    auto dhww1 = DHWW1_.get_mutable_device_data(); auto dhww2 = DHWW2_.get_mutable_device_data();

    Kokkos::deep_copy(dhuu1, 1.); Kokkos::deep_copy(dhuu2, 1.);
    Kokkos::deep_copy(dhuv1, 1.); Kokkos::deep_copy(dhuv2, 1.);
    Kokkos::deep_copy(dhuw1, 1.); Kokkos::deep_copy(dhuw2, 1.);
    Kokkos::deep_copy(dhvu1, 1.); Kokkos::deep_copy(dhvu2, 1.);
    Kokkos::deep_copy(dhvv1, 1.); Kokkos::deep_copy(dhvv2, 1.);
    Kokkos::deep_copy(dhvw1, 1.); Kokkos::deep_copy(dhvw2, 1.);
    Kokkos::deep_copy(dhwu1, 1.); Kokkos::deep_copy(dhwu2, 1.);
    Kokkos::deep_copy(dhwv1, 1.); Kokkos::deep_copy(dhwv2, 1.);
    Kokkos::deep_copy(dhww1, 1.); Kokkos::deep_copy(dhww2, 1.);

    Kokkos::parallel_for("Init_DH_Topo_Inside",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int NN = static_cast<int>(hx(j, i)); 
            if (NN != 0) { 
                for (int k = h; k < NN; ++k) { 
                    dhuu1(k,j,i)=0.; dhuv1(k,j,i)=0.; dhuw1(k,j,i)=0.;
                    dhvu1(k,j,i)=0.; dhvv1(k,j,i)=0.; dhvw1(k,j,i)=0.;
                    dhuu2(k,j,i)=0.; dhuv2(k,j,i)=0.; dhuw2(k,j,i)=0.;
                    dhvu2(k,j,i)=0.; dhvv2(k,j,i)=0.; dhvw2(k,j,i)=0.;
                    dhwu1(k,j,i)=0.; dhwv1(k,j,i)=0.; dhww1(k,j,i)=0.;
                    dhwu2(k,j,i)=0.; dhwv2(k,j,i)=0.; dhww2(k,j,i)=0.;
                }
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEW",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{max_topo+1, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k, j, i) != 1) {
                dhuu1(k, j, i-1) = 0.; 
                dhuv1(k, j, i-1) = 0.; 
                dhuw1(k, j, i-1) = 0.;
                dhvu1(k, j-1, i) = 0.; 
                dhvv1(k, j-1, i) = 0.; 
                dhvw1(k, j-1, i) = 0.;
                
                dhuu2(k, j, i-1) = 0.; 
                dhuv2(k, j, i-1) = 0.; 
                dhuw2(k, j, i-1) = 0.;
                dhvu2(k, j-1, i) = 0.; 
                dhvv2(k, j-1, i) = 0.; 
                dhvw2(k, j-1, i) = 0.;
                
                dhwu2(k, j, i+1) = 0.;
                dhwu1(k, j, i-1) = 0.; 
                dhwv2(k, j+1, i) = 0.;
                dhwv1(k, j-1, i) = 0.;
                dhww2(k+1, j, i) = 0.;
                dhww1(k-1, j, i) = 0.;
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEU",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}), 
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEU(k, j, i) != 1) {
                dhuu2(k, j, i+1) = 0.;
                dhuu1(k, j, i-1) = 0.;
                dhuv2(k, j+1, i) = 0.;
                dhuv1(k, j-1, i) = 0.;
                dhuw2(k+1, j, i) = 0.;
                dhuw1(k-1, j, i) = 0.;
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEV",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEV(k, j, i) != 1) {
                dhvu2(k, j, i+1) = 0.;
                dhvu1(k, j, i-1) = 0.;
                dhvv2(k, j+1, i) = 0.;
                dhvv1(k, j-1, i) = 0.;
                dhvw2(k+1, j, i) = 0.;
                dhvw1(k-1, j, i) = 0.;
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEW",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{nz, nx}}),
        KOKKOS_LAMBDA(const int k, const int i) {
            // North Boundary
            if (ITYPEW(k, ny-h, i) != 1) {
                dhvu1(k, ny-h-1, i) = 0.; dhvv1(k, ny-h-1, i) = 0.; dhvw1(k, ny-h-1, i) = 0.;
                dhvu2(k, ny-h-1, i) = 0.; dhvv2(k, ny-h-1, i) = 0.; dhvw2(k, ny-h-1, i) = 0.;
                dhwv1(k, ny-h-1, i) = 0.;
            }
            // South Boundary
            if (ITYPEW(k, h-1, i) != 1) dhwv2(k, h, i) = 0.;

            if (ITYPEU(k, ny-h, i) != 1) dhuv1(k, ny-h-1, i) = 0.;
            if (ITYPEU(k, h-1, i) != 1) dhuv2(k, h, i) = 0.;
            
            if (ITYPEV(k, ny-h, i) != 1) dhvv1(k, ny-h-1, i) = 0.;
            if (ITYPEV(k, h-1, i) != 1) dhvv2(k, h, i) = 0.;
        }
    );

    Kokkos::parallel_for("Init_DH_Edge_X",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{nz, ny}}),
        KOKKOS_LAMBDA(const int k, const int j) {
            // East Boundary
            if (ITYPEW(k, j, nx-h) != 1) {
                dhuu1(k, j, nx-h-1) = 0.; dhuv1(k, j, nx-h-1) = 0.; dhuw1(k, j, nx-h-1) = 0.;
                dhuu2(k, j, nx-h-1) = 0.; dhuv2(k, j, nx-h-1) = 0.; dhuw2(k, j, nx-h-1) = 0.;
                dhwu1(k, j, nx-h-1) = 0.;
            }
            // West Boundary
            if (ITYPEW(k, j, h-1) != 1) dhwu2(k, j, h) = 0.;

            if (ITYPEU(k, j, nx-h) != 1) dhuu1(k, j, nx-h-1) = 0.;
            if (ITYPEU(k, j, h-1) != 1) dhuu2(k, j, h) = 0.;

            if (ITYPEV(k, j, nx-h) != 1) dhvu1(k, j, nx-h-1) = 0.;
            if (ITYPEV(k, j, h-1) != 1) dhvu2(k, j, h) = 0.;
        }
    );
    
    Kokkos::parallel_for("Init_DH_WW_Bound",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{ny, nx}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int NN = static_cast<int>(hx(j, i)); 
            if (NN>1 && ITYPEW(NN-1,j,i) == 0) {
                 dhww1(NN,j,i) = 0.;
                 dhwu1(NN,j,i) = 0.;
                 dhwu2(NN,j,i) = 0.;
                 dhwv1(NN,j,i) = 0.;
                 dhwv2(NN,j,i) = 0.;
            }
            if (NN>1 && ITYPEV(NN,j,i) == 0) {
                 dhvu1(NN,j,i) = 0.;
                 dhvu2(NN,j,i) = 0.;
                 dhvv1(NN,j,i) = 0.;
                 dhvw1(NN,j,i) = 0.;
            }
            if (NN>1 && ITYPEU(NN,j,i) == 0) {
                 dhuu1(NN,j,i) = 0.;
                 dhuu2(NN,j,i) = 0.;
                 dhuv1(NN,j,i) = 0.;
                 dhuv2(NN,j,i) = 0.;
                 dhuw1(NN,j,i) = 0.;
            }
        }
    );

    auto dims = std::array<int, 3>{
          grid_.get_local_total_points_z(),
          grid_.get_local_total_points_y(),
          grid_.get_local_total_points_x()
    };

    int k_print = 15;
    if (!state.has_field("DHUU1")) state.add_field<3>("DHUU1", dims);
    if (!state.has_field("DHUV1")) state.add_field<3>("DHUV1", dims);
    if (!state.has_field("DHUW1")) state.add_field<3>("DHUW1", dims);
    Kokkos::deep_copy(state.get_field<3>("DHUU1").get_mutable_device_data(), dhuu1);
    Kokkos::deep_copy(state.get_field<3>("DHUV1").get_mutable_device_data(), dhuv1);
    Kokkos::deep_copy(state.get_field<3>("DHUW1").get_mutable_device_data(), dhuw1);
    halo_exchanger_.exchange_halos(state.get_field<3>("DHUU1"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHUV1"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHUW1"));
    // state.get_field<3>("DHUU1").print_slice_z_at_k(grid_, 0, k_print, halo);
    // state.get_field<3>("DHUV1").print_slice_z_at_k(grid_, 0, k_print, halo);
    // state.get_field<3>("DHUW1").print_slice_z_at_k(grid_, 0, k_print, halo);
    //
    if (!state.has_field("DHUU2")) state.add_field<3>("DHUU2", dims);
    if (!state.has_field("DHUV2")) state.add_field<3>("DHUV2", dims);
    if (!state.has_field("DHUW2")) state.add_field<3>("DHUW2", dims);
    Kokkos::deep_copy(state.get_field<3>("DHUU2").get_mutable_device_data(), dhuu2);
    Kokkos::deep_copy(state.get_field<3>("DHUV2").get_mutable_device_data(), dhuv2);
    Kokkos::deep_copy(state.get_field<3>("DHUW2").get_mutable_device_data(), dhuw2);
    halo_exchanger_.exchange_halos(state.get_field<3>("DHUU2"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHUV2"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHUW2"));
    // state.get_field<3>("DHUU2").print_slice_z_at_k(grid_, 0, k_print, halo);
    // state.get_field<3>("DHUV2").print_slice_z_at_k(grid_, 0, k_print, halo);
    // state.get_field<3>("DHUW2").print_slice_z_at_k(grid_, 0, k_print, halo);

    
    if (!state.has_field("DHVU1")) state.add_field<3>("DHVU1", dims);
    if (!state.has_field("DHVV1")) state.add_field<3>("DHVV1", dims);
    if (!state.has_field("DHVW1")) state.add_field<3>("DHVW1", dims);
    Kokkos::deep_copy(state.get_field<3>("DHVU1").get_mutable_device_data(), dhvu1);
    Kokkos::deep_copy(state.get_field<3>("DHVV1").get_mutable_device_data(), dhvv1);
    Kokkos::deep_copy(state.get_field<3>("DHVW1").get_mutable_device_data(), dhvw1);
    halo_exchanger_.exchange_halos(state.get_field<3>("DHVU1"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHVV1"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHVW1"));
    // state.get_field<3>("DHVU1").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHVV1").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHVW1").print_slice_z_at_k(grid_, 0, 12, halo);
    //
    if (!state.has_field("DHVU2")) state.add_field<3>("DHVU2", dims);
    if (!state.has_field("DHVV2")) state.add_field<3>("DHVV2", dims);
    if (!state.has_field("DHVW2")) state.add_field<3>("DHVW2", dims);
    Kokkos::deep_copy(state.get_field<3>("DHVU2").get_mutable_device_data(), dhvu2);
    Kokkos::deep_copy(state.get_field<3>("DHVV2").get_mutable_device_data(), dhvv2);
    Kokkos::deep_copy(state.get_field<3>("DHVW2").get_mutable_device_data(), dhvw2);
    halo_exchanger_.exchange_halos(state.get_field<3>("DHVU2"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHVV2"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHVW2"));
    // state.get_field<3>("DHVU2").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHVV2").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHVW2").print_slice_z_at_k(grid_, 0, 12, halo);

    if (!state.has_field("DHWU1")) state.add_field<3>("DHWU1", dims);
    if (!state.has_field("DHWV1")) state.add_field<3>("DHWV1", dims);
    if (!state.has_field("DHWW1")) state.add_field<3>("DHWW1", dims);
    Kokkos::deep_copy(state.get_field<3>("DHWU1").get_mutable_device_data(), dhwu1);
    Kokkos::deep_copy(state.get_field<3>("DHWV1").get_mutable_device_data(), dhwv1);
    Kokkos::deep_copy(state.get_field<3>("DHWW1").get_mutable_device_data(), dhww1);
    halo_exchanger_.exchange_halos(state.get_field<3>("DHWU1"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHWV1"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHWW1"));
    // state.get_field<3>("DHWU1").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHWV1").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHWW1").print_slice_z_at_k(grid_, 0, k_print, halo);
    state.get_field<3>("ITYPEW").print_xz_cross_at_j(grid_, 0, 4);
    //
    if (!state.has_field("DHWU2")) state.add_field<3>("DHWU2", dims);
    if (!state.has_field("DHWV2")) state.add_field<3>("DHWV2", dims);
    if (!state.has_field("DHWW2")) state.add_field<3>("DHWW2", dims);
    Kokkos::deep_copy(state.get_field<3>("DHWU2").get_mutable_device_data(), dhwu2);
    Kokkos::deep_copy(state.get_field<3>("DHWV2").get_mutable_device_data(), dhwv2);
    Kokkos::deep_copy(state.get_field<3>("DHWW2").get_mutable_device_data(), dhww2);
    halo_exchanger_.exchange_halos(state.get_field<3>("DHWU2"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHWV2"));
    halo_exchanger_.exchange_halos(state.get_field<3>("DHWW2"));
    // state.get_field<3>("DHWU2").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHWV2").print_slice_z_at_k(grid_, 0, 12, halo);
    // state.get_field<3>("DHWW2").print_slice_z_at_k(grid_, 0, k_print, halo);
    state.get_field<3>("DHWU1").print_xz_cross_at_j(grid_, 0, 4, 0);
    state.get_field<3>("DHWU2").print_xz_cross_at_j(grid_, 0, 4, 0);
    state.get_field<3>("DHWV1").print_xz_cross_at_j(grid_, 0, 4, 0);
    state.get_field<3>("DHWV2").print_xz_cross_at_j(grid_, 0, 4, 0);
    state.get_field<3>("DHWW1").print_xz_cross_at_j(grid_, 0, 4, 0);
    state.get_field<3>("DHWW2").print_xz_cross_at_j(grid_, 0, 4, 0);

    // DHWW1_.print_slice_z_at_k(grid_, 0, 2, 2);

    state.get_field<3>("ITYPEW").print_xz_cross_at_j(grid_, 0, 4, h);
    state.get_field<3>("DHWW2").print_slice_z_at_k(grid_, 0, 1, 0);
    state.get_field<3>("ITYPEU").print_slice_z_at_k(grid_, 0, 1, 0);
    state.get_field<3>("ITYPEV").print_slice_z_at_k(grid_, 0, 1, 0);
    state.get_field<3>("ITYPEW").print_slice_z_at_k(grid_, 0, 1, 0);
}


void TurbulenceProcess::init_boundary_masks(Core::State& state) {
    const auto& ITYPEU = state.get_field<3>("ITYPEU").get_device_data();
    const auto& ITYPEV = state.get_field<3>("ITYPEV").get_device_data();
    const auto& ITYPEW = state.get_field<3>("ITYPEW").get_device_data();
    const auto& hx     = state.get_field<2>("topo").get_device_data();

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    int max_topo = params_.max_topo_idx; 

    auto masks = masks_;

    masks_.reset_to_ones();

    Kokkos::parallel_for("Init_DH_Topo_Inside",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int NN = static_cast<int>(hx(j, i)); 
            if (NN != 0) { 
                for (int k = h; k < NN; ++k) { 
                    masks.turn_off_all(k, j, i);
                }
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEW",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{max_topo+1, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEW(k, j, i) != 1) {
                masks.turn_off(k, j, i-1, UU1); 
                masks.turn_off(k, j, i-1, UV1); 
                masks.turn_off(k, j, i-1, UW1);
                masks.turn_off(k, j-1, i, VU1); 
                masks.turn_off(k, j-1, i, VV1); 
                masks.turn_off(k, j-1, i, VW1);
                
                masks.turn_off(k, j, i-1, UU2); 
                masks.turn_off(k, j, i-1, UV2); 
                masks.turn_off(k, j, i-1, UW2);
                masks.turn_off(k, j-1, i, VU2); 
                masks.turn_off(k, j-1, i, VV2); 
                masks.turn_off(k, j-1, i, VW2);
                
                masks.turn_off(k, j, i+1, WU2);
                masks.turn_off(k, j, i-1, WU1); 
                masks.turn_off(k, j+1, i, WV2);
                masks.turn_off(k, j-1, i, WV1);
                masks.turn_off(k+1, j, i, WW2);
                masks.turn_off(k-1, j, i, WW1);
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEU",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}), 
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEU(k, j, i) != 1) {
                masks.turn_off(k, j, i+1, UU2);
                masks.turn_off(k, j, i-1, UU1);
                masks.turn_off(k, j+1, i, UV2);
                masks.turn_off(k, j-1, i, UV1);
                masks.turn_off(k+1, j, i, UW2);
                masks.turn_off(k-1, j, i, UW1);
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEV",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            if (ITYPEV(k, j, i) != 1) {
                masks.turn_off(k, j, i+1, VU2);
                masks.turn_off(k, j, i-1, VU1);
                masks.turn_off(k, j+1, i, VV2);
                masks.turn_off(k, j-1, i, VV1);
                masks.turn_off(k+1, j, i, VW2);
                masks.turn_off(k-1, j, i, VW1);
            }
        }
    );

    Kokkos::parallel_for("Init_DH_ITYPEW_2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{nz, nx}}),
        KOKKOS_LAMBDA(const int k, const int i) {
            // North Boundary
            if (ITYPEW(k, ny-h, i) != 1) {
                masks.turn_off(k, ny-h-1, i, VU1); masks.turn_off(k, ny-h-1, i, VV1); masks.turn_off(k, ny-h-1, i, VW1);
                masks.turn_off(k, ny-h-1, i, VU2); masks.turn_off(k, ny-h-1, i, VV2); masks.turn_off(k, ny-h-1, i, VW2);
                masks.turn_off(k, ny-h-1, i, WV1);
            }
            // South Boundary
            if (ITYPEW(k, h-1, i) != 1) masks.turn_off(k, h, i, WV2);

            if (ITYPEU(k, ny-h, i) != 1) masks.turn_off(k, ny-h-1, i, UV1);
            if (ITYPEU(k, h-1, i) != 1) masks.turn_off(k, h, i, UV2);
            
            if (ITYPEV(k, ny-h, i) != 1) masks.turn_off(k, ny-h-1, i, VV1);
            if (ITYPEV(k, h-1, i) != 1) masks.turn_off(k, h, i, VV2);
        }
    );

    Kokkos::parallel_for("Init_DH_Edge_X",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{nz, ny}}),
        KOKKOS_LAMBDA(const int k, const int j) {
            // East Boundary
            if (ITYPEW(k, j, nx-h) != 1) {
                masks.turn_off(k, j, nx-h-1, UU1); masks.turn_off(k, j, nx-h-1, UV1); masks.turn_off(k, j, nx-h-1, UW1);
                masks.turn_off(k, j, nx-h-1, UU2); masks.turn_off(k, j, nx-h-1, UV2); masks.turn_off(k, j, nx-h-1, UW2);
                masks.turn_off(k, j, nx-h-1, WU1);
            }
            // West Boundary
            if (ITYPEW(k, j, h-1) != 1) masks.turn_off(k, j, h, WU2);

            if (ITYPEU(k, j, nx-h) != 1) masks.turn_off(k, j, nx-h-1, UU1);
            if (ITYPEU(k, j, h-1) != 1) masks.turn_off(k, j, h, UU2);

            if (ITYPEV(k, j, nx-h) != 1) masks.turn_off(k, j, nx-h-1, VU1);
            if (ITYPEV(k, j, h-1) != 1) masks.turn_off(k, j, h, VU2);
        }
    );
    
    Kokkos::parallel_for("Init_DH_WW_Bound",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{ny, nx}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int NN = static_cast<int>(hx(j, i)); 
            if (NN>1 && ITYPEW(NN-1,j,i) == 0) {
                 masks.turn_off(NN,j,i, WW1);
                 masks.turn_off(NN,j,i, WU1);
                 masks.turn_off(NN,j,i, WU2);
                 masks.turn_off(NN,j,i, WV1);
                 masks.turn_off(NN,j,i, WV2);
            }
            if (NN>1 && ITYPEV(NN,j,i) == 0) {
                 masks.turn_off(NN,j,i, VU1);
                 masks.turn_off(NN,j,i, VU2);
                 masks.turn_off(NN,j,i, VV1);
                 masks.turn_off(NN,j,i, VW1);
            }
            if (NN>1 && ITYPEU(NN,j,i) == 0) {
                 masks.turn_off(NN,j,i, UU1);
                 masks.turn_off(NN,j,i, UU2);
                 masks.turn_off(NN,j,i, UV1);
                 masks.turn_off(NN,j,i, UV2);
                 masks.turn_off(NN,j,i, UW1);
            }
        }
    );
}

void TurbulenceProcess::compute_coefficients(Core::State& state, double dt) 
{
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& v = state.get_field<3>("v").get_device_data();
    const auto& w = state.get_field<3>("w").get_device_data();
    const auto& R_xi = state.get_field<3>("R_xi").get_device_data();
    const auto& R_eta = state.get_field<3>("R_eta").get_device_data();
    const auto& R_zeta = state.get_field<3>("R_zeta").get_device_data();
    const auto& th = state.get_field<3>("th").get_device_data();
    const auto& z_mid = params_.z_mid.get_device_data(); 
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& ITYPEU = state.get_field<3>("ITYPEU").get_device_data();
    const auto& ITYPEV = state.get_field<3>("ITYPEV").get_device_data();
    const auto& ITYPEW = state.get_field<3>("ITYPEW").get_device_data();

    auto& rkm = state.get_field<3>("RKM").get_mutable_device_data();
    auto& rkh = state.get_field<3>("RKH").get_mutable_device_data();

    const double rdx = rdx_;
    const double rdy = rdy_;
    const double rdz = rdz_;
    
    const double grav = grav_;
    const double vk = vk_;
    const double ramd0s = ramd0s_;
    const double critmn = critmn_;
    const double critmx = 0.8 * deld_ * deld_ / dt;

    auto DHWW1 = DHWW1_.get_device_data(); auto DHWW2 = DHWW2_.get_device_data();
    Kokkos::parallel_for("ShuttsGray_Coeffs",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            double du_dx = (u(k, j, i) - u(k, j, i-1)) * rdx;
            double dv_dy = (v(k, j, i) - v(k, j-1, i)) * rdy;
            double dw_dz = flex_height_coef_mid(k) * (w(k, j, i) - w(k-1, j, i)) * rdz;

            double TERM1 =  Kokkos::pow(R_zeta(k,j-1,i-1), 2.) + Kokkos::pow(R_zeta(k,j,i-1), 2.)
                          + Kokkos::pow(R_zeta(k,j-1,  i), 2.) + Kokkos::pow(R_zeta(k,j,  i), 2.)
                          + Kokkos::pow(R_eta(k,  j,i-1), 2.) + Kokkos::pow(R_eta(k,  j,i), 2.)
                          + Kokkos::pow(R_eta(k-1,j,i-1), 2.) + Kokkos::pow(R_eta(k-1,j,i), 2.)
                          + Kokkos::pow(R_xi(k,  j-1,i), 2.) + Kokkos::pow(R_xi(k,  j,i), 2.)
                          + Kokkos::pow(R_xi(k-1,j-1,i), 2.) + Kokkos::pow(R_xi(k-1,j,i), 2.);
            TERM1 = 0.25 * TERM1 + 2.0 * (du_dx*du_dx + dv_dy*dv_dy + dw_dz*dw_dz);
            
            if (ITYPEW(k,j,i) != 1) TERM1 = 0.;


            // N^2
            // double DDY_top = grav*flex_height_coef_up(k) * (th(k+1,j,i)-th(k,j,i)) * rdz
            //              / (th(k+1,j,i)+th(k,j,i)) * mask_view(k, j, i).u_top();;
            //
            // double DDY_bot = grav*flex_height_coef_up(k-1) * (th(k,j,i)-th(k-1,j,i)) * rdz
            //              / (th(k,j,i)+th(k-1,j,i)) * mask_view(k, j, i).u_bot();


            double DDY_top = grav*flex_height_coef_up(k) * (th(k+1,j,i)-th(k,j,i)) * rdz
                         / (th(k+1,j,i)+th(k,j,i)) * DHWW1(k, j, i);;

            double DDY_bot = grav*flex_height_coef_up(k-1) * (th(k,j,i)-th(k-1,j,i)) * rdz
                         / (th(k,j,i)+th(k-1,j,i)) * DHWW2(k, j, i);

            double DDY = DDY_top + DDY_bot;


            // Mixing Length
            double z = z_mid(k);
            double ZROUGH = 2e-4;
            double DDX = (ramd0s * vk*vk*Kokkos::pow(z+ZROUGH, 2.)) / (ramd0s + vk*vk*Kokkos::pow(z+ZROUGH, 2.));

            // Richardson Number
            double Ri = DDY / TERM1;

            // E. Km, Kh
            double rkm_val = 0.0;
            double rkh_val = 0.0;
            double sqrt_TERM1 = Kokkos::sqrt(TERM1);

            if (TERM1 == 0.) {
                rkm_val = 0.;
                rkh_val = 0.;
            }
            else {
                if (Ri < 0.0) { 
                    rkm_val = sqrt_TERM1 * DDX * Kokkos::sqrt(1.0 - 16.0 * Ri);
                    rkh_val = sqrt_TERM1 * DDX * 1.4 * Kokkos::sqrt(1.0 - 40.0 * Ri);
                } 
                else if (Ri < 0.25) { 
                    rkm_val = sqrt_TERM1 * DDX * Kokkos::pow(1.-4.*Ri, 4.);
                    rkh_val = sqrt_TERM1 * DDX * 1.4 * (1.0 - 1.2 * Ri) * Kokkos::pow(1.-4.*Ri, 4.); 
                } 
                else { 
                    rkm_val = 0.0;
                    rkh_val = 0.0;
                }
            }

            // Limiter
            rkm_val = Kokkos::max(rkm_val, critmn);
            rkh_val = Kokkos::max(rkh_val, critmn);
            rkm_val = Kokkos::min(rkm_val, critmx);
            rkh_val = Kokkos::min(rkh_val, critmx);

            if (ITYPEW(k,j,i) != 1) {
                rkh_val = 0.; 
                rkm_val = 0.; 
            }

            rkm(k, j, i) = rkm_val;
            rkh(k, j, i) = rkh_val;
        }
    );
    Kokkos::deep_copy(rkm, 100.);
    Kokkos::deep_copy(rkh, 100.);

    halo_exchanger_.exchange_halos(state.get_field<3>("RKM"));
    halo_exchanger_.exchange_halos(state.get_field<3>("RKH"));


    // This is for debugging
    double local_max_rkm = 0.0;
    double local_max_rkh = 0.0;

    Kokkos::parallel_reduce("Find_Max_RKM",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i, double& lmax) {
            if (rkm(k, j, i) > lmax) lmax = rkm(k, j, i);
        },
        Kokkos::Max<double>(local_max_rkm)
    );

    Kokkos::parallel_reduce("Find_Max_RKH",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int k, const int j, const int i, double& lmax) {
            if (rkh(k, j, i) > lmax) lmax = rkh(k, j, i);
        },
        Kokkos::Max<double>(local_max_rkh)
    );

    double global_max_rkm = 0.0;
    double global_max_rkh = 0.0;

    MPI_Reduce(&local_max_rkm, &global_max_rkm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_rkh, &global_max_rkh, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (grid_.get_mpi_rank() == 0) {
        std::cout << "[Turbulence] Step " << state.get_step() 
                  << " | Max RKM: " << global_max_rkm 
                  << " | Max RKH: " << global_max_rkh << std::endl;
    }
}

template<size_t Dim>
void TurbulenceProcess::calculate_tendencies(Core::State& state, 
                                             const std::string& var_name, 
                                             Core::Field<Dim>& out_tendency) 
{
    const auto& RKM = state.get_field<3>("RKM").get_device_data();
    const auto& RKH = state.get_field<3>("RKH").get_device_data();
    const auto& var = state.get_field<3>(var_name).get_device_data();
    auto& tend = out_tendency.get_mutable_device_data();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& hx     = state.get_field<2>("topo").get_device_data();
    const auto& hxv     = state.get_field<2>("topov").get_device_data();

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();
    
    const double rdx2 = rdx2_;
    const double rdy2 = rdy2_;
    const double rdz2 = rdz2_;

    auto DHUU1 = DHUU1_.get_device_data(); auto DHUU2 = DHUU2_.get_device_data();
    auto DHUV1 = DHUV1_.get_device_data(); auto DHUV2 = DHUV2_.get_device_data();
    auto DHUW1 = DHUW1_.get_device_data(); auto DHUW2 = DHUW2_.get_device_data();
    auto DHVU1 = DHVU1_.get_device_data(); auto DHVU2 = DHVU2_.get_device_data();
    auto DHVV1 = DHVV1_.get_device_data(); auto DHVV2 = DHVV2_.get_device_data();
    auto DHVW1 = DHVW1_.get_device_data(); auto DHVW2 = DHVW2_.get_device_data();
    auto DHWU1 = DHWU1_.get_device_data(); auto DHWU2 = DHWU2_.get_device_data();
    auto DHWV1 = DHWV1_.get_device_data(); auto DHWV2 = DHWV2_.get_device_data();
    auto DHWW1 = DHWW1_.get_device_data(); auto DHWW2 = DHWW2_.get_device_data();

    int NK2 = nz-h-1;
    int NK1 = nz-h-2;
    const auto& ITYPEW = state.get_field<3>("ITYPEW").get_device_data();
    auto step = state.get_step();
    if constexpr (Dim == 3) {
        if (var_name == "xi") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h-1, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    // double d2dx2 = ((RKM(k,  j,i)+RKM(k,  j,i+1)+RKM(k,  j+1,i)+RKM(k,  j+1,i+1)
                    //                 +RKM(k+1,j,i)+RKM(k+1,j,i+1)+RKM(k+1,j+1,i)+RKM(k+1,j+1,i+1))
                    //                     *(var(k,j,i+1)-var(k,j,i))*mask_view(k,j,i).v_right() -
                    //                 (RKM(k,  j,i-1)+RKM(k,  j,i)+RKM(k,j+1,i-1)+RKM(k,j+1,i)
                    //                 +RKM(k+1,j,i-1)+RKM(k+1,j,i)+RKM(k+1,j+1,i-1)+RKM(k+1,j+1,i)) 
                    //                     *(var(k,j,i)-var(k,j,i-1))*mask_view(k,j,i).v_left() )
                    //                 * 0.125 * rdx2;
                    // 
                    // double d2dy2 = ((RKM(k,j+1,i)+RKM(k+1,j+1,i))*(var(k,j+1,i)-var(k,j,  i))*mask_view(k,j,i).v_front()
                    //                -(RKM(k,j,  i)+RKM(k+1,j,  i))*(var(k,j,  i)-var(k,j-1,i))*mask_view(k,j,i).v_back())
                    //                 * 0.5*rdy2;
                    //
                    // double d2dz2 = (flex_height_coef_mid(k+1)*rhobar(k+1)*(RKM(k+1,j,i)+RKM(k+1,j+1,i))
                    //                     *(var(k+1,j,i)-var(k,  j,i))*mask_view(k,j,i).v_top()
                    //               - flex_height_coef_mid(k)*rhobar(k)*(RKM(k,j,i)+RKM(k,j+1,i))
                    //                     *(var(k,  j,i)-var(k-1,j,i))*mask_view(k,j,i).v_bot())
                    //                 *0.5 * rdz2 / (rhobar_up(k))*flex_height_coef_up(k);


                    double d2dx2 = ((RKM(k,  j,i)+RKM(k,  j,i+1)+RKM(k,  j+1,i)+RKM(k,  j+1,i+1)
                                    +RKM(k+1,j,i)+RKM(k+1,j,i+1)+RKM(k+1,j+1,i)+RKM(k+1,j+1,i+1))
                                        *(var(k,j,i+1)-var(k,j,i))*DHVU1(k,j,i) -
                                    (RKM(k,  j,i-1)+RKM(k,  j,i)+RKM(k,j+1,i-1)+RKM(k,j+1,i)
                                    +RKM(k+1,j,i-1)+RKM(k+1,j,i)+RKM(k+1,j+1,i-1)+RKM(k+1,j+1,i)) 
                                        *(var(k,j,i)-var(k,j,i-1))*DHVU2(k,j,i) )
                                    * 0.125 * rdx2;
                    
                    double d2dy2 = ((RKM(k,j+1,i)+RKM(k+1,j+1,i))*(var(k,j+1,i)-var(k,j,  i))*DHVV1(k,j,i)
                                   -(RKM(k,j,  i)+RKM(k+1,j,  i))*(var(k,j,  i)-var(k,j-1,i))*DHVV2(k,j,i))
                                    * 0.5*rdy2;

                    double d2dz2 = (flex_height_coef_mid(k+1)*rhobar(k+1)*(RKM(k+1,j,i)+RKM(k+1,j+1,i))
                                        *(var(k+1,j,i)-var(k,  j,i))*DHVW1(k,j,i)
                                  - flex_height_coef_mid(k)*rhobar(k)*(RKM(k,j,i)+RKM(k,j+1,i))
                                        *(var(k,  j,i)-var(k-1,j,i))*DHVW2(k,j,i))
                                    *0.5 * rdz2 / (rhobar_up(k))*flex_height_coef_up(k);

                    tend(k,j,i) = d2dx2 + d2dy2 + d2dz2;
                }
            );

            // This is for surface flux
            /*
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    int hxp = hx(j,i) + 1;
                    int hxvp = hxv(j,i) + 1;

                    if (hxvp > 1) tend(hxvp,j,i) = tend(hxvp,j,i)+flex_height_coef_up(hxvp)*flex_height_coef_mid(hxvp)*WV(I,J)/(RHOU(hxvp)*DZ*DZ)

                }
            );
            */
        }
        else if (var_name == "eta") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h-1, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    
                    // double d2dx2 = ((RKM(k,j,i+1)+RKM(k+1,j,i+1))*(var(k,j,i+1)-var(k,j,i  )) * mask_view(k,j,i).u_right()
                    //                -(RKM(k,j,i  )+RKM(k+1,j,i  ))*(var(k,j,i  )-var(k,j,i-1)) * mask_view(k,j,i).u_left())
                    //                 * 0.5 * rdx2;
                    //
                    // double d2dy2 = ((RKM(k,j,i)+RKM(k,j,i+1)+RKM(k,j+1,i)+RKM(k,j+1,i+1) 
                    //                 +RKM(k+1,j,i)+RKM(k+1,j,i+1)+RKM(k+1,j+1,i)+RKM(k+1,j+1,i+1))
                    //                             *(var(k,j+1,i)-var(k,j,i))*mask_view(k,j,i).u_front() -      
                    //                 (RKM(k,j-1,i)+RKM(k,j-1,i+1)+RKM(k,j,i)+RKM(k,j,i+1) 
                    //                 +RKM(k+1,j-1,i)+RKM(k+1,j-1,i+1)+RKM(k+1,j,i)+RKM(k+1,j,i+1))
                    //                             *(var(k,j,i)-var(k,j-1,i))*mask_view(k,j,i).u_back())
                    //                 * 0.125 * rdy2;
                    //
                    //
                    // double d2dz2 = (flex_height_coef_mid(k+1)*rhobar(k+1)*(RKM(k+1,j,i)+RKM(k+1,j,i+1))  
                    //                    *(var(k+1,j,i)-var(k,  j,i))*mask_view(k,j,i).u_top()
                    //                -flex_height_coef_mid(k)*rhobar(k)*(RKM(k,j,i)+RKM(k,j,i+1))
                    //                    *(var(k,  j,i)-var(k-1,j,i))*mask_view(k,j,i).u_bot())
                    //                 * 0.5 * rdz2 / rhobar_up(k) * flex_height_coef_up(k);
                                  

                    double d2dx2 = ((RKM(k,j,i+1)+RKM(k+1,j,i+1))*(var(k,j,i+1)-var(k,j,i  )) * DHUU1(k,j,i)
                                   -(RKM(k,j,i  )+RKM(k+1,j,i  ))*(var(k,j,i  )-var(k,j,i-1)) * DHUU2(k,j,i))
                                    * 0.5 * rdx2;

                    double d2dy2 = ((RKM(k,j,i)+RKM(k,j,i+1)+RKM(k,j+1,i)+RKM(k,j+1,i+1) 
                                    +RKM(k+1,j,i)+RKM(k+1,j,i+1)+RKM(k+1,j+1,i)+RKM(k+1,j+1,i+1))
                                                *(var(k,j+1,i)-var(k,j,i))*DHUV1(k,j,i) -      
                                    (RKM(k,j-1,i)+RKM(k,j-1,i+1)+RKM(k,j,i)+RKM(k,j,i+1) 
                                    +RKM(k+1,j-1,i)+RKM(k+1,j-1,i+1)+RKM(k+1,j,i)+RKM(k+1,j,i+1))
                                                *(var(k,j,i)-var(k,j-1,i))*DHUV2(k,j,i))
                                    * 0.125 * rdy2;


                    double d2dz2 = (flex_height_coef_mid(k+1)*rhobar(k+1)*(RKM(k+1,j,i)+RKM(k+1,j,i+1))  
                                       *(var(k+1,j,i)-var(k,  j,i))*DHUW1(k,j,i)
                                   -flex_height_coef_mid(k)*rhobar(k)*(RKM(k,j,i)+RKM(k,j,i+1))
                                       *(var(k,  j,i)-var(k-1,j,i))*DHUW2(k,j,i))
                                    * 0.5 * rdz2 / rhobar_up(k) * flex_height_coef_up(k);
                    tend(k,j,i) = d2dx2 + d2dy2 + d2dz2;
                }
            );
        }
        else {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    // double d2dx2 = 0.5*( (RKH(k,j,i+1)+RKH(k,j,i))
                    //                         *(var(k,j,i+1)-var(k,j,i))*mask_view(k,j,i).w_right()
                    //                    - (RKH(k,j,i)+RKH(k,j,i-1))
                    //                         *(var(k,j,i)-var(k,j,i-1))*mask_view(k,j,i).w_left() ) * rdx2;
                    //
                    // double d2dy2 = 0.5*( (RKH(k,j+1,i)+RKH(k,j,i))
                    //                         *(var(k,j+1,i)-var(k,j  ,i))*mask_view(k,j,i).w_front()
                    //                    - (RKH(k,j,i)+RKH(k,j-1,i))
                    //                         *(var(k,j  ,i)-var(k,j-1,i))*mask_view(k,j,i).w_back() ) * rdy2;
                    //  
                    // double d2dz2 = 0.;
                    //
                    // if (k != nz-h-1) {
                    //     d2dz2 =  0.5 * flex_height_coef_mid(k)*(flex_height_coef_up(k)*rhobar_up(k)*(RKH(k+1,j,i)+RKH(k,j,i))
                    //                         *(var(k+1,j,i)-var(k,j,i))*mask_view(k,j,i).w_top()
                    //                   -flex_height_coef_up(k-1)*rhobar_up(k-1)*(RKH(k,j,i)+RKH(k-1,j,i))
                    //                         *(var(k,j,i)-var(k-1,j,i))*mask_view(k,j,i).w_bot() ) / flex_height_coef_mid(k) * rdz2;
                    // }
                    // else {
                    //     d2dz2 = -0.5*flex_height_coef_mid(NK2)*(flex_height_coef_up(NK1)*rhobar_up(NK1)*(RKH(NK2,j,i)+RKH(NK1,j,i))
                    //                 *(var(NK2,j,i)-var(NK1,j,i))) / rhobar(NK2) * rdz2; 
                    // }
                     

                    double d2dx2 = 0.5*( (RKH(k,j,i+1)+RKH(k,j,i))
                                            *(var(k,j,i+1)-var(k,j,i))*DHWU1(k,j,i)
                                       - (RKH(k,j,i)+RKH(k,j,i-1))
                                            *(var(k,j,i)-var(k,j,i-1))*DHWU2(k,j,i) ) * rdx2;

                    double d2dy2 = 0.5*( (RKH(k,j+1,i)+RKH(k,j,i))
                                            *(var(k,j+1,i)-var(k,j  ,i))*DHWV1(k,j,i)
                                       - (RKH(k,j,i)+RKH(k,j-1,i))
                                            *(var(k,j  ,i)-var(k,j-1,i))*DHWV2(k,j,i) ) * rdy2;
                     
                    double d2dz2 = 0.;

                    if (k == nz-h-1) {
                        d2dz2 = -0.5*flex_height_coef_mid(NK2)*(flex_height_coef_up(NK1)*rhobar_up(NK1)*(RKH(NK2,j,i)+RKH(NK1,j,i))
                                    *(var(NK2,j,i)-var(NK1,j,i))) / rhobar(NK2) * rdz2; 
                    }
                    else {
                        d2dz2 =  0.5 * flex_height_coef_mid(k)*(
                                       flex_height_coef_up(k)*rhobar_up(k)*(RKH(k+1,j,i)+RKH(k,j,i))
                                            *(var(k+1,j,i)-var(k,j,i))*DHWW1(k,j,i)
                                      -flex_height_coef_up(k-1)*rhobar_up(k-1)*(RKH(k,j,i)+RKH(k-1,j,i))
                                            *(var(k,j,i)-var(k-1,j,i))*DHWW2(k,j,i) ) / rhobar(k) * rdz2;
                    }

                    tend(k,j,i) = d2dx2 + d2dy2 + d2dz2;
                    // tend(k,j,i) = d2dz2;
                }
            );
        }
    }
    else if constexpr (Dim == 2) {
        if (var_name == "zeta") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                
                    double d2dx2 = ((RKM(NK2,j,i+1)+RKM(NK2,j+1,i+1))*(var(NK2,j,i+1)-var(NK2,j,i  ))
                                   -(RKM(NK2,j,i  )+RKM(NK2,j+1,i  ))*(var(NK2,j,i  )-var(NK2,j,i-1)))  
                                    * 0.5 * rdx2;

                    double d2dy2 = ((RKM(NK2,j+1,i)+RKM(NK2,j+1,i+1))*(var(NK2,j+1,i)-var(NK2,j,  i))
                                   -(RKM(NK2,j  ,i)+RKM(NK2,j,  i+1))*(var(NK2,j  ,i)-var(NK2,j-1,i)))
                                    * 0.5 * rdy2;

                    double d2dz2 = -(flex_height_coef_up(NK1)*rhobar_up(NK1)*
                                     (RKM(NK2,j,i)+RKM(NK2,j,i+1)+RKM(NK2,j+1,i)+RKM(NK2,j+1,i+1) 
                                     +RKM(NK1,j,i)+RKM(NK1,j,i+1)+RKM(NK1,j+1,i)+RKM(NK1,j+1,i+1))
                                    *(var(NK2,j,i)-var(NK1,j,i))) * 0.125 * rdz2 / rhobar(NK2) * flex_height_coef_mid(NK2); 

                    tend(j,i) = d2dx2 + d2dy2 + d2dz2;
                }
            );
        }
    }

}

void TurbulenceProcess::process_thermodynamics(Core::State& state, double dt) {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();

    for (const auto& var_name : thermodynamics_vars_) {
        auto& var_data = state.get_field<3>(var_name).get_mutable_device_data();
        temp3d_tendency_.initialize_to_zero();
        calculate_tendencies(state, var_name, temp3d_tendency_);
        const auto& tend_data = temp3d_tendency_.get_device_data(); 

        Kokkos::parallel_for("Turbulence_Update_" + var_name,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                var_data(k, j, i) += dt * tend_data(k, j, i);
            }
        );
        halo_exchanger_.exchange_halos(state.get_field<3>(var_name));
        // TODO: This is a temporary soution. Make it a method for boundary process
        if (var_name == "th" || var_name == "qv") {
            Kokkos::parallel_for("Boundary" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{ny, nx}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    var_data(h-1, j, i) = var_data(h, j, i);
                    var_data(nz-h, j, i) = var_data(nz-h-1, j, i);
                }
            );
        }
        else {
            Kokkos::parallel_for("Boundary" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{ny, nx}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    var_data(h-1, j, i) = var_data(h, j, i);
                    var_data(nz-h, j, i) = 0.;
                }
            );
        }
    }
}

void TurbulenceProcess::process_dynamics(Core::State& state, double dt) {
    const int h = grid_.get_halo_cells();
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    int NK2 = nz-h-1;

    for (const auto& var_name : dynamics_vars_) {
        auto& var_data = state.get_field<3>(var_name).get_mutable_device_data();
        if (var_name == "zeta") {
            calculate_tendencies(state, var_name, temp2d_tendency_);
            const auto& tend_data = temp2d_tendency_.get_device_data(); 

            Kokkos::parallel_for("Turbulence_Update_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    var_data(NK2, j, i) += dt * tend_data(j, i);
                }
            );
        }
        else {
            calculate_tendencies(state, var_name, temp3d_tendency_);
            const auto& tend_data = temp3d_tendency_.get_device_data(); 

            Kokkos::parallel_for("Turbulence_Update_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h-1, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    var_data(k, j, i) += dt * tend_data(k, j, i);
                }
            );
        }
        halo_exchanger_.exchange_halos(state.get_field<3>(var_name));
        if (var_name == "xi" || var_name == "eta") {
            Kokkos::parallel_for("Boundary" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{0, 0}}, {{ny, nx}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    var_data(h-1, j, i) = 0.;
                    var_data(nz-h-1, j, i) = 0.;
                }
            );
        }
    }
}

} // namespace Physics
} // namespace VVM
