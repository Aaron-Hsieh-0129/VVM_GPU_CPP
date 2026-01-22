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
    thermodynamics_vars_ = {"th", "qv"};
    if (config.get_value<bool>("physics.p3.enable_p3", false)) {
        thermodynamics_vars_.insert(thermodynamics_vars_.end(), {"qc", "qr", "qi", "nc", "nr", "ni", "bm", "qm"});
    }
}


void TurbulenceProcess::initialize(Core::State& state) {
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    auto dims = std::array<int, 3>{nz, ny, nx};
    for (const auto& var_name : thermodynamics_vars_) {
        std::string fe_tendency_name = "fe_tendency_" + var_name;
        if (!state.has_field(fe_tendency_name)) {
            state.add_field<3>(fe_tendency_name, dims);
        }
    }
    for (const auto& var_name : dynamics_vars_) {
        std::string fe_tendency_name = "fe_tendency_" + var_name;
        if (!state.has_field(fe_tendency_name)) {
            if (var_name == "zeta") state.add_field<2>(fe_tendency_name, {ny, nx});
            else state.add_field<3>(fe_tendency_name, dims);
        }
    }
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

    init_boundary_masks(state);
    return;
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

    const auto& masks = masks_;
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
            double DDY_top = grav*flex_height_coef_up(k) * (th(k+1,j,i)-th(k,j,i)) * rdz
                         / (th(k+1,j,i)+th(k,j,i)) * masks.val(k,j,i,WW1);

            double DDY_bot = grav*flex_height_coef_up(k-1) * (th(k,j,i)-th(k-1,j,i)) * rdz
                         / (th(k,j,i)+th(k-1,j,i)) * masks.val(k,j,i,WW2);

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

    halo_exchanger_.exchange_halos(state.get_field<3>("RKM"));
    halo_exchanger_.exchange_halos(state.get_field<3>("RKH"));
}

template<size_t Dim>
void TurbulenceProcess::calculate_tendencies(Core::State& state, 
                                             const std::string& var_name, 
                                             Core::Field<Dim>& out_tendency) 
{
    VVM::Utils::Timer turbulence_timer("Turbulence");
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

    const auto masks = masks_;

    int NK2 = nz-h-1;
    int NK1 = nz-h-2;
    const auto& ITYPEW = state.get_field<3>("ITYPEW").get_device_data();
    auto step = state.get_step();
    if constexpr (Dim == 3) {
        if (var_name == "xi") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h-1, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    double d2dx2 = ((RKM(k,  j,i)+RKM(k,  j,i+1)+RKM(k,  j+1,i)+RKM(k,  j+1,i+1)
                                    +RKM(k+1,j,i)+RKM(k+1,j,i+1)+RKM(k+1,j+1,i)+RKM(k+1,j+1,i+1))
                                        *(var(k,j,i+1)-var(k,j,i))*masks.val(k, j, i, VU1) -
                                    (RKM(k,  j,i-1)+RKM(k,  j,i)+RKM(k,j+1,i-1)+RKM(k,j+1,i)
                                    +RKM(k+1,j,i-1)+RKM(k+1,j,i)+RKM(k+1,j+1,i-1)+RKM(k+1,j+1,i)) 
                                        *(var(k,j,i)-var(k,j,i-1))*masks.val(k, j, i, VU2) )
                                    * 0.125 * rdx2;
                    
                    double d2dy2 = ((RKM(k,j+1,i)+RKM(k+1,j+1,i))*(var(k,j+1,i)-var(k,j,  i))*masks.val(k, j, i, VV1)
                                   -(RKM(k,j,  i)+RKM(k+1,j,  i))*(var(k,j,  i)-var(k,j-1,i))*masks.val(k, j, i, VV2))
                                    * 0.5*rdy2;

                    double d2dz2 = (flex_height_coef_mid(k+1)*rhobar(k+1)*(RKM(k+1,j,i)+RKM(k+1,j+1,i))
                                        *(var(k+1,j,i)-var(k,  j,i))*masks.val(k, j, i, VW1)
                                  - flex_height_coef_mid(k)*rhobar(k)*(RKM(k,j,i)+RKM(k,j+1,i))
                                        *(var(k,  j,i)-var(k-1,j,i))*masks.val(k, j, i, VW2))
                                    *0.5 * rdz2 / (rhobar_up(k))*flex_height_coef_up(k);

                    tend(k,j,i) = d2dx2 + d2dy2 + d2dz2;
                }
            );
        }
        else if (var_name == "eta") {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h-1, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    double d2dx2 = ((RKM(k,j,i+1)+RKM(k+1,j,i+1))*(var(k,j,i+1)-var(k,j,i  )) * masks.val(k, j, i, UU1)
                                   -(RKM(k,j,i  )+RKM(k+1,j,i  ))*(var(k,j,i  )-var(k,j,i-1)) * masks.val(k, j, i, UU2))
                                    * 0.5 * rdx2;

                    double d2dy2 = ((RKM(k,j,i)+RKM(k,j,i+1)+RKM(k,j+1,i)+RKM(k,j+1,i+1) 
                                    +RKM(k+1,j,i)+RKM(k+1,j,i+1)+RKM(k+1,j+1,i)+RKM(k+1,j+1,i+1))
                                                *(var(k,j+1,i)-var(k,j,i))*masks.val(k, j, i, UV1) -      
                                    (RKM(k,j-1,i)+RKM(k,j-1,i+1)+RKM(k,j,i)+RKM(k,j,i+1) 
                                    +RKM(k+1,j-1,i)+RKM(k+1,j-1,i+1)+RKM(k+1,j,i)+RKM(k+1,j,i+1))
                                                *(var(k,j,i)-var(k,j-1,i))*masks.val(k, j, i, UV2))
                                    * 0.125 * rdy2;


                    double d2dz2 = (flex_height_coef_mid(k+1)*rhobar(k+1)*(RKM(k+1,j,i)+RKM(k+1,j,i+1))  
                                       *(var(k+1,j,i)-var(k,  j,i))*masks.val(k, j, i, UW1)
                                   -flex_height_coef_mid(k)*rhobar(k)*(RKM(k,j,i)+RKM(k,j,i+1))
                                       *(var(k,  j,i)-var(k-1,j,i))*masks.val(k, j, i, UW2))
                                    * 0.5 * rdz2 / rhobar_up(k) * flex_height_coef_up(k);

                    tend(k,j,i) = d2dx2 + d2dy2 + d2dz2;
                }
            );
        }
        else {
            Kokkos::parallel_for("Compute_Diff_Tendency_" + var_name,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({{h, h, h}}, {{nz-h, ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    double d2dx2 = 0.5*( (RKH(k,j,i+1)+RKH(k,j,i))
                                            *(var(k,j,i+1)-var(k,j,i))*masks.val(k, j, i, WU1)
                                       - (RKH(k,j,i)+RKH(k,j,i-1))
                                            *(var(k,j,i)-var(k,j,i-1))*masks.val(k, j, i, WU2) ) * rdx2;

                    double d2dy2 = 0.5*( (RKH(k,j+1,i)+RKH(k,j,i))
                                            *(var(k,j+1,i)-var(k,j  ,i))*masks.val(k, j, i, WV1)
                                       - (RKH(k,j,i)+RKH(k,j-1,i))
                                            *(var(k,j  ,i)-var(k,j-1,i))*masks.val(k, j, i, WV2) ) * rdy2;
                     
                    double d2dz2 = 0.;

                    if (k == nz-h-1) {
                        d2dz2 = -0.5*flex_height_coef_mid(NK2)*(flex_height_coef_up(NK1)*rhobar_up(NK1)*(RKH(NK2,j,i)+RKH(NK1,j,i))
                                    *(var(NK2,j,i)-var(NK1,j,i))) / rhobar(NK2) * rdz2; 
                    }
                    else {
                        d2dz2 =  0.5 * flex_height_coef_mid(k)*(
                                       flex_height_coef_up(k)*rhobar_up(k)*(RKH(k+1,j,i)+RKH(k,j,i))
                                            *(var(k+1,j,i)-var(k,j,i))*masks.val(k, j, i, WW1)
                                      -flex_height_coef_up(k-1)*rhobar_up(k-1)*(RKH(k,j,i)+RKH(k-1,j,i))
                                            *(var(k,j,i)-var(k-1,j,i))*masks.val(k, j, i, WW2) ) / rhobar(k) * rdz2;
                    }
                    tend(k,j,i) = d2dx2 + d2dy2 + d2dz2;
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

template void TurbulenceProcess::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<2ul>& out_tendency);
template void TurbulenceProcess::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Physics
} // namespace VVM
