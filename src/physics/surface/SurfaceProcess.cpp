#include "SurfaceProcess.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace VVM {
namespace Physics {

SurfaceProcess::SurfaceProcess(const Utils::ConfigurationManager& config, 
                                     const Core::Grid& grid, 
                                     const Core::Parameters& params,
                                     Core::HaloExchanger& halo_exchanger,
                                     Core::State& state)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger) {

    v_coord_type_ = config_.get_value<std::string>("grid.vertical_coordinate_type", "default");

    if (v_coord_type_ == "rcemip") speed1_filter_ = 1;
    else speed1_filter_ = 1e-3;
    return;
}

void SurfaceProcess::initialize(Core::State& state) {
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    
    if (!state.has_field("qc")) state.add_field<3>("qc", {nz, ny, nx});
    if (!state.has_field("qi")) state.add_field<3>("qi", {nz, ny, nx});

    if (!state.has_field("sfc_flux_th")) state.add_field<2>("sfc_flux_th", {ny, nx});
    if (!state.has_field("sfc_flux_qv")) state.add_field<2>("sfc_flux_qv", {ny, nx});
    if (!state.has_field("sfc_flux_u"))  state.add_field<2>("sfc_flux_u", {ny, nx});
    if (!state.has_field("sfc_flux_v"))  state.add_field<2>("sfc_flux_v", {ny, nx});
    
    if (!state.has_field("gwet")) state.add_field<2>("gwet", {ny, nx}); // Surface Wetness
    if (!state.has_field("zrough")) state.add_field<2>("zrough", {ny, nx}); // Roughness Length
    if (!state.has_field("VEN2D")) state.add_field<2>("VEN2D", {ny, nx}); // Roughness Length
    Kokkos::deep_copy(state.get_field<2>("gwet").get_mutable_device_data(), -1.);
    Kokkos::deep_copy(state.get_field<2>("zrough").get_mutable_device_data(), 2e-4);
    
    if (!state.has_field("ustar")) state.add_field<2>("ustar", {ny, nx});
    if (!state.has_field("molen")) state.add_field<2>("molen", {ny, nx});

    if (!state.has_field("sea_land_ice_mask")) state.add_field<2>("sea_land_ice_mask", {ny, nx});

    mode_ = config_.get_value<std::string>("physics.surface_process.ocean_scheme", "none");
    land_scheme_ = config_.get_value<std::string>("physics.surface_process.land_scheme", "none");
}

KOKKOS_INLINE_FUNCTION
VVM::Real SurfaceProcess::compute_es(VVM::Real t) {
    VVM::Real es = real(611.2) * Kokkos::exp(real(17.67) * (t - real(273.15)) / (t - real(29.65)));
    return es; 
}

KOKKOS_INLINE_FUNCTION
void SurfaceProcess::sflux_2d(VVM::Real sigmau, VVM::Real thvm, VVM::Real thvsm, VVM::Real speed1, 
                              VVM::Real zr, VVM::Real zrough, VVM::Real speed1_filter, 
                              VVM::Real& ustar, VVM::Real ventfc[2], VVM::Real& molen) {
    const VVM::Real bus = real(0.74);
    const VVM::Real crit = real(0.003);
    const int maxit = 5;

    bool stopit = false;
    VVM::Real speedm = (speed1 > speed1_filter) ? speed1 : speed1_filter;

    const VVM::Real vk = real(0.4);
    const VVM::Real pi = real(3.141592653589793);
    VVM::Real grav = real(9.806);

    VVM::Real tem1 = Kokkos::log(zr / zrough);
    VVM::Real cuni = tem1 / vk;
    VVM::Real ctni = cuni * bus;

    bool stable = (thvsm < 0.0);

    // START ITERATION
    int it = 0;
    VVM::Real cu = real(1.0) / cuni;
    VVM::Real ct = real(1.0) / ctni;

    if (!stable) speedm = (speedm > sigmau) ? speedm : sigmau;

    VVM::Real cui, cti;
    while (!stopit) {
        it++;
        VVM::Real zeta = -zr * ct * vk * grav * thvsm / (thvm * cu * cu * speedm * speedm);

        if (stable) {
            if (zeta >= real(2.45)) {
                stopit = true;
                zeta = real(2.45);
            }
            VVM::Real tem2 = tem1 + real(4.7) * zeta;
            VVM::Real tem3 = tem1 + real(4.7) / bus * zeta;

            cui = tem2 / vk;
            cti = bus * tem3 / vk;
        }
        else {
            // UNSTABLE OR NEUTRAL CASE
            VVM::Real x = Kokkos::pow(real(1.0) - real(15.0) * zeta, real(0.25));
            VVM::Real y = Kokkos::pow(real(1.0) - real(9.0) * zeta, real(0.25));

            VVM::Real tem2 = tem1 - (Kokkos::log((real(1.0) + x * x) / real(2.0))
                                + real(2.0) * Kokkos::log((real(1.0) + x) / real(2.0))
                                - real(2.0) * Kokkos::atan(x) + pi / real(2.0));
            
            VVM::Real tem3 = tem1 - real(2.0) * Kokkos::log((real(1.0) + y * y) / real(2.0));

            cui = tem2 / vk;
            cui = (cui > real(0.5) * cuni) ? cui : real(0.5) * cuni; // MAX(CUI, 0.5*CUNI)

            cti = bus * tem3 / vk;
            cti = (cti > real(0.3) * ctni) ? cti : real(0.3) * ctni; // MAX(CTI, 0.3*CTNI)
        }

        // STOPIT = STOPIT .OR. IT .EQ. MAXIT
        if (it == maxit) stopit = true;

        if (stopit) {
            cu = real(1.0) / cui;
            ct = real(1.0) / cti;
        } 
        else {
            // CHECK FOR CONVERGENCE
            VVM::Real custar = cu;
            VVM::Real ctstar = ct;
            cu = real(1.0) / cui;
            ct = real(1.0) / cti;
            
            if (Kokkos::abs(cu / custar - real(1.0)) <= crit && Kokkos::abs(ct / ctstar - real(1.0)) <= crit) {
                stopit = true;
            }
        }
    }

    // ITERATION COMPLETED. CALCULATE USTAR AND VENTFC
    ustar = cu * speedm;
    ventfc[0] = cu * ustar;
    ventfc[1] = ct * ustar;

    // CHECK TOWNSEND'S LIMIT (Unstable only)
    if (!stable && cti < real(0.3) * ctni) {
        ventfc[1] = Kokkos::max(ventfc[1], real(0.0019) * Kokkos::pow(thvsm, real(1.)/real(3.)) ); 
    }

    // MONIN-OBUKHOV LENGTH
    VVM::Real zeta = -zr * ct * vk * grav * thvsm / (thvm * cu * cu * speedm * speedm);
    zeta = Kokkos::max(Kokkos::abs(zeta), real(1e-6)) * Kokkos::copysign(real(1.), zeta);
    molen = zr / Kokkos::min(zeta, real(2.45));
}


KOKKOS_INLINE_FUNCTION
void SurfaceProcess::sflux_tc_2d(VVM::Real sigmau, VVM::Real thvm, VVM::Real thvsm, VVM::Real speed1, 
                                 VVM::Real zr, VVM::Real zrough, VVM::Real speed1_filter, 
                                 VVM::Real& ustar, VVM::Real ventfc[2], VVM::Real& molen) {
    const VVM::Real crit = real(0.003);
    const int maxit = 20;
    const VVM::Real verysmall = real(1.e-6);
    const VVM::Real z0m_min = real(1.27e-7);
    const VVM::Real z0m_max = real(2.85e-3);
    const VVM::Real z0s = real(1.0e-4); // scalar roughness (fixed)
    
    const VVM::Real vk = real(0.4);
    const VVM::Real pi = real(3.141592653589793);
    const VVM::Real grav = real(9.806);

    bool stopit = false;
    VVM::Real speedm = (speed1 > speed1_filter) ? speed1 : speed1_filter;

    bool stable = (thvsm < 0.0);

    // Unstable case (Gustiness floor for unstable case)
    if (!stable) speedm = (speedm > sigmau) ? speedm : sigmau;

    // Initial guess for coefficients
    VVM::Real z0m = Kokkos::max(Kokkos::min(zrough, z0m_max), z0m_min);

    VVM::Real den_m = Kokkos::log(zr / z0m);
    VVM::Real den_s = Kokkos::log(zr / z0s);

    VVM::Real cd = (vk / den_m) * (vk / den_m);
    VVM::Real cs = (vk * vk) / (den_m * den_s);

    int it = 0;
    bool cap_stable = false;
    bool converged = false;

    // START ITERATION
    while (!stopit) {
        it++;
        cap_stable = false;

        VVM::Real cd_old = cd;
        VVM::Real cs_old = cs;

        VVM::Real ustar_tmp = Kokkos::sqrt(Kokkos::max(cd_old, verysmall)) * speedm;

        // Update z0m (WRF-like high-wind formula)
        VVM::Real zw = Kokkos::min(real(1.0), Kokkos::pow(ustar_tmp / real(1.06), real(0.3)));
        VVM::Real z1 = real(0.011) * ustar_tmp * ustar_tmp / grav + real(1.59e-5);
        VVM::Real z2 = real(10.0) * Kokkos::exp(-real(9.5) * Kokkos::pow(ustar_tmp, -real(1.0)/real(3.0))) 
                  + real(1.65e-6) / Kokkos::max(ustar_tmp, real(0.01));

        z0m = (real(1.0) - zw) * z1 + zw * z2;
        z0m = Kokkos::max(Kokkos::min(z0m, z0m_max), z0m_min);

        // zeta = z / L
        VVM::Real zeta = -zr * cs_old * vk * grav * thvsm / 
                      (thvm * Kokkos::pow(Kokkos::max(cd_old, verysmall), real(1.5)) * speedm * speedm);

        VVM::Real psi_m, psi_h;
        if (stable) {
            // STABLE CASE
            if (zeta >= real(2.45)) {
                cap_stable = true;
                zeta = real(2.45);
            }
            psi_m = -real(4.7) * zeta;
            psi_h = -real(4.7) * zeta;
        } 
        else {
            // UNSTABLE OR NEUTRAL CASE
            VVM::Real x = Kokkos::pow(real(1.0) - real(15.0) * zeta, real(0.25));
            VVM::Real y = Kokkos::pow(real(1.0) - real(9.0) * zeta, real(0.25));

            psi_m = Kokkos::log((real(1.0) + x * x) / real(2.0)) 
                  + real(2.0) * Kokkos::log((real(1.0) + x) / real(2.0)) 
                  - real(2.0) * Kokkos::atan(x) + pi / real(2.0);

            psi_h = real(2.0) * Kokkos::log((real(1.0) + y * y) / real(2.0));
        }

        // Effective denominators
        VVM::Real den_m_neu = Kokkos::log(zr / z0m);
        VVM::Real den_s_neu = Kokkos::log(zr / z0s);
        den_m = den_m_neu - psi_m;
        den_s = den_s_neu - psi_h;

        if (!stable) {
            den_m = Kokkos::max(den_m, real(0.5) * den_m_neu);
            den_s = Kokkos::max(den_s, real(0.3) * den_s_neu);
        }

        // Update coefficients
        VVM::Real cd_new = Kokkos::pow(vk / den_m, real(2.0));
        VVM::Real cs_new = (vk * vk) / (den_m * den_s);

        cd_new = Kokkos::max(cd_new, verysmall);
        cs_new = Kokkos::max(cs_new, verysmall);

        // Convergence check
        VVM::Real res_cd = Kokkos::abs(cd_new / cd_old - real(1.0));
        VVM::Real res_cs = Kokkos::abs(cs_new / cs_old - real(1.0));

        converged = (res_cd <= crit) && (res_cs <= crit);
        stopit = cap_stable || converged || (it >= maxit);

        cd = cd_new;
        cs = cs_new;
    }

    // ITERATION COMPLETED. FINAL OUTPUTS
    ustar = Kokkos::sqrt(Kokkos::max(cd, verysmall)) * speedm;

    ventfc[0] = cd * speedm; // Cd * U
    ventfc[1] = cs * speedm; // Cs * U

    // FINAL MONIN-OBUKHOV LENGTH
    VVM::Real zeta = -zr * cs * vk * grav * thvsm / 
                  (thvm * Kokkos::pow(Kokkos::max(cd, verysmall), real(1.5)) * speedm * speedm);
                  
    zeta = Kokkos::max(Kokkos::abs(zeta), real(1e-6)) * Kokkos::copysign(real(1.0), zeta);
    molen = zr / Kokkos::min(zeta, real(2.45));
}


void SurfaceProcess::compute_coefficients(Core::State& state) {
    const auto& u = state.get_field<3>("u").get_device_data();
    const auto& v = state.get_field<3>("v").get_device_data();
    const auto& th = state.get_field<3>("th").get_device_data();
    const auto& qv = state.get_field<3>("qv").get_device_data();
    const auto& qc = state.get_field<3>("qc").get_device_data();
    const auto& qi = state.get_field<3>("qi").get_device_data();
    
    const auto& pbar = state.get_field<1>("pbar").get_device_data();
    const auto& pibar = state.get_field<1>("pibar").get_device_data();
    const auto& z_mid = params_.z_mid.get_device_data(); const auto& z_up = params_.z_up.get_device_data();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data();
    const auto& rhobar_up = state.get_field<1>("rhobar_up").get_device_data();
    const auto& thbar = state.get_field<1>("thbar").get_device_data();
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();

    const auto& Tg = state.get_field<2>("Tg").get_device_data();
    const auto& gwet = state.get_field<2>("gwet").get_device_data();
    const auto& zrough = state.get_field<2>("zrough").get_device_data();
    const auto& sea_land_ice_mask = state.get_field<2>("sea_land_ice_mask").get_device_data();

    auto& sfc_flux_th = state.get_field<2>("sfc_flux_th").get_mutable_device_data();
    auto& sfc_flux_qv = state.get_field<2>("sfc_flux_qv").get_mutable_device_data();
    auto& sfc_flux_u = state.get_field<2>("sfc_flux_u").get_mutable_device_data();
    auto& sfc_flux_v = state.get_field<2>("sfc_flux_v").get_mutable_device_data();
    auto& ustar_view = state.get_field<2>("ustar").get_mutable_device_data();
    auto& molen_view = state.get_field<2>("molen").get_mutable_device_data();
    auto& VEN2D = state.get_field<2>("VEN2D").get_mutable_device_data();

    const auto& hx     = state.get_field<2>("topo").get_device_data();
    const auto& hxu    = state.get_field<2>("topou").get_device_data();
    const auto& hxv    = state.get_field<2>("topov").get_device_data();

    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells(); 
    const auto& dz = params_.dz; 
    
    // Constants from 
    const VVM::Real cp = real(1004.5);
    const auto& grav = params_.gravity;
    const VVM::Real hlf = real(2.5e6);
    const VVM::Real hlm = real(3.336e5);
    const VVM::Real delta = real(0.608);

    const VVM::Real local_speed1_filter = this->speed1_filter_;
    if (mode_ == "sflux_2d") {
        Kokkos::parallel_for("SFlux_3D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // If not sea, return
                if (sea_land_ice_mask(j,i) != 0) return;

                // NOTE: Need to check about the difference between original VVM and this
                int hx1 = hx(j,i);
                int hxp = hx(j,i)+1;

                VVM::Real ztmp = real(0.5) * dz() / flex_height_coef_mid(hxp);
                VVM::Real speedtp = real(0.5) * Kokkos::sqrt(
                                Kokkos::pow(u(hxp,j,i-1) + u(hxp,j,i), real(2)) + 
                                Kokkos::pow(v(hxp,j-1,i) + v(hxp,j,i), real(2))
                             );

                // TODO: check compute_es
                VVM::Real es1 = compute_es(Tg(j, i)); 
                VVM::Real qsfc = es1 * real(0.622) / (pbar(hx1) - es1);
                VVM::Real ts = cp * Tg(j, i) + grav() * z_up(hx1); 

                VVM::Real Q = qv(hxp, j, i) + qc(hxp, j, i) + qi(hxp, j, i);
                VVM::Real T = cp * th(hxp, j, i) * pibar(hxp) 
                             - hlf * qc(hxp, j, i)
                             + grav() * z_mid(hxp)
                             - (hlf + hlm) * qi(hxp, j, i);
                VVM::Real thvsm = Tg(j,i) / pibar(hx1) - th(hxp,j,i) + 
                               Kokkos::abs(gwet(j,i))*thbar(hxp) * (delta * (qsfc-qv(hxp,j,i)));

                VVM::Real sigmau = real(0.0);
                VVM::Real ustar, molen;
                VVM::Real ventfc[2];
                sflux_2d(sigmau, thbar(hxp), thvsm, speedtp, ztmp, zrough(j, i), local_speed1_filter, ustar, ventfc, molen);

                VVM::Real wt = ventfc[1] * (ts - T);
                VVM::Real wq = ventfc[1] * Kokkos::abs(gwet(j, i)) * (qsfc - Q);
                VEN2D(j,i) = ventfc[0];

                sfc_flux_qv(j,i) = wq * rhobar_up(hx1);
                sfc_flux_th(j,i) = wt * rhobar_up(hx1) / (cp * pibar(hx1));
            }
        );
    }
    else if (mode_ == "sflux_tc_2d") {
        Kokkos::parallel_for("SFlux_3D",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // If not sea, return
                if (sea_land_ice_mask(j,i) != 0) return;

                // NOTE: Need to check about the difference between original VVM and this
                int hx1 = hx(j,i);
                int hxp = hx(j,i)+1;

                VVM::Real ztmp = real(0.5) * dz() / flex_height_coef_mid(hxp);
                VVM::Real speedtp = real(0.5) * Kokkos::sqrt(
                                Kokkos::pow(u(hxp,j,i-1) + u(hxp,j,i), real(2)) + 
                                Kokkos::pow(v(hxp,j-1,i) + v(hxp,j,i), real(2))
                             );

                VVM::Real es1 = compute_es(Tg(j, i)); 
                VVM::Real qsfc = es1 * real(0.622) / (pbar(hx1) - es1);
                VVM::Real ts = cp * Tg(j, i) + grav() * z_up(hx1); 

                VVM::Real Q = qv(hxp, j, i) + qc(hxp, j, i) + qi(hxp, j, i);
                VVM::Real T = cp * th(hxp, j, i) * pibar(hxp) 
                             - hlf * qc(hxp, j, i)
                             + grav() * z_mid(hxp)
                             - (hlf + hlm) * qi(hxp, j, i);
                VVM::Real thvsm = Tg(j,i) / pibar(hx1) - th(hxp,j,i) + 
                               Kokkos::abs(gwet(j,i))*thbar(hxp) * (delta * (qsfc-qv(hxp,j,i)));

                VVM::Real sigmau = real(0.0);
                VVM::Real ustar, molen;
                VVM::Real ventfc[2];
                sflux_tc_2d(sigmau, thbar(hxp), thvsm, speedtp, ztmp, zrough(j, i), local_speed1_filter, ustar, ventfc, molen);

                VVM::Real wt = ventfc[1] * (ts - T);
                VVM::Real wq = ventfc[1] * Kokkos::abs(gwet(j, i)) * (qsfc - Q);
                VEN2D(j,i) = ventfc[0];

                sfc_flux_qv(j,i) = wq * rhobar_up(hx1);
                sfc_flux_th(j,i) = wt * rhobar_up(hx1) / (cp * pibar(hx1));
            }
        );
    }

    if (land_scheme_ == "noahlsm") {
        const auto& cmx = state.get_field<2>("cmx").get_device_data();

        if (mode_ == "sflux_2d" || mode_ == "sflux_tc_2d") {
            Kokkos::parallel_for("OverwriteLandVEN2D",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    if (sea_land_ice_mask(j, i) != 0) {
                        VEN2D(j, i) = cmx(j, i);
                    }
                }
            );
        }
        else if (mode_ == "tco_ocean") {
            Kokkos::parallel_for("OverwriteLandVEN2D",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
                KOKKOS_LAMBDA(const int j, const int i) {
                    VEN2D(j, i) = cmx(j, i);
                }
            );
        }
    }
    halo_exchanger_.exchange_halos(state.get_field<2>("VEN2D"));

    bool has_topo = (params_.max_topo_idx > h);
    Kokkos::parallel_for("SFlux_uv",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int hx1 = hx(j,i);
            int hxp = hx(j,i)+1;
            int hxup = hxu(j,i) + 1;
            int hxvp = hxv(j,i) + 1;

            if (has_topo) {
                if (hxup > 1) sfc_flux_u(j,i) = -VEN2D(j,i+1) * u(hxup,j,i);
                else sfc_flux_u(j,i) = -VEN2D(j,i) * u(hxp,j,i);
                if (hxvp > 1) sfc_flux_v(j,i) = -VEN2D(j,i) * v(hxvp,j,i);
                else sfc_flux_v(j,i) = -VEN2D(j+1,i) * v(hxp,j,i);
            }
            else {
                sfc_flux_u(j,i) = -real(0.5) * (VEN2D(j,i)+VEN2D(j,i+1)) * u(h,j,i);
                sfc_flux_v(j,i) = -real(0.5) * (VEN2D(j,i)+VEN2D(j+1,i)) * v(h,j,i);
            }
        }
    );

    Kokkos::parallel_for("SFlux_uv",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int hx1 = hx(j,i);
            int hxu1 = hxu(j,i);
            int hxv1 = hxv(j,i);

            if (hxu1 > 0) sfc_flux_u(j,i) = sfc_flux_u(j,i) * rhobar_up(hxu1);
            else sfc_flux_u(j,i) = sfc_flux_u(j,i) * rhobar_up(hx1);
            if (hxv1 > 0) sfc_flux_v(j,i) = sfc_flux_v(j,i) * rhobar_up(hxv1);
            else sfc_flux_v(j,i) = sfc_flux_v(j,i) * rhobar_up(hx1);
        }
    );
}

template<size_t Dim>
void SurfaceProcess::calculate_tendencies(Core::State& state, 
                                          const std::string& var_name, 
                                          Core::Field<Dim>& out_tendency) {
    VVM::Utils::Timer surface_timer("Surface");
    if (var_name != "th" && var_name != "qv" && var_name != "xi" && var_name != "eta") return;

    auto tend = out_tendency.get_mutable_device_data();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    int h = grid_.get_halo_cells();
    const auto& hxu    = state.get_field<2>("topou").get_device_data();
    const auto& hxv    = state.get_field<2>("topov").get_device_data();

    const auto& rhobar = state.get_field<1>("rhobar").get_device_data(); // Density
    const auto& hx     = state.get_field<2>("topo").get_device_data();
    const auto& rdz = params_.rdz; 
    const auto& rdz2 = params_.rdz2; 
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();
    const auto& flex_height_coef_up = params_.flex_height_coef_up.get_device_data();
    const auto& sea_land_ice_mask = state.get_field<2>("sea_land_ice_mask").get_device_data();

    if (var_name == "th") {
        const auto& flux = state.get_field<2>("sfc_flux_th").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_TH",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // If not sea, return
                if (sea_land_ice_mask(j,i) != 0) return;

                int hxp = hx(j,i)+1;
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    } 
    else if (var_name == "qv") {
        const auto& flux = state.get_field<2>("sfc_flux_qv").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_QV",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                // If not sea, return
                if (sea_land_ice_mask(j,i) != 0) return;

                int hxp = hx(j,i)+1;
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    }
    else if (var_name == "xi") {
        const auto& flux = state.get_field<2>("sfc_flux_v").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_XI",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i) + 1;
                int hxvp = hxv(j,i) + 1;
                int target_k = (hxvp > 1) ? hxvp : hxp;
                
                tend(target_k, j, i) += flux(j, i) * flex_height_coef_mid(target_k) * flex_height_coef_up(target_k) * rdz2() / rhobar(target_k);
            }
        );
    }
    else if (var_name == "eta") {
        const auto& flux = state.get_field<2>("sfc_flux_u").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_ETA",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i) + 1;
                int hxup = hxu(j,i) + 1;
                int target_k = (hxup > 1) ? hxup : hxp;
                
                tend(target_k, j, i) += flux(j, i) * flex_height_coef_mid(target_k) * flex_height_coef_up(target_k) * rdz2() / rhobar(target_k);
            }
        );
    }
    return;
}

template void SurfaceProcess::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Physics
} // namespace VVM
