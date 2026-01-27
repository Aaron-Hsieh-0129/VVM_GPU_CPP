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
    // NOTE: sfc_flux_u, sfc_flux_v are not used for now. They are supposed to be used in land model.
    if (!state.has_field("sfc_flux_u"))  state.add_field<2>("sfc_flux_u", {ny, nx});
    if (!state.has_field("sfc_flux_v"))  state.add_field<2>("sfc_flux_v", {ny, nx});
    
    if (!state.has_field("Tg")) state.add_field<2>("Tg", {ny, nx}); // Surface Temperature (K)
    if (!state.has_field("gwet")) state.add_field<2>("gwet", {ny, nx}); // Surface Wetness
    if (!state.has_field("zrough")) state.add_field<2>("zrough", {ny, nx}); // Roughness Length
    if (!state.has_field("VEN2D")) state.add_field<2>("VEN2D", {ny, nx}); // Roughness Length
    Kokkos::deep_copy(state.get_field<2>("Tg").get_mutable_device_data(), 305.);
    Kokkos::deep_copy(state.get_field<2>("gwet").get_mutable_device_data(), 0.8);
    Kokkos::deep_copy(state.get_field<2>("zrough").get_mutable_device_data(), 2e-4);
    
    if (!state.has_field("ustar")) state.add_field<2>("ustar", {ny, nx});
    if (!state.has_field("molen")) state.add_field<2>("molen", {ny, nx});
}

KOKKOS_INLINE_FUNCTION
double SurfaceProcess::compute_es(double t) {
    double es = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
    return es; 
}

KOKKOS_INLINE_FUNCTION
void SurfaceProcess::sflux_2d(double sigmau, double thvm, double thvsm, double speed1, 
                              double zr, double zrough, 
                              double& ustar, double ventfc[2], double& molen) {
    const double bus = 0.74;
    const double crit = 0.003;
    const int maxit = 5;

    bool stopit = false;
    double speedm = (speed1 > 1.e-3) ? speed1 : 1.e-3;

    const double vk = 0.4;
    const double pi = 3.141592653589793;
    double grav = 9.806;

    double tem1 = Kokkos::log(zr / zrough);
    double cuni = tem1 / vk;
    double ctni = cuni * bus;

    bool stable = (thvsm < 0.0);

    // START ITERATION
    int it = 0;
    double cu = 1.0 / cuni;
    double ct = 1.0 / ctni;

    if (!stable) speedm = (speedm > sigmau) ? speedm : sigmau;

    double cui, cti;
    while (!stopit) {
        it++;
        double zeta = -zr * ct * vk * grav * thvsm / (thvm * cu * cu * speedm * speedm);

        if (stable) {
            if (zeta >= 2.45) {
                stopit = true;
                zeta = 2.45;
            }
            double tem2 = tem1 + 4.7 * zeta;
            double tem3 = tem1 + 4.7 / bus * zeta;

            cui = tem2 / vk;
            cti = bus * tem3 / vk;
        }
        else {
            // UNSTABLE OR NEUTRAL CASE
            double x = Kokkos::pow(1.0 - 15.0 * zeta, 0.25);
            double y = Kokkos::pow(1.0 - 9.0 * zeta, 0.25);

            double tem2 = tem1 - (Kokkos::log((1.0 + x * x) / 2.0)
                                + 2.0 * Kokkos::log((1.0 + x) / 2.0)
                                - 2.0 * Kokkos::atan(x) + pi / 2.0);
            
            double tem3 = tem1 - 2.0 * Kokkos::log((1.0 + y * y) / 2.0);

            cui = tem2 / vk;
            cui = (cui > 0.5 * cuni) ? cui : 0.5 * cuni; // MAX(CUI, 0.5*CUNI)

            cti = bus * tem3 / vk;
            cti = (cti > 0.3 * ctni) ? cti : 0.3 * ctni; // MAX(CTI, 0.3*CTNI)
        }

        // STOPIT = STOPIT .OR. IT .EQ. MAXIT
        if (it == maxit) stopit = true;

        if (stopit) {
            cu = 1.0 / cui;
            ct = 1.0 / cti;
        } 
        else {
            // CHECK FOR CONVERGENCE
            double custar = cu;
            double ctstar = ct;
            cu = 1.0 / cui;
            ct = 1.0 / cti;
            
            if (std::abs(cu / custar - 1.0) <= crit && std::abs(ct / ctstar - 1.0) <= crit) {
                stopit = true;
            }
        }
    }

    // ITERATION COMPLETED. CALCULATE USTAR AND VENTFC
    ustar = cu * speedm;
    ventfc[0] = cu * ustar;
    ventfc[1] = ct * ustar;

    // CHECK TOWNSEND'S LIMIT (Unstable only)
    if (!stable && cti < 0.3 * ctni) {
        ventfc[1] = Kokkos::max(ventfc[1], 0.0019 * Kokkos::pow(thvsm,1./3.) ); 
    }

    // MONIN-OBUKHOV LENGTH
    double zeta = -zr * ct * vk * grav * thvsm / (thvm * cu * cu * speedm * speedm);
    zeta = Kokkos::max(Kokkos::abs(zeta), 1e-6) * std::copysign(1., zeta);
    molen = zr / Kokkos::min(zeta, 2.45);
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

    auto sfc_flux_th = state.get_field<2>("sfc_flux_th").get_mutable_device_data();
    auto sfc_flux_qv = state.get_field<2>("sfc_flux_qv").get_mutable_device_data();
    auto sfc_flux_u = state.get_field<2>("sfc_flux_u").get_mutable_device_data();
    auto sfc_flux_v = state.get_field<2>("sfc_flux_v").get_mutable_device_data();
    auto ustar_view = state.get_field<2>("ustar").get_mutable_device_data();
    auto molen_view = state.get_field<2>("molen").get_mutable_device_data();
    auto VEN2D = state.get_field<2>("VEN2D").get_mutable_device_data();

    const auto& hx     = state.get_field<2>("topo").get_device_data();
    const auto& hxu    = state.get_field<2>("topou").get_device_data();
    const auto& hxv    = state.get_field<2>("topov").get_device_data();

    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells(); 
    const auto& dz = params_.dz; 
    
    // Constants from 
    const double cp = 1004.5;
    const auto& grav = params_.gravity;
    const double hlf = 2.5e6;
    const double hlm = 3.336e5;
    const double delta = 0.608;

    Kokkos::parallel_for("SFlux_3D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            // NOTE: Need to check about the difference between original VVM and this
            int hx1 = hx(j,i) - 1;
            int hxp = hx(j,i);

            double ztmp = 0.5 * dz() / flex_height_coef_mid(hxp);
            double speedtp = 0.5 * Kokkos::sqrt(
                            Kokkos::pow(u(hxp,j,i-1) + u(hxp,j,i), 2) + 
                            Kokkos::pow(v(hxp,j-1,i) + v(hxp,j,i), 2)
                         );

            double es1 = compute_es(Tg(j, i)); 
            double qsfc = es1 * 0.622 / (pbar(hx1) - es1);
            double ts = cp * Tg(j, i) + grav() * z_up(hx1); 

            double Q = qv(hxp, j, i) + qc(hxp, j, i) + qi(hxp, j, i);
            double T = cp * th(hxp, j, i) * pibar(hxp) 
                         - hlf * qc(hxp, j, i)
                         + grav() * z_mid(hxp)
                         - (hlf + hlm) * qi(hxp, j, i);
            double thvsm = Tg(j,i) / pibar(hx1) - th(hxp,j,i) + 
                           Kokkos::abs(gwet(j,i))*thbar(hxp) * (delta * (qsfc-qv(hxp,j,i)));

            double sigmau = 0.0;
            double ustar, molen;
            double ventfc[2];
            sflux_2d(sigmau, thbar(hxp), thvsm, speedtp, ztmp, zrough(j, i), ustar, ventfc, molen);

            double wt = ventfc[1] * (ts - T);
            double wq = ventfc[1] * std::abs(gwet(j, i)) * (qsfc - Q);
            VEN2D(j,i) = ventfc[0];

            sfc_flux_qv(j,i) = wq * rhobar_up(hx1);
            sfc_flux_th(j,i) = wt * rhobar_up(hx1) / (cp * pibar(hx1));
        }
    );
    halo_exchanger_.exchange_halos(state.get_field<2>("VEN2D"));

    Kokkos::parallel_for("SFlux_3D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int hx1 = hx(j,i)-1;
            int hxp = hx(j,i);
            int hxup = hxu(j,i) + 1;
            int hxvp = hxv(j,i) + 1;

            if (hxup > 1) sfc_flux_u(j,i) = -VEN2D(j,i+1) * u(hxup,j,i);
            else sfc_flux_u(j,i) = -VEN2D(j,i) * u(hxp,j,i);
            if (hxvp > 1) sfc_flux_v(j,i) = -VEN2D(j,i+1) * v(hxvp,j,i);
            else sfc_flux_v(j,i) = -VEN2D(j+1,i) * v(hxp,j,i);
        }
    );

    Kokkos::parallel_for("SFlux_3D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
        KOKKOS_LAMBDA(const int j, const int i) {
            int hx1 = hx(j,i)-1;
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
    if (var_name != "th" && var_name != "qv") return;

    auto tend = out_tendency.get_mutable_device_data();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data(); // Density
    const auto& hx     = state.get_field<2>("topo").get_device_data();
    const auto& rdz = params_.rdz; 
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();

    if (var_name == "th") {
        const auto& flux = state.get_field<2>("sfc_flux_th").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_TH",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i);
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    } 
    else if (var_name == "qv") {
        const auto& flux = state.get_field<2>("sfc_flux_qv").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_QV",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i);
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    }
    return;
}

template void SurfaceProcess::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Physics
} // namespace VVM
