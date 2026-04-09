#include "physics/land/LandProcess.hpp"
#include <cmath>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#define MAP_KOKKOS_DEVICE(view) acc_map_data(view.data(), view.data(), view.span() * sizeof(view.data()[0]))                                           
#define UNMAP_KOKKOS_DEVICE(view) acc_unmap_data(view.data())

namespace VVM {
namespace Physics {

LandProcess::LandProcess(const Utils::ConfigurationManager& config, 
                         const Core::Grid& grid, 
                         const Core::Parameters& params, 
                         Core::HaloExchanger& halo_exchanger, 
                         Core::State& state)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger), state_(state)
{
    m_nx = grid_.get_local_physical_points_x();
    m_ny = grid_.get_local_physical_points_y();
    m_halo_x = grid_.get_halo_cells(); 
    m_halo_y = grid_.get_halo_cells();
    m_nsoil = 4;

    m_islimsk = view_2d_int_ll("lsm_islimsk", m_nx, m_ny);
    m_vegtype = view_2d_int_ll("lsm_vegtype", m_nx, m_ny);
    m_soiltype = view_2d_int_ll("lsm_soiltype", m_nx, m_ny);
    m_slopetype = view_2d_int_ll("lsm_slopetype", m_nx, m_ny);

    m_t1 = view_2d_ll("lsm_t1", m_nx, m_ny);
    m_q1 = view_2d_ll("lsm_q1", m_nx, m_ny);
    m_u1 = view_2d_ll("lsm_u1", m_nx, m_ny);
    m_v1 = view_2d_ll("lsm_v1", m_nx, m_ny);
    m_ps = view_2d_ll("lsm_ps", m_nx, m_ny);
    m_prcp = view_2d_ll("lsm_prcp", m_nx, m_ny);
    m_swdn = view_2d_ll("lsm_swdn", m_nx, m_ny);
    m_lwdn = view_2d_ll("lsm_lwdn", m_nx, m_ny);
    m_hgt = view_2d_ll("lsm_hgt", m_nx, m_ny);
    m_prslki = view_2d_ll("lsm_prslki", m_nx, m_ny);

    m_stc = view_3d_ll("lsm_stc", m_nx, m_nsoil, m_ny);
    m_smc = view_3d_ll("lsm_smc", m_nx, m_nsoil, m_ny);
    m_slc = view_3d_ll("lsm_slc", m_nx, m_nsoil, m_ny);

    m_tskin = view_2d_ll("lsm_tskin", m_nx, m_ny);
    m_canopy = view_2d_ll("lsm_canopy", m_nx, m_ny);
    m_snwdph = view_2d_ll("lsm_snwdph", m_nx, m_ny);
    m_zorl = view_2d_ll("lsm_zorl", m_nx, m_ny);

    m_sigmaf = view_2d_ll("lsm_sigmaf", m_nx, m_ny); // Green Vegetation Fraction
    m_sfemis = view_2d_ll("lsm_sfemis", m_nx, m_ny); // Surface Emissivity
    m_alb    = view_2d_ll("lsm_alb", m_nx, m_ny); // Surface Albedo
    m_shdmin = view_2d_ll("lsm_shdmin", m_nx, m_ny); // Minimum Fractional Coverage
    m_shdmax = view_2d_ll("lsm_shdmax", m_nx, m_ny); // Maximum Fractional Coverage

    m_hflux = view_2d_ll("lsm_hflux", m_nx, m_ny);
    m_qflux = view_2d_ll("lsm_qflux", m_nx, m_ny);
    m_evap = view_2d_ll("lsm_evap", m_nx, m_ny);

    int ny = m_ny+2*m_halo_y;
    int nx = m_nx+2*m_halo_x;

    if (!state.has_field("hfx")) state.add_field<2>("hfx", {ny, nx});
    if (!state.has_field("le")) state.add_field<2>("le", {ny, nx});
    if (!state.has_field("sea_land_ice_mask")) state.add_field<2>("sea_land_ice_mask", {ny, nx});
    if (!state.has_field("canopy")) state.add_field<2>("canopy", {ny, nx});
    if (!state.has_field("snwdph")) state.add_field<2>("snwdph", {ny, nx});
    if (!state.has_field("zorl")) state.add_field<2>("zorl", {ny, nx});
    if (!state.has_field("vegtype")) state.add_field<2>("vegtype", {ny, nx});
    if (!state.has_field("soiltype")) state.add_field<2>("soiltype", {ny, nx});
    if (!state.has_field("slopetype")) state.add_field<2>("slopetype", {ny, nx});
    if (!state.has_field("stc")) state.add_field<3>("stc", {m_nsoil, ny, nx});
    if (!state.has_field("smc")) state.add_field<3>("smc", {m_nsoil, ny, nx});
    if (!state.has_field("slc")) state.add_field<3>("slc", {m_nsoil, ny, nx});
}

void LandProcess::init() {
    auto& th_v = state_.get_field<3>("th").get_device_data();
    auto& pibar_v = state_.get_field<1>("pibar").get_device_data();
    auto& topo_v = state_.get_field<2>("topo").get_device_data();
    auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data(); 
    
    Kokkos::parallel_for("InitLandStates", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;
            int hxp = topo_v(vj, vi) + 1;

            m_tskin(i, j) = Tg(vj, vi);

            m_islimsk(i, j) = 1;
            m_vegtype(i, j) = 2;
            m_soiltype(i, j) = 2;
            m_slopetype(i, j) = 1;
            
            m_prcp(i,j) = 0.0;

            m_zorl(i, j) = 0.1;
            m_canopy(i, j) = 0.0;
            m_snwdph(i, j) = 0.0;

            m_t1(i, j) = th_v(hxp, vj, vi) * pibar_v(hxp);

            for(int k=0; k<m_nsoil; ++k) {
                m_stc(i, k, j) = m_tskin(i, j) - k * 0.5; // soil temperature
                m_smc(i, k, j) = 0.3; // volumetric soil moisture content
                m_slc(i, k, j) = 0.3; // liquid soil moisture
            }
        }
    );
    prepare_static_data();
    register_openacc();
}

void LandProcess::prepare_static_data() {
    auto& topo_v = state_.get_field<2>("topo").get_device_data();
    auto& pbar_v = state_.get_field<1>("pbar").get_device_data();
    auto z_mid_v = params_.z_mid.get_device_data();
    auto z_up_v = params_.z_up.get_device_data();

    auto& pibar_v = state_.get_field<1>("pibar").get_device_data();
    auto& pibar_up_v = state_.get_field<1>("pibar_up").get_device_data();

    auto& sea_land_ice_mask = state_.get_field<2>("sea_land_ice_mask").get_device_data();
    auto& vegtype = state_.get_field<2>("vegtype").get_device_data();
    auto& soiltype = state_.get_field<2>("soiltype").get_device_data();
    auto& slopetype = state_.get_field<2>("slopetype").get_device_data();

    Kokkos::parallel_for("PrepareLandStaticData", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            int hx = topo_v(vj, vi);
            int hxp = topo_v(vj, vi) + 1;
            double dz = z_mid_v(hxp) - z_up_v(hx);
            m_hgt(i, j) = (dz < 2.0) ? 2.0 : dz;

            m_ps(i, j) = pbar_v(hxp); 
            m_prslki(i, j) = pibar_up_v(hx) / pibar_v(hxp);

            m_sigmaf(i, j) = 0.8;  // Green Vegetation Fraction
            m_sfemis(i, j) = 0.98; // Surface Emissivity
            m_alb(i, j)    = 0.2;  // Surface Emissivity
            m_shdmin(i, j) = 0.01; // Minimum Fractional Coverage
            m_shdmax(i, j) = 0.99; // Maximum Fractional Coverage

            // sea/land/ice, 0/1/2
            m_islimsk(i, j) = sea_land_ice_mask(vj, vi);
            m_vegtype(i, j) = vegtype(vj, vi); // vegetation type 20 types
            m_soiltype(i, j) = soiltype(vj, vi); // soil type 19 types
            m_slopetype(i, j) = slopetype(vj, vi); // slope 9 types
            m_zorl(i, j) = 0.1; // surface roughness (m)
        }
    );
}

void LandProcess::preprocessing_and_packing() {
    auto& u_v  = state_.get_field<3>("u").get_device_data();
    auto& v_v  = state_.get_field<3>("v").get_device_data();
    auto& qv_v = state_.get_field<3>("qv").get_device_data();
    auto& swdn_v = state_.get_field<3>("swdn").get_device_data();
    auto& lwdn_v = state_.get_field<3>("lwdn").get_device_data();
    auto& th_v = state_.get_field<3>("th").get_device_data();

    auto& pibar_v = state_.get_field<1>("pibar").get_device_data();
    auto& pr_v = state_.get_field<1>("pbar").get_device_data();
    auto& topo_v = state_.get_field<2>("topo").get_device_data();
    
    auto& canopy_v = state_.get_field<2>("canopy").get_device_data();
    auto& snwdph_v = state_.get_field<2>("snwdph").get_device_data();
    auto& stc_v    = state_.get_field<3>("stc").get_device_data();
    auto& smc_v    = state_.get_field<3>("smc").get_device_data();
    auto& slc_v    = state_.get_field<3>("slc").get_device_data();
    auto& precip_liq_surf_2d = state_.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();
    auto& precip_ice_surf_2d = state_.get_field<2>("precip_ice_surf_mass").get_mutable_device_data();

    Kokkos::parallel_for("PackToLand", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            int hx = topo_v(vj, vi);
            int hxp = topo_v(vj, vi) + 1;
            

            m_t1(i, j) = th_v(hxp, vj, vi) * pibar_v(hxp);
            m_u1(i, j) = 0.5 * (u_v(hxp, vj, vi) + u_v(hxp, vj, vi-1));
            m_v1(i, j) = 0.5 * (v_v(hxp, vj, vi) + v_v(hxp, vj-1, vi));
            // m_u1(i, j) = u_v(hxp, vj, vi);
            // m_v1(i, j) = v_v(hxp, vj, vi);
            m_q1(i, j) = qv_v(hxp, vj, vi);
            m_ps(i, j) = pr_v(hxp);

            m_swdn(i,j) = swdn_v(hxp, vj, vi);
            m_lwdn(i,j) = lwdn_v(hxp, vj, vi);
            m_prcp(i,j) = precip_liq_surf_2d(vj, vi) + precip_ice_surf_2d(vj, vi);
        }
    );
}

void LandProcess::postprocessing_and_unpacking() {
    auto& hfx_v    = state_.get_field<2>("hfx").get_mutable_device_data();
    auto& le_v     = state_.get_field<2>("le").get_mutable_device_data();
    
    auto& canopy_v = state_.get_field<2>("canopy").get_mutable_device_data();
    auto& snwdph_v = state_.get_field<2>("snwdph").get_mutable_device_data();
    auto& stc_v    = state_.get_field<3>("stc").get_mutable_device_data();
    auto& smc_v    = state_.get_field<3>("smc").get_mutable_device_data();
    auto& slc_v    = state_.get_field<3>("slc").get_mutable_device_data();
    auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data(); 


    Kokkos::parallel_for("UnpackToVVM", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            hfx_v(vj, vi) = m_hflux(i, j);
            le_v(vj, vi)  = m_evap(i, j);

            Tg(vj, vi) = m_tskin(i, j);
            canopy_v(vj, vi) = m_canopy(i, j);
            snwdph_v(vj, vi) = m_snwdph(i, j);

            for(int k=0; k<m_nsoil; ++k) {
                stc_v(k, vj, vi) = m_stc(i, k, j);
                smc_v(k, vj, vi) = m_smc(i, k, j);
                slc_v(k, vj, vi) = m_slc(i, k, j);
            }
        }
    );
}


void LandProcess::run(double dt) {
    preprocessing_and_packing();

    Kokkos::fence();

// #if defined(KOKKOS_ENABLE_CUDA)
//     cudaStream_t stream = Kokkos::DefaultExecutionSpace().cuda_stream();
//     acc_set_cuda_stream(1, stream);
// #elif defined(KOKKOS_ENABLE_HIP)
//     hipStream_t stream = Kokkos::DefaultExecutionSpace().hip_stream();
// #endif

    run_vvm_land_wrapper(m_nx, m_ny, m_nsoil, dt,
        m_islimsk.data(), m_vegtype.data(), m_soiltype.data(), m_slopetype.data(),
        m_sigmaf.data(), m_sfemis.data(), m_alb.data(), m_shdmin.data(), m_shdmax.data(),
        m_t1.data(), m_q1.data(), m_u1.data(), m_v1.data(), m_ps.data(), 
        m_prcp.data(), m_swdn.data(), m_lwdn.data(), m_hgt.data(), m_prslki.data(),
        m_stc.data(), m_smc.data(), m_slc.data(), m_tskin.data(), 
        m_canopy.data(), m_snwdph.data(),
        m_hflux.data(), m_qflux.data(), m_evap.data(), m_zorl.data());

    postprocessing_and_unpacking();
}

void LandProcess::finalize() {
    unregister_openacc();
    m_islimsk = {}; m_vegtype = {}; m_soiltype = {}; m_slopetype = {};
    m_zorl = {}; m_t1 = {}; m_q1 = {}; m_u1 = {}; m_v1 = {};
    m_ps = {}; m_prcp = {}; m_swdn = {}; m_lwdn = {};
    m_stc = {}; m_smc = {}; m_slc = {};
    m_tskin = {}; m_canopy = {}; m_snwdph = {};
    m_hflux = {}; m_qflux = {}; m_evap = {};
}

void LandProcess::register_openacc() {
    MAP_KOKKOS_DEVICE(m_islimsk); MAP_KOKKOS_DEVICE(m_vegtype); 
    MAP_KOKKOS_DEVICE(m_soiltype); MAP_KOKKOS_DEVICE(m_slopetype); MAP_KOKKOS_DEVICE(m_zorl);
    MAP_KOKKOS_DEVICE(m_t1); MAP_KOKKOS_DEVICE(m_q1); 
    MAP_KOKKOS_DEVICE(m_u1); MAP_KOKKOS_DEVICE(m_v1);
    MAP_KOKKOS_DEVICE(m_ps); MAP_KOKKOS_DEVICE(m_prcp); 
    MAP_KOKKOS_DEVICE(m_swdn); MAP_KOKKOS_DEVICE(m_lwdn); MAP_KOKKOS_DEVICE(m_hgt); MAP_KOKKOS_DEVICE(m_prslki);
    MAP_KOKKOS_DEVICE(m_sigmaf); MAP_KOKKOS_DEVICE(m_sfemis); MAP_KOKKOS_DEVICE(m_alb); MAP_KOKKOS_DEVICE(m_shdmin); MAP_KOKKOS_DEVICE(m_shdmax);
    MAP_KOKKOS_DEVICE(m_stc); MAP_KOKKOS_DEVICE(m_smc); MAP_KOKKOS_DEVICE(m_slc);
    MAP_KOKKOS_DEVICE(m_tskin); MAP_KOKKOS_DEVICE(m_canopy); MAP_KOKKOS_DEVICE(m_snwdph);
    MAP_KOKKOS_DEVICE(m_hflux); MAP_KOKKOS_DEVICE(m_qflux); MAP_KOKKOS_DEVICE(m_evap);
}

void LandProcess::unregister_openacc() {
    UNMAP_KOKKOS_DEVICE(m_islimsk); UNMAP_KOKKOS_DEVICE(m_vegtype); 
    UNMAP_KOKKOS_DEVICE(m_soiltype); UNMAP_KOKKOS_DEVICE(m_slopetype); UNMAP_KOKKOS_DEVICE(m_zorl);
    UNMAP_KOKKOS_DEVICE(m_t1); UNMAP_KOKKOS_DEVICE(m_q1); 
    UNMAP_KOKKOS_DEVICE(m_u1); UNMAP_KOKKOS_DEVICE(m_v1);
    UNMAP_KOKKOS_DEVICE(m_ps); UNMAP_KOKKOS_DEVICE(m_prcp); 
    UNMAP_KOKKOS_DEVICE(m_swdn); UNMAP_KOKKOS_DEVICE(m_lwdn); UNMAP_KOKKOS_DEVICE(m_hgt); UNMAP_KOKKOS_DEVICE(m_prslki);
    UNMAP_KOKKOS_DEVICE(m_sigmaf); UNMAP_KOKKOS_DEVICE(m_sfemis); UNMAP_KOKKOS_DEVICE(m_alb); UNMAP_KOKKOS_DEVICE(m_shdmin); UNMAP_KOKKOS_DEVICE(m_shdmax);
    UNMAP_KOKKOS_DEVICE(m_stc); UNMAP_KOKKOS_DEVICE(m_smc); UNMAP_KOKKOS_DEVICE(m_slc);
    UNMAP_KOKKOS_DEVICE(m_tskin); UNMAP_KOKKOS_DEVICE(m_canopy); UNMAP_KOKKOS_DEVICE(m_snwdph);
    UNMAP_KOKKOS_DEVICE(m_hflux); UNMAP_KOKKOS_DEVICE(m_qflux); UNMAP_KOKKOS_DEVICE(m_evap);
}


template<size_t Dim>
void LandProcess::calculate_tendencies(const std::string& var_name, 
                                      Core::Field<Dim>& out_tendency) {
    if (var_name != "th" && var_name != "qv") return;

    auto tend = out_tendency.get_mutable_device_data();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();
    const auto& rhobar = state_.get_field<1>("rhobar").get_device_data(); // Density
    const auto& hx     = state_.get_field<2>("topo").get_device_data();
    const auto& rdz = params_.rdz; 
    const auto& flex_height_coef_mid = params_.flex_height_coef_mid.get_device_data();

    if (var_name == "th") {
        const auto& flux = state_.get_field<2>("hfx").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_TH",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i)+1;
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    } 
    else if (var_name == "qv") {
        const auto& flux = state_.get_field<2>("le").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_QV",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i)+1;
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    }
    return;
}

template void LandProcess::calculate_tendencies(const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Physics
} // namespace VVM
