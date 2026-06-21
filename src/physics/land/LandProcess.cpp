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
                         Core::State& state, 
                         std::string ocean_scheme)
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
    m_ps = view_2d_ll("lsm_ps", m_nx, m_ny); // Surface pressure
    m_prsl1 = view_2d_ll("lsm_prsl1", m_nx, m_ny); // Pressure at first layer of atmosphere
    m_prcp = view_2d_ll("lsm_prcp", m_nx, m_ny);
    m_swdn = view_2d_ll("lsm_swdn", m_nx, m_ny);
    m_swnet = view_2d_ll("lsm_swnet", m_nx, m_ny);
    m_lwdn = view_2d_ll("lsm_lwdn", m_nx, m_ny);
    m_hgt = view_2d_ll("lsm_hgt", m_nx, m_ny);
    m_prslki = view_2d_ll("lsm_prslki", m_nx, m_ny);

    m_stc = view_3d_ll("lsm_stc", m_nx, m_nsoil, m_ny);
    m_smc = view_3d_ll("lsm_smc", m_nx, m_nsoil, m_ny);
    m_slc = view_3d_ll("lsm_slc", m_nx, m_nsoil, m_ny);

    m_tskin = view_2d_ll("lsm_tskin", m_nx, m_ny);
    m_canopy = view_2d_ll("lsm_canopy", m_nx, m_ny);
    m_snwdph = view_2d_ll("lsm_snwdph", m_nx, m_ny);
    m_sneqv = view_2d_ll("lsm_sneqv", m_nx, m_ny);
    m_zorl = view_2d_ll("lsm_zorl", m_nx, m_ny);
    m_cmx = view_2d_ll("lsm_cmx", m_nx, m_ny);
    m_chx = view_2d_ll("lsm_chx", m_nx, m_ny);

    m_sigmaf = view_2d_ll("lsm_sigmaf", m_nx, m_ny); // Green Vegetation Fraction
    m_lai = view_2d_ll("lsm_lai", m_nx, m_ny); // leaf area index (m^2/m^2)
    m_sfemis = view_2d_ll("lsm_sfemis", m_nx, m_ny); // Surface Emissivity
    m_alb    = view_2d_ll("lsm_alb", m_nx, m_ny); // Surface Albedo
    m_shdmin = view_2d_ll("lsm_shdmin", m_nx, m_ny); // Minimum Fractional Coverage
    m_shdmax = view_2d_ll("lsm_shdmax", m_nx, m_ny); // Maximum Fractional Coverage

    m_hflux = view_2d_ll("lsm_hflux", m_nx, m_ny);
    m_qflux = view_2d_ll("lsm_qflux", m_nx, m_ny);
    m_evap = view_2d_ll("lsm_evap", m_nx, m_ny);
    m_gfx = view_2d_ll("lsm_gfx", m_nx, m_ny);

    int ny = m_ny+2*m_halo_y;
    int nx = m_nx+2*m_halo_x;

    if (!state.has_field("hfx")) state.add_field<2>("hfx", {ny, nx});
    if (!state.has_field("le")) state.add_field<2>("le", {ny, nx});
    if (!state.has_field("gfx")) state.add_field<2>("gfx", {ny, nx});
    if (!state.has_field("sea_land_ice_mask")) state.add_field<2>("sea_land_ice_mask", {ny, nx});
    if (!state.has_field("canopy")) state.add_field<2>("canopy", {ny, nx});
    if (!state.has_field("snwdph")) state.add_field<2>("snwdph", {ny, nx});
    if (!state.has_field("sneqv")) state.add_field<2>("sneqv", {ny, nx});
    if (!state.has_field("zorl")) state.add_field<2>("zorl", {ny, nx});
    if (!state.has_field("cmx")) state.add_field<2>("cmx", {ny, nx});
    if (!state.has_field("chx")) state.add_field<2>("chx", {ny, nx});
    if (!state.has_field("vegtype")) state.add_field<2>("vegtype", {ny, nx});
    if (!state.has_field("soiltype")) state.add_field<2>("soiltype", {ny, nx});
    if (!state.has_field("slopetype")) state.add_field<2>("slopetype", {ny, nx});
    if (!state.has_field("shdmin")) state.add_field<2>("shdmin", {ny, nx});
    if (!state.has_field("shdmax")) state.add_field<2>("shdmax", {ny, nx});
    if (!state.has_field("albedo")) state.add_field<2>("albedo", {ny, nx});
    if (!state.has_field("gvf")) state.add_field<2>("gvf", {ny, nx});
    if (!state.has_field("lai")) state.add_field<2>("lai", {ny, nx});
    if (!state.has_field("st1")) state.add_field<2>("st1", {ny, nx});
    if (!state.has_field("st2")) state.add_field<2>("st2", {ny, nx});
    if (!state.has_field("st3")) state.add_field<2>("st3", {ny, nx});
    if (!state.has_field("st4")) state.add_field<2>("st4", {ny, nx});
    if (!state.has_field("sm1")) state.add_field<2>("sm1", {ny, nx});
    if (!state.has_field("sm2")) state.add_field<2>("sm2", {ny, nx});
    if (!state.has_field("sm3")) state.add_field<2>("sm3", {ny, nx});
    if (!state.has_field("sm4")) state.add_field<2>("sm4", {ny, nx});
    if (!state.has_field("sl1")) state.add_field<2>("sl1", {ny, nx});
    if (!state.has_field("sl2")) state.add_field<2>("sl2", {ny, nx});
    if (!state.has_field("sl3")) state.add_field<2>("sl3", {ny, nx});
    if (!state.has_field("sl4")) state.add_field<2>("sl4", {ny, nx});
    if (!state.has_field("sfemis")) state.add_field<2>("sfemis", {ny, nx});


    if (!state.has_field("swdn_sfc")) state.add_field<2>("swdn_sfc", {ny, nx});
    if (!state.has_field("swup_sfc")) state.add_field<2>("swup_sfc", {ny, nx});
    if (!state.has_field("lwdn_sfc")) state.add_field<2>("lwdn_sfc", {ny, nx});

    m_use_tco_ocean = (ocean_scheme == "tco_ocean") ? 1 : 0; // false/true: 0/1 
}

void LandProcess::init() {
    auto& th_v = state_.get_field<3>("th").get_device_data();
    auto& pibar_v = state_.get_field<1>("pibar").get_device_data();
    auto& topo_v = state_.get_field<2>("topo").get_device_data();
    auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data(); 
    auto& sm1_v = state_.get_field<2>("sm1").get_device_data();
    auto& sm2_v = state_.get_field<2>("sm2").get_device_data();
    auto& sm3_v = state_.get_field<2>("sm3").get_device_data();
    auto& sm4_v = state_.get_field<2>("sm4").get_device_data();

    auto& sl1_v = state_.get_field<2>("sl1").get_device_data();
    auto& sl2_v = state_.get_field<2>("sl2").get_device_data();
    auto& sl3_v = state_.get_field<2>("sl3").get_device_data();
    auto& sl4_v = state_.get_field<2>("sl4").get_device_data();
    
    Kokkos::parallel_for("InitLandStates", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;
            int hx = topo_v(vj, vi);
            int hxp = topo_v(vj, vi) + 1;

            // NOTE: These values will be overwritten in preprocessing_and_packing
            m_islimsk(i, j) = 1;
            m_vegtype(i, j) = 2;
            m_soiltype(i, j) = 2;
            m_slopetype(i, j) = 1;
            m_prcp(i,j) = 0.0;
            m_zorl(i, j) = 0.1;
            m_cmx(i, j) = 0.;
            m_chx(i, j) = 0.;
            m_canopy(i, j) = 0.0;
            m_snwdph(i, j) = 0.0;
            m_sneqv(i, j) = 0.0;

            m_t1(i, j) = th_v(hx, vj, vi) * pibar_v(hx);
            m_tskin(i, j) = m_t1(i, j); // It it supposed to be equal to Tg but Fortran VVM use m_t1 so we follow it.

            m_smc(i, 0, j) = sm1_v(vj, vi);
            m_smc(i, 1, j) = sm2_v(vj, vi);
            m_smc(i, 2, j) = sm3_v(vj, vi);
            m_smc(i, 3, j) = sm4_v(vj, vi);

            m_slc(i, 0, j) = sl1_v(vj, vi);
            m_slc(i, 1, j) = sl2_v(vj, vi);
            m_slc(i, 2, j) = sl3_v(vj, vi);
            m_slc(i, 3, j) = sl4_v(vj, vi);

            for(int k=0; k<m_nsoil; ++k) {
                m_stc(i, k, j) = m_t1(i, j); // soil temperature
            }
        }
    );
    prepare_static_data();
    register_openacc();

    init_vvm_land_sfcdif_wrf();
}

void LandProcess::prepare_static_data() {
    auto& topo_v = state_.get_field<2>("topo").get_device_data();
    auto& pbar_v = state_.get_field<1>("pbar").get_device_data();
    auto& pbar_up_v = state_.get_field<1>("pbar_up").get_device_data();

    auto z_mid_v = params_.z_mid.get_device_data();
    auto z_up_v = params_.z_up.get_device_data();

    auto& pibar_v = state_.get_field<1>("pibar").get_device_data();
    auto& pibar_up_v = state_.get_field<1>("pibar_up").get_device_data();

    auto& sea_land_ice_mask = state_.get_field<2>("sea_land_ice_mask").get_device_data();
    auto& vegtype = state_.get_field<2>("vegtype").get_device_data();
    auto& soiltype = state_.get_field<2>("soiltype").get_device_data();
    auto& slopetype = state_.get_field<2>("slopetype").get_device_data();
    auto& shdmin = state_.get_field<2>("shdmin").get_device_data();
    auto& shdmax = state_.get_field<2>("shdmax").get_device_data();
    auto& albedo = state_.get_field<2>("albedo").get_device_data();
    auto& gvf = state_.get_field<2>("gvf").get_device_data();
    auto& lai = state_.get_field<2>("lai").get_device_data();

    Kokkos::parallel_for("PrepareLandStaticData", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            int hx = topo_v(vj, vi);
            int hxp = topo_v(vj, vi) + 1;
            VVM::Real dz = z_mid_v(hxp) - z_up_v(hx);
            m_hgt(i, j) = (dz < real(2.0)) ? real(2.0) : dz;

            m_ps(i, j) = pbar_v(hxp); 
            m_prsl1(i, j) = pbar_v(hxp);
            m_prslki(i, j) = pibar_up_v(hx) / pibar_v(hxp);

            m_sigmaf(i, j) = gvf(vj, vi) / real(100.);  // Green Vegetation Fraction
            m_lai(i, j) = lai(vj, vi);
            m_sfemis(i, j) = real(0.98); // Surface Emissivity (the distribution will be given in vvm_land_interface.f90)
            m_alb(i, j)    = albedo(vj, vi) / real(100.);  // Surface albedo
            m_shdmin(i, j) = shdmin(vj, vi) / real(100.); // Minimum Fractional Coverage
            m_shdmax(i, j) = shdmax(vj, vi) / real(100.); // Maximum Fractional Coverage

            // sea/land/ice, 0/1/2
            m_islimsk(i, j) = sea_land_ice_mask(vj, vi);
            m_vegtype(i, j) = vegtype(vj, vi); // vegetation type 20 types
            m_soiltype(i, j) = soiltype(vj, vi); // soil type 19 types
            m_slopetype(i, j) = slopetype(vj, vi); // slope 9 types
            m_zorl(i, j) = real(0.1); // surface roughness (m)
            m_cmx(i, j) = real(0.); // Exchange coefficient for momentum (m s-1)
            m_chx(i, j) = real(0.); // Exchange coefficient for thermal (m s-1)
        }
    );
}

void LandProcess::preprocessing_and_packing() {
    auto& u_v  = state_.get_field<3>("u").get_device_data();
    auto& v_v  = state_.get_field<3>("v").get_device_data();
    auto& qv_v = state_.get_field<3>("qv").get_device_data();
    // The surface radiation has considered topography
    auto& swdn_sfc_v = state_.get_field<2>("swdn_sfc").get_device_data();
    auto& swup_sfc_v = state_.get_field<2>("swup_sfc").get_device_data();
    auto& lwdn_sfc_v = state_.get_field<2>("lwdn_sfc").get_device_data();
    auto& th_v = state_.get_field<3>("th").get_device_data();

    auto& pibar_v = state_.get_field<1>("pibar").get_device_data();
    auto& topo_v = state_.get_field<2>("topo").get_device_data();
    
    auto& canopy_v = state_.get_field<2>("canopy").get_device_data();
    auto& precip_liq_surf_2d = state_.get_field<2>("precip_liq_surf_flux").get_mutable_device_data();
    auto& precip_ice_surf_2d = state_.get_field<2>("precip_ice_surf_flux").get_mutable_device_data();

    Kokkos::parallel_for("PackToLand", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            int hx = topo_v(vj, vi);
            int hxp = topo_v(vj, vi) + 1;
            
            m_t1(i, j) = th_v(hxp, vj, vi) * pibar_v(hxp);
            m_u1(i, j) = real(0.5) * (u_v(hxp, vj, vi) + u_v(hxp, vj, vi-1));
            m_v1(i, j) = real(0.5) * (v_v(hxp, vj, vi) + v_v(hxp, vj-1, vi));
            // m_u1(i, j) = u_v(hxp, vj, vi);
            // m_v1(i, j) = v_v(hxp, vj, vi);
            m_q1(i, j) = qv_v(hxp, vj, vi) / (1 + qv_v(hxp, vj, vi));

            m_swdn(i,j) = swdn_sfc_v(vj, vi);
            m_swnet(i,j) = swdn_sfc_v(vj, vi) - swup_sfc_v(vj, vi);
            m_lwdn(i,j) = lwdn_sfc_v(vj, vi);
            m_prcp(i,j) = precip_liq_surf_2d(vj, vi) + precip_ice_surf_2d(vj, vi);

            /*
            m_t1(i, j) = 299.644;
            m_u1(i, j) = 3.;
            m_v1(i, j) = 0.;
            // m_u1(i, j) = u_v(hxp, vj, vi);
            // m_v1(i, j) = v_v(hxp, vj, vi);
            m_q1(i, j) = 0.0161989;

            m_swdn(i,j) = 936.75;
            m_swnet(i,j) = 0.;
            m_lwdn(i,j) = 410.554;
            m_prcp(i,j) = 0.;
            */
        }
    );
}

void LandProcess::postprocessing_and_unpacking() {
    auto& hfx_v    = state_.get_field<2>("hfx").get_mutable_device_data();
    auto& le_v     = state_.get_field<2>("le").get_mutable_device_data();
    auto& gfx_v     = state_.get_field<2>("gfx").get_mutable_device_data();
    
    auto& canopy_v = state_.get_field<2>("canopy").get_mutable_device_data();
    auto& snwdph_v = state_.get_field<2>("snwdph").get_mutable_device_data();
    auto& sneqv_v = state_.get_field<2>("sneqv").get_mutable_device_data();
    auto& st1_v    = state_.get_field<2>("st1").get_mutable_device_data();
    auto& st2_v    = state_.get_field<2>("st2").get_mutable_device_data();
    auto& st3_v    = state_.get_field<2>("st3").get_mutable_device_data();
    auto& st4_v    = state_.get_field<2>("st4").get_mutable_device_data();
    auto& sm1_v    = state_.get_field<2>("sm1").get_mutable_device_data();
    auto& sm2_v    = state_.get_field<2>("sm2").get_mutable_device_data();
    auto& sm3_v    = state_.get_field<2>("sm3").get_mutable_device_data();
    auto& sm4_v    = state_.get_field<2>("sm4").get_mutable_device_data();
    auto& sl1_v    = state_.get_field<2>("sl1").get_mutable_device_data();
    auto& sl2_v    = state_.get_field<2>("sl2").get_mutable_device_data();
    auto& sl3_v    = state_.get_field<2>("sl3").get_mutable_device_data();
    auto& sl4_v    = state_.get_field<2>("sl4").get_mutable_device_data();
    auto& Tg = state_.get_field<2>("Tg").get_mutable_device_data(); 
    auto& zorl = state_.get_field<2>("zorl").get_mutable_device_data(); 
    auto& cmx = state_.get_field<2>("cmx").get_mutable_device_data(); 
    auto& chx = state_.get_field<2>("chx").get_mutable_device_data(); 
    auto& sfemis = state_.get_field<2>("sfemis").get_mutable_device_data(); 


    Kokkos::parallel_for("UnpackToVVM", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            hfx_v(vj, vi) = m_hflux(i, j);
            le_v(vj, vi)  = m_evap(i, j);
            gfx_v(vj, vi)  = m_gfx(i, j);


            Tg(vj, vi) = m_tskin(i, j);
            cmx(vj, vi) = m_cmx(i, j);
            chx(vj, vi) = m_chx(i, j);
            // NOTE: The following varaibles don't need to be unpacked unless they need to be output
            // canopy_v(vj, vi) = m_canopy(i, j);
            // snwdph_v(vj, vi) = m_snwdph(i, j);
            // sneqv_v(vj, vi) = m_sneqv(i, j);
            zorl(vj, vi) = m_zorl(i, j);
            sfemis(vj, vi) = m_sfemis(i, j);
            st1_v(vj, vi) = m_stc(i, 0, j);
            st2_v(vj, vi) = m_stc(i, 1, j);
            st3_v(vj, vi) = m_stc(i, 2, j);
            st4_v(vj, vi) = m_stc(i, 3, j);
            sm1_v(vj, vi) = m_smc(i, 0, j);
            sm2_v(vj, vi) = m_smc(i, 1, j);
            sm3_v(vj, vi) = m_smc(i, 2, j);
            sm4_v(vj, vi) = m_smc(i, 3, j);
            sl1_v(vj, vi) = m_slc(i, 0, j);
            sl2_v(vj, vi) = m_slc(i, 1, j);
            sl3_v(vj, vi) = m_slc(i, 2, j);
            sl4_v(vj, vi) = m_slc(i, 3, j);
        }
    );
}


void LandProcess::run(VVM::Real dt) {
    preprocessing_and_packing();

    Kokkos::fence();

// #if defined(KOKKOS_ENABLE_CUDA)
//     cudaStream_t stream = Kokkos::DefaultExecutionSpace().cuda_stream();
//     acc_set_cuda_stream(1, stream);
// #elif defined(KOKKOS_ENABLE_HIP)
//     hipStream_t stream = Kokkos::DefaultExecutionSpace().hip_stream();
// #endif




    run_vvm_land_wrapper(m_use_tco_ocean, m_nx, m_ny, m_nsoil, dt,
        m_islimsk.data(), m_vegtype.data(), m_soiltype.data(), m_slopetype.data(),
        m_sigmaf.data(), m_sfemis.data(), m_alb.data(), m_shdmin.data(), m_shdmax.data(),
        m_t1.data(), m_q1.data(), m_u1.data(), m_v1.data(), m_ps.data(), m_prsl1.data(), 
        m_prcp.data(), m_swdn.data(), m_lwdn.data(), m_swnet.data(), m_hgt.data(), m_prslki.data(),
        m_stc.data(), m_smc.data(), m_slc.data(), m_tskin.data(), 
        m_canopy.data(), m_snwdph.data(), m_sneqv.data(),
        m_hflux.data(), m_qflux.data(), m_evap.data(), m_gfx.data(), m_zorl.data(), m_cmx.data(), m_chx.data(),
        m_lai.data(), true);

    postprocessing_and_unpacking();
}

void LandProcess::finalize() {
    unregister_openacc();
    m_islimsk = {}; m_vegtype = {}; m_soiltype = {}; m_slopetype = {};
    m_zorl = {}; m_cmx = {}; m_chx = {}; m_t1 = {}; m_q1 = {}; m_u1 = {}; m_v1 = {};
    m_ps = {}; m_prsl1 = {}; m_prcp = {}; m_swdn = {}; m_lwdn = {}; m_swnet = {};
    m_stc = {}; m_smc = {}; m_slc = {};
    m_tskin = {}; m_canopy = {}; m_snwdph = {}; m_sneqv = {};
    m_hflux = {}; m_qflux = {}; m_evap = {}; m_gfx = {}; m_lai = {};
}

void LandProcess::register_openacc() {
    MAP_KOKKOS_DEVICE(m_islimsk); MAP_KOKKOS_DEVICE(m_vegtype); 
    MAP_KOKKOS_DEVICE(m_soiltype); MAP_KOKKOS_DEVICE(m_slopetype); MAP_KOKKOS_DEVICE(m_zorl); MAP_KOKKOS_DEVICE(m_cmx); MAP_KOKKOS_DEVICE(m_chx);
    MAP_KOKKOS_DEVICE(m_t1); MAP_KOKKOS_DEVICE(m_q1); 
    MAP_KOKKOS_DEVICE(m_u1); MAP_KOKKOS_DEVICE(m_v1);
    MAP_KOKKOS_DEVICE(m_ps); MAP_KOKKOS_DEVICE(m_prsl1); MAP_KOKKOS_DEVICE(m_prcp); 
    MAP_KOKKOS_DEVICE(m_swdn); MAP_KOKKOS_DEVICE(m_lwdn); MAP_KOKKOS_DEVICE(m_swnet); MAP_KOKKOS_DEVICE(m_hgt); MAP_KOKKOS_DEVICE(m_prslki);
    MAP_KOKKOS_DEVICE(m_sigmaf); MAP_KOKKOS_DEVICE(m_sfemis); MAP_KOKKOS_DEVICE(m_alb); MAP_KOKKOS_DEVICE(m_shdmin); MAP_KOKKOS_DEVICE(m_shdmax); MAP_KOKKOS_DEVICE(m_lai);
    MAP_KOKKOS_DEVICE(m_stc); MAP_KOKKOS_DEVICE(m_smc); MAP_KOKKOS_DEVICE(m_slc);
    MAP_KOKKOS_DEVICE(m_tskin); MAP_KOKKOS_DEVICE(m_canopy); MAP_KOKKOS_DEVICE(m_snwdph); MAP_KOKKOS_DEVICE(m_sneqv);
    MAP_KOKKOS_DEVICE(m_hflux); MAP_KOKKOS_DEVICE(m_qflux); MAP_KOKKOS_DEVICE(m_evap); MAP_KOKKOS_DEVICE(m_gfx);
}

void LandProcess::unregister_openacc() {
    UNMAP_KOKKOS_DEVICE(m_islimsk); UNMAP_KOKKOS_DEVICE(m_vegtype); 
    UNMAP_KOKKOS_DEVICE(m_soiltype); UNMAP_KOKKOS_DEVICE(m_slopetype); UNMAP_KOKKOS_DEVICE(m_zorl); UNMAP_KOKKOS_DEVICE(m_cmx); UNMAP_KOKKOS_DEVICE(m_chx);
    UNMAP_KOKKOS_DEVICE(m_t1); UNMAP_KOKKOS_DEVICE(m_q1); 
    UNMAP_KOKKOS_DEVICE(m_u1); UNMAP_KOKKOS_DEVICE(m_v1);
    UNMAP_KOKKOS_DEVICE(m_ps); UNMAP_KOKKOS_DEVICE(m_prsl1); UNMAP_KOKKOS_DEVICE(m_prcp); 
    UNMAP_KOKKOS_DEVICE(m_swdn); UNMAP_KOKKOS_DEVICE(m_lwdn); UNMAP_KOKKOS_DEVICE(m_swnet); UNMAP_KOKKOS_DEVICE(m_hgt); UNMAP_KOKKOS_DEVICE(m_prslki);
    UNMAP_KOKKOS_DEVICE(m_sigmaf); UNMAP_KOKKOS_DEVICE(m_sfemis); UNMAP_KOKKOS_DEVICE(m_alb); UNMAP_KOKKOS_DEVICE(m_shdmin); UNMAP_KOKKOS_DEVICE(m_shdmax); UNMAP_KOKKOS_DEVICE(m_lai);
    UNMAP_KOKKOS_DEVICE(m_stc); UNMAP_KOKKOS_DEVICE(m_smc); UNMAP_KOKKOS_DEVICE(m_slc);
    UNMAP_KOKKOS_DEVICE(m_tskin); UNMAP_KOKKOS_DEVICE(m_canopy); UNMAP_KOKKOS_DEVICE(m_snwdph); UNMAP_KOKKOS_DEVICE(m_sneqv);
    UNMAP_KOKKOS_DEVICE(m_hflux); UNMAP_KOKKOS_DEVICE(m_qflux); UNMAP_KOKKOS_DEVICE(m_evap); UNMAP_KOKKOS_DEVICE(m_gfx);
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
