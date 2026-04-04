#include "physics/land/LandProcess.hpp"
#include <cmath>

#define MAP_KOKKOS_DEVICE(view) acc_map_data(view.data(), view.data(), view.span() * sizeof(view.data()[0]))                                           
#define UNMAP_KOKKOS_DEVICE(view) acc_unmap_data(view.data())

namespace VVM {
namespace Physics {

LandProcess::LandProcess(const Utils::ConfigurationManager& config, 
                         const Core::Grid& grid, 
                         const Core::Parameters& params, 
                         Core::HaloExchanger& halo_exchanger)
    : m_config(config), m_grid(grid), m_params(params), m_halo_exchanger(halo_exchanger)
{
    m_nx = m_grid.get_local_physical_points_x();
    m_ny = m_grid.get_local_physical_points_y();
    m_halo_x = m_grid.get_halo_cells(); 
    m_halo_y = m_grid.get_halo_cells();
    m_nsoil = 4;

    m_islimsk = view_2d_int_ll("lsm_islimsk", m_nx, m_ny);
    m_vegtype = view_2d_int_ll("lsm_vegtype", m_nx, m_ny);
    m_soiltyp = view_2d_int_ll("lsm_soiltyp", m_nx, m_ny);
    m_slopetyp = view_2d_int_ll("lsm_slopetyp", m_nx, m_ny);

    m_t1 = view_2d_ll("lsm_t1", m_nx, m_ny);
    m_q1 = view_2d_ll("lsm_q1", m_nx, m_ny);
    m_u1 = view_2d_ll("lsm_u1", m_nx, m_ny);
    m_v1 = view_2d_ll("lsm_v1", m_nx, m_ny);
    m_ps = view_2d_ll("lsm_ps", m_nx, m_ny);
    m_prcp = view_2d_ll("lsm_prcp", m_nx, m_ny);
    m_swdn = view_2d_ll("lsm_swdn", m_nx, m_ny);
    m_lwdn = view_2d_ll("lsm_lwdn", m_nx, m_ny);
    m_hgt = view_2d_ll("lsm_hgt", m_nx, m_ny);

    m_stc = view_3d_ll("lsm_stc", m_nx, m_nsoil, m_ny);
    m_smc = view_3d_ll("lsm_smc", m_nx, m_nsoil, m_ny);
    m_slc = view_3d_ll("lsm_slc", m_nx, m_nsoil, m_ny);

    m_tskin = view_2d_ll("lsm_tskin", m_nx, m_ny);
    m_canopy = view_2d_ll("lsm_canopy", m_nx, m_ny);
    m_snwdph = view_2d_ll("lsm_snwdph", m_nx, m_ny);
    m_zorl = view_2d_ll("lsm_zorl", m_nx, m_ny);

    m_hflux = view_2d_ll("lsm_hflux", m_nx, m_ny);
    m_qflux = view_2d_ll("lsm_qflux", m_nx, m_ny);
    m_evap = view_2d_ll("lsm_evap", m_nx, m_ny);
}

void LandProcess::init(Core::State& state) {
    int ny = m_ny+2*m_halo_y;
    int nx = m_nx+2*m_halo_x;
    state.add_field<2>("tskin", {ny, nx});
    state.add_field<2>("canopy", {ny, nx});
    state.add_field<2>("snwdph", {ny, nx});
    state.add_field<2>("zorl", {ny, nx});

    state.add_field<3>("stc", {m_nsoil, ny, nx});
    state.add_field<3>("smc", {m_nsoil, ny, nx});
    state.add_field<3>("slc", {m_nsoil, ny, nx});

    if (!state.has_field("hfx")) state.add_field<2>("hfx", {ny, nx});
    if (!state.has_field("le"))  state.add_field<2>("le", {ny, nx});

    auto& th_v = state.get_field<3>("th").get_device_data();
    auto& pibar_v = state.get_field<1>("pibar").get_device_data();
    auto& topo_v = state.get_field<2>("topo").get_device_data();
    
    Kokkos::parallel_for("InitLandStates", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;
            int hxp = topo_v(vi, vj) + 1;

            double atm_t1 = th_v(hxp, vj, vi) * pibar_v(hxp);
            m_tskin(i, j) = atm_t1;

            m_islimsk(i, j) = 1;
            m_vegtype(i, j) = 2;
            m_soiltyp(i, j) = 2;
            m_slopetyp(i, j) = 1;
            
            m_prcp(i,j) = 0.0;

            m_zorl(i, j) = 0.1;
            m_canopy(i, j) = 0.0;
            m_snwdph(i, j) = 0.0;

            m_t1(i, j) = th_v(hxp, vj, vi) * pibar_v(hxp);

            for(int k=0; k<m_nsoil; ++k) {
                m_stc(i, k, j) = atm_t1 - k * 0.5; // soil temperature
                m_smc(i, k, j) = 0.3; // volumetric soil moisture content
                m_slc(i, k, j) = 0.3; // liquid soil moisture
            }
        }
    );
    prepare_static_data(state);
    register_openacc();
}

void LandProcess::prepare_static_data(Core::State& state) {
    auto& topo_v = state.get_field<2>("topo").get_device_data();
    auto& pbar_v = state.get_field<1>("pbar").get_device_data();
    auto z_mid_v = m_params.z_mid.get_device_data();
    auto z_up_v = m_params.z_up.get_device_data();

    Kokkos::parallel_for("PrepareLandStaticData", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            int hx = topo_v(vi, vj);
            int hxp = topo_v(vi, vj) + 1;
            
            double dz = z_mid_v(hxp) - z_up_v(hx);
            m_hgt(i, j) = (dz < 2.0) ? 2.0 : dz;

            m_ps(i, j) = pbar_v(hxp); 

            // m_islimsk(i, j) = 1; // sea/land/ice, 0/1/2
            if (i < (m_nx+m_halo_x)/2) m_islimsk(i, j) = 0;
            else m_islimsk(i, j) = 1;
            m_vegtype(i, j) = 2; // vegetation type 20 types
            m_soiltyp(i, j) = 1; // soil type 19 types
            m_slopetyp(i, j) = 1; // slope 9 types
            m_zorl(i, j) = 0.1; // surface roughness (m)
        }
    );
}

void LandProcess::preprocessing_and_packing(Core::State& state) {
    auto& u_v  = state.get_field<3>("u").get_device_data();
    auto& v_v  = state.get_field<3>("v").get_device_data();
    auto& qv_v = state.get_field<3>("qv").get_device_data();
    auto& swdn_v = state.get_field<3>("swdn").get_device_data();
    auto& lwdn_v = state.get_field<3>("lwdn").get_device_data();

    auto& pr_v = state.get_field<1>("pbar").get_device_data();
    auto& topo_v = state.get_field<2>("topo").get_device_data();
    
    auto& tskin_v  = state.get_field<2>("tskin").get_device_data();
    auto& canopy_v = state.get_field<2>("canopy").get_device_data();
    auto& snwdph_v = state.get_field<2>("snwdph").get_device_data();
    auto& stc_v    = state.get_field<3>("stc").get_device_data();
    auto& smc_v    = state.get_field<3>("smc").get_device_data();
    auto& slc_v    = state.get_field<3>("slc").get_device_data();
    auto& precip_liq_surf_2d = state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();
    auto& precip_ice_surf_2d = state.get_field<2>("precip_ice_surf_mass").get_mutable_device_data();

    Kokkos::parallel_for("PackToLand", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            int hx = topo_v(vi, vj);
            int hxp = topo_v(vi, vj) + 1;
            
            m_u1(i, j) = u_v(hxp, vj, vi);
            m_v1(i, j) = v_v(hxp, vj, vi);
            m_q1(i, j) = qv_v(hxp, vj, vi);
            m_ps(i, j) = pr_v(hxp);

            m_swdn(i,j) = swdn_v(hxp, vj, vi);
            m_lwdn(i,j) = lwdn_v(hxp, vj, vi);
            m_prcp(i,j) = precip_liq_surf_2d(vj, vi) + precip_ice_surf_2d(vj, vi);
        }
    );
}

void LandProcess::postprocessing_and_unpacking(Core::State& state) {
    auto& hfx_v    = state.get_field<2>("hfx").get_mutable_device_data();
    auto& le_v     = state.get_field<2>("le").get_mutable_device_data();
    
    auto& tskin_v  = state.get_field<2>("tskin").get_mutable_device_data();
    auto& canopy_v = state.get_field<2>("canopy").get_mutable_device_data();
    auto& snwdph_v = state.get_field<2>("snwdph").get_mutable_device_data();
    auto& stc_v    = state.get_field<3>("stc").get_mutable_device_data();
    auto& smc_v    = state.get_field<3>("smc").get_mutable_device_data();
    auto& slc_v    = state.get_field<3>("slc").get_mutable_device_data();

    Kokkos::parallel_for("UnpackToVVM", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {m_nx, m_ny}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            const int vi = i + m_halo_x;
            const int vj = j + m_halo_y;

            hfx_v(vj, vi) = m_hflux(i, j);
            le_v(vj, vi)  = m_evap(i, j);

            tskin_v(vj, vi) = m_tskin(i, j);
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


void LandProcess::run(Core::State& state, double dt) {
    preprocessing_and_packing(state);

    Kokkos::fence();

    run_vvm_land_wrapper(m_nx, m_ny, m_nsoil, dt,
        m_islimsk.data(), m_vegtype.data(), m_soiltyp.data(), m_slopetyp.data(),
        m_t1.data(), m_q1.data(), m_u1.data(), m_v1.data(), m_ps.data(), 
        m_prcp.data(), m_swdn.data(), m_lwdn.data(), m_hgt.data(),
        m_stc.data(), m_smc.data(), m_slc.data(), m_tskin.data(), 
        m_canopy.data(), m_snwdph.data(),
        m_hflux.data(), m_qflux.data(), m_evap.data(), m_zorl.data());

    postprocessing_and_unpacking(state);
}

void LandProcess::finalize() {
    unregister_openacc();
    m_islimsk = {}; m_vegtype = {}; m_soiltyp = {}; m_slopetyp = {};
    m_zorl = {}; m_t1 = {}; m_q1 = {}; m_u1 = {}; m_v1 = {};
    m_ps = {}; m_prcp = {}; m_swdn = {}; m_lwdn = {};
    m_stc = {}; m_smc = {}; m_slc = {};
    m_tskin = {}; m_canopy = {}; m_snwdph = {};
    m_hflux = {}; m_qflux = {}; m_evap = {};
}

void LandProcess::register_openacc() {
    MAP_KOKKOS_DEVICE(m_islimsk); MAP_KOKKOS_DEVICE(m_vegtype); 
    MAP_KOKKOS_DEVICE(m_soiltyp); MAP_KOKKOS_DEVICE(m_slopetyp); MAP_KOKKOS_DEVICE(m_zorl);
    MAP_KOKKOS_DEVICE(m_t1); MAP_KOKKOS_DEVICE(m_q1); 
    MAP_KOKKOS_DEVICE(m_u1); MAP_KOKKOS_DEVICE(m_v1);
    MAP_KOKKOS_DEVICE(m_ps); MAP_KOKKOS_DEVICE(m_prcp); 
    MAP_KOKKOS_DEVICE(m_swdn); MAP_KOKKOS_DEVICE(m_lwdn); MAP_KOKKOS_DEVICE(m_hgt);
    MAP_KOKKOS_DEVICE(m_stc); MAP_KOKKOS_DEVICE(m_smc); MAP_KOKKOS_DEVICE(m_slc);
    MAP_KOKKOS_DEVICE(m_tskin); MAP_KOKKOS_DEVICE(m_canopy); MAP_KOKKOS_DEVICE(m_snwdph);
    MAP_KOKKOS_DEVICE(m_hflux); MAP_KOKKOS_DEVICE(m_qflux); MAP_KOKKOS_DEVICE(m_evap);
}

void LandProcess::unregister_openacc() {
    UNMAP_KOKKOS_DEVICE(m_islimsk); UNMAP_KOKKOS_DEVICE(m_vegtype); 
    UNMAP_KOKKOS_DEVICE(m_soiltyp); UNMAP_KOKKOS_DEVICE(m_slopetyp); UNMAP_KOKKOS_DEVICE(m_zorl);
    UNMAP_KOKKOS_DEVICE(m_t1); UNMAP_KOKKOS_DEVICE(m_q1); 
    UNMAP_KOKKOS_DEVICE(m_u1); UNMAP_KOKKOS_DEVICE(m_v1);
    UNMAP_KOKKOS_DEVICE(m_ps); UNMAP_KOKKOS_DEVICE(m_prcp); 
    UNMAP_KOKKOS_DEVICE(m_swdn); UNMAP_KOKKOS_DEVICE(m_lwdn); UNMAP_KOKKOS_DEVICE(m_hgt);
    UNMAP_KOKKOS_DEVICE(m_stc); UNMAP_KOKKOS_DEVICE(m_smc); UNMAP_KOKKOS_DEVICE(m_slc);
    UNMAP_KOKKOS_DEVICE(m_tskin); UNMAP_KOKKOS_DEVICE(m_canopy); UNMAP_KOKKOS_DEVICE(m_snwdph);
    UNMAP_KOKKOS_DEVICE(m_hflux); UNMAP_KOKKOS_DEVICE(m_qflux); UNMAP_KOKKOS_DEVICE(m_evap);
}


template<size_t Dim>
void LandProcess::calculate_tendencies(Core::State& state, 
                                          const std::string& var_name, 
                                          Core::Field<Dim>& out_tendency) {
    if (var_name != "th" && var_name != "qv") return;

    auto tend = out_tendency.get_mutable_device_data();
    int ny = m_grid.get_local_total_points_y();
    int nx = m_grid.get_local_total_points_x();
    int h = m_grid.get_halo_cells();
    const auto& rhobar = state.get_field<1>("rhobar").get_device_data(); // Density
    const auto& hx     = state.get_field<2>("topo").get_device_data();
    const auto& rdz = m_params.rdz; 
    const auto& flex_height_coef_mid = m_params.flex_height_coef_mid.get_device_data();

    if (var_name == "th") {
        const auto& flux = state.get_field<2>("hfx").get_device_data();
        Kokkos::parallel_for("SfcFlux_Tendency_TH",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({{h, h}}, {{ny-h, nx-h}}),
            KOKKOS_LAMBDA(const int j, const int i) {
                int hxp = hx(j,i)+1;
                tend(hxp, j, i) += flux(j, i) * flex_height_coef_mid(hxp) * rdz() / rhobar(hxp);
            }
        );
    } 
    else if (var_name == "qv") {
        const auto& flux = state.get_field<2>("le").get_device_data();
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

template void LandProcess::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency);

} // namespace Physics
} // namespace VVM
