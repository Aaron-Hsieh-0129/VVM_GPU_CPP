#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "physics/rrtmgp/VVM_rrtmgp_process_interface.hpp"
#include "physics/rrtmgp/rrtmgp_utils.hpp"
#include "physics/rrtmgp/shr_orb_mod_c2f.hpp"
#include "physics/share/scream_trcmix.hpp"

// #include "share/io/scream_scorpio_interface.hpp"
// #include "share/property_checks/field_within_interval_check.hpp"
#include "share/util/scream_common_physics_functions.hpp"
#include "share/util/scream_column_ops.hpp"

#include "ekat/ekat_assert.hpp"

#include "cpp/rrtmgp/mo_gas_concentrations.h"
#ifdef RRTMGP_ENABLE_YAKL
#include "YAKL.h"
#endif

namespace VVM {
namespace Physics {

using DefaultDevice = scream::DefaultDevice;
using KT = ekat::KokkosTypes<DefaultDevice>;
using ExeSpace = KT::ExeSpace;
using MemberType = KT::MemberType;

// Helper namespace for local utilities
namespace {
    // Helper to create subviews for RRTMGP which expects (col, lay)
    // Adapting EAMXX logic to VVM's context
    struct ConvertToRrtmgpSubview {
        int beg;
        int ncol;
        
        template <typename View>
        View subview1d(const View& v) const {
            return View(v, std::make_pair(beg, beg+ncol));
        }
        
        // Specialization for layout left buffer views (RRTMGP native)
        template <typename BufferView>
        BufferView subview2d_buffer(const BufferView& buffer_view) const {
            // Buffer views are (col, lay) in LayoutLeft
            return BufferView(buffer_view, std::make_pair(0, ncol), Kokkos::ALL);
        }
        
        template <typename BufferView>
        BufferView subview3d_buffer(const BufferView& buffer_view) const {
            // Buffer views are (col, lay, dim3) in LayoutLeft
            return BufferView(buffer_view, std::make_pair(0, ncol), Kokkos::ALL, Kokkos::ALL);
        }
    };
}

// =========================================================================================
// Constructor
// =========================================================================================
VVM_RRTMGP_Interface::VVM_RRTMGP_Interface(const VVM::Utils::ConfigurationManager& config, 
                                           const VVM::Core::Grid& grid, 
                                           const VVM::Core::Parameters& params)
    : m_lat("lat", {grid.get_local_physical_points_y(), grid.get_local_physical_points_x()})
    , m_lon("lon", {grid.get_local_physical_points_y(), grid.get_local_physical_points_x()})
{
    // Set grid dimensions from VVM Grid
    m_nx = grid.get_local_physical_points_x();
    m_ny = grid.get_local_physical_points_y();
    m_nz = grid.get_local_physical_points_z(); // Vertical levels
    m_ncol = m_nx * m_ny;
    m_nlay = m_nz;

    // Figure out radiation column chunks stats
    // For VVM, we can default to processing the whole grid at once if GPU memory allows,
    // or set a specific chunk size.
    m_col_chunk_size = m_ncol; // Default to full grid
    // if (config.has("column_chunk_size")) m_col_chunk_size = config.get<int>("column_chunk_size");

    m_num_col_chunks = (m_ncol + m_col_chunk_size - 1) / m_col_chunk_size;
    m_col_chunk_beg.resize(m_num_col_chunks + 1, 0);
    for (int i = 0; i < m_num_col_chunks; ++i) {
        m_col_chunk_beg[i + 1] = std::min(m_ncol, m_col_chunk_beg[i] + m_col_chunk_size);
    }

    // Initialize list of active gases
    // Hardcoded standard set for now, should ideally come from config
    m_gas_names = {"h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2"};
    m_ngas = m_gas_names.size();
    
    // Initialize logger
    if(spdlog::get("VVM_RRTMGP")) {
        m_logger = spdlog::get("VVM_RRTMGP");
    } 
    else {
        m_logger = spdlog::stdout_color_mt("VVM_RRTMGP");
    }
}

// =========================================================================================
// Helper: Calculate buffer size
// =========================================================================================
size_t VVM_RRTMGP_Interface::requested_buffer_size_in_bytes() const
{
  // This calculation mimics EAMXX to ensure enough memory for intermediate RRTMGP variables
  // Since VVM uses LayoutRight for fields, but RRTMGP prefers LayoutLeft, we need these buffers.
  const size_t interface_request =
    Buffer::num_1d_ncol*m_col_chunk_size +
    Buffer::num_2d_nlay*m_col_chunk_size*m_nlay +
    Buffer::num_2d_nlay_p1*m_col_chunk_size*(m_nlay+1) +
    Buffer::num_2d_nswbands*m_col_chunk_size*m_nswbands +
    Buffer::num_3d_nlev_nswbands*m_col_chunk_size*(m_nlay+1)*m_nswbands +
    Buffer::num_3d_nlev_nlwbands*m_col_chunk_size*(m_nlay+1)*m_nlwbands +
    Buffer::num_3d_nlay_nswbands*m_col_chunk_size*(m_nlay)*m_nswbands +
    Buffer::num_3d_nlay_nlwbands*m_col_chunk_size*(m_nlay)*m_nlwbands +
    Buffer::num_3d_nlay_nswgpts*m_col_chunk_size*(m_nlay)*m_nswgpts +
    Buffer::num_3d_nlay_nlwgpts*m_col_chunk_size*(m_nlay)*m_nlwgpts;

  // Add a safety margin or check alignment
  return interface_request * sizeof(Real);
}

// =========================================================================================
// Initialize Buffers
// =========================================================================================
void VVM_RRTMGP_Interface::init_buffers() {
    size_t request_bytes = requested_buffer_size_in_bytes();
    size_t request_size = request_bytes / sizeof(Real);
    
    // Allocate the storage view
    Kokkos::resize(m_buffer_storage, request_size);
    Real* mem = m_buffer_storage.data();

#ifdef RRTMGP_ENABLE_KOKKOS
    // Use placement new or view constructors to map the raw memory to Kokkos Views
    // Note: We assume LayoutLeft for these buffer views as per the header definition

    // 1D arrays
    m_buffer.mu0_k = decltype(m_buffer.mu0_k)(mem, m_col_chunk_size); mem += m_buffer.mu0_k.size();
    m_buffer.sfc_alb_dir_vis_k = decltype(m_buffer.sfc_alb_dir_vis_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_alb_dir_vis_k.size();
    m_buffer.sfc_alb_dir_nir_k = decltype(m_buffer.sfc_alb_dir_nir_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_alb_dir_nir_k.size();
    m_buffer.sfc_alb_dif_vis_k = decltype(m_buffer.sfc_alb_dif_vis_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_alb_dif_vis_k.size();
    m_buffer.sfc_alb_dif_nir_k = decltype(m_buffer.sfc_alb_dif_nir_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_alb_dif_nir_k.size();
    m_buffer.sfc_flux_dir_vis_k = decltype(m_buffer.sfc_flux_dir_vis_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_flux_dir_vis_k.size();
    m_buffer.sfc_flux_dir_nir_k = decltype(m_buffer.sfc_flux_dir_nir_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_flux_dir_nir_k.size();
    m_buffer.sfc_flux_dif_vis_k = decltype(m_buffer.sfc_flux_dif_vis_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_flux_dif_vis_k.size();
    m_buffer.sfc_flux_dif_nir_k = decltype(m_buffer.sfc_flux_dif_nir_k)(mem, m_col_chunk_size); mem += m_buffer.sfc_flux_dif_nir_k.size();
    m_buffer.cosine_zenith = decltype(m_buffer.cosine_zenith)(mem, m_col_chunk_size); mem += m_buffer.cosine_zenith.size();

    // 2D arrays
    m_buffer.p_lay_k = decltype(m_buffer.p_lay_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.p_lay_k.size();
    m_buffer.t_lay_k = decltype(m_buffer.t_lay_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.t_lay_k.size();
    m_buffer.z_del_k = decltype(m_buffer.z_del_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.z_del_k.size();
    m_buffer.p_del_k = decltype(m_buffer.p_del_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.p_del_k.size();
    m_buffer.qc_k = decltype(m_buffer.qc_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.qc_k.size();
    m_buffer.nc_k = decltype(m_buffer.nc_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.nc_k.size();
    m_buffer.qi_k = decltype(m_buffer.qi_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.qi_k.size();
    m_buffer.cldfrac_tot_k = decltype(m_buffer.cldfrac_tot_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.cldfrac_tot_k.size();
    m_buffer.eff_radius_qc_k = decltype(m_buffer.eff_radius_qc_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.eff_radius_qc_k.size();
    m_buffer.eff_radius_qi_k = decltype(m_buffer.eff_radius_qi_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.eff_radius_qi_k.size();
    m_buffer.tmp2d_k = decltype(m_buffer.tmp2d_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.tmp2d_k.size();
    m_buffer.lwp_k = decltype(m_buffer.lwp_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.lwp_k.size();
    m_buffer.iwp_k = decltype(m_buffer.iwp_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.iwp_k.size();
    m_buffer.sw_heating_k = decltype(m_buffer.sw_heating_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.sw_heating_k.size();
    m_buffer.lw_heating_k = decltype(m_buffer.lw_heating_k)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.lw_heating_k.size();
    m_buffer.p_lev_k = decltype(m_buffer.p_lev_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.p_lev_k.size();
    m_buffer.t_lev_k = decltype(m_buffer.t_lev_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.t_lev_k.size();
    m_buffer.d_tint = decltype(m_buffer.d_tint)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.d_tint.size();
    m_buffer.d_dz  = decltype(m_buffer.d_dz)(mem, m_col_chunk_size, m_nlay); mem += m_buffer.d_dz.size();

    // Fluxes
    m_buffer.sw_flux_up_k = decltype(m_buffer.sw_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_flux_up_k.size();
    m_buffer.sw_flux_dn_k = decltype(m_buffer.sw_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_flux_dn_k.size();
    m_buffer.sw_flux_dn_dir_k = decltype(m_buffer.sw_flux_dn_dir_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_flux_dn_dir_k.size();
    m_buffer.lw_flux_up_k = decltype(m_buffer.lw_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_flux_up_k.size();
    m_buffer.lw_flux_dn_k = decltype(m_buffer.lw_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_flux_dn_k.size();

    // Additional diagnostic fluxes (allocating them to avoid null pointers, even if not always used)
    m_buffer.sw_clnclrsky_flux_up_k = decltype(m_buffer.sw_clnclrsky_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clnclrsky_flux_up_k.size();
    m_buffer.sw_clnclrsky_flux_dn_k = decltype(m_buffer.sw_clnclrsky_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clnclrsky_flux_dn_k.size();
    m_buffer.sw_clnclrsky_flux_dn_dir_k = decltype(m_buffer.sw_clnclrsky_flux_dn_dir_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clnclrsky_flux_dn_dir_k.size();
    m_buffer.sw_clrsky_flux_up_k = decltype(m_buffer.sw_clrsky_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clrsky_flux_up_k.size();
    m_buffer.sw_clrsky_flux_dn_k = decltype(m_buffer.sw_clrsky_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clrsky_flux_dn_k.size();
    m_buffer.sw_clrsky_flux_dn_dir_k = decltype(m_buffer.sw_clrsky_flux_dn_dir_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clrsky_flux_dn_dir_k.size();
    m_buffer.sw_clnsky_flux_up_k = decltype(m_buffer.sw_clnsky_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clnsky_flux_up_k.size();
    m_buffer.sw_clnsky_flux_dn_k = decltype(m_buffer.sw_clnsky_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clnsky_flux_dn_k.size();
    m_buffer.sw_clnsky_flux_dn_dir_k = decltype(m_buffer.sw_clnsky_flux_dn_dir_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.sw_clnsky_flux_dn_dir_k.size();

    m_buffer.lw_clnclrsky_flux_up_k = decltype(m_buffer.lw_clnclrsky_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_clnclrsky_flux_up_k.size();
    m_buffer.lw_clnclrsky_flux_dn_k = decltype(m_buffer.lw_clnclrsky_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_clnclrsky_flux_dn_k.size();
    m_buffer.lw_clrsky_flux_up_k = decltype(m_buffer.lw_clrsky_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_clrsky_flux_up_k.size();
    m_buffer.lw_clrsky_flux_dn_k = decltype(m_buffer.lw_clrsky_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_clrsky_flux_dn_k.size();
    m_buffer.lw_clnsky_flux_up_k = decltype(m_buffer.lw_clnsky_flux_up_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_clnsky_flux_up_k.size();
    m_buffer.lw_clnsky_flux_dn_k = decltype(m_buffer.lw_clnsky_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1); mem += m_buffer.lw_clnsky_flux_dn_k.size();

    // 3D arrays (Spectral bands)
    m_buffer.sw_bnd_flux_up_k = decltype(m_buffer.sw_bnd_flux_up_k)(mem, m_col_chunk_size, m_nlay+1, m_nswbands); mem += m_buffer.sw_bnd_flux_up_k.size();
    m_buffer.sw_bnd_flux_dn_k = decltype(m_buffer.sw_bnd_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1, m_nswbands); mem += m_buffer.sw_bnd_flux_dn_k.size();
    m_buffer.sw_bnd_flux_dir_k = decltype(m_buffer.sw_bnd_flux_dir_k)(mem, m_col_chunk_size, m_nlay+1, m_nswbands); mem += m_buffer.sw_bnd_flux_dir_k.size();
    m_buffer.sw_bnd_flux_dif_k = decltype(m_buffer.sw_bnd_flux_dif_k)(mem, m_col_chunk_size, m_nlay+1, m_nswbands); mem += m_buffer.sw_bnd_flux_dif_k.size();
    m_buffer.lw_bnd_flux_up_k = decltype(m_buffer.lw_bnd_flux_up_k)(mem, m_col_chunk_size, m_nlay+1, m_nlwbands); mem += m_buffer.lw_bnd_flux_up_k.size();
    m_buffer.lw_bnd_flux_dn_k = decltype(m_buffer.lw_bnd_flux_dn_k)(mem, m_col_chunk_size, m_nlay+1, m_nlwbands); mem += m_buffer.lw_bnd_flux_dn_k.size();

    m_buffer.sfc_alb_dir_k = decltype(m_buffer.sfc_alb_dir_k)(mem, m_col_chunk_size, m_nswbands); mem += m_buffer.sfc_alb_dir_k.size();
    m_buffer.sfc_alb_dif_k = decltype(m_buffer.sfc_alb_dif_k)(mem, m_col_chunk_size, m_nswbands); mem += m_buffer.sfc_alb_dif_k.size();

    // Aerosols
    m_buffer.aero_tau_sw_k = decltype(m_buffer.aero_tau_sw_k)(mem, m_col_chunk_size, m_nlay, m_nswbands); mem += m_buffer.aero_tau_sw_k.size();
    m_buffer.aero_ssa_sw_k = decltype(m_buffer.aero_ssa_sw_k)(mem, m_col_chunk_size, m_nlay, m_nswbands); mem += m_buffer.aero_ssa_sw_k.size();
    m_buffer.aero_g_sw_k = decltype(m_buffer.aero_g_sw_k)(mem, m_col_chunk_size, m_nlay, m_nswbands); mem += m_buffer.aero_g_sw_k.size();
    m_buffer.aero_tau_lw_k = decltype(m_buffer.aero_tau_lw_k)(mem, m_col_chunk_size, m_nlay, m_nlwbands); mem += m_buffer.aero_tau_lw_k.size();

    // Clouds
    m_buffer.cld_tau_sw_bnd_k = decltype(m_buffer.cld_tau_sw_bnd_k)(mem, m_col_chunk_size, m_nlay, m_nswbands); mem += m_buffer.cld_tau_sw_bnd_k.size();
    m_buffer.cld_tau_lw_bnd_k = decltype(m_buffer.cld_tau_lw_bnd_k)(mem, m_col_chunk_size, m_nlay, m_nlwbands); mem += m_buffer.cld_tau_lw_bnd_k.size();
    m_buffer.cld_tau_sw_gpt_k = decltype(m_buffer.cld_tau_sw_gpt_k)(mem, m_col_chunk_size, m_nlay, m_nswgpts); mem += m_buffer.cld_tau_sw_gpt_k.size();
    m_buffer.cld_tau_lw_gpt_k = decltype(m_buffer.cld_tau_lw_gpt_k)(mem, m_col_chunk_size, m_nlay, m_nlwgpts); mem += m_buffer.cld_tau_lw_gpt_k.size();
#endif
}

// =========================================================================================
// Initialize
// =========================================================================================
void VVM_RRTMGP_Interface::initialize(VVM::Core::State& state) {
    using PC = scream::physics::Constants<Real>;

    // Configuration (Ideally get these from config object)
    m_rad_freq_in_steps = 1;
    m_orbital_year = 2000; 
    m_orbital_eccen = -9999;
    m_orbital_obliq = -9999;
    m_orbital_mvelp = -9999;
    m_fixed_solar_zenith_angle = -9999; // Negative means calc from orbit
    
    // Surface gas concentrations (Mole Fraction)
    m_co2vmr = 388.717e-6;
    m_n2ovmr = 323.141e-9;
    m_ch4vmr = 1807.851e-9;
    m_f11vmr = 768.7644e-12;
    m_f12vmr = 531.2820e-12;
    m_n2vmr = 0.7906;
    m_covmr = 1.0e-7;

    m_do_subcol_sampling = true;
    m_do_aerosol_rad = false;
    m_extra_clnclrsky_diag = false;
    m_extra_clnsky_diag = false;

    // Initialize molecular weights on device
    m_gas_mol_weights = real1dk("gas_mol_weights", m_ngas);
    auto gas_mol_w_host = Kokkos::create_mirror_view(m_gas_mol_weights);
    for (int igas = 0; igas < m_ngas; igas++) {
        gas_mol_w_host[igas] = PC::get_gas_mol_weight(m_gas_names[igas]);
    }
    Kokkos::deep_copy(m_gas_mol_weights, gas_mol_w_host);

    // Allocate buffers
    init_buffers();

    // Initialize RRTMGP
    // Note: Ensure these files are available in your running directory or point to correct path
    std::string coefficients_file_sw = "rrtmgp-data-sw-g224-2018-12-04.nc";
    std::string coefficients_file_lw = "rrtmgp-data-lw-g256-2018-12-04.nc";
    std::string cloud_optics_file_sw = "rrtmgp-cloud-optics-coeffs-sw.nc";
    std::string cloud_optics_file_lw = "rrtmgp-cloud-optics-coeffs-lw.nc";

#ifdef RRTMGP_ENABLE_KOKKOS
    // Setup gas names view for GasConcsK
    auto gas_names_view = string1dv(m_ngas);
    for(int i=0; i<m_ngas; ++i) gas_names_view(i) = m_gas_names[i];

    m_gas_concs_k.init(gas_names_view, m_col_chunk_size, m_nlay);
    
    // Call static initialization for RRTMGP
    // This reads the NetCDF coefficient files
    interface_t::rrtmgp_initialize(
          m_gas_concs_k,
          coefficients_file_sw, coefficients_file_lw,
          cloud_optics_file_sw, cloud_optics_file_lw,
          m_logger
    );

    VALIDATE_KOKKOS(m_gas_concs, m_gas_concs_k);
#endif

    // TODO: Initialize m_lat and m_lon from State if needed
    // VVM::Core::State should ideally provide access to grid coordinates if they vary.
    // For now, assuming VVM Grid or State handles initialization of lat/lon elsewhere,
    // or we can calculate them here if the grid is regular.
}

// =========================================================================================
// Run
// =========================================================================================
void VVM_RRTMGP_Interface::run(VVM::Core::State& state, const double dt, const double current_time) {
#ifdef RRTMGP_ENABLE_KOKKOS
    using PC = scream::physics::Constants<Real>;

    // Access VVM Fields
    // We assume VVM::Core::State has get_field methods returning Field objects
    // VVM layout is assumed to be (k, j, i) or (z, y, x) which maps to (level, col) conceptually
    // However, VVM's Field implementation likely uses (k, j, i) = (z, y, x).
    
    auto f_pmid = state.get_field<3>("p_mid").get_device_data(); // Pressure at mid-points
    auto f_temp = state.get_field<3>("T_mid").get_device_data(); // Temperature
    auto f_qv   = state.get_field<3>("qv").get_device_data();    // Water vapor specific humidity
    auto f_qc   = state.get_field<3>("qc").get_device_data();    // Cloud water mixing ratio
    auto f_qi   = state.get_field<3>("qi").get_device_data();    // Cloud ice mixing ratio
    auto f_pdel = state.get_field<3>("p_del").get_device_data(); // Pressure thickness (delta p)
    
    // Optional fields (if not present, you might need to handle logic)
    // auto f_cld  = state.get_field<3>("cldfrac").get_device_data(); 
    
    // Outputs
    auto f_sw_heating = state.get_field<3>("sw_heating").get_mutable_device_data();
    auto f_lw_heating = state.get_field<3>("lw_heating").get_mutable_device_data();
    auto f_sfc_flux_sw = state.get_field<2>("sfc_flux_sw_net").get_mutable_device_data();
    auto f_sfc_flux_lw = state.get_field<2>("sfc_flux_lw_dn").get_mutable_device_data();

    // Surface albedo (usually from land model or fixed)
    // Assuming 2D fields exists in State
    // auto f_sfc_alb = state.get_field<2>("sfc_alb").get_device_data();

    int nx = m_nx;
    // int ny = m_ny; // Unused variable warning fix
    int nz = m_nlay;
    
    // Calculate Cosine Zenith Angle (mu0)
    // For simplicity, just use a fixed angle or placeholder if orbital logic isn't fully wired
    // Real cos_z_val = 0.5; // 60 degrees
    
    // Loop over chunks
    for (int ic=0; ic<m_num_col_chunks; ++ic) {
        const int beg  = m_col_chunk_beg[ic];
        const int ncol = m_col_chunk_beg[ic+1] - beg;

        // Helper to manage subviews (mapping chunk local index to global index)
        ConvertToRrtmgpSubview conv = {beg, ncol};
        
        // Get subviews of RRTMGP buffers (LayoutLeft)
        auto p_lay_k = conv.subview2d_buffer(m_buffer.p_lay_k);
        auto t_lay_k = conv.subview2d_buffer(m_buffer.t_lay_k);
        auto p_del_k = conv.subview2d_buffer(m_buffer.p_del_k);
        auto qc_k    = conv.subview2d_buffer(m_buffer.qc_k);
        auto qi_k    = conv.subview2d_buffer(m_buffer.qi_k);
        auto mu0_k   = conv.subview1d(m_buffer.mu0_k);
        
        // Output buffers
        auto sw_heating_k = conv.subview2d_buffer(m_buffer.sw_heating_k);
        auto lw_heating_k = conv.subview2d_buffer(m_buffer.lw_heating_k);
        auto sw_flux_up_k = conv.subview2d_buffer(m_buffer.sw_flux_up_k);
        auto sw_flux_dn_k = conv.subview2d_buffer(m_buffer.sw_flux_dn_k);
        auto lw_flux_dn_k = conv.subview2d_buffer(m_buffer.lw_flux_dn_k);

        // 1. Pre-processing: Flatten VVM (k, j, i) -> RRTMGP (col, lay)
        //    RRTMGP expects LayoutLeft (stride 1 on columns typically if (col, lay), wait...)
        //    Note: In scream_rrtmgp_interface definition, layout is LayoutLeft.
        //    For LayoutLeft, stride 1 is on the first dimension.
        //    If real2d is (col, lay), then consecutive columns are contiguous in memory.
        //    We need to ensure the mapping is correct.
        
        Kokkos::parallel_for(Kokkos::RangePolicy<ExeSpace>(0, ncol * nz), KOKKOS_LAMBDA(const int idx) {
            int k = idx % nz;           // Level index
            int col_local = idx / nz;   // Local column index in this chunk
            
            int col_global = beg + col_local;
            int j = col_global / nx;
            int i = col_global % nx;
            
            // Map VVM data to RRTMGP buffers
            p_lay_k(col_local, k) = f_pmid(k, j, i);
            t_lay_k(col_local, k) = f_temp(k, j, i);
            p_del_k(col_local, k) = f_pdel(k, j, i);
            qc_k(col_local, k)    = f_qc(k, j, i);
            qi_k(col_local, k)    = f_qi(k, j, i);
            
            // Initialize other fields if necessary (e.g. effective radius)
            // m_buffer.eff_radius_qc_k(col_local, k) = 10.0e-6; // placeholder
            
            // Set surface albedo (broadband)
            if (k == 0) { // Do once per column
                 // Placeholder for surface albedo mapping
                 // m_buffer.sfc_alb_dir_vis_k(col_local) = ...
                 mu0_k(col_local) = 0.5; // Fixed zenith angle for testing
            }
        });

        // 2. Setup Gas Concentrations
        //    For a first pass, we can set spatially uniform concentrations using the VMRs in m_gas_concs_k
        m_gas_concs_k.ncol = ncol;
        // Note: To set actual values, we need to call m_gas_concs_k.set_vmr(...)
        // Since we don't have trcmix running here yet, we assume constant profiles initialized in run_impl of EAMXX
        // but since we skipped that, we should probably set them here or in initialize.
        // Ideally, m_gas_concs_k should be populated with 3D data if VMR varies.
        
        // 3. Run RRTMGP
        //    We need to populate all arguments. 
        //    For placeholders (like aerosols), we ensure they are zeroed in init_buffers or here.
        
        // Interface requires full argument list.
        // Using placeholder logic for optional/unused fields.
        
        interface_t::rrtmgp_main(
            ncol, m_nlay,
            p_lay_k, t_lay_k, 
            m_buffer.p_lev_k, m_buffer.t_lev_k, // p_lev, t_lev (need to be calculated if not in VVM)
            m_gas_concs_k,
            m_buffer.sfc_alb_dir_k, m_buffer.sfc_alb_dif_k, mu0_k,
            m_buffer.lwp_k, m_buffer.iwp_k, m_buffer.eff_radius_qc_k, m_buffer.eff_radius_qi_k, m_buffer.cldfrac_tot_k,
            m_buffer.aero_tau_sw_k, m_buffer.aero_ssa_sw_k, m_buffer.aero_g_sw_k, m_buffer.aero_tau_lw_k,
            m_buffer.cld_tau_sw_bnd_k, m_buffer.cld_tau_lw_bnd_k,
            m_buffer.cld_tau_sw_gpt_k, m_buffer.cld_tau_lw_gpt_k,
            m_buffer.sw_flux_up_k, m_buffer.sw_flux_dn_k, m_buffer.sw_flux_dn_dir_k,
            m_buffer.lw_flux_up_k, m_buffer.lw_flux_dn_k,
            m_buffer.sw_clnclrsky_flux_up_k, m_buffer.sw_clnclrsky_flux_dn_k, m_buffer.sw_clnclrsky_flux_dn_dir_k,
            m_buffer.sw_clrsky_flux_up_k, m_buffer.sw_clrsky_flux_dn_k, m_buffer.sw_clrsky_flux_dn_dir_k,
            m_buffer.sw_clnsky_flux_up_k, m_buffer.sw_clnsky_flux_dn_k, m_buffer.sw_clnsky_flux_dn_dir_k,
            m_buffer.lw_clnclrsky_flux_up_k, m_buffer.lw_clnclrsky_flux_dn_k,
            m_buffer.lw_clrsky_flux_up_k, m_buffer.lw_clrsky_flux_dn_k,
            m_buffer.lw_clnsky_flux_up_k, m_buffer.lw_clnsky_flux_dn_k,
            m_buffer.sw_bnd_flux_up_k, m_buffer.sw_bnd_flux_dn_k, m_buffer.sw_bnd_flux_dir_k,
            m_buffer.lw_bnd_flux_up_k, m_buffer.lw_bnd_flux_dn_k,
            1.0, // tsi_scaling
            m_logger
        );
        
        // 4. Compute Heating Rates (if not done inside rrtmgp_main, EAMXX calculates it after)
        //    EAMXX calculates heating rate separately using `compute_heating_rate`
        //    Here we calculate it inline or assume `sw_heating_k` is updated if we added that logic.
        //    Wait, `rrtmgp_main` in EAMXX interface calculates fluxes. Heating rate is derived.
        
        // Calculate heating rate: dT/dt = (g/Cp) * d(Flux)/dp
        // d(Flux) = (Flux_net_top - Flux_net_bot)
        // Note: VVM might expect K/s.
        Real grav = PC::gravit;
        Real Cp   = PC::Cpair;
        
        Kokkos::parallel_for(Kokkos::RangePolicy<ExeSpace>(0, ncol * nz), KOKKOS_LAMBDA(const int idx) {
            int k = idx % nz;
            int col_local = idx / nz;
            
            // Net flux convergence for layer k
            // Fluxes are defined at interfaces (p_lev). Layer k is between k and k+1?
            // EAMXX/RRTMGP usually: 0 is TOA.
            // Net Flux SW = dn - up
            Real net_sw_top = sw_flux_dn_k(col_local, k)   - sw_flux_up_k(col_local, k);
            Real net_sw_bot = sw_flux_dn_k(col_local, k+1) - sw_flux_up_k(col_local, k+1);
            
            Real net_lw_top = lw_flux_dn_k(col_local, k)   - lw_flux_up_k(col_local, k);
            Real net_lw_bot = lw_flux_dn_k(col_local, k+1) - lw_flux_up_k(col_local, k+1);
            
            Real heating_sw = (grav / Cp) * (net_sw_top - net_sw_bot) / p_del_k(col_local, k);
            Real heating_lw = (grav / Cp) * (net_lw_top - net_lw_bot) / p_del_k(col_local, k);
            
            sw_heating_k(col_local, k) = heating_sw;
            lw_heating_k(col_local, k) = heating_lw;
        });
        
        // 5. Copy back to VVM Fields
        Kokkos::parallel_for(Kokkos::RangePolicy<ExeSpace>(0, ncol * nz), KOKKOS_LAMBDA(const int idx) {
            int k = idx % nz;
            int col_local = idx / nz;
            int col_global = beg + col_local;
            int j = col_global / nx;
            int i = col_global % nx;
            
            f_sw_heating(k, j, i) = sw_heating_k(col_local, k);
            f_lw_heating(k, j, i) = lw_heating_k(col_local, k);
            
            if (k == 0) {
                // Surface fluxes (Net SW, Down LW)
                f_sfc_flux_sw(j, i) = sw_flux_dn_k(col_local, nz) - sw_flux_up_k(col_local, nz);
                f_sfc_flux_lw(j, i) = lw_flux_dn_k(col_local, nz);
            }
        });
    }
#endif
}
//
// =========================================================================================
// Finalize
// =========================================================================================
void VVM_RRTMGP_Interface::finalize() {
#ifdef RRTMGP_ENABLE_KOKKOS
    m_gas_concs_k.reset();
    interface_t::rrtmgp_finalize();
#endif
}

} // namespace Physics
} // namespace VVM
