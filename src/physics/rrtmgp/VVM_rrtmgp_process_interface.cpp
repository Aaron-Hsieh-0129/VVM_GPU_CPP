#include "physics/rrtmgp/VVM_rrtmgp_process_interface.hpp"
#include "physics/rrtmgp/rrtmgp_utils.hpp"
#include "physics/rrtmgp/shr_orb_mod_c2f.hpp"
#include "share/physics/eamxx_trcmix.hpp"
#include "share/physics/eamxx_common_physics_functions.hpp"

#include <ekat_assert.hpp>
#include <ekat_units.hpp>

namespace VVM {
namespace Physics {
namespace RRTMGP {

using Real = scream::Real;
using Int = scream::Int;
using PF = scream::PhysicsFunctions<DefaultDevice>;
using PC = scream::physics::Constants<Real>;

RRTMGPRadiation::RRTMGPRadiation(const VVM::Core::Grid& grid, const VVM::Utils::ConfigurationManager& config)
    : m_grid(grid), m_config(config)
{
    // Initialize dimensions
    m_ncol = m_grid.get_local_physical_points_x() * m_grid.get_local_physical_points_y();
    m_nlay = m_grid.get_local_physical_points_z();

    // Column chunking (mimic EAMxx logic or just use ncol)
    m_col_chunk_size = m_config.get_value<int>("physics.rrtmgp.column_chunk_size", m_ncol);
    if (m_col_chunk_size <= 0) m_col_chunk_size = m_ncol;
    
    m_num_col_chunks = (m_ncol + m_col_chunk_size - 1) / m_col_chunk_size;
    m_col_chunk_beg.resize(m_num_col_chunks + 1, 0);
    for (int i = 0; i < m_num_col_chunks; ++i) {
        m_col_chunk_beg[i + 1] = std::min(m_ncol, m_col_chunk_beg[i] + m_col_chunk_size);
    }

    // Configuration
    m_do_aerosol_rad = m_config.get_value<bool>("physics.rrtmgp.do_aerosol_rad", false);
    m_extra_clnsky_diag = m_config.get_value<bool>("physics.rrtmgp.extra_clnsky_diag", false);
    m_extra_clnclrsky_diag = m_config.get_value<bool>("physics.rrtmgp.extra_clnclrsky_diag", false);

    m_orbital_year = m_config.get_value<int>("physics.rrtmgp.orbital_year", -9999);
    m_orbital_eccen = m_config.get_value<double>("physics.rrtmgp.orbital_eccentricity", -9999.0);
    m_orbital_obliq = m_config.get_value<double>("physics.rrtmgp.orbital_obliquity", -9999.0);
    m_orbital_mvelp = m_config.get_value<double>("physics.rrtmgp.orbital_mvelp", -9999.0);
    m_fixed_total_solar_irradiance = m_config.get_value<double>("physics.rrtmgp.fixed_total_solar_irradiance", -9999.0);
    m_fixed_solar_zenith_angle = m_config.get_value<double>("physics.rrtmgp.fixed_solar_zenith_angle", -9999.0);

    m_rad_freq_in_steps = m_config.get_value<int>("physics.rrtmgp.rad_frequency", 1);
    m_do_subcol_sampling = m_config.get_value<bool>("physics.rrtmgp.do_subcol_sampling", true);

    // Gas configuration
    std::vector<std::string> default_gases = {"h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2"};
    for (const auto& gas : default_gases) {
        m_gas_names.push_back(gas);
    }
    m_ngas = m_gas_names.size();

    // Prescribed gas values
    m_co2vmr = m_config.get_value<double>("physics.rrtmgp.co2vmr", 388.717e-6);
    m_n2ovmr = m_config.get_value<double>("physics.rrtmgp.n2ovmr", 323.141e-9);
    m_ch4vmr = m_config.get_value<double>("physics.rrtmgp.ch4vmr", 1807.851e-9);
    m_f11vmr = m_config.get_value<double>("physics.rrtmgp.f11vmr", 768.7644e-12);
    m_f12vmr = m_config.get_value<double>("physics.rrtmgp.f12vmr", 531.2820e-12);
    m_n2vmr  = m_config.get_value<double>("physics.rrtmgp.n2vmr", 0.7906);
    m_covmr  = m_config.get_value<double>("physics.rrtmgp.covmr", 1.0e-7);

    // Initialize Lat/Lon fields
    // m_lat = VVM::Core::Field<2>("lat", {m_grid.get_local_total_points_y(), m_grid.get_local_total_points_x()});
    // m_lon = VVM::Core::Field<2>("lon", {m_grid.get_local_total_points_y(), m_grid.get_local_total_points_x()});

    m_nswgpts = m_config.get_value<int>("physics.rrtmgp.nswgpts", 112);
    m_nlwgpts = m_config.get_value<int>("physics.rrtmgp.nlwgpts", 128);
}

RRTMGPRadiation::~RRTMGPRadiation() {}

void RRTMGPRadiation::initialize(const VVM::Core::State& state) {
    using PC = scream::physics::Constants<Real>;

    // Determine rad timestep, specified as number of atm steps
    m_rad_freq_in_steps = m_config.get_value<Int>("physics.rrtmgp.rad_frequency", 1);

    // Determine orbital year. If orbital_year is negative, use current year
    // from timestamp for orbital year; if positive, use provided orbital year
    // for duration of simulation.
    m_orbital_year = m_config.get_value<Int>("orbital_year",-9999);
    // Get orbital parameters from yaml file
    m_orbital_eccen = m_config.get_value<double>("orbital_eccentricity",-9999);
    m_orbital_obliq = m_config.get_value<double>("orbital_obliquity"   ,-9999);
    m_orbital_mvelp = m_config.get_value<double>("orbital_mvelp"       ,-9999);


    // Value for prescribing an invariant solar constant (i.e. total solar irradiance at
    // TOA).  Used for idealized experiments such as RCE. Disabled when value is less than 0.
    m_fixed_total_solar_irradiance = m_config.get_value<double>("fixed_total_solar_irradiance", -9999);

    // Determine whether or not we are using a fixed solar zenith angle (positive value)
    m_fixed_solar_zenith_angle = m_config.get_value<double>("fixed_solar_zenith_angle", -9999);

    // Get prescribed surface values of greenhouse gases
    m_co2vmr     = m_config.get_value<double>("co2vmr", 388.717e-6);
    m_n2ovmr     = m_config.get_value<double>("n2ovmr", 323.141e-9);
    m_ch4vmr     = m_config.get_value<double>("ch4vmr", 1807.851e-9);
    m_f11vmr     = m_config.get_value<double>("f11vmr", 768.7644e-12);
    m_f12vmr     = m_config.get_value<double>("f12vmr", 531.2820e-12);
    m_n2vmr      = m_config.get_value<double>("n2vmr", 0.7906);
    m_covmr      = m_config.get_value<double>("covmr", 1.0e-7);

    // Whether or not to do MCICA subcolumn sampling
    m_do_subcol_sampling = m_config.get_value<bool>("do_subcol_sampling",true);

    // Initialize kokkos
    init_kls();

    // Names of active gases
    auto gas_names_offset = string1dv(m_ngas);
    m_gas_mol_weights     = real1dk("gas_mol_weights",m_ngas);
    // the lookup function for getting the gas mol weights doesn't work on device
    auto gas_mol_w_host = Kokkos::create_mirror_view(m_gas_mol_weights);
    for (int igas = 0; igas < m_ngas; igas++) {
        const auto& gas_name = m_gas_names[igas];

        gas_names_offset[igas] = gas_name;
        gas_mol_w_host[igas]   = PC::get_gas_mol_weight(gas_name);

    }
    Kokkos::deep_copy(m_gas_mol_weights,gas_mol_w_host);

    std::string coefficients_file_sw = m_config.get_value<std::string>("physics.rrtmgp.coefficients_file_sw", "../rundata/rrtmgp/rrtmgp-data-sw-g112-210809.nc");
    std::string coefficients_file_lw = m_config.get_value<std::string>("physics.rrtmgp.coefficients_file_lw", "../rundata/rrtmgp/rrtmgp-data-lw-g128-210809.nc");
    std::string cloud_optics_file_sw = m_config.get_value<std::string>("physics.rrtmgp.cloud_optics_file_sw", "../rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-sw.nc");
    std::string cloud_optics_file_lw = m_config.get_value<std::string>("physics.rrtmgp.cloud_optics_file_lw", "../rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-lw.nc");
    const double multiplier = m_config.get_value<double>("pool_size_multiplier", 1.0);

    m_gas_concs_k.init(gas_names_offset,m_col_chunk_size,m_nlay);
    interface_t::rrtmgp_initialize(
        m_gas_concs_k,
        coefficients_file_sw, coefficients_file_lw,
        cloud_optics_file_sw, cloud_optics_file_lw,
        nullptr,
        multiplier
    );

    // Set property checks for fields in this process
    // add_invariant_check<FieldWithinIntervalCheck>(get_field_out("T_mid"),m_grid,100.0, 500.0,false);

    // VMR of n2 and co is currently prescribed as a constant value, read from file
    // if (has_computed_field("n2_volume_mix_ratio",m_grid->name())) {
    //     auto n2_vmr = get_field_out("n2_volume_mix_ratio").get_view<Real**>();
    //     Kokkos::deep_copy(n2_vmr, m_params.get<double>("n2vmr", 0.7906));
    // }
    // if (has_computed_field("co_volume_mix_ratio",m_grid->name())) {
    //     auto co_vmr = get_field_out("co_volume_mix_ratio").get_view<Real**>();
    //     Kokkos::deep_copy(co_vmr, m_params.get<double>("covmr", 1.0e-7));
    // }
    
    init_buffers();
}

size_t RRTMGPRadiation::requested_buffer_size_in_bytes() const {
    const size_t interface_request =
        Buffer::num_1d_ncol * m_col_chunk_size +
        Buffer::num_2d_nlay * m_col_chunk_size * m_nlay +
        Buffer::num_2d_nlay_p1 * m_col_chunk_size * (m_nlay + 1) +
        Buffer::num_2d_nswbands * m_col_chunk_size * m_nswbands +
        Buffer::num_3d_nlev_nswbands * m_col_chunk_size * (m_nlay + 1) * m_nswbands +
        Buffer::num_3d_nlev_nlwbands * m_col_chunk_size * (m_nlay + 1) * m_nlwbands +
        Buffer::num_3d_nlay_nswbands * m_col_chunk_size * m_nlay * m_nswbands +
        Buffer::num_3d_nlay_nlwbands * m_col_chunk_size * m_nlay * m_nlwbands +
        Buffer::num_3d_nlay_nswgpts * m_col_chunk_size * m_nlay * m_nswgpts +
        Buffer::num_3d_nlay_nlwgpts * m_col_chunk_size * m_nlay * m_nlwgpts;

    return interface_request * sizeof(Real);
}

void RRTMGPRadiation::init_buffers() {
    size_t bytes = requested_buffer_size_in_bytes();
    size_t size = bytes / sizeof(Real);
    m_buffer_storage = Kokkos::View<Real*, DefaultDevice>("rrtmgp_buffer", size);
    
    Real* mem = m_buffer_storage.data();

    // Helper lambda to allocate unmanaged views
    auto alloc_1d = [&](int n) {
        auto v = ureal1dk(mem, n);
        mem += n;
        return v;
    };
    auto alloc_2d = [&](int n1, int n2) {
        auto v = ureal2dk(mem, n1, n2);
        mem += n1 * n2;
        return v;
    };
    auto alloc_3d = [&](int n1, int n2, int n3) {
        auto v = ureal3dk(mem, n1, n2, n3);
        mem += n1 * n2 * n3;
        return v;
    };

    // 1d arrays
    m_buffer.sfc_alb_dir_vis_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_alb_dir_nir_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_alb_dif_vis_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_alb_dif_nir_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_flux_dir_vis_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_flux_dir_nir_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_flux_dif_vis_k = alloc_1d(m_col_chunk_size);
    m_buffer.sfc_flux_dif_nir_k = alloc_1d(m_col_chunk_size);

    // 2d arrays (ncol, nlay)
    m_buffer.d_dz = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.p_lay_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.t_lay_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.z_del_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.p_del_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.qc_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.nc_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.qi_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.cldfrac_tot_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.eff_radius_qc_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.eff_radius_qi_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.tmp2d_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.lwp_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.iwp_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.sw_heating_k = alloc_2d(m_col_chunk_size, m_nlay);
    m_buffer.lw_heating_k = alloc_2d(m_col_chunk_size, m_nlay);

    // 2d arrays (ncol, nlay+1)
    m_buffer.d_tint = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.p_lev_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.t_lev_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_flux_dn_dir_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clnclrsky_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clnclrsky_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clnclrsky_flux_dn_dir_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clrsky_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clrsky_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clrsky_flux_dn_dir_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clnsky_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clnsky_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.sw_clnsky_flux_dn_dir_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_clnclrsky_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_clnclrsky_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_clrsky_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_clrsky_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_clnsky_flux_up_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.lw_clnsky_flux_dn_k = alloc_2d(m_col_chunk_size, m_nlay + 1);

    // 3d arrays
    m_buffer.sw_bnd_flux_up_k = alloc_3d(m_col_chunk_size, m_nlay + 1, m_nswbands);
    m_buffer.sw_bnd_flux_dn_k = alloc_3d(m_col_chunk_size, m_nlay + 1, m_nswbands);
    m_buffer.sw_bnd_flux_dir_k = alloc_3d(m_col_chunk_size, m_nlay + 1, m_nswbands);
    m_buffer.sw_bnd_flux_dif_k = alloc_3d(m_col_chunk_size, m_nlay + 1, m_nswbands);
    m_buffer.lw_bnd_flux_up_k = alloc_3d(m_col_chunk_size, m_nlay + 1, m_nlwbands);
    m_buffer.lw_bnd_flux_dn_k = alloc_3d(m_col_chunk_size, m_nlay + 1, m_nlwbands);

    m_buffer.sfc_alb_dir_k = alloc_2d(m_col_chunk_size, m_nswbands);
    m_buffer.sfc_alb_dif_k = alloc_2d(m_col_chunk_size, m_nswbands);

    m_buffer.aero_tau_sw_k = alloc_3d(m_col_chunk_size, m_nlay, m_nswbands);
    m_buffer.aero_ssa_sw_k = alloc_3d(m_col_chunk_size, m_nlay, m_nswbands);
    m_buffer.aero_g_sw_k = alloc_3d(m_col_chunk_size, m_nlay, m_nswbands);
    m_buffer.aero_tau_lw_k = alloc_3d(m_col_chunk_size, m_nlay, m_nlwbands);

    m_buffer.cld_tau_sw_bnd_k = alloc_3d(m_col_chunk_size, m_nlay, m_nswbands);
    m_buffer.cld_tau_lw_bnd_k = alloc_3d(m_col_chunk_size, m_nlay, m_nlwbands);

    m_buffer.cld_tau_sw_gpt_k = alloc_3d(m_col_chunk_size, m_nlay, m_nswgpts);
    m_buffer.cld_tau_lw_gpt_k = alloc_3d(m_col_chunk_size, m_nlay, m_nlwgpts);
}

void RRTMGPRadiation::finalize() {
    m_buffer_storage = Kokkos::View<Real*, DefaultDevice>();
}

void RRTMGPRadiation::run(const double dt, VVM::Core::State& state) {
    const int nx = m_grid.get_local_physical_points_x();
    const int halo = m_grid.get_halo_cells();
    const int nlay = m_nlay;

    // Get VVM fields
    auto pbar = state.get_field<1>("pbar").get_device_data(); 
    auto qc = state.get_field<3>("qc").get_device_data();
    auto nc = state.get_field<3>("nc").get_device_data();
    auto qi = state.get_field<3>("qi").get_device_data();
    auto qv = state.get_field<3>("qv").get_device_data();
    auto th = state.get_field<3>("th").get_device_data();
    auto pibar = state.get_field<1>("pibar").get_device_data(); 

    // Orbital parameters and Zenith Angle
    double eccf = 1.0; 
    double delta = 0.0; 
    
    if (m_fixed_total_solar_irradiance > 0) {
        eccf = m_fixed_total_solar_irradiance / 1360.9;
    }

    // Loop over chunks
    for (int ic = 0; ic < m_num_col_chunks; ++ic) {
        int beg = m_col_chunk_beg[ic];
        int ncol = m_col_chunk_beg[ic + 1] - beg;

        // Calculate Zenith Angle (mu0) on Host
        Kokkos::View<Real*, Kokkos::HostSpace> h_mu0("h_mu0", ncol);
        
        if (m_fixed_solar_zenith_angle > 0) {
            for (int i = 0; i < ncol; ++i) h_mu0(i) = m_fixed_solar_zenith_angle;
        } else {
            for (int i = 0; i < ncol; ++i) h_mu0(i) = 0.5; 
        }
        
        Kokkos::View<Real*, DefaultDevice> mu0_k("mu0_k", ncol);
        Kokkos::deep_copy(mu0_k, h_mu0);

        auto buffer = m_buffer; // Local copy for lambda capture

        // Pack data for this chunk
        const int nlay_local = m_nlay;
        Kokkos::parallel_for("pack_chunk_data", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;

                for (int k = 0; k < nlay_local; ++k) {
                    buffer.p_lay_k(i, k) = pbar(k + halo); 
                    buffer.t_lay_k(i, k) = th(k + halo, iy + halo, ix + halo) * pibar(k + halo);
                    buffer.qc_k(i, k) = qc(k + halo, iy + halo, ix + halo);
                    buffer.nc_k(i, k) = nc(k + halo, iy + halo, ix + halo);
                    buffer.qi_k(i, k) = qi(k + halo, iy + halo, ix + halo);
                    buffer.cldfrac_tot_k(i, k) = (buffer.qc_k(i, k) > 1e-6 || buffer.qi_k(i, k) > 1e-6) ? 1.0 : 0.0;
                    buffer.eff_radius_qc_k(i, k) = 10.0e-6; 
                    buffer.eff_radius_qi_k(i, k) = 25.0e-6;
                }
                
                for (int k = 0; k <= nlay_local; ++k) {
                     buffer.p_lev_k(i, k) = pbar(std::min(k + halo, nlay_local + halo - 1)); 
                }
                
                for (int k = 0; k < nlay_local; ++k) {
                    buffer.p_del_k(i, k) = buffer.p_lev_k(i, k+1) - buffer.p_lev_k(i, k); 
                }
            });
        
        // Fill gas concentrations
        auto gas_concs = m_gas_concs_k; 
        Kokkos::parallel_for("fill_gas_concs", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;
                
                for (int k = 0; k < nlay_local; ++k) {
                    Real qv_val = qv(k + halo, iy + halo, ix + halo);
                    // TODO: Set VMRs correctly using gas_concs
                }
            });
            
        // Run RRTMGP
        std::shared_ptr<spdlog::logger> logger = nullptr;
        
        interface_t::rrtmgp_main(
            ncol, m_nlay,
            m_buffer.p_lay_k, m_buffer.t_lay_k, m_buffer.p_lev_k, m_buffer.t_lev_k,
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
            eccf, logger,
            m_extra_clnclrsky_diag, m_extra_clnsky_diag
        );

        // Unpack data and compute heating
        Kokkos::parallel_for("unpack_chunk_data", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;
                
                Real g = 9.80665;
                Real cp = 1004.64;
                
                for (int k = 0; k < nlay; ++k) {
                    Real flux_net_top = buffer.sw_flux_dn_k(i, k) - buffer.sw_flux_up_k(i, k) +
                                        buffer.lw_flux_dn_k(i, k) - buffer.lw_flux_up_k(i, k);
                    Real flux_net_bot = buffer.sw_flux_dn_k(i, k+1) - buffer.sw_flux_up_k(i, k+1) +
                                        buffer.lw_flux_dn_k(i, k+1) - buffer.lw_flux_up_k(i, k+1);
                    
                    Real dp = buffer.p_lev_k(i, k+1) - buffer.p_lev_k(i, k);
                    
                    if (abs(dp) > 1e-10) {
                        Real heating = (g / cp) * (flux_net_top - flux_net_bot) / dp;
                        th(k + halo, iy + halo, ix + halo) += heating * dt / pibar(k + halo);
                    }
                }
            });
    }
}

} // namespace RRTMGP
} // namespace Physics
} // namespace VVM
