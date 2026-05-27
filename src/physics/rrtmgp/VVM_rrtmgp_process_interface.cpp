#include "physics/rrtmgp/VVM_rrtmgp_process_interface.hpp"
#include "physics/rrtmgp/rrtmgp_utils.hpp"
#include "physics/rrtmgp/shr_orb_mod_c2f.hpp"
#include "share/physics/eamxx_trcmix.hpp"
#include "share/physics/eamxx_common_physics_functions.hpp"
#include "share/util/eamxx_column_ops.hpp"
#include "share/util/eamxx_utils.hpp"
#include "utils/Timer.hpp"

#include <ekat_assert.hpp>
#include <ekat_units.hpp>

namespace VVM {
namespace Physics {
namespace RRTMGP {

using Real = scream::Real;
using Int = scream::Int;
using PF = scream::PhysicsFunctions<DefaultDevice>;
using PC = scream::physics::Constants<Real>;
using CO = scream::ColumnOps<DefaultDevice, Real>;

bool is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

double calculate_calday(int year, int month, int day, int hour, int minute, double second) {
    if (year < 0 || month < 0 || day < 0 || hour < 0 || minute < 0 || second < 0) {
        std::cout << "WARNING: The timestamp is negative, please check!!!" << std::endl;
        return 1;
    }
    int days_in_month[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    
    if (is_leap_year(year)) {
        days_in_month[2] = 29;
    }

    int day_of_year = 0;
    for (int i = 1; i < month; ++i) {
        day_of_year += days_in_month[i];
    }
    day_of_year += day;

    double fractional_day = (hour + minute / 60.0 + second / 3600.0) / 24.0;

    return day_of_year + fractional_day;
}


RRTMGPRadiation::RRTMGPRadiation(const VVM::Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params)
    : m_grid(grid), m_config(config), m_params(params)
{
    // Initialize dimensions
    m_ncol = m_grid.get_local_physical_points_x() * m_grid.get_local_physical_points_y();
    m_nlay = m_grid.get_local_physical_points_z() + 1; // NOTE: Add a layer for TOA

    m_col_chunk_size = m_config.get_value<int>("physics.rrtmgp.column_chunk_size", m_ncol);
    
    std::cout << "m_col_chunk_size: " << m_col_chunk_size << std::endl;

    m_num_col_chunks = (m_ncol + m_col_chunk_size - 1) / m_col_chunk_size;
    m_col_chunk_beg.resize(m_num_col_chunks + 1, 0);
    for (int i = 0; i < m_num_col_chunks; ++i) {
        m_col_chunk_beg[i + 1] = std::min(m_ncol, m_col_chunk_beg[i] + m_col_chunk_size);
    }

    // Configuration
    m_nswgpts = m_config.get_value<int>("nswgpts",112);
    m_nlwgpts = m_config.get_value<int>("nlwgpts",128);
    m_do_aerosol_rad = m_config.get_value<bool>("physics.rrtmgp.do_aerosol_rad", false);
    m_extra_clnsky_diag = m_config.get_value<bool>("physics.rrtmgp.extra_clnsky_diag", false);
    m_extra_clnclrsky_diag = m_config.get_value<bool>("physics.rrtmgp.extra_clnclrsky_diag", false);

    const auto& active_gases = m_config.get_value<std::vector<std::string>>("physics.rrtmgp.active_gases", {"h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2"});
    for (const auto& gas : active_gases) {
        bool already_present = false;
        for (const auto& existing : m_gas_names) {
            if (std::string(existing) == gas) already_present = true;
        }
        if (!already_present) {
            m_gas_names.push_back(gas);
        }
    }
    m_ngas = m_gas_names.size();
}

RRTMGPRadiation::~RRTMGPRadiation() {}

void RRTMGPRadiation::initialize(VVM::Core::State& state) {
    int nx_total = m_grid.get_local_total_points_x();
    int ny_total = m_grid.get_local_total_points_y();
    int nz_total = m_grid.get_local_total_points_z();

    if (!state.has_field("sw_heating")) state.add_field<3>("sw_heating", {nz_total, ny_total, nx_total});
    if (!state.has_field("lw_heating")) state.add_field<3>("lw_heating", {nz_total, ny_total, nx_total});
    if (!state.has_field("net_heating")) state.add_field<3>("net_heating", {nz_total, ny_total, nx_total});
    if (!state.has_field("net_sw_flux")) state.add_field<3>("net_sw_flux", {nz_total, ny_total, nx_total});
    if (!state.has_field("net_lw_flux")) state.add_field<3>("net_lw_flux", {nz_total, ny_total, nx_total});
    if (!state.has_field("swdn")) state.add_field<3>("swdn", {nz_total, ny_total, nx_total});
    if (!state.has_field("lwdn")) state.add_field<3>("lwdn", {nz_total, ny_total, nx_total});
    if (!state.has_field("lwup")) state.add_field<3>("lwup", {nz_total, ny_total, nx_total});

    if (!state.has_field("swdn_toa")) state.add_field<2>("swdn_toa", {ny_total, nx_total});
    if (!state.has_field("lwup_toa")) state.add_field<2>("lwup_toa", {ny_total, nx_total});


    using PC = scream::physics::Constants<Real>;

    // Determine rad timestep, specified as number of atm steps

    // Determine orbital year. If orbital_year is negative, use current year
    // from timestamp for orbital year; if positive, use provided orbital year
    // for duration of simulation.
    m_orbital_year = m_config.get_value<int>("physics.rrtmgp.time.year", -9999);
    m_orbital_month = m_config.get_value<int>("physics.rrtmgp.time.month", -9999);
    m_orbital_day = m_config.get_value<int>("physics.rrtmgp.time.day", -9999);
    m_orbital_hour = m_config.get_value<int>("physics.rrtmgp.time.hour", -9999);
    m_orbital_minute = m_config.get_value<int>("physics.rrtmgp.time.minute", -9999);
    m_orbital_second = m_config.get_value<int>("physics.rrtmgp.time.second", -9999);
    // Get orbital parameters from yaml file
    m_orbital_eccen = m_config.get_value<VVM::Real>("physics.rrtmgp.orbital_eccentricity", -9999.0);
    m_orbital_obliq = m_config.get_value<VVM::Real>("physics.rrtmgp.orbital_obliquity", -9999.0);
    m_orbital_mvelp = m_config.get_value<VVM::Real>("physics.rrtmgp.orbital_mvelp", -9999.0);


    // Value for prescribing an invariant solar constant (i.e. total solar irradiance at
    // TOA).  Used for idealized experiments such as RCE. Disabled when value is less than 0.
    m_fixed_total_solar_irradiance = m_config.get_value<double>("physics.rrtmgp.fixed_total_solar_irradiance", -9999.0);

    // Determine whether or not we are using a fixed solar zenith angle (positive value)
    m_fixed_solar_zenith_angle = m_config.get_value<double>("physics.rrtmgp.fixed_solar_zenith_angle", -9999.0);

    // Get prescribed surface values of greenhouse gases
    m_o2vmr      = m_config.get_value<double>("physics.rrtmgp.o2vmr", 0.209);
    m_co2vmr     = m_config.get_value<double>("physics.rrtmgp.co2vmr", 355.03e-6);
    m_n2ovmr     = m_config.get_value<double>("physics.rrtmgp.n2ovmr", 320e-9);
    m_ch4vmr     = m_config.get_value<double>("physics.rrtmgp.ch4vmr", 1700e-9);
    m_f11vmr     = m_config.get_value<double>("physics.rrtmgp.f11vmr", 0.);
    m_f12vmr     = m_config.get_value<double>("physics.rrtmgp.f12vmr", 0.);
    m_n2vmr      = m_config.get_value<double>("physics.rrtmgp.n2vmr", 0.7906);
    m_covmr      = m_config.get_value<double>("physics.rrtmgp.covmr", 1.0e-7);
    m_o3vmr      = m_config.get_value<double>("physics.rrtmgp.o3vmr", 0.3017e-7);

    // Whether or not to do MCICA subcolumn sampling
    m_do_subcol_sampling = m_config.get_value<bool>("do_subcol_sampling",true);

    // Initialize kokkos
    // init_kls();

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
    Kokkos::deep_copy(m_gas_mol_weights, gas_mol_w_host);

    std::string coefficients_file_sw = m_config.get_value<std::string>("physics.rrtmgp.coefficients_file_sw", "../rundata/rrtmgp/rrtmgp-data-sw-g112-210809.nc");
    std::string coefficients_file_lw = m_config.get_value<std::string>("physics.rrtmgp.coefficients_file_lw", "../rundata/rrtmgp/rrtmgp-data-lw-g128-210809.nc");
    std::string cloud_optics_file_sw = m_config.get_value<std::string>("physics.rrtmgp.cloud_optics_file_sw", "../rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-sw.nc");
    std::string cloud_optics_file_lw = m_config.get_value<std::string>("physics.rrtmgp.cloud_optics_file_lw", "../rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-lw.nc");
    const double multiplier = m_config.get_value<double>("physics.rrtmgp.pool_size_multiplier", 1.0);

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
    
    const auto& h = m_grid.get_halo_cells();
    const auto& ny = m_grid.get_local_physical_points_y();
    const auto& nx = m_grid.get_local_physical_points_x();
    const auto& lon = state.get_field<2>("lon").get_device_data();
    const auto& lat = state.get_field<2>("lat").get_device_data();

    const auto& lon_h = state.get_field<2>("lon").get_host_data();
    const auto& lat_h = state.get_field<2>("lat").get_host_data();

    int rank = m_grid.get_mpi_rank();
    if (rank == 0) std::cout << "(lon, lat): (" << lon_h(0,0) << ", " << lat_h(0,0) << ")" << std::endl;


    m_lat = Kokkos::View<double*>("m_lat", m_ncol);
    m_lon = Kokkos::View<double*>("m_lon", m_ncol);
    auto m_lat_view = m_lat; 
    auto m_lon_view = m_lon;

    Kokkos::parallel_for("init_latlon_2d", Kokkos::RangePolicy<>(0, m_ncol),
        KOKKOS_LAMBDA(const int k) {
            int ix = k % nx;
            int iy = k / nx;

            m_lon_view(k) = lon(0, ix + h);
            m_lat_view(k) = lat(iy + h, 0);
        }
    );

    m_o3_profile  = Kokkos::View<Real*, DefaultDevice>("m_o3_profile", m_nlay);
    m_co2_profile = Kokkos::View<Real*, DefaultDevice>("m_co2_profile", m_nlay);
    m_ch4_profile = Kokkos::View<Real*, DefaultDevice>("m_ch4_profile", m_nlay);
    m_n2o_profile = Kokkos::View<Real*, DefaultDevice>("m_n2o_profile", m_nlay);
    m_o2_profile  = Kokkos::View<Real*, DefaultDevice>("m_o2_profile",  m_nlay);

    auto h_o3  = Kokkos::create_mirror_view(m_o3_profile);
    auto h_co2 = Kokkos::create_mirror_view(m_co2_profile);
    auto h_ch4 = Kokkos::create_mirror_view(m_ch4_profile);
    auto h_n2o = Kokkos::create_mirror_view(m_n2o_profile);
    auto h_o2  = Kokkos::create_mirror_view(m_o2_profile);

    VVM::Real g1 = 3.6478, g2 = 0.83209, g3 = 11.3515;
    const auto& h_pbar = state.get_field<1>("pbar").get_host_data(); 
    const int nz = m_grid.get_local_physical_points_z();
    // NOTE: The gas should be upside down, which means low index representing high level.
    for (int k = 0; k < m_nlay; ++k) {
        int k_vvm = h + nz - k;
        double p_val = h_pbar(k_vvm);
        if (k == 0) p_val = 1.01; // TOA pressure
        h_o3(k)  = g1 * std::pow(p_val/100., g2) * std::exp(-p_val/100. / g3) * 1e-6;
        // h_o3(k)  = 3.0170e-8;
        h_co2(k) = 348.0e-6; 
        h_ch4(k) = 1650.0e-9; 
        h_n2o(k) = 306.0e-9; 
        h_o2(k)  = 0.2090; 

        if (rank == 0) std::cout << "k = " << k << ", p_lay = " << p_val << ", o3 = " << h_o3(k) << std::endl;
    }
    std::cout << std::endl;
    Kokkos::deep_copy(m_o3_profile,  h_o3);
    Kokkos::deep_copy(m_co2_profile, h_co2);
    Kokkos::deep_copy(m_ch4_profile, h_ch4);
    Kokkos::deep_copy(m_n2o_profile, h_n2o);
    Kokkos::deep_copy(m_o2_profile,  h_o2);

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
    m_buffer.p_lev_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.t_lev_k = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.d_tint = alloc_2d(m_col_chunk_size, m_nlay + 1);
    m_buffer.d_dz = alloc_2d(m_col_chunk_size, m_nlay);

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
    m_gas_concs_k.reset();
    m_buffer_storage = Kokkos::View<Real*, DefaultDevice>();
    bool is_root = (m_grid.get_mpi_rank() == 0);
    interface_t::rrtmgp_finalize(is_root);
    m_gas_mol_weights = real1dk();
    m_lon = real1dk();
    m_lat = real1dk();
}

void RRTMGPRadiation::run(VVM::Core::State& state, const double dt) {
    VVM::Utils::Timer rrtmgp_timer("RRTMGP_timer");
    const int nx = m_grid.get_local_physical_points_x();
    const int halo = m_grid.get_halo_cells();
    const int nlay = m_nlay;

    // Get VVM fields
    auto& pbar = state.get_field<1>("pbar").get_device_data(); 
    auto& pbar_up = state.get_field<1>("pbar_up").get_device_data(); 
    auto& dpbar_mid = state.get_field<1>("dpbar_mid").get_device_data(); 
    auto& dz_mid = m_params.dz_mid.get_device_data(); 
    auto& qc = state.get_field<3>("qc").get_device_data();
    auto& nc = state.get_field<3>("nc").get_device_data();
    auto& qi = state.get_field<3>("qi").get_device_data();
    auto& qv = state.get_field<3>("qv").get_device_data();
    auto& th = state.get_field<3>("th").get_device_data();
    const auto& tg = state.get_field<2>("Tg").get_device_data();
    const auto& pibar = state.get_field<1>("pibar").get_device_data(); 
    const auto& thbar = state.get_field<1>("thbar").get_device_data();
    const auto& diag_eff_radius_qc = state.get_field<3>("diag_eff_radius_qc").get_device_data();
    const auto& diag_eff_radius_qi = state.get_field<3>("diag_eff_radius_qi").get_device_data();

    VVM::Real Rd = m_config.get_value<VVM::Real>("constants.Rd");
    VVM::Real g = m_config.get_value<VVM::Real>("constants.gravity");

    // Output fields
    auto& sw_heating = state.get_field<3>("sw_heating").get_mutable_device_data();
    auto& lw_heating = state.get_field<3>("lw_heating").get_mutable_device_data();
    auto& net_heating = state.get_field<3>("net_heating").get_mutable_device_data();
    auto& net_sw_flux = state.get_field<3>("net_sw_flux").get_mutable_device_data();
    auto& net_lw_flux = state.get_field<3>("net_lw_flux").get_mutable_device_data();
    auto& swdn = state.get_field<3>("swdn").get_mutable_device_data();
    auto& lwdn = state.get_field<3>("lwdn").get_mutable_device_data();
    auto& lwup = state.get_field<3>("lwup").get_mutable_device_data();
    auto& swdn_toa = state.get_field<2>("swdn_toa").get_mutable_device_data();
    auto& lwup_toa = state.get_field<2>("lwup_toa").get_mutable_device_data();

    // Orbital parameters and Zenith Angle
    double obliqr, lambm0, mvelpp;
    Int orbital_year = m_orbital_year;
    double eccen = m_orbital_eccen;
    double obliq = m_orbital_obliq;
    double mvelp = m_orbital_mvelp;
    
    // double calday = 1.0;       // Placeholder for Jan 1st
    calday_ = calculate_calday(m_orbital_year, m_orbital_month, m_orbital_day, m_orbital_hour, m_orbital_minute, m_orbital_second);
    calday_ += (state.get_time() / 86400.0);

    if (eccen >= 0 && obliq >= 0 && mvelp >= 0) {
        orbital_year = shr_orb_undef_int_c2f;
    }

    // If we had a timestamp, we would set orbital_year based on it if m_orbital_year < 0
    shr_orb_params_c2f(&orbital_year, &eccen, &obliq, &mvelp,
                       &obliqr, &lambm0, &mvelpp);

    double delta, eccf;
    shr_orb_decl_c2f(calday_, eccen, mvelpp, lambm0,
                     obliqr, &delta, &eccf);
    
    if (m_fixed_total_solar_irradiance > 0) {
        eccf = m_fixed_total_solar_irradiance / 1360.9;
    }

    const auto gas_mol_weights = m_gas_mol_weights;
    auto h_lat = Kokkos::create_mirror_view(m_lat); // Need to sync lat/lon if they change
    Kokkos::deep_copy(h_lat, m_lat);
    auto h_lon = Kokkos::create_mirror_view(m_lon);
    Kokkos::deep_copy(h_lon, m_lon);

    // Loop over chunks
    for (int ic = 0; ic < m_num_col_chunks; ++ic) {
        int beg = m_col_chunk_beg[ic];
        int ncol = m_col_chunk_beg[ic + 1] - beg;

        auto buffer = m_buffer;

        // Calculate Zenith Angle (mu0) on Host
        Kokkos::View<Real*, Kokkos::HostSpace> h_mu0("h_mu0", ncol);
        
        if (m_fixed_solar_zenith_angle > 0) {
            for (int i = 0; i < ncol; ++i) h_mu0(i) = m_fixed_solar_zenith_angle;
        } 
        else {
            for (int i = 0; i < ncol; ++i) {
                // EAMxx uses physics::Constants::Pi, need to verify VVM namespace
                double lat_rad = h_lat(beg + i) * PC::Pi / 180.0;
                double lon_rad = h_lon(beg + i) * PC::Pi / 180.0;
                // double lat_rad = 23.5 * PC::Pi / 180.0;
                // double lon_rad = 121. * PC::Pi / 180.0;
                h_mu0(i) = shr_orb_cosz_c2f(calday_, lat_rad, lon_rad, delta, dt);
            }
        }
        
        Kokkos::View<Real*, DefaultDevice> mu0_k("mu0_k", ncol);
        Kokkos::deep_copy(mu0_k, h_mu0);

        const int nswbands = m_nswbands;
        const int nlwbands = m_nlwbands;

        // Pack data for this chunk
        const int nz = m_nlay - 1;
        const int h  = halo;
        Kokkos::parallel_for("pack_chunk_data", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;

                buffer.p_lev_k(i, 0) = 1.01; // TOA pressure (1 Pa)
                buffer.p_lev_k(i, 1) = pbar_up(h + nz - 1);
                
                buffer.p_lay_k(i, 0) = 0.5 * (buffer.p_lev_k(i, 0) + buffer.p_lev_k(i, 1));
                buffer.p_del_k(i, 0) = buffer.p_lev_k(i, 1) - buffer.p_lev_k(i, 0);
                buffer.d_dz(i, 0) = 0.0;
                
                int top_vvm = h + nz - 1;
                int subtop_vvm = h + nz - 2;
                Real t_top = th(top_vvm, iy + h, ix + h) * pibar(top_vvm);
                Real t_subtop = th(subtop_vvm, iy + h, ix + h) * pibar(subtop_vvm);
                buffer.t_lay_k(i, 0) = 2.0 * t_top - t_subtop;
                
                buffer.qc_k(i, 0) = 0.0;
                buffer.nc_k(i, 0) = 0.0;
                buffer.qi_k(i, 0) = 0.0;
                buffer.cldfrac_tot_k(i, 0) = 0.0;
                buffer.eff_radius_qc_k(i, 0) = 10.0;
                buffer.eff_radius_qi_k(i, 0) = 50.0;

                for (int k = 1; k <= nz; ++k) {
                    int k_vvm = h + nz - k;

                    buffer.p_lay_k(i, k) = pbar(k_vvm); 
                    buffer.t_lay_k(i, k) = th(k_vvm, iy + h, ix + h) * pibar(k_vvm);
                    buffer.qc_k(i, k) = qc(k_vvm, iy + h, ix + h);
                    buffer.nc_k(i, k) = nc(k_vvm, iy + h, ix + h);
                    buffer.qi_k(i, k) = qi(k_vvm, iy + h, ix + h);
                    buffer.eff_radius_qc_k(i, k) = diag_eff_radius_qc(k_vvm, iy+h, ix+h);
                    buffer.eff_radius_qi_k(i, k) = diag_eff_radius_qi(k_vvm, iy+h, ix+h);
                    buffer.p_del_k(i, k) = dpbar_mid(k_vvm); 
                    buffer.d_dz(i, k) = dz_mid(k_vvm);

                    if (buffer.qc_k(i, k) + buffer.qi_k(i, k) > 1e-12) {
                        buffer.cldfrac_tot_k(i, k) = 1.0;
                    }
                    else {
                        buffer.cldfrac_tot_k(i, k) = 0.0;
                    }
                }
                buffer.t_lev_k(i, nlay) = tg(iy + h, ix + h);
                
                for (int k = 1; k <= nz+1; ++k) {
                    int k_vvm = h + nz - k;
                    buffer.p_lev_k(i, k) = pbar_up(k_vvm);
                }
                
                // Initialize Broadband Surface Albedo  (TODO: Get from State)
                // For now, assuming ocean-like albedo
                buffer.sfc_alb_dir_vis_k(i) = 0.3;
                buffer.sfc_alb_dir_nir_k(i) = 0.3;
                buffer.sfc_alb_dif_vis_k(i) = 0.3;
                buffer.sfc_alb_dif_nir_k(i) = 0.3;

                for(int k=0; k < nlay; ++k) {
                    for(int b=0; b<nswbands; ++b) {
                        buffer.aero_tau_sw_k(i,k,b) = 0.0;
                        buffer.aero_ssa_sw_k(i,k,b) = 0.0;
                        buffer.aero_g_sw_k(i,k,b) = 0.0;
                    }
                    for(int b=0; b<nlwbands; ++b) {
                        buffer.aero_tau_lw_k(i,k,b) = 0.0;
                    }
                }
            });


        // Compute T_int Manually
        Kokkos::parallel_for("compute_t_int", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                for (int k = 1; k < nlay; ++k) {
                    buffer.t_lev_k(i, k) = 0.5 * (buffer.t_lay_k(i, k-1) + buffer.t_lay_k(i, k));
                }
                buffer.t_lev_k(i, 0) = 2.0 * buffer.t_lay_k(i, 0) - buffer.t_lev_k(i, 1);
            });

        interface_t::mixing_ratio_to_cloud_mass(buffer.qc_k, buffer.cldfrac_tot_k, buffer.p_del_k, buffer.lwp_k);
        interface_t::mixing_ratio_to_cloud_mass(buffer.qi_k, buffer.cldfrac_tot_k, buffer.p_del_k, buffer.iwp_k);

        // Convert kg/m2 to g/m2 as required by RRTMGP
        Kokkos::parallel_for("convert_cld_mass_units", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                for(int k=0; k<nlay; ++k) {
                    buffer.lwp_k(i,k) *= 1e3;
                    buffer.iwp_k(i,k) *= 1e3;
                    if (buffer.lwp_k(i,k) < 0) buffer.lwp_k(i,k) = 0.;
                    if (buffer.iwp_k(i,k) < 0) buffer.iwp_k(i,k) = 0.;
                       
                }
            }
        );

        // Compute Band-by-Band Surface Albedos
        interface_t::compute_band_by_band_surface_albedos(
            ncol, m_nswbands,
            buffer.sfc_alb_dir_vis_k, buffer.sfc_alb_dir_nir_k,
            buffer.sfc_alb_dif_vis_k, buffer.sfc_alb_dif_nir_k,
            buffer.sfc_alb_dir_k, buffer.sfc_alb_dif_k
        );


        m_gas_concs_k.ncol = ncol;

        const auto co2_profile = m_co2_profile;
        const auto ch4_profile = m_ch4_profile;
        const auto n2o_profile = m_n2o_profile;
        const auto o2_profile = m_o2_profile;
        const auto o3_profile = m_o3_profile;
        for (int igas = 0; igas < m_ngas; igas++) {
            std::string name = m_gas_names[igas];
            auto vmr_view = Kokkos::subview(buffer.tmp2d_k, std::make_pair(0, ncol), Kokkos::ALL());

            if (name == "h2o") {
                Kokkos::parallel_for("set_h2o_vmr", Kokkos::RangePolicy<>(0, ncol),
                   KOKKOS_LAMBDA(int i) {
                       int col_idx = beg + i;
                       int ix = col_idx % nx;
                       int iy = col_idx / nx;

                       vmr_view(i, 0) = PF::calculate_vmr_from_mmr(gas_mol_weights(igas), 1e-6, 1e-6);

                       for (int k = 1; k < nlay; ++k) {
                           int k_vvm = h + nz - k; 
                           Real qv_val = qv(k_vvm, iy + h, ix + h);
                           vmr_view(i, k) = PF::calculate_vmr_from_mmr(gas_mol_weights(igas), qv_val, qv_val);
                       }
                   });
            } 
            else {
                bool is_profile = false;
                Kokkos::View<Real*, DefaultDevice> profile;
                Real scalar_val = 0.0;

                if      (name == "co2") { is_profile = true; profile = m_co2_profile; }
                else if (name == "ch4") { is_profile = true; profile = m_ch4_profile; }
                else if (name == "n2o") { is_profile = true; profile = m_n2o_profile; }
                else if (name == "o2")  { is_profile = true; profile = m_o2_profile;  }
                else if (name == "o3")  { is_profile = true; profile = m_o3_profile;  }
                else if (name == "n2")  { scalar_val = m_n2vmr; }
                else if (name == "co")  { scalar_val = m_covmr; }

                if (is_profile) {
                    Kokkos::parallel_for("set_gas_profile", Kokkos::RangePolicy<>(0, ncol),
                       KOKKOS_LAMBDA(int i) {
                           for (int k = 0; k < nlay; ++k) {
                               vmr_view(i, k) = profile(k);
                           }
                       });
                } 
                else {
                    Kokkos::parallel_for("set_gas_scalar", Kokkos::RangePolicy<>(0, ncol),
                       KOKKOS_LAMBDA(int i) {
                           for (int k = 0; k < nlay; ++k) {
                               vmr_view(i, k) = scalar_val;
                           }
                       });
                }
            }

            Kokkos::parallel_for("copy_gas_to_toa", Kokkos::RangePolicy<>(0, ncol),
                KOKKOS_LAMBDA(int i) {
                    vmr_view(i, 0) = vmr_view(i, 1); 
                }
            );

            m_gas_concs_k.set_vmr(name, vmr_view);
        }
        std::shared_ptr<spdlog::logger> logger = nullptr;

        interface_t::rrtmgp_main(
            ncol, m_nlay,
            buffer.p_lay_k, buffer.t_lay_k, buffer.p_lev_k, buffer.t_lev_k,
            m_gas_concs_k,
            buffer.sfc_alb_dir_k, buffer.sfc_alb_dif_k, mu0_k,
            buffer.lwp_k, buffer.iwp_k, buffer.eff_radius_qc_k, buffer.eff_radius_qi_k, buffer.cldfrac_tot_k,
            buffer.aero_tau_sw_k, buffer.aero_ssa_sw_k, buffer.aero_g_sw_k, buffer.aero_tau_lw_k,
            buffer.cld_tau_sw_bnd_k, buffer.cld_tau_lw_bnd_k,
            buffer.cld_tau_sw_gpt_k, buffer.cld_tau_lw_gpt_k,
            buffer.sw_flux_up_k, buffer.sw_flux_dn_k, buffer.sw_flux_dn_dir_k, buffer.lw_flux_up_k, buffer.lw_flux_dn_k,
            buffer.sw_clnclrsky_flux_up_k, buffer.sw_clnclrsky_flux_dn_k, buffer.sw_clnclrsky_flux_dn_dir_k,
            buffer.sw_clrsky_flux_up_k, buffer.sw_clrsky_flux_dn_k, buffer.sw_clrsky_flux_dn_dir_k,
            buffer.sw_clnsky_flux_up_k, buffer.sw_clnsky_flux_dn_k, buffer.sw_clnsky_flux_dn_dir_k,
            buffer.lw_clnclrsky_flux_up_k, buffer.lw_clnclrsky_flux_dn_k,
            buffer.lw_clrsky_flux_up_k, buffer.lw_clrsky_flux_dn_k,
            buffer.lw_clnsky_flux_up_k, buffer.lw_clnsky_flux_dn_k,
            buffer.sw_bnd_flux_up_k, buffer.sw_bnd_flux_dn_k, buffer.sw_bnd_flux_dir_k,
            buffer.lw_bnd_flux_up_k, buffer.lw_bnd_flux_dn_k,
            eccf, logger,
            m_extra_clnclrsky_diag, m_extra_clnsky_diag
        );

        // Compute Heating Rates (Using RRTMGP helper)
        scream::rrtmgp::compute_heating_rate(buffer.sw_flux_up_k, buffer.sw_flux_dn_k, buffer.p_del_k, buffer.sw_heating_k);
        scream::rrtmgp::compute_heating_rate(buffer.lw_flux_up_k, buffer.lw_flux_dn_k, buffer.p_del_k, buffer.lw_heating_k);

        // Unpack data and compute heating
        Kokkos::parallel_for("unpack_chunk_data", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;

                swdn_toa(iy + h, ix + h) = buffer.sw_flux_dn_k(i, 0);
                lwup_toa(iy + h, ix + h) = buffer.lw_flux_up_k(i, 0);

                for (int k = 1; k <= nz; ++k) {
                    int k_vvm = h + nz - k;

                    Real net_heating_val = buffer.sw_heating_k(i, k) + buffer.lw_heating_k(i, k); 
                    sw_heating(k_vvm, iy + h, ix + h) = buffer.sw_heating_k(i, k);
                    lw_heating(k_vvm, iy + h, ix + h) = buffer.lw_heating_k(i, k);
                    net_heating(k_vvm, iy + h, ix + h) = net_heating_val;
                }

                for (int k = 1; k <= nz + 1; ++k) {
                    int k_vvm = h + nz - k;
                    
                    Real net_sw = buffer.sw_flux_dn_k(i, k) - buffer.sw_flux_up_k(i, k);
                    Real net_lw = buffer.lw_flux_dn_k(i, k) - buffer.lw_flux_up_k(i, k);

                    net_sw_flux(k_vvm, iy + h, ix + h) = net_sw;
                    net_lw_flux(k_vvm, iy + h, ix + h) = net_lw;
                    swdn(k_vvm, iy + h, ix + h) = buffer.sw_flux_dn_k(i, k);
                    lwdn(k_vvm, iy + h, ix + h) = buffer.lw_flux_dn_k(i, k);
                    lwup(k_vvm, iy + h, ix + h) = buffer.lw_flux_up_k(i, k);
                }
            }
        );
    }
}

void RRTMGPRadiation::calculate_tendencies(VVM::Core::State& state) {
    const int nz = m_grid.get_local_total_points_z();
    const int ny = m_grid.get_local_total_points_y();
    const int nx = m_grid.get_local_total_points_x();
    const int h = m_grid.get_halo_cells();

    state.get_field<3>("fe_tendency_th").set_to_zero();
    auto fe_tend = state.get_field<3>("fe_tendency_th").get_mutable_device_data();
    
    const auto& net_heating = state.get_field<3>("net_heating").get_device_data(); 
    const auto& pibar = state.get_field<1>("pibar").get_device_data();

    Kokkos::parallel_for("Apply_RRTMGP_Heating_FE",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            fe_tend(k, j, i) += net_heating(k, j, i) / pibar(k);
        }
    );
}

} // namespace RRTMGP
} // namespace Physics
} // namespace VVM

