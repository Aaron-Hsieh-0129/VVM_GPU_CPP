#include "physics/rrtmgp/VVM_rrtmgp_process_interface.hpp"
#include "physics/rrtmgp/rrtmgp_utils.hpp"
#include "physics/rrtmgp/shr_orb_mod_c2f.hpp"
#include "share/physics/eamxx_trcmix.hpp"
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

    if (!state.has_field("swup_toa")) state.add_field<2>("swup_toa", {ny_total, nx_total});
    if (!state.has_field("swdn_toa")) state.add_field<2>("swdn_toa", {ny_total, nx_total});
    if (!state.has_field("lwup_toa")) state.add_field<2>("lwup_toa", {ny_total, nx_total});
    if (!state.has_field("lwdn_toa")) state.add_field<2>("lwdn_toa", {ny_total, nx_total});

    if (!state.has_field("swup_sfc")) state.add_field<2>("swup_sfc", {ny_total, nx_total});
    if (!state.has_field("swdn_sfc")) state.add_field<2>("swdn_sfc", {ny_total, nx_total});
    if (!state.has_field("lwup_sfc")) state.add_field<2>("lwup_sfc", {ny_total, nx_total});
    if (!state.has_field("lwdn_sfc")) state.add_field<2>("lwdn_sfc", {ny_total, nx_total});

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

    std::string coefficients_file_sw = m_config.get_value<std::string>("physics.rrtmgp.coefficients_file_sw", VVM_ROOT_DIR "/rundata/rrtmgp/rrtmgp-data-sw-g112-210809.nc");
    std::string coefficients_file_lw = m_config.get_value<std::string>("physics.rrtmgp.coefficients_file_lw", VVM_ROOT_DIR "/rundata/rrtmgp/rrtmgp-data-lw-g128-210809.nc");
    std::string cloud_optics_file_sw = m_config.get_value<std::string>("physics.rrtmgp.cloud_optics_file_sw", VVM_ROOT_DIR "/rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-sw.nc");
    std::string cloud_optics_file_lw = m_config.get_value<std::string>("physics.rrtmgp.cloud_optics_file_lw", VVM_ROOT_DIR "/rundata/rrtmgp/rrtmgp-cloud-optics-coeffs-lw.nc");
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
    const auto& albedo = state.get_field<2>("albedo").get_device_data();

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
    auto& swup_toa = state.get_field<2>("swup_toa").get_mutable_device_data();
    auto& swdn_toa = state.get_field<2>("swdn_toa").get_mutable_device_data();
    auto& lwup_toa = state.get_field<2>("lwup_toa").get_mutable_device_data();
    auto& lwdn_toa = state.get_field<2>("lwdn_toa").get_mutable_device_data();

    auto& swup_sfc = state.get_field<2>("swup_sfc").get_mutable_device_data();
    auto& swdn_sfc = state.get_field<2>("swdn_sfc").get_mutable_device_data();
    auto& lwup_sfc = state.get_field<2>("lwup_sfc").get_mutable_device_data();
    auto& lwdn_sfc = state.get_field<2>("lwdn_sfc").get_mutable_device_data();

    const auto& topo = state.get_field<2>("topo").get_device_data();

    // Fortran VVM RRTMG uses h2ovmr = mwdry/mwh2o*qv, not qv/(1-qv).
    constexpr Real mwdry = 28.966;
    constexpr Real mwh2o = 18.016;
    constexpr Real qcmin_rrtmg = 1e-7;
    constexpr Real qimin_rrtmg = 1e-8;

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

                int hx = topo(iy + h, ix + h);
                int num_dummy = hx - 1;

                Real p_toa_real = pbar_up(h + nz - 1);
                Real p_top = 1.01;
                Real dp_dummy = (num_dummy >= 0) ? (p_toa_real - p_top) / (num_dummy + 1.0) : 0.0;

                for (int k = 0; k <= num_dummy; ++k) {
                    buffer.p_lev_k(i, k) = p_top + k * dp_dummy;
                }

                for (int k = num_dummy + 1; k <= nz + 1; ++k) {
                    int k_vvm = h + nz - k + num_dummy; 
                    buffer.p_lev_k(i, k) = pbar_up(k_vvm);
                }
                
                for (int k = 0; k <= nz; ++k) {
                    int k_init_vvm = h+nz-k;
                    buffer.t_lay_k(i, k) = th(k_init_vvm, iy+h, ix+h) * pibar(k_init_vvm);

                    if (k == 0) {
                        // original TOA process
                        buffer.p_lay_k(i, 0) = 0.5 * (buffer.p_lev_k(i, 0) + buffer.p_lev_k(i, 1));
                        buffer.p_del_k(i, 0) = buffer.p_lev_k(i, 1) - buffer.p_lev_k(i, 0);
                        buffer.d_dz(i, 0) = 0.0;
                        int top_vvm = h + nz - 1;
                        int subtop_vvm = h + nz - 2;
                        buffer.t_lay_k(i, 0) = 2.0 * th(top_vvm, iy+h, ix+h) * pibar(top_vvm) - th(subtop_vvm, iy+h, ix+h) * pibar(subtop_vvm);
                        buffer.qc_k(i, 0) = 0.0;
                        buffer.qi_k(i, 0) = 0.0;
                        buffer.eff_radius_qc_k(i, 0) = 10.0;
                        buffer.eff_radius_qi_k(i, 0) = 50.0;
                        buffer.cldfrac_tot_k(i, 0) = 0.0;
                    } 
                    else if (k <= num_dummy) {
                        // pseudo grid for topo
                        buffer.p_lay_k(i, k) = 0.5 * (buffer.p_lev_k(i, k) + buffer.p_lev_k(i, k+1));
                        buffer.p_del_k(i, k) = buffer.p_lev_k(i, k+1) - buffer.p_lev_k(i, k);
                        buffer.d_dz(i, k) = 0.0;
                        buffer.qc_k(i, k) = 0.0;
                        buffer.qi_k(i, k) = 0.0;
                        buffer.eff_radius_qc_k(i, k) = 10.0;
                        buffer.eff_radius_qi_k(i, k) = 50.0;
                        buffer.cldfrac_tot_k(i, k) = 0.0;
                    } 
                    else {
                        // real atmosphere move downward
                        int k_vvm = h + nz - k + num_dummy;
                        buffer.p_lay_k(i, k) = pbar(k_vvm); 
                        buffer.p_del_k(i, k) = buffer.p_lev_k(i, k+1) - buffer.p_lev_k(i, k);
                        buffer.d_dz(i, k) = dz_mid(k_vvm);
                        buffer.t_lay_k(i, k) = th(k_vvm, iy + h, ix + h) * pibar(k_vvm);
                        Real qc_raw = qc(k_vvm, iy + h, ix + h);
                        Real qi_raw = qi(k_vvm, iy + h, ix + h);

                        // The unit here is mu m
                        Real re_qc_microns = diag_eff_radius_qc(k_vvm, iy+h, ix+h);
                        Real re_qi_microns = diag_eff_radius_qi(k_vvm, iy+h, ix+h);

                        buffer.eff_radius_qc_k(i, k) = Kokkos::max(real(2.5), Kokkos::min(real(60.0), re_qc_microns));
                        buffer.eff_radius_qi_k(i, k) = Kokkos::max(real(5.0), Kokkos::min(real(140.0), re_qi_microns));

                        // radiation_rrtmg.f90 floors qcl/qci before rad_full.
                        // rad_driver then sets cloudFrac where the resulting LWP/IWP is positive.
                        buffer.qc_k(i, k) = Kokkos::max(qc_raw, qcmin_rrtmg);
                        buffer.qi_k(i, k) = Kokkos::max(qi_raw, qimin_rrtmg);
                        buffer.cldfrac_tot_k(i, k) = 1.0;
                    }
                }
                buffer.t_lev_k(i, nlay) = tg(iy + h, ix + h);

                buffer.sfc_alb_dir_vis_k(i) = albedo(iy+h, ix+h)/100.;
                buffer.sfc_alb_dir_nir_k(i) = albedo(iy+h, ix+h)/100.;
                buffer.sfc_alb_dif_vis_k(i) = albedo(iy+h, ix+h)/100.;
                buffer.sfc_alb_dif_nir_k(i) = albedo(iy+h, ix+h)/100.;

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
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;

                for (int k = 1; k < nlay; ++k) {
                    buffer.t_lev_k(i, k) = 0.5 * (buffer.t_lay_k(i, k-1) + buffer.t_lay_k(i, k));
                }
                buffer.t_lev_k(i, 0) = 2.0 * buffer.t_lay_k(i, 0) - buffer.t_lev_k(i, 1);
                
                buffer.t_lev_k(i, nlay) = tg(iy + h, ix + h);
            }
        );

        // Fortran VVM computes LWP/IWP directly from qcl/qci:
        // q * 1e3 * (dp/g). qcl/qci have already followed radiation_rrtmg.f90
        // qcmin/qimin floors for real atmosphere layers.
        Kokkos::parallel_for("compute_fortran_like_cloud_paths", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                for(int k=0; k<nlay; ++k) {
                    Real layer_mass = buffer.p_del_k(i,k) / g;
                    buffer.lwp_k(i,k) = buffer.qc_k(i,k) * real(1e3) * layer_mass;
                    buffer.iwp_k(i,k) = buffer.qi_k(i,k) * real(1e3) * layer_mass;
                    if (buffer.lwp_k(i,k) < 0) buffer.lwp_k(i,k) = 0.0;
                    if (buffer.iwp_k(i,k) < 0) buffer.iwp_k(i,k) = 0.0;
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
                        int hx = topo(iy + h, ix + h);

                        vmr_view(i, 0) = (mwdry / mwh2o) * 1e-6;

                        int num_dummy = hx - 1;
                        for (int k = 1; k < nlay; ++k) {
                            if (k <= num_dummy) {
                                int k_vvm_top = h + nz - 1;
                                Real qv_val = Kokkos::max(qv(k_vvm_top, iy + h, ix + h), real(1e-6));
                                vmr_view(i, k) = (mwdry / mwh2o) * qv_val;
                            } 
                            else {
                                int k_vvm = h + nz - k + num_dummy;
                                Real qv_val = Kokkos::max(qv(k_vvm, iy + h, ix + h), real(1e-6));
                                vmr_view(i, k) = (mwdry / mwh2o) * qv_val;
                            }
                        }
                    }
                );
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
                            int col_idx = beg + i;
                            int ix = col_idx % nx;
                            int iy = col_idx / nx;
                            int hx = topo(iy + h, ix + h);
                            int num_dummy = hx - 1;

                            for (int k = 0; k < nlay; ++k) {
                                if (k == 0) {
                                    vmr_view(i, k) = profile(0);
                                }
                                else if (k <= num_dummy) {
                                    vmr_view(i, k) = profile(1); 
                                } 
                                else {
                                    int k_prof = k - num_dummy; 
                                    vmr_view(i, k) = profile(k_prof);
                                }
                            }
                        }
                    );
                } 
                else {
                    Kokkos::parallel_for("set_gas_scalar", Kokkos::RangePolicy<>(0, ncol),
                        KOKKOS_LAMBDA(int i) {
                            int col_idx = beg + i;
                            int ix = col_idx % nx;
                            int iy = col_idx / nx;
                            int hx = topo(iy + h, ix + h);
                            int num_dummy = hx - 1;

                            for (int k = 0; k < nlay; ++k) {
                                if (k == 0) {
                                    vmr_view(i, k) = scalar_val;
                                }
                                else if (k <= num_dummy) {
                                    vmr_view(i, k) = scalar_val;
                                } 
                                else {
                                    vmr_view(i, k) = scalar_val;
                                }
                            }
                        }
                    );
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

        scream::rrtmgp::compute_heating_rate(buffer.sw_flux_up_k, buffer.sw_flux_dn_k, buffer.p_del_k, buffer.sw_heating_k);
        scream::rrtmgp::compute_heating_rate(buffer.lw_flux_up_k, buffer.lw_flux_dn_k, buffer.p_del_k, buffer.lw_heating_k);

#ifdef VVM_RRTMGP_DEBUG
        auto h_lw_heating = Kokkos::create_mirror_view(buffer.lw_heating_k);
        auto h_p_lev      = Kokkos::create_mirror_view(buffer.p_lev_k);
        auto h_p_del      = Kokkos::create_mirror_view(buffer.p_del_k);
        auto h_lw_up      = Kokkos::create_mirror_view(buffer.lw_flux_up_k);
        auto h_lw_dn      = Kokkos::create_mirror_view(buffer.lw_flux_dn_k);
        auto h_t_lay      = Kokkos::create_mirror_view(buffer.t_lay_k);
        auto h_t_lev      = Kokkos::create_mirror_view(buffer.t_lev_k);
        auto h_cldfrac    = Kokkos::create_mirror_view(buffer.cldfrac_tot_k);

        auto h_qc_pack    = Kokkos::create_mirror_view(buffer.qc_k);
        auto h_qi_pack    = Kokkos::create_mirror_view(buffer.qi_k);
        auto h_lwp        = Kokkos::create_mirror_view(buffer.lwp_k);
        auto h_iwp        = Kokkos::create_mirror_view(buffer.iwp_k);
        auto h_rel        = Kokkos::create_mirror_view(buffer.eff_radius_qc_k);
        auto h_rei        = Kokkos::create_mirror_view(buffer.eff_radius_qi_k);

        auto h_gas_concs  = Kokkos::create_mirror_view(m_gas_concs_k.concs);

        Kokkos::deep_copy(h_lw_heating, buffer.lw_heating_k);
        Kokkos::deep_copy(h_p_lev,      buffer.p_lev_k);
        Kokkos::deep_copy(h_p_del,      buffer.p_del_k);
        Kokkos::deep_copy(h_lw_up,      buffer.lw_flux_up_k);
        Kokkos::deep_copy(h_lw_dn,      buffer.lw_flux_dn_k);
        Kokkos::deep_copy(h_t_lay,      buffer.t_lay_k);
        Kokkos::deep_copy(h_t_lev,      buffer.t_lev_k);
        Kokkos::deep_copy(h_cldfrac,    buffer.cldfrac_tot_k);

        Kokkos::deep_copy(h_qc_pack,    buffer.qc_k);
        Kokkos::deep_copy(h_qi_pack,    buffer.qi_k);
        Kokkos::deep_copy(h_lwp,        buffer.lwp_k);
        Kokkos::deep_copy(h_iwp,        buffer.iwp_k);
        Kokkos::deep_copy(h_rel,        buffer.eff_radius_qc_k);
        Kokkos::deep_copy(h_rei,        buffer.eff_radius_qi_k);

        Kokkos::deep_copy(h_gas_concs,  m_gas_concs_k.concs);

        auto h_topo = state.get_field<2>("topo").get_host_data();
        auto h_tg   = state.get_field<2>("Tg").get_host_data();
        auto h_qc   = state.get_field<3>("qc").get_host_data();
        auto h_qi   = state.get_field<3>("qi").get_host_data();
        auto h_qv   = state.get_field<3>("qv").get_host_data();

        const int ih2o = m_gas_concs_k.find_gas("h2o");

        constexpr double grav = 9.80665;
        constexpr double cp   = 1004.0;

        for (int i = 0; i < ncol; ++i) {
            int col_idx = beg + i;
            int ix = col_idx % nx;
            int iy = col_idx / nx;

            int hx = h_topo(iy + h, ix + h);
            int num_dummy = hx - 1;

            if (ix == 641 && iy == 63) {
                std::cout << "\nVVM_RRTMGP_LAND_ANOMALY_DIAG"
                          << " ix=" << ix
                          << " iy=" << iy
                          << " hx=" << hx
                          << " num_dummy=" << num_dummy
                          << " Tg=" << h_tg(iy + h, ix + h)
                          << " ih2o=" << ih2o
                          << "\n";

                for (int kv = h + hx - 1; kv <= h + hx + 3; ++kv) {
                    if (kv > h + nz - 1) continue;

                    int kr = h + nz - kv + num_dummy;

                    double plev_top = h_p_lev(i, kr);
                    double plev_bot = h_p_lev(i, kr + 1);
                    double pdel     = h_p_del(i, kr);

                    double lwup_top = h_lw_up(i, kr);
                    double lwdn_top = h_lw_dn(i, kr);
                    double lwup_bot = h_lw_up(i, kr + 1);
                    double lwdn_bot = h_lw_dn(i, kr + 1);

                    double net_bot = lwdn_bot - lwup_bot;
                    double net_top = lwdn_top - lwup_top;
                    double dnet    = net_bot - net_top;

                    double heating_output = h_lw_heating(i, kr) * 86400.0;
                    double heating_recomputed_pos =  dnet * grav * 86400.0 / (cp * pdel);
                    double heating_recomputed_neg = -dnet * grav * 86400.0 / (cp * pdel);

                    double layer_mass = pdel / grav;

                    double raw_qv = h_qv(kv, iy + h, ix + h);
                    double raw_qc = h_qc(kv, iy + h, ix + h);
                    double raw_qi = h_qi(kv, iy + h, ix + h);

                    double h2ovmr = (ih2o >= 0) ? h_gas_concs(i, kr, ih2o) : -999.0;

                    std::cout << "  VVM_k=" << (kv - h + 1)
                              << " RRTM_k=" << kr
                              << " topo/hx=" << hx
                              << " num_dummy=" << num_dummy

                              << " Tg=" << h_tg(iy + h, ix + h)

                              << " plev_bot=" << plev_bot
                              << " plev_top=" << plev_top
                              << " pdel=" << pdel
                              << " layer_mass=" << layer_mass

                              << " t_lay=" << h_t_lay(i, kr)
                              << " t_lev_bot=" << h_t_lev(i, kr + 1)
                              << " t_lev_top=" << h_t_lev(i, kr)

                              << " lwup_bot=" << lwup_bot
                              << " lwdn_bot=" << lwdn_bot
                              << " lwup_top=" << lwup_top
                              << " lwdn_top=" << lwdn_top

                              << " net_bot=" << net_bot
                              << " net_top=" << net_top
                              << " dnet=" << dnet

                              << " heating_output(K/day)=" << heating_output
                              << " heating_recomputed_pos(K/day)=" << heating_recomputed_pos
                              << " heating_recomputed_neg(K/day)=" << heating_recomputed_neg

                              << " raw_qv=" << raw_qv
                              << " raw_qc=" << raw_qc
                              << " raw_qi=" << raw_qi

                              << " h2ovmr=" << h2ovmr

                              << " qc_pack=" << h_qc_pack(i, kr)
                              << " qi_pack=" << h_qi_pack(i, kr)
                              << " cldfrac=" << h_cldfrac(i, kr)
                              << " lwp=" << h_lwp(i, kr)
                              << " iwp=" << h_iwp(i, kr)
                              << " rel=" << h_rel(i, kr)
                              << " rei=" << h_rei(i, kr)

                              << "\n";
                }
            }
        }
#endif

        // Unpack data and compute heating
        Kokkos::parallel_for("unpack_chunk_data", Kokkos::RangePolicy<>(0, ncol),
            KOKKOS_LAMBDA(int i) {
                int col_idx = beg + i;
                int ix = col_idx % nx;
                int iy = col_idx / nx;

                int hx = topo(iy + h, ix + h); 
                int num_dummy = hx - 1;

                swdn_sfc(iy + h, ix + h) = buffer.sw_flux_dn_k(i, nz + 1);
                swup_sfc(iy + h, ix + h) = buffer.sw_flux_up_k(i, nz + 1);
                lwdn_sfc(iy + h, ix + h) = buffer.lw_flux_dn_k(i, nz + 1);
                lwup_sfc(iy + h, ix + h) = buffer.lw_flux_up_k(i, nz + 1);

                swup_toa(iy + h, ix + h) = buffer.sw_flux_up_k(i, 0);
                swdn_toa(iy + h, ix + h) = buffer.sw_flux_dn_k(i, 0);
                lwup_toa(iy + h, ix + h) = buffer.lw_flux_up_k(i, 0);
                lwdn_toa(iy + h, ix + h) = buffer.lw_flux_dn_k(i, 0);

                for (int k_vvm = h; k_vvm <= h + nz - 1; ++k_vvm) {
                    if (k_vvm < h + hx - 1) {
                        sw_heating(k_vvm, iy + h, ix + h) = 0.0;
                        lw_heating(k_vvm, iy + h, ix + h) = 0.0;
                        net_heating(k_vvm, iy + h, ix + h) = 0.0;
                    } 
                    else {
                        int k = h + nz - k_vvm + num_dummy;
                        Real net_heating_val = buffer.sw_heating_k(i, k) + buffer.lw_heating_k(i, k); 
                        sw_heating(k_vvm, iy + h, ix + h) = buffer.sw_heating_k(i, k);
                        lw_heating(k_vvm, iy + h, ix + h) = buffer.lw_heating_k(i, k);
                        net_heating(k_vvm, iy + h, ix + h) = net_heating_val;
                    }
                }

                for (int k_vvm = h - 1; k_vvm <= h + nz - 1; ++k_vvm) {
                    if (k_vvm < h + hx - 2) {
                        net_sw_flux(k_vvm, iy+h, ix+h) = 0.0;
                        net_lw_flux(k_vvm, iy+h, ix+h) = 0.0;
                        swdn(k_vvm, iy+h, ix+h) = 0.0;
                        lwdn(k_vvm, iy+h, ix+h) = 0.0;
                        lwup(k_vvm, iy+h, ix+h) = 0.0;
                    } 
                    else {
                        int k = h + nz - k_vvm + num_dummy;
                        Real net_sw = buffer.sw_flux_dn_k(i, k) - buffer.sw_flux_up_k(i, k);
                        Real net_lw = buffer.lw_flux_dn_k(i, k) - buffer.lw_flux_up_k(i, k);

                        net_sw_flux(k_vvm, iy+h, ix+h) = net_sw;
                        net_lw_flux(k_vvm, iy+h, ix+h) = net_lw;
                        swdn(k_vvm, iy+h, ix+h) = buffer.sw_flux_dn_k(i, k);
                        lwdn(k_vvm, iy+h, ix+h) = buffer.lw_flux_dn_k(i, k);
                        lwup(k_vvm, iy+h, ix+h) = buffer.lw_flux_up_k(i, k);
                    }
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
