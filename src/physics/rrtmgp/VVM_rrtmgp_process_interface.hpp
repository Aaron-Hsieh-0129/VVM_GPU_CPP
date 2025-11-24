#ifndef VVM_RRTMGP_PROCESS_INTERFACE_HPP
#define VVM_RRTMGP_PROCESS_INTERFACE_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>

// VVM Includes
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Field.hpp"
#include "utils/ConfigurationManager.hpp"

// RRTMGP/EAMxx Includes
#include "physics/rrtmgp/eamxx_rrtmgp_interface.hpp"
#include "cpp/rrtmgp/mo_gas_concentrations.h"

namespace VVM {
namespace Physics {
namespace RRTMGP {

// RRTMGP is performance tuned for layout left views
#define RRTMGP_LAYOUT_LEFT

using ekat::KokkosTypes;
using ekat::Unmanaged;
class RRTMGPRadiation {
public:
    using DefaultDevice = scream::DefaultDevice;
    using Real = scream::Real;
    using Int = scream::Int;
    using KT = ekat::KokkosTypes<DefaultDevice>;

#ifdef RRTMGP_LAYOUT_LEFT
    using layout_t   = Kokkos::LayoutLeft;
#else
    using layout_t   = typename ekat::KokkosTypes<DefaultDevice>::Layout;
#endif

    using real1dk    = Kokkos::View<Real*, DefaultDevice>;
    using real2dk    = Kokkos::View<Real**, layout_t, DefaultDevice>;
    using real3dk    = Kokkos::View<Real***, layout_t, DefaultDevice>;
    using creal1dk   = Kokkos::View<const Real*, DefaultDevice>;
    using creal2dk   = Kokkos::View<const Real**, layout_t, DefaultDevice>;
    using creal3dk   = Kokkos::View<const Real***, layout_t, DefaultDevice>;
    using ureal1dk  = Unmanaged<real1dk>;
    using ureal2dk  = Unmanaged<real2dk>;
    using ureal3dk  = Unmanaged<real3dk>;
    using cureal1dk = Unmanaged<creal1dk>;
    using cureal2dk = Unmanaged<creal2dk>;
    using cureal3dk = Unmanaged<creal3dk>;

    using ci_string = ekat::CaseInsensitiveString;

    using lrreal2dk   = typename KT::template view_2d<Real>;
    using ulrreal2dk  = Unmanaged<lrreal2dk>;

    using interface_t = VVM::Physics::RRTMGP::rrtmgp_interface<Real, layout_t, DefaultDevice>;

    // Constructors
    RRTMGPRadiation(const VVM::Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params);
    ~RRTMGPRadiation();

    std::string name() const { return "rrtmgp"; }

    void initialize(const VVM::Core::State& state);
    void run(VVM::Core::State& state, const double dt);
    void finalize();


    // Dimensions
    int m_ncol;
    int m_num_col_chunks;
    int m_col_chunk_size;
    std::vector<int> m_col_chunk_beg;
    int m_nlay;
    real1dk m_lat;
    real1dk m_lon;

    // Configuration Flags
    // Whether we use aerosol forcing in radiation
    bool m_do_aerosol_rad;
    // Whether we do extra aerosol forcing calls
    bool m_extra_clnsky_diag;
    bool m_extra_clnclrsky_diag;

    // The orbital year, used for zenith angle calculations:
    // If > 0, use constant orbital year for duration of simulation
    // If < 0, use year from timestamp for orbital parameters
    Int m_orbital_year;
    // Orbital parameters, used for zenith angle calculations.
    // If >= 0, bypass computation based on orbital year and use fixed parameters
    // If <  0, compute based on orbital year, specified above
    Real m_orbital_eccen;  // Eccentricity
    Real m_orbital_obliq;  // Obliquity
    Real m_orbital_mvelp;  // Vernal Equinox Mean Longitude of Perihelion
                         //
    // Value for prescribing an invariant solar constant (i.e. total solar irradiance
    // at TOA).  Used for idealized experiments such as RCE. This is only used when a
    // positive value is supplied.
    Real m_fixed_total_solar_irradiance;

    // Fixed solar zenith angle to use for shortwave calculations
    // This is only used if a positive value is supplied
    Real m_fixed_solar_zenith_angle;

    // Band Dimensions
    const int m_nswbands = 14;
    const int m_nlwbands = 16;
    int m_nswgpts;
    int m_nlwgpts;

    // Gas Configuration
    int m_ngas;
    std::vector<ci_string>   m_gas_names;
    real1dk                  m_gas_mol_weights;
    GasConcsK<Real, layout_t, DefaultDevice> m_gas_concs_k;

    // Prescribed Gas Concentrations
    Real m_co2vmr;
    Real m_n2ovmr;
    Real m_ch4vmr;
    Real m_f11vmr;
    Real m_f12vmr;
    Real m_n2vmr;
    Real m_covmr;

    // Radiation Frequency
    int m_rad_freq_in_steps;

    // Subcolumn Sampling Flag
    bool m_do_subcol_sampling;

    struct Buffer {
        static constexpr int num_1d_ncol        = 8;
        static constexpr int num_2d_nlay        = 16;
        static constexpr int num_2d_nlay_p1     = 23;
        static constexpr int num_2d_nswbands    = 2;
        static constexpr int num_3d_nlev_nswbands = 4;
        static constexpr int num_3d_nlev_nlwbands = 2;
        static constexpr int num_3d_nlay_nswbands = 4;
        static constexpr int num_3d_nlay_nlwbands = 2;
        static constexpr int num_3d_nlay_nswgpts = 1;
        static constexpr int num_3d_nlay_nlwgpts = 1;

        // 1d size (ncol)
        ureal1dk sfc_alb_dir_vis_k;
        ureal1dk sfc_alb_dir_nir_k;
        ureal1dk sfc_alb_dif_vis_k;
        ureal1dk sfc_alb_dif_nir_k;
        ureal1dk sfc_flux_dir_vis_k;
        ureal1dk sfc_flux_dir_nir_k;
        ureal1dk sfc_flux_dif_vis_k;
        ureal1dk sfc_flux_dif_nir_k;

        // 2d size (ncol, nlay)
        ureal2dk d_dz;
        ureal2dk p_lay_k;
        ureal2dk t_lay_k;
        ureal2dk z_del_k;
        ureal2dk p_del_k;
        ureal2dk qc_k;
        ureal2dk nc_k;
        ureal2dk qi_k;
        ureal2dk cldfrac_tot_k;
        ureal2dk eff_radius_qc_k;
        ureal2dk eff_radius_qi_k;
        ureal2dk tmp2d_k;
        ureal2dk lwp_k;
        ureal2dk iwp_k;
        ureal2dk sw_heating_k;
        ureal2dk lw_heating_k;

        // 2d size (ncol, nlay+1)
        ureal2dk d_tint;
        ureal2dk p_lev_k;
        ureal2dk t_lev_k;
        ureal2dk sw_flux_up_k;
        ureal2dk sw_flux_dn_k;
        ureal2dk sw_flux_dn_dir_k;
        ureal2dk lw_flux_up_k;
        ureal2dk lw_flux_dn_k;
        ureal2dk sw_clnclrsky_flux_up_k;
        ureal2dk sw_clnclrsky_flux_dn_k;
        ureal2dk sw_clnclrsky_flux_dn_dir_k;
        ureal2dk sw_clrsky_flux_up_k;
        ureal2dk sw_clrsky_flux_dn_k;
        ureal2dk sw_clrsky_flux_dn_dir_k;
        ureal2dk sw_clnsky_flux_up_k;
        ureal2dk sw_clnsky_flux_dn_k;
        ureal2dk sw_clnsky_flux_dn_dir_k;
        ureal2dk lw_clnclrsky_flux_up_k;
        ureal2dk lw_clnclrsky_flux_dn_k;
        ureal2dk lw_clrsky_flux_up_k;
        ureal2dk lw_clrsky_flux_dn_k;
        ureal2dk lw_clnsky_flux_up_k;
        ureal2dk lw_clnsky_flux_dn_k;

        // 3d size (ncol, nlay+1, nswbands)
        ureal3dk sw_bnd_flux_up_k;
        ureal3dk sw_bnd_flux_dn_k;
        ureal3dk sw_bnd_flux_dir_k;
        ureal3dk sw_bnd_flux_dif_k;

        // 3d size (ncol, nlay+1, nlwbands)
        ureal3dk lw_bnd_flux_up_k;
        ureal3dk lw_bnd_flux_dn_k;

        // 2d size (ncol, nswbands)
        ureal2dk sfc_alb_dir_k;
        ureal2dk sfc_alb_dif_k;

        // 3d size (ncol, nlay, n[sw,lw]bands)
        ureal3dk aero_tau_sw_k;
        ureal3dk aero_ssa_sw_k;
        ureal3dk aero_g_sw_k;
        ureal3dk aero_tau_lw_k;

        // 3d size (ncol, nlay, n[sw,lw]bnds)
        ureal3dk cld_tau_sw_bnd_k;
        ureal3dk cld_tau_lw_bnd_k;

        // 3d size (ncol, nlay, n[sw,lw]gpts)
        ureal3dk cld_tau_sw_gpt_k;
        ureal3dk cld_tau_lw_gpt_k;
    };

protected:
    // Helpers for buffer management (replace ATMBufferManager)
    size_t requested_buffer_size_in_bytes() const;
    void init_buffers();

    // VVM References
    const VVM::Core::Grid& m_grid;
    const VVM::Utils::ConfigurationManager& m_config;
    const VVM::Core::Parameters& m_params;

    // Backing storage for the Buffer struct
    Kokkos::View<Real*, DefaultDevice> m_buffer_storage;

    // Struct which contains local variables (Unmanaged views pointing to m_buffer_storage)
    Buffer m_buffer;
};

} // namespace RRTMGP
} // namespace Physics
} // namespace VVM

#endif // VVM_RRTMGP_PROCESS_INTERFACE_HPP
