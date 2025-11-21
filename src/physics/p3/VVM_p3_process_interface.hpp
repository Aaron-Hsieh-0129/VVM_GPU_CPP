#ifndef VVM_PHYSICS_P3_PROCESS_INTERFACE_HPP
#define VVM_PHYSICS_P3_PROCESS_INTERFACE_HPP

#include <mpi.h>
#include <iostream>
#include <string>
#include <map>

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"

#include "share/core/eamxx_types.hpp" 
#include "share/physics/eamxx_common_physics_functions.hpp"
#include <ekat_pack.hpp>
#include <ekat_kokkos_types.hpp>
// #include <ekat_kokkos_utils.hpp>
#include <ekat_parameter_list.hpp>
#include <ekat_assert.hpp>
#include <ekat_pack_kokkos.hpp>
#include <ekat_workspace.hpp>
// #include "ekat/ekat_workspace_impl.hpp"

// #include "p3_f90.hpp"
// #include "p3_main_wrap.hpp"
#include "p3_functions.hpp" 
#include "share/physics/physics_constants.hpp"



using ekat::KokkosTypes;
using ekat::Unmanaged;

namespace ekat {
    using Real = scream::Real;
    using Spack = scream::p3::Functions<Real, scream::DefaultDevice>::Spack;

} // namespace ekat

namespace VVM {
namespace Physics {

using Real = scream::Real;
using Int = scream::Int;
using DefaultDevice = scream::DefaultDevice;

using P3F           = scream::p3::Functions<Real, DefaultDevice>;
using Spack         = typename P3F::Spack;
using Smask         = typename P3F::Smask;
using IntSpack      = typename P3F::IntSmallPack;
using Pack          = ekat::Pack<Real,Spack::n>;
using PF            = scream::PhysicsFunctions<DefaultDevice>;
using PC            = scream::physics::Constants<Real>;
using KT            = ekat::KokkosTypes<DefaultDevice>;
using WSM          = ekat::WorkspaceManager<Spack, KT::Device>;

using view_1d  = typename P3F::view_1d<Real>;
using view_1d_const  = typename P3F::view_1d<const Real>;
using view_2d  = typename P3F::view_2d<Spack>;
using view_2d_const  = typename P3F::view_2d<const Spack>;
using sview_2d = typename ekat::KokkosTypes<DefaultDevice>::template view_2d<Real>;

using uview_1d  = Unmanaged<view_1d>;
using uview_2d  = Unmanaged<view_2d>;
using suview_2d = Unmanaged<sview_2d>;

using TeamPolicy    = typename P3F::KT::TeamPolicy;

using P3LookupTables    = typename P3F::P3LookupTables;
using P3PrognosticState = typename P3F::P3PrognosticState;
using P3DiagnosticInputs= typename P3F::P3DiagnosticInputs;
using P3DiagnosticOutputs = typename P3F::P3DiagnosticOutputs;
using P3HistoryOnly     = typename P3F::P3HistoryOnly;
using P3Infrastructure  = typename P3F::P3Infrastructure;
using P3Runtime         = typename P3F::P3Runtime;

struct p3_preamble {
    p3_preamble() = default;
    // Functor for Kokkos loop to pre-process every run step
    KOKKOS_INLINE_FUNCTION
    void operator()(const KT::MemberType& team) const {
      const int icol = team.league_rank();
      const auto npack = ekat::npack<Spack>(m_nlev);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, npack), [&] (const Int& ipack) {
        // The ipack slice of input variables used more than once
        const Spack& pmid_pack(pmid(icol,ipack));
        const Spack& T_atm_pack(T_atm(icol,ipack));
        const Spack& cld_frac_t_in_pack(cld_frac_t_in(icol,ipack));
        const Spack& cld_frac_l_in_pack(cld_frac_l_in(icol,ipack));
        const Spack& cld_frac_i_in_pack(cld_frac_i_in(icol,ipack));
        const Spack& pseudo_density_pack(pseudo_density(icol,ipack));
        const Spack& pseudo_density_dry_pack(pseudo_density_dry(icol,ipack));

        //compute dz from full pressure
        // dz(icol,ipack) = PF::calculate_dz(pseudo_density_pack, pmid_pack, T_atm_pack, qv(icol,ipack));

        // Wet to dry mixing ratios
        qc(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(qc(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        nc(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(nc(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qr(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(qr(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        nr(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(nr(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qi(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(qi(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        ni(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(ni(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qm(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(qm(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        bm(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(bm(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qv(icol, ipack)      = PF::calculate_drymmr_from_wetmmr_dp_based(qv(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qv_prev(icol, ipack) = PF::calculate_drymmr_from_wetmmr_dp_based(qv_prev(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);

        // Exner from full pressure
        const auto& exner = PF::exner_function(pmid_pack);
        inv_exner(icol,ipack) = 1.0/exner;
        // Potential temperature, from full pressure
        th_atm(icol,ipack) = PF::calculate_theta_from_T(T_atm_pack,pmid_pack);


        if (runtime_opts.use_separate_ice_liq_frac){
          cld_frac_l(icol,ipack) = runtime_opts.set_cld_frac_l_to_one ? 1 : ekat::max(cld_frac_l_in_pack,mincld);
          cld_frac_i(icol,ipack) = runtime_opts.set_cld_frac_i_to_one ? 1 : ekat::max(cld_frac_i_in_pack,mincld);
        } else {
          cld_frac_l(icol,ipack) = runtime_opts.set_cld_frac_l_to_one ? 1 : ekat::max(cld_frac_t_in_pack,mincld);
          cld_frac_i(icol,ipack) = runtime_opts.set_cld_frac_i_to_one ? 1 : ekat::max(cld_frac_t_in_pack,mincld);
        }
        cld_frac_r(icol,ipack) = runtime_opts.set_cld_frac_r_to_one ? 1 : ekat::max(cld_frac_t_in_pack, mincld);

        // update rain cloud fraction given neighboring levels using max-overlap approach.
        if ( not runtime_opts.set_cld_frac_r_to_one ) {
          // Get the cld_frac_t_in(icol, ilev-1) entries
          const auto& cld_frac_t_in_s = ekat::scalarize(ekat::subview(cld_frac_t_in, icol));
          Spack cld_frac_t_in_k, cld_frac_t_in_km1;
          auto range_pack1 = ekat::range<IntSpack>(ipack*Spack::n);
          auto range_pack2 = range_pack1;
          range_pack2.set(range_pack1 < 1, 1); // don't want the shift to go below zero. we mask out that result anyway
          ekat::index_and_shift<-1>(cld_frac_t_in_s, range_pack2, cld_frac_t_in_k, cld_frac_t_in_km1);

          // Hard-coded max-overlap cloud fraction calculation.  Cycle through the layers from top to bottom and
          // determine if the rain fraction needs to be updated to match the cloud fraction in the layer above.
          const auto active_range = range_pack1 > 0 && range_pack1 < m_nlev;
          if (active_range.any()) {
            const auto set_to_t_in = cld_frac_t_in_km1 > cld_frac_r(icol, ipack);
            cld_frac_r(icol,ipack).set(active_range and set_to_t_in, cld_frac_t_in_km1);
          }
        }
      });
    } // operator
    // Local variables
    int m_ncol, m_nlev;
    Real mincld = 0.0001; 
    view_2d_const pmid;
    view_2d_const pmid_dry;
    view_2d_const pseudo_density;
    view_2d_const pseudo_density_dry;
    view_2d       T_atm;
    view_2d_const cld_frac_t_in;
    view_2d_const cld_frac_l_in;
    view_2d_const cld_frac_i_in;
    view_2d       qv;
    view_2d       qc;
    view_2d       nc;
    view_2d       qr;
    view_2d       nr;
    view_2d       qi;
    view_2d       qm;
    view_2d       ni;
    view_2d       bm;
    view_2d       qv_prev;
    view_2d       inv_exner;
    view_2d       th_atm;
    view_2d       cld_frac_l;
    view_2d       cld_frac_i;
    view_2d       cld_frac_r;
    view_2d       dz;

    view_2d       diag_eff_radius_qc;
    view_2d       diag_eff_radius_qi;
    view_2d       diag_eff_radius_qr;
    view_1d_const precip_liq_surf_flux;
    view_1d_const precip_ice_surf_flux;
    view_1d       precip_liq_surf_mass;
    view_1d       precip_ice_surf_mass;
    P3F::P3Runtime runtime_opts;

    // Assigning local variables
    void set_variables(const int ncol, const int nlev,
           const view_2d_const& pmid_, const view_2d_const& pmid_dry_,
           const view_2d_const& pseudo_density_,
           const view_2d_const& pseudo_density_dry_, const view_2d& T_atm_,
           const view_2d_const& cld_frac_t_in_, const view_2d_const& cld_frac_l_in_, const view_2d_const& cld_frac_i_in_,
           const view_2d& qv_, const view_2d& qc_,
           const view_2d& nc_, const view_2d& qr_, const view_2d& nr_, const view_2d& qi_,
           const view_2d& qm_, const view_2d& ni_, const view_2d& bm_, const view_2d& qv_prev_,
           const view_2d& inv_exner_, const view_2d& th_atm_, const view_2d& cld_frac_l_,
           const view_2d& cld_frac_i_, const view_2d& cld_frac_r_, const view_2d& dz_,
           const P3F::P3Runtime& runtime_opts_
           )
    {
        m_ncol = ncol; 
        m_nlev = nlev;
        // IN
        pmid = pmid_; 
        pmid_dry = pmid_dry_; 
        pseudo_density = pseudo_density_;
        pseudo_density_dry = pseudo_density_dry_; 
        T_atm = T_atm_; 
        cld_frac_t_in     = cld_frac_t_in_;
        cld_frac_l_in     = cld_frac_l_in_;
        cld_frac_i_in     = cld_frac_i_in_;
        // OUT
        qv = qv_; 
        qc = qc_; 
        nc = nc_; 
        qr = qr_; 
        nr = nr_; 
        qi = qi_; 
        qm = qm_; 
        ni = ni_;
        bm = bm_; 
        qv_prev = qv_prev_; 
        inv_exner = inv_exner_; 
        th_atm = th_atm_;
        cld_frac_l = cld_frac_l_; 
        cld_frac_i = cld_frac_i_; 
        cld_frac_r = cld_frac_r_; 
        dz = dz_;
        runtime_opts = runtime_opts_;
    } 
}; 

struct p3_postamble {
    p3_postamble() = default;
    // Functor for Kokkos loop to pre-process every run step
    KOKKOS_INLINE_FUNCTION
    void operator()(const KT::MemberType& team) const {
      const int icol = team.league_rank();
      const auto npack = m_npack;
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, npack), [&] (const Int& ipack) {
        const Spack& pseudo_density_pack(pseudo_density(icol,ipack));
        const Spack& pseudo_density_dry_pack(pseudo_density_dry(icol,ipack));

        // Update the atmospheric temperature and the previous temperature.
        {
          const Spack T_atm_before_p3 = T_atm(icol,ipack);
          T_atm(icol,ipack)  = (PF::calculate_T_from_theta(th_atm(icol,ipack),pmid(icol,ipack)) - T_atm_before_p3)
                             * pseudo_density_dry(icol,ipack) / pseudo_density(icol,ipack);
          T_atm(icol,ipack)  +=  T_atm_before_p3;
        }
        T_prev(icol,ipack) = T_atm(icol,ipack); // Update T_prev

        // DRY-TO-WET MMRs
        qc(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(qc(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        nc(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(nc(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qr(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(qr(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        nr(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(nr(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qi(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(qi(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        ni(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(ni(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qm(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(qm(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        bm(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(bm(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qv(icol,ipack) = PF::calculate_wetmmr_from_drymmr_dp_based(qv(icol,ipack),pseudo_density_pack,pseudo_density_dry_pack);
        qv_prev(icol,ipack) = qv(icol,ipack); // Update qv_prev

        // Rescale effective radius' into microns
        diag_eff_radius_qc(icol,ipack) *= 1e6;
        diag_eff_radius_qi(icol,ipack) *= 1e6;
        diag_eff_radius_qr(icol,ipack) *= 1e6;
      }); // for ipack

      // Microphysics can be subcycled together during a single physics timestep,
      // therefore we must accumulate these fluxes.
      // Note: we need to ensure that only a single thread within the team is
      //       updating the mass value.
      Kokkos::single(Kokkos::PerTeam(team), [&] {
        precip_liq_surf_mass(icol) += precip_liq_surf_flux(icol) * PC::RHO_H2O * m_dt;
        precip_ice_surf_mass(icol) += precip_ice_surf_flux(icol) * PC::RHO_H2O * m_dt;
      });
    } // operator
    
    // Local variables
    int m_ncol, m_npack;
    double m_dt;
    view_2d       T_atm; // IN/OUT
    view_2d_const pmid;
    view_2d_const pmid_dry;
    view_2d_const pseudo_density;
    view_2d_const pseudo_density_dry;
    view_2d       th_atm; // IN
    view_2d       T_prev; // OUT
    view_2d       qv;     // IN/OUT
    view_2d       qc;     // IN/OUT
    view_2d       nc;     // IN/OUT
    view_2d       qr;     // IN/OUT
    view_2d       nr;     // IN/OUT
    view_2d       qi;     // IN/OUT
    view_2d       qm;     // IN/OUT
    view_2d       ni;     // IN/OUT
    view_2d       bm;     // IN/OUT
    view_2d       qv_prev;// OUT
    view_2d       diag_eff_radius_qc;
    view_2d       diag_eff_radius_qi;
    view_2d       diag_eff_radius_qr;
    view_1d_const precip_liq_surf_flux;
    view_1d_const precip_ice_surf_flux;
    view_1d       precip_liq_surf_mass;
    view_1d       precip_ice_surf_mass;

    void set_variables(const int ncol, const int npack,
                    const view_2d& th_atm_, const view_2d_const& pmid_, const view_2d_const& pmid_dry_,
                    const view_2d& T_atm_, const view_2d& T_prev_,
                    const view_2d_const& pseudo_density_, const view_2d_const& pseudo_density_dry_,
                    const view_2d& qv_, const view_2d& qc_, const view_2d& nc_, const view_2d& qr_, const view_2d& nr_,
                    const view_2d& qi_, const view_2d& qm_, const view_2d& ni_, const view_2d& bm_,
                    const view_2d& qv_prev_, const view_2d& diag_eff_radius_qc_,
                    const view_2d& diag_eff_radius_qi_, const view_2d& diag_eff_radius_qr_,
                    const view_1d_const& precip_liq_surf_flux_, const view_1d_const& precip_ice_surf_flux_,
                    const view_1d& precip_liq_surf_mass_, const view_1d& precip_ice_surf_mass_)
    {
      // IN
      m_ncol  = ncol; 
      m_npack = npack;
      th_atm = th_atm_; 
      pmid = pmid_; 
      pmid_dry = pmid_dry_;
      pseudo_density = pseudo_density_;
      pseudo_density_dry = pseudo_density_dry_;
      qv = qv_; 
      qc = qc_; 
      nc = nc_; 
      qr = qr_; 
      nr = nr_;
      qi = qi_; 
      qm = qm_; 
      ni = ni_; 
      bm = bm_;
      precip_liq_surf_flux = precip_liq_surf_flux_; 
      precip_ice_surf_flux = precip_ice_surf_flux_;
      // OUT
      T_atm = T_atm_; 
      T_prev = T_prev_; 
      qv_prev = qv_prev_;
      diag_eff_radius_qc   = diag_eff_radius_qc_;
      diag_eff_radius_qi   = diag_eff_radius_qi_;
      diag_eff_radius_qr   = diag_eff_radius_qr_;
      precip_liq_surf_mass = precip_liq_surf_mass_; 
      precip_ice_surf_mass = precip_ice_surf_mass_;
    } 
};


class VVM_P3_Interface {
public:
    VVM_P3_Interface(const VVM::Utils::ConfigurationManager& config, 
                     const VVM::Core::Grid& grid, 
                     const VVM::Core::Parameters& params);

    void initialize(VVM::Core::State& state);
    void finalize();

    void run(VVM::Core::State& state, const double dt);

    /**
     * @brief Pack: VVM 3D (z,y,x) + halo -> P3 2D (col,lev_packs)
     */
    template<typename VVMViewType, typename P3ViewType>
    void pack_3d_to_2d_packed(const VVMViewType& vvm_view, const P3ViewType& p3_view);

    /**
     * @brief Unpack: P3 2D (col,lev_packs) -> VVM 3D (z,y,x) + halo
     */
    template<typename P3ViewType, typename VVMViewType>
    void unpack_2d_packed_to_3d(const P3ViewType& p3_view, VVMViewType& vvm_view);

    /**
     * @brief Unpack: P3 1D (col) -> VVM 2D (y,x) + halo (for surface fields)
     */
    template<typename P3ViewType, typename VVMViewType>
    void unpack_1d_to_2d(const P3ViewType& p3_view, VVMViewType& vvm_view);

    /**
     * @brief Pack: VVM 2D (y,x) + halo -> P3 1D (col) (for surface fields)
     */
    template<typename VVMViewType, typename P3ViewType>
    void pack_2d_to_1d(const VVMViewType& vvm_view, const P3ViewType& p3_view);
    void initialize_constant_buffers(VVM::Core::State& initial_state);

protected:
    void allocate_p3_buffers();

    const VVM::Core::Grid& grid_;
    const VVM::Utils::ConfigurationManager& config_;
    const VVM::Core::Parameters& params_;

    int m_num_cols; // nx * ny
    int m_num_levs; // nz
    int m_num_lev_packs; // Spacks dim for P3 

    P3LookupTables            m_lookup_tables; 
    P3Runtime                 m_runtime_options; 
    
    // P3 data structure
    P3PrognosticState   m_prog_state;
    P3DiagnosticInputs  m_diag_inputs;
    P3DiagnosticOutputs m_diag_outputs;
    P3HistoryOnly       m_history_only;
    P3Infrastructure    m_infrastructure;

    // premable/postamble instance
    p3_preamble  m_p3_preproc;
    p3_postamble m_p3_postproc;

    // Kokkos execution
    TeamPolicy m_policy;
    int m_team_size;
    ekat::WorkspaceManager<Spack, KT::Device> workspace_mgr;
    Spack* m_wsm_data;
    Kokkos::View<Spack*> m_wsm_view_storage;

    // P3 2D buffer (col, lev_pakcs)
    // Prognostic state (IN/OUT)
    view_2d m_qv_view;
    view_2d m_qc_view;
    view_2d m_qr_view;
    view_2d m_qi_view;
    view_2d m_qm_view;
    view_2d m_nc_view;
    view_2d m_nr_view;
    view_2d m_ni_view;
    view_2d m_bm_view;
    view_2d m_th_view; 

    // Diagnostic input (IN)
    view_2d m_pmid_view;           // Total pressure (pmid)
    view_2d m_pmid_dry_view;       // Dry pressure (pmid_dry)
    view_2d m_pseudo_density_view; // Total pseudo-density
    view_2d m_pseudo_density_dry_view; // Dry pseudo-density (dpres)
    view_2d m_T_atm_view;          // Temperature (T_atm)
    view_2d m_cld_frac_t_view;     // Cloud fraction 
    view_2d m_qv_prev_view;        // qv_prev_micro_step
    view_2d m_t_prev_view;         // T_prev_micro_step

    view_2d m_nc_nuceat_tend_view;
    view_2d m_nccn_view;
    view_2d m_ni_activated_view;
    view_2d m_inv_qc_relvar_view;

    view_2d m_dz_view;       // delta height [m]
    view_2d m_inv_exner_view;
    view_2d m_cld_frac_i_view;
    view_2d m_cld_frac_l_view;
    view_2d m_cld_frac_r_view;
    
    // Diagnostic (OUT)
    view_1d m_precip_liq_surf_mass_view; // accumulate
    view_1d m_precip_ice_surf_mass_view; // accumulate
    view_1d m_precip_liq_surf_flux_view;
    view_1d m_precip_ice_surf_flux_view;
    view_2d m_qv2qi_depos_tend_view;
    view_2d m_precip_liq_flux_view;
    view_2d m_precip_ice_flux_view;
    view_2d m_precip_total_tend_view;
    view_2d m_nevapr_view;
    view_2d m_rho_qi_view;

    view_2d m_diag_eff_radius_qc_view;
    view_2d m_diag_eff_radius_qi_view;
    view_2d m_diag_eff_radius_qr_view;

    view_2d m_liq_ice_exchange_view;
    view_2d m_vap_liq_exchange_view;
    view_2d m_vap_ice_exchange_view;
    view_2d m_diag_equiv_reflectivity_view;

    view_2d m_unused;

    sview_2d m_col_location_view;

    using MemberType = typename P3F::KT::MemberType;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_P3_PROCESS_INTERFACE_HPP
