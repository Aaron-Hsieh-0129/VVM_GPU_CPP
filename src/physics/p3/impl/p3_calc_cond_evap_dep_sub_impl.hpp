#ifndef P3_CALC_COND_EVAP_DEP_SUB_IMPL_HPP
#define P3_CALC_COND_EVAP_DEP_SUB_IMPL_HPP

// Aaron - This is the combinition of evaporation_rain and ice_deposition_sublimation.
// This function processes condensation/evaporation/sublimation/deposition that EAMxx deleted.

#include "p3_functions.hpp"
#include "share/physics/physics_functions.hpp"

namespace scream {
namespace p3 {

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::calc_cond_evap_dep_sub(
    const Spack& qc_incld, const Spack& qr_incld, const Spack& qi_incld, 
    const Spack& nr_incld, const Spack& ni_incld,
    const Spack& cld_frac_l, const Spack& cld_frac_r, const Spack& cld_frac_i,
    const Spack& qv, const Spack& qv_prev, const Spack& qv_sat_l, const Spack& qv_sat_i,
    const Spack& ab, const Spack& abi, const Spack& epsc, const Spack& epsr, 
    const Spack& epsi, const Spack& epsi_tot,
    const Spack& T_atm, const Spack& t_prev, const Spack& dqsdt,
    const Scalar& dt, const Scalar& inv_dt, const bool do_ice_production,
    Spack& qv2qc_conden_tend, Spack& qc2qv_evap_tend, 
    Spack& qv2qr_conden_tend, Spack& qr2qv_evap_tend, Spack& nr_evap_tend,
    Spack& qv2qi_vapdep_tend, Spack& qi2qv_sublim_tend, Spack& ni_sublim_tend, 
    Spack& qc2qi_berg_tend, const Smask& context)
{
  qv2qc_conden_tend = 0.0;
  qc2qv_evap_tend   = 0.0;
  qv2qr_conden_tend = 0.0;
  qr2qv_evap_tend   = 0.0;
  nr_evap_tend      = 0.0;
  qv2qi_vapdep_tend = 0.0;
  qi2qv_sublim_tend = 0.0;
  ni_sublim_tend    = 0.0;
  qc2qi_berg_tend   = 0.0;

  constexpr Scalar QSMALL   = C::QSMALL;
  constexpr Scalar Tmelt  = C::Tmelt;
  constexpr Scalar T_zerodegc = C::T_zerodegc;
  constexpr Scalar inv_cp = 1.0/C::Cpair;
  constexpr Scalar latvap = C::LatVap;
  constexpr Scalar latice = C::LatIce;
  constexpr Scalar cp     = C::CP;
  constexpr Scalar g      = C::gravit;
  constexpr Scalar clbfact_dep = 1.0;
  constexpr Scalar clbfact_sub = 1.0;

  Spack ssat_r = qv_prev - qv_sat_l;
  // Aaron - add supersaturation and related parameters
  Spack ssat_cld = qv_prev - qv_sat_l;
  Spack sup_r = qv_prev/qv_sat_l-1.;
  Spack sup_cld = qv_prev/qv_sat_l-1.;
  Spack supi_cld = qv_prev/qv_sat_i-1.;


  const Smask is_freezing  = (T_atm < T_zerodegc) && context && do_ice_production;
  const Smask not_freezing = !is_freezing && context;
  const auto qi_incld_not_small = qi_incld > QSMALL && context;
  
  // NOTE: Aaron: Calculate epsc like Fortran P3. It actually has been calculated in calc_liq_relaxation_timescale

  Spack eps_eff(0.), A_c(0.);
  if (is_freezing.any()){
    // Aaron: Add epsc here
    eps_eff.set(is_freezing, epsc + epsr + epsi_tot*(1 + (latvap+latice)*inv_cp*dqsdt)/abi);
    A_c.set(is_freezing,(qv - qv_prev)*inv_dt - dqsdt*(T_atm-t_prev)*inv_dt
     - (qv_sat_l - qv_sat_i)*(1 + (latvap+latice)*inv_cp*dqsdt)/abi*epsi_tot );
  }
  if (not_freezing.any()){
    // Aaron: Add epsc here
    eps_eff.set(not_freezing, epsc + epsr);
    A_c.set(not_freezing, (qv - qv_prev)*inv_dt - dqsdt*(T_atm-t_prev)*inv_dt );
  }

  if (context.any()) {
    //Set lower bound on eps_eff to prevent division by zero
    eps_eff.set(eps_eff<1e-20 && context, 1e-20);
    const Spack tau_eff = 1.0/eps_eff;

    // Aaron - Add qc condensation and qr condensation
    Spack qccon(0), qrcon(0);
    qccon.set(qc_incld > QSMALL, (A_c*epsc*tau_eff + (ssat_cld*cld_frac_l - A_c*tau_eff) * inv_dt*epsc*tau_eff * (1.0 - exp(-eps_eff * dt))) / ab);
    qrcon.set(qr_incld > QSMALL, (A_c*epsr*tau_eff + (ssat_r*cld_frac_r - A_c*tau_eff) * inv_dt*epsr*tau_eff * (1.0 - exp(-eps_eff * dt))) / ab);
    
    // Aaron - qi
    Spack qidep_base = (A_c*epsi*tau_eff + (ssat_cld*cld_frac_l - A_c*tau_eff) * inv_dt*epsi*tau_eff * (1.0 - exp(-eps_eff * dt))) / abi;
    Spack qidep_corr = (qv_sat_l - qv_sat_i) * epsi / abi;
    Spack qidep(0.0);
    qidep.set(is_freezing && qi_incld_not_small, qidep_base + qidep_corr);

    // Aaron - Add qc, qr small value evaporation just like qr2qv
    Smask is_qc_tiny = qc_incld < 1e-12 && sup_cld < -0.001;
    qccon.set(is_qc_tiny, -qc_incld * inv_dt);

    Smask is_qr_tiny = qr_incld < 1e-12 && sup_r < -0.001;
    qrcon.set(is_qr_tiny, -qr_incld * inv_dt);

    Smask is_qi_tiny = qi_incld < 1e-12 && supi_cld < -0.001;
    qidep.set(is_freezing && is_qi_tiny, -qi_incld * inv_dt);


    // Cloud tendency
    Smask qc_is_neg = qccon < 0.0;
    qc2qv_evap_tend.set(qc_is_neg, -qccon);
    qv2qc_conden_tend.set(qc_is_neg, 0.);
    qv2qc_conden_tend.set(!qc_is_neg, min(qccon, qv * inv_dt));

    // Rain tendency
    Smask qr_is_neg = qrcon < 0.0;
    qr2qv_evap_tend.set(qr_is_neg, -qrcon);
    qv2qr_conden_tend.set(qr_is_neg, 0.);
    qv2qr_conden_tend.set(!qr_is_neg, min(qrcon, qv * inv_dt));
    // nr tendency
    Spack inv_qr_incld(0.0);
    inv_qr_incld.set(qr_incld > 0.0, 1.0 / qr_incld);
    nr_evap_tend.set(qr_is_neg, qr2qv_evap_tend * (nr_incld * inv_qr_incld));
    
    // qi tendency
    Smask qi_is_neg = qidep < 0.0;
    Spack qisub(0.0);
    qisub.set(is_freezing && qi_is_neg, -qidep * clbfact_sub);
    qi2qv_sublim_tend.set(is_freezing && qi_is_neg, min(qisub, qi_incld * inv_dt));
    
    Spack qidep_pos(0.0);
    qidep_pos.set(is_freezing && !qi_is_neg, qidep * clbfact_dep);
    qv2qi_vapdep_tend.set(is_freezing && !qi_is_neg, min(qidep_pos, qv * inv_dt));

    // ni tendency
    Spack inv_qi_incld(0.0);
    inv_qi_incld.set(qi_incld > 0.0, 1.0 / qi_incld);
    ni_sublim_tend.set(is_freezing && qi_is_neg, qi2qv_sublim_tend * (ni_incld * inv_qi_incld));
  }
}

} // namespace p3
} // namespace scream

#endif
