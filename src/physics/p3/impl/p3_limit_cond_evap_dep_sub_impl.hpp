#ifndef P3_LIMIT_COND_EVAP_DEP_SUB_IMPL_HPP
#define P3_LIMIT_COND_EVAP_DEP_SUB_IMPL_HPP

#include "p3_functions.hpp" // for ETI only but harmless for GPU
#include "share/physics/physics_functions.hpp" // also for ETI not on GPUs
#include "share/physics/physics_saturation_impl.hpp"

namespace scream {
namespace p3 {

template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::limit_cond_evap_dep_sub(
      const Spack& T_atm, const Spack& pres, const Spack& qv, const Spack& cld_frac_l, 
      const Scalar& dt, const Scalar& inv_dt, Spack& qv2qc_conden_tend, Spack& nc_nucleat_tend, Spack& qv2qr_conden_tend, 
      Spack& qv2qc_nucleat_tend, Spack& qc2qv_evap_tend, Spack& qr2qv_evap_tend, 
      Spack& nr_evap_tend, Spack& qv2qi_depos_tend, Spack& qv2qi_nucleat_tend, 
      Spack& ni_nucleat_tend, Spack& qi2qv_sublim_tend, Spack& ni_sublim_tend, const Smask& context)
{
  using physics = scream::physics::Functions<Scalar, Device>;

  constexpr Scalar inv_cp       = C::INV_CP;
  constexpr Scalar cp           = C::CP;
  constexpr Scalar rv           = C::RV;
  constexpr Scalar latvap       = C::LatVap;
  constexpr Scalar latice       = C::LatIce;
  constexpr Scalar latvapice    = latvap + latice;

  // Spack dumqvs = physics::qv_sat_dry(T_atm, pres, false, context, physics::MurphyKoop, "p3::limit");
  Spack dumqvs = physics::qv_sat_dry(T_atm, pres, false, context, physics::Polysvp1, "p3::limit");
  Spack qcon_satadj = (qv-dumqvs)/(1.+latvap*latvap*dumqvs/(cp*rv*T_atm*T_atm)) * inv_dt * cld_frac_l;

  // Limit total condensation (incl. activation) and evaporation to saturation adjustment
  Spack total_cond = qv2qc_conden_tend + qv2qc_conden_tend + qv2qc_nucleat_tend; 
  Spack total_evap = qc2qv_evap_tend + qr2qv_evap_tend;

  Smask mask_kill_cond = (total_cond > 0.0) && (qcon_satadj < 0.0);
  Smask mask_scale_cond = !mask_kill_cond && (total_cond > 0.0) && (total_cond > qcon_satadj);
  Smask mask_scale_evap = !mask_kill_cond && !mask_scale_cond && (total_evap > 0.0);

  qv2qc_conden_tend.set(mask_kill_cond, 0.);
  qv2qr_conden_tend.set(mask_kill_cond, 0.);
  qv2qc_nucleat_tend.set(mask_kill_cond, 0.);
  nc_nucleat_tend.set(mask_kill_cond, 0.);

  Spack cond_ratio = 0.0;
  cond_ratio.set(mask_scale_cond, max(0.0, qcon_satadj) / total_cond);
  cond_ratio.set(mask_scale_cond, min(1.0, cond_ratio));
  qv2qc_conden_tend.set(mask_scale_cond, qv2qc_conden_tend*cond_ratio);
  qv2qr_conden_tend.set(mask_scale_cond, qv2qr_conden_tend*cond_ratio);
  qv2qc_nucleat_tend.set(mask_scale_cond, qv2qc_nucleat_tend*cond_ratio);
  nc_nucleat_tend.set(mask_scale_cond, nc_nucleat_tend*cond_ratio);

  Spack evap_ratio = 0.0;
  evap_ratio.set(mask_scale_evap, max(0.0, -qcon_satadj) / total_evap);
  evap_ratio.set(mask_scale_evap, min(1.0, evap_ratio));
  qc2qv_evap_tend.set(mask_scale_evap, qc2qv_evap_tend*evap_ratio);
  qr2qv_evap_tend.set(mask_scale_evap, qr2qv_evap_tend*evap_ratio);
  nr_evap_tend.set(mask_scale_evap, nr_evap_tend*evap_ratio);

  // Limit total deposition (incl. nucleation) and sublimation to saturation adjustment
  Spack qv_tmp = qv + (-qv2qc_nucleat_tend - qv2qc_conden_tend - qv2qr_conden_tend + qc2qv_evap_tend + qr2qv_evap_tend) * dt;
  Spack t_tmp = T_atm + (qv2qc_nucleat_tend+qv2qc_conden_tend+qv2qr_conden_tend - qc2qv_evap_tend - qr2qv_evap_tend)*latvap*inv_cp*dt;
  // Spack dumqvi = physics::qv_sat_dry(t_tmp, pres, true, context, physics::MurphyKoop, "p3::limit");
  Spack dumqvi = physics::qv_sat_dry(t_tmp, pres, true, context, physics::Polysvp1, "p3::limit");
  Spack qdep_satadj = (qv_tmp-dumqvi)/(1.+latvapice*latvapice*dumqvi/(cp*rv*t_tmp*t_tmp)) * inv_dt * cld_frac_l;

  Spack total_dep = qv2qi_depos_tend + qv2qi_nucleat_tend;

  Smask mask_kill_dep = (total_dep > 0.0) && (qdep_satadj < 0.0);
  Smask mask_scale_dep = !mask_kill_dep && (total_dep > 0.0) && (total_dep > qdep_satadj);

  qv2qi_depos_tend.set(mask_kill_dep, 0.);
  qv2qi_nucleat_tend.set(mask_kill_dep, 0.);
  ni_nucleat_tend.set(mask_kill_dep, 0.);

  Spack dep_ratio = 0.0;
  dep_ratio.set(mask_scale_dep, max(0.0, qdep_satadj) / total_dep);
  dep_ratio.set(mask_scale_dep, min(1.0, dep_ratio));
  qv2qi_depos_tend.set(mask_scale_dep, qv2qi_depos_tend * dep_ratio);
  qv2qi_nucleat_tend.set(mask_scale_dep, qv2qi_nucleat_tend * dep_ratio);
  ni_nucleat_tend.set(mask_scale_dep, ni_nucleat_tend * dep_ratio);

  Spack dum(0);
  dum.set(!mask_kill_dep, max(qi2qv_sublim_tend, 1e-20));
  qi2qv_sublim_tend.set(!mask_kill_dep, qi2qv_sublim_tend*min(1.,max(0.,-qdep_satadj)/max(qi2qv_sublim_tend, 1.e-20)) );
  ni_sublim_tend.set(!mask_kill_dep, ni_sublim_tend*min(1., qi2qv_sublim_tend/dum));
}

} // namespace p3
} // namespace scream

#endif
