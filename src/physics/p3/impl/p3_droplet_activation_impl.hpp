#ifndef P3_DROPLET_ACTIVATION_IMPL_HPP
#define P3_DROPLET_ACTIVATION_IMPL_HPP

#include "p3_functions.hpp"
#include "share/physics/physics_functions.hpp"
#include "share/physics/physics_saturation_impl.hpp"

namespace scream {
namespace p3 {


template<typename S, typename D>
KOKKOS_FUNCTION
void Functions<S,D>
::droplet_activation(
    const Spack& T_atm, const Spack& th_atm, const Spack& pres,
    const Spack& qv, const Spack& qv_prev, const Spack& qv_sat_l, const Spack& nc_incld,
    const Spack& cld_frac_l, const Int& it, const Scalar& inv_dt,
    Spack& qv2qc_nucleat_tend, Spack& nc_nuclet_tend, Spack& qv2qc_conden_tend,
    const Smask& context)
{
  using physics = scream::physics::Functions<S, D>;
  
  qv2qc_nucleat_tend = 0.0;
  nc_nuclet_tend     = 0.0;
  // no need to initalize qv2qc_conden_tend because it might be given some values in semi analystic part before

  constexpr Scalar thrd = C::THIRD;
  constexpr Scalar mw   = C::MW;    // molecule weight of water [kg/mol]
  constexpr Scalar rr   = C::RR;    // ideal gas constant [J/mol/K]
  constexpr Scalar rhow = C::RHOW;
  constexpr Scalar cons7 = C::CONS7;
  constexpr Scalar vi = C::VI; // number of ions in solution nu
  constexpr Scalar osm = C::OSM; // osmotic potential phi_s [ ]
  constexpr Scalar epsm = C::EPSM; // mass fraction of soluble material [ ]
  constexpr Scalar rhoa = C::RHOA; // density of (dry) aerosol [kg/m3]
  constexpr Scalar map = C::MAP; // molecular weight of aerosol M_s [kg/mol]
  constexpr Scalar bact = vi*osm*epsm*mw*rhoa/(map*rhow);
  constexpr Scalar inv_rm1 = C::INV_RM1;
  constexpr Scalar inv_rm2 = C::INV_RM2;
  constexpr Scalar sig1 = C::SIG1;
  constexpr Scalar sig2 = C::SIG2;
  constexpr Scalar nanew1 = C::NANEW1;
  constexpr Scalar nanew2 = C::NANEW2;
  constexpr Scalar LatVap = C::LatVap; 
  constexpr Scalar rd     = C::Rair;
  constexpr Scalar cp     = C::CP;
  constexpr Scalar inv_cp = C::INV_CP;
  constexpr Scalar rv     = C::RV;

  // Spack dumqvs = physics::qv_sat_dry(T_atm, pres, false, context, physics::MurphyKoop, "p3_act");
  Spack dumqvs = physics::qv_sat_dry(T_atm, pres, false, context, physics::Polysvp1, "p3_act");
  Smask act_mask = context && (qv > dumqvs);

  Spack sup_cld = qv_prev / qv_sat_l - 1.0;
  Smask is_activating = context && (sup_cld > 1.e-6);

  if (is_activating.any()) {
    Spack dum1  = 1.0 / sqrt(bact);
    Spack sigvl = 0.0761 - 1.55e-4 * (T_atm - 273.15);
    Spack aact  = 2.0 * mw / (rhow * rr * T_atm) * sigvl;
    Spack sm1 = 2.0 * dum1 * pow(aact * thrd * inv_rm1, 1.5);
    Spack sm2 = 2.0 * dum1 * pow(aact * thrd * inv_rm2, 1.5);
    Spack uu1 = 2.0 * log(sm1 / sup_cld) / (4.242 * log(sig1));
    Spack uu2 = 2.0 * log(sm2 / sup_cld) / (4.242 * log(sig2));
    dum1 = nanew1 * 0.5 * (1.0 - erf(uu1));
    Spack dum2 = nanew2 * 0.5 * (1.0 - erf(uu2));
    dum2 = min(nanew1+nanew2, dum1+dum2);
    dum2 = (dum2 - nc_incld) * inv_dt * cld_frac_l;
    dum2 = max(0.0, dum2);
    nc_nuclet_tend.set(is_activating, dum2);

    Spack qcnuc(0.0);
    if (it > 1) qcnuc = nc_nuclet_tend * cons7;
    qv2qc_nucleat_tend.set(is_activating, qcnuc);
  }

  if (it <= 1 && context.any()) {
    Spack dumt = th_atm * pow(pres * 1.e-5, rd * inv_cp);
    Spack dumqv = qv;
    // Spack dumqvs = physics::qv_sat_dry(dumt, pres, false, context, physics::MurphyKoop, "p3::droplet_activation");
    Spack dumqvs = physics::qv_sat_dry(dumt, pres, false, context, physics::Polysvp1, "p3::droplet_activation");
    Spack dums = dumqv - dumqvs;
    Spack qccon = (dums / (1.0 + (LatVap * LatVap * dumqvs) / (cp * rv * dumt * dumt))) * inv_dt * cld_frac_l;
    qccon = max(0.0, qccon);
    qccon.set(qccon <= 1.e-7, 0.0);
    qv2qc_conden_tend.set(context, qccon);
  }
}

} // namespace p3
} // namespace scream

#endif // P3_DROPLET_ACTIVATION_IMPL_HPP
