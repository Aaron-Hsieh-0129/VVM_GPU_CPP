#include "eamxx_rrtmgp_interface.hpp"
#include "share/physics/physics_constants.hpp"

namespace VVM {
namespace Physics {

void init_kls ()
{
  // Initialize kokkos
  if(!Kokkos::is_initialized()) { Kokkos::initialize(); }
}

void finalize_kls()
{
  //Kokkos::finalize(); We do the kokkos finalization elsewhere
}

}  // namespace Physics
}  // namespace VVM
