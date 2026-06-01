#ifndef VVM_PHYSICS_LAND_PROCESS_HPP
#define VVM_PHYSICS_LAND_PROCESS_HPP

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "core/HaloExchanger.hpp"
#include "core/vvm_types.hpp"
#include "utils/ConfigurationManager.hpp"
#include <Kokkos_Core.hpp>
#include <openacc.h>

extern "C" {
    void run_vvm_land_wrapper(int use_tco_ocean, int nx, int ny, int nsoil, VVM::Real dt,
        int* islimsk, int* vegtype, int* soiltyp, int* slopetyp,
        VVM::Real* sigmaf, VVM::Real* sfemis, VVM::Real* alb, VVM::Real* shdmin, VVM::Real* shdmax,
        VVM::Real* t1, VVM::Real* q1, VVM::Real* u1, VVM::Real* v1, VVM::Real* ps, 
        VVM::Real* prcp, VVM::Real* swdn, VVM::Real* lwdn, VVM::Real* swnet, VVM::Real* hgt, VVM::Real* prslki_in,
        VVM::Real* stc, VVM::Real* smc, VVM::Real* slc, VVM::Real* tskin, VVM::Real* canopy, VVM::Real* snwdph, VVM::Real* sneqv,
        VVM::Real* hflux, VVM::Real* qflux, VVM::Real* evap, VVM::Real* zorl, VVM::Real* cmx, VVM::Real* lai, bool rdlai2d);
}

namespace VVM {
namespace Physics {

class LandProcess {
public:
    LandProcess(const Utils::ConfigurationManager& config, 
                const Core::Grid& grid, 
                const Core::Parameters& params, 
                Core::HaloExchanger& halo_exchanger, 
                Core::State& state,
                std::string ocean_scheme);
    
    ~LandProcess() = default;

    void init();
    void run(VVM::Real dt);
    void finalize();
    void prepare_static_data();
    void preprocessing_and_packing();
    void postprocessing_and_unpacking();
    template<size_t Dim>
    void calculate_tendencies(const std::string& var_name, 
                              Core::Field<Dim>& out_tendency);

private:
    void register_openacc();
    void unregister_openacc();

    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    Core::HaloExchanger& halo_exchanger_;
    Core::State& state_;

    int m_nx;
    int m_ny;
    int m_nsoil;

    int m_halo_x;
    int m_halo_y;

    int m_use_tco_ocean = 0;

    using view_2d_ll = Kokkos::View<VVM::Real**, Kokkos::LayoutLeft>;
    using view_2d_int_ll = Kokkos::View<int**, Kokkos::LayoutLeft>;
    using view_3d_ll = Kokkos::View<VVM::Real***, Kokkos::LayoutLeft>;

    view_2d_int_ll m_islimsk, m_vegtype, m_soiltype, m_slopetype;

    view_2d_ll m_t1, m_q1, m_u1, m_v1, m_ps, m_prcp, m_swdn, m_swnet, m_lwdn, m_hgt, m_prslki;
    view_2d_ll m_sigmaf, m_sfemis, m_alb, m_shdmin, m_shdmax;

    view_3d_ll m_stc, m_smc, m_slc;
    view_2d_ll m_tskin, m_canopy, m_snwdph, m_sneqv, m_zorl, m_cmx, m_lai;

    view_2d_ll m_hflux, m_qflux, m_evap;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_LAND_PROCESS_HPP
