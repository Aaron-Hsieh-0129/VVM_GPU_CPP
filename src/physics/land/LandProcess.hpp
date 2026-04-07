#ifndef VVM_PHYSICS_LAND_PROCESS_HPP
#define VVM_PHYSICS_LAND_PROCESS_HPP

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "core/Parameters.hpp"
#include "core/HaloExchanger.hpp"
#include "utils/ConfigurationManager.hpp"
#include <Kokkos_Core.hpp>
#include <openacc.h>

extern "C" {
    void run_vvm_land_wrapper(int nx, int ny, int nsoil, double dt,
        int* islimsk, int* vegtype, int* soiltyp, int* slopetyp,
        double* t1, double* q1, double* u1, double* v1, double* ps, 
        double* prcp, double* swdn, double* lwdn, double* hgt, double* prslki_in,
        double* stc, double* smc, double* slc, double* tskin, double* canopy, double* snwdph,
        double* hflux, double* qflux, double* evap, double* zorl);
}

namespace VVM {
namespace Physics {

class LandProcess {
public:
    LandProcess(const Utils::ConfigurationManager& config, 
                const Core::Grid& grid, 
                const Core::Parameters& params, 
                Core::HaloExchanger& halo_exchanger, 
                Core::State& state);
    
    ~LandProcess() = default;

    void init();
    void run(double dt);
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

    using view_2d_ll = Kokkos::View<double**, Kokkos::LayoutLeft>;
    using view_2d_int_ll = Kokkos::View<int**, Kokkos::LayoutLeft>;
    using view_3d_ll = Kokkos::View<double***, Kokkos::LayoutLeft>;

    view_2d_int_ll m_islimsk, m_vegtype, m_soiltyp, m_slopetyp;

    view_2d_ll m_t1, m_q1, m_u1, m_v1, m_ps, m_prcp, m_swdn, m_lwdn, m_hgt, m_prslki;

    view_3d_ll m_stc, m_smc, m_slc;
    view_2d_ll m_tskin, m_canopy, m_snwdph, m_zorl;

    view_2d_ll m_hflux, m_qflux, m_evap;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_LAND_PROCESS_HPP
