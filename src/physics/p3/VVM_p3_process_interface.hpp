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

#include "share/scream_types.hpp" 
#include "share/util/scream_common_physics_functions.hpp"
#include "ekat/ekat_pack.hpp"
#include "ekat/kokkos/ekat_kokkos_types.hpp"
#include "ekat/kokkos/ekat_kokkos_utils.hpp" 
#include "ekat/ekat_parameter_list.hpp"
#include "ekat/ekat_assert.hpp"
#include "ekat/ekat_pack_kokkos.hpp"

#include "p3_f90.hpp"
#include "p3_functions.hpp" 
#include "p3_main_wrap.hpp"


namespace VVM {
namespace Physics {

using Real = scream::Real;
using Int = scream::Int;
using DefaultDevice = scream::DefaultDevice;

using P3F           = scream::p3::Functions<Real, DefaultDevice>;
using CP3           = scream::physics::P3_Constants<Real>;
using Spack         = typename P3F::Spack;
using Smask         = typename P3F::Smask;
using TeamPolicy    = typename P3F::KT::TeamPolicy;
using Pack          = ekat::Pack<Real,Spack::n>;
using PF            = scream::PhysicsFunctions<DefaultDevice>;
using PC            = scream::physics::Constants<Real>;
using KT            = ekat::KokkosTypes<DefaultDevice>;
using WSM          = ekat::WorkspaceManager<Spack, KT::Device>;


using P3LookupTables    = typename P3F::P3LookupTables;
using P3PrognosticState = typename P3F::P3PrognosticState;
using P3DiagnosticInputs= typename P3F::P3DiagnosticInputs;
using P3DiagnosticOutputs = typename P3F::P3DiagnosticOutputs;
using P3HistoryOnly     = typename P3F::P3HistoryOnly;
using P3Infrastructure  = typename P3F::P3Infrastructure;
using P3Runtime         = typename P3F::P3Runtime;

using view_2d_spack = typename P3F::view_2d<Spack>;
using view_1d_scalar= typename P3F::view_1d<Real>;

template<size_t Dim>
using VVMView = typename VVM::Core::Field<Dim>::ViewType;
using VVMView3D = VVMView<3>;
using VVMView2D = VVMView<2>;


class VVM_P3_Interface {
public:
    VVM_P3_Interface(const VVM::Utils::ConfigurationManager& config, 
                     const VVM::Core::Grid& grid, 
                     const VVM::Core::Parameters& params);

    void initialize(VVM::Core::State& state);
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
    void unpack_2d_packed_to_3d(const P3ViewType& p3_view, const VVMViewType& vvm_view);

    /**
     * @brief Unpack: P3 1D (col) -> VVM 2D (y,x) + halo (for surface fields)
     */
    template<typename P3ViewType, typename VVMViewType>
    void unpack_1d_to_2d(const P3ViewType& p3_view, const VVMViewType& vvm_view);

    /**
     * @brief Pack: VVM 2D (y,x) + halo -> P3 1D (col) (for surface fields)
     */
    template<typename VVMViewType, typename P3ViewType>
    void pack_2d_to_1d(const VVMViewType& vvm_view, const P3ViewType& p3_view);

protected:
    void allocate_p3_buffers();

private:
    const VVM::Core::Grid& grid_;
    const VVM::Utils::ConfigurationManager& config_;
    const VVM::Core::Parameters& params_;

    int m_num_cols; // nx * ny
    int m_num_levs; // nz
    int m_num_lev_packs; // Spacks dim for P3 

    P3LookupTables            m_lookup_tables; 
    CP3                       m_p3constants;
    P3Runtime                 m_runtime_options; 
    
    // P3 data structure
    P3PrognosticState   m_prog_state;
    P3DiagnosticInputs  m_diag_inputs;
    P3DiagnosticOutputs m_diag_outputs;
    P3HistoryOnly       m_history_only;
    P3Infrastructure    m_infrastructure;

    // Kokkos execution
    TeamPolicy m_policy;
    int m_team_size;
    ekat::WorkspaceManager<Spack, KT::Device> workspace_mgr;

    // P3 2D buffer (col, lev_pakcs)
    // Prognostic state (IN/OUT)
    view_2d_spack m_qv_view;
    view_2d_spack m_qc_view;
    view_2d_spack m_qr_view;
    view_2d_spack m_qi_view;
    view_2d_spack m_qm_view;
    view_2d_spack m_nc_view;
    view_2d_spack m_nr_view;
    view_2d_spack m_ni_view;
    view_2d_spack m_bm_view;
    view_2d_spack m_th_view; 

    // Diagnostic input (IN)
    view_2d_spack m_pres_view;     // pressure [Pa]
    view_2d_spack m_dpres_view;    // delta pressure [Pa]
    view_2d_spack m_dz_view;       // delta height [m]
    view_2d_spack m_inv_exner_view;
    view_2d_spack m_qv_prev_view;
    view_2d_spack m_t_prev_view;   // T (not theta)
    view_2d_spack m_cld_frac_i_view;
    view_2d_spack m_cld_frac_l_view;
    view_2d_spack m_cld_frac_r_view;
    view_2d_spack m_nc_nuceat_tend_view;
    view_2d_spack m_nccn_view;
    view_2d_spack m_ni_activated_view;
    view_2d_spack m_inv_qc_relvar_view;
    
    // Diagnostic (OUT)
    view_1d_scalar m_precip_liq_surf_view;
    view_1d_scalar m_precip_ice_surf_view;
    view_2d_spack  m_qv2qi_depos_tend_view;

    using MemberType = typename P3F::KT::MemberType;

};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_P3_PROCESS_INTERFACE_HPP
