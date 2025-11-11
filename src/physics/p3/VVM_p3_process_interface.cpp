#include "physics/p3/VVM_p3_process_interface.hpp"
#include "ekat/util/ekat_units.hpp"
#include "impl/p3_main_impl.hpp"

namespace VVM {
namespace Physics {

VVM_P3_Interface::VVM_P3_Interface(const VVM::Utils::ConfigurationManager &config, const VVM::Core::Grid &grid, const VVM::Core::Parameters &params)
    : config_(config), grid_(grid), params_(params), m_p3constants(CP3()) {

    m_num_cols = grid_.get_local_physical_points_x() * grid_.get_local_physical_points_y();
    m_num_levs = grid_.get_local_physical_points_z();
    m_num_lev_packs = ekat::npack<Spack>(m_num_levs);

    // Infrastructure initialization
    // dt is passed as an argument to run
    m_infrastructure.it = 0;
    m_infrastructure.its = 0;
    m_infrastructure.ite = m_num_cols - 1;
    m_infrastructure.kts = 0;
    m_infrastructure.kte = m_num_levs - 1;
    // Get runtime options from config (mimicking m_params.get)
    m_infrastructure.predictNc = config_.get_value<bool>("physics.p3.do_predict_nc");
    m_infrastructure.prescribedCCN = config_.get_value<bool>("physics.p3.do_prescribed_ccn");

    // Set Kokkos execution policy
    m_policy = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, m_num_lev_packs);
    m_team_size = m_policy.team_size();

    allocate_p3_buffers();
}

void VVM_P3_Interface::allocate_p3_buffers() {
    // Mimic the 2d packed views from EAMxx Buffer struct
    m_qv_view = view_2d_spack("qv", m_num_cols, m_num_lev_packs);
    m_qc_view = view_2d_spack("qc", m_num_cols, m_num_lev_packs);
    m_qr_view = view_2d_spack("qr", m_num_cols, m_num_lev_packs);
    m_qi_view = view_2d_spack("qi", m_num_cols, m_num_lev_packs);
    m_qm_view = view_2d_spack("qm", m_num_cols, m_num_lev_packs);
    m_nc_view = view_2d_spack("nc", m_num_cols, m_num_lev_packs);
    m_nr_view = view_2d_spack("nr", m_num_cols, m_num_lev_packs);
    m_ni_view = view_2d_spack("ni", m_num_cols, m_num_lev_packs);
    m_bm_view = view_2d_spack("bm", m_num_cols, m_num_lev_packs);
    m_th_view = view_2d_spack("th_atm", m_num_cols, m_num_lev_packs); // Corresponds to th_atm in EAMxx

    m_pres_view      = view_2d_spack("p_mid", m_num_cols, m_num_lev_packs);
    m_dpres_view     = view_2d_spack("pseudo_density_dry", m_num_cols, m_num_lev_packs);
    m_dz_view        = view_2d_spack("dz", m_num_cols, m_num_lev_packs);
    m_inv_exner_view = view_2d_spack("inv_exner", m_num_cols, m_num_lev_packs);
    m_qv_prev_view   = view_2d_spack("qv_prev_micro_step", m_num_cols, m_num_lev_packs);
    m_t_prev_view    = view_2d_spack("T_prev_micro_step", m_num_cols, m_num_lev_packs);

    m_cld_frac_i_view = view_2d_spack("cld_frac_i", m_num_cols, m_num_lev_packs);
    m_cld_frac_l_view = view_2d_spack("cld_frac_l", m_num_cols, m_num_lev_packs);
    m_cld_frac_r_view = view_2d_spack("cld_frac_r", m_num_cols, m_num_lev_packs);

    m_nc_nuceat_tend_view = view_2d_spack("nc_nuceat_tend", m_num_cols, m_num_lev_packs);
    m_nccn_view           = view_2d_spack("nccn", m_num_cols, m_num_lev_packs);
    m_ni_activated_view   = view_2d_spack("ni_activated", m_num_cols, m_num_lev_packs);
    m_inv_qc_relvar_view  = view_2d_spack("inv_qc_relvar", m_num_cols, m_num_lev_packs);

    m_qv2qi_depos_tend_view = view_2d_spack("qv2qi_depos_tend", m_num_cols, m_num_lev_packs);

    // 1d scalar views
    m_precip_liq_surf_view = view_1d_scalar("precip_liq_surf_mass", m_num_cols);
    m_precip_ice_surf_view = view_1d_scalar("precip_ice_surf_mass", m_num_cols);
}

void VVM_P3_Interface::initialize(VVM::Core::State& state) {
    // Gather runtime options
    m_runtime_options.max_total_ni = config_.get_value<double>("physics.p3.max_total_ni"); 

    // Initialize p3
    bool is_root = (grid_.get_mpi_rank() == 0);
    scream::p3::p3_init(false, is_root);

    // --Prognostic State Variables:
    m_prog_state.qc = m_qc_view;
    m_prog_state.nc = m_nc_view;
    m_prog_state.qr = m_qr_view;
    m_prog_state.nr = m_nr_view;
    m_prog_state.qi = m_qi_view;
    m_prog_state.qm = m_qm_view;
    m_prog_state.ni = m_ni_view;
    m_prog_state.bm = m_bm_view;
    m_prog_state.th = m_th_view;
    m_prog_state.qv = m_qv_view;

    // --Diagnostic Input Variables:
    m_diag_inputs.nc_nuceat_tend = m_nc_nuceat_tend_view;
    m_diag_inputs.nccn           = m_nccn_view;
    m_diag_inputs.ni_activated   = m_ni_activated_view;
    m_diag_inputs.inv_qc_relvar  = m_inv_qc_relvar_view;
    m_diag_inputs.pres           = m_pres_view;
    m_diag_inputs.dpres          = m_dpres_view; 
    m_diag_inputs.qv_prev        = m_qv_prev_view;
    m_diag_inputs.t_prev         = m_t_prev_view;
    m_diag_inputs.cld_frac_l     = m_cld_frac_l_view;
    m_diag_inputs.cld_frac_i     = m_cld_frac_i_view;
    m_diag_inputs.cld_frac_r     = m_cld_frac_r_view;
    m_diag_inputs.dz             = m_dz_view;
    m_diag_inputs.inv_exner      = m_inv_exner_view;

    // --Diagnostic Outputs:
    m_diag_outputs.precip_liq_surf = m_precip_liq_surf_view;
    m_diag_outputs.precip_ice_surf = m_precip_ice_surf_view;
    m_diag_outputs.qv2qi_depos_tend= m_qv2qi_depos_tend_view;

    // Load tables
    P3F::init_kokkos_ice_lookup_tables(m_lookup_tables.ice_table_vals, m_lookup_tables.collect_table_vals);
    P3F::init_kokkos_tables(m_lookup_tables.vn_table_vals, m_lookup_tables.vm_table_vals,
                            m_lookup_tables.revap_table_vals, m_lookup_tables.mu_r_table_vals,
                            m_lookup_tables.dnu_table_vals);

}

template<typename VVMViewType, typename P3ViewType>
void VVM_P3_Interface::pack_3d_to_2d_packed(const VVMViewType& vvm_view, const P3ViewType& p3_view) {
    const int nz_phys = m_num_levs;
    const int ny_phys = grid_.get_local_physical_points_y();
    const int nx_phys = grid_.get_local_physical_points_x();
    const int nlev_packs = m_num_lev_packs;

    const int halo_offset = grid_.get_halo_cells(); 

    Kokkos::parallel_for("pack_3d_to_2d", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();
            
            const int ix_phys = icol % nx_phys;
            const int iy_phys = icol / nx_phys;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs),
                [&](const int k_pack) {
                    auto& pack = p3_view(icol, k_pack);
                    const int k_offset = k_pack * Spack::n;
                    
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, (int)Spack::n),
                        [&](const int k_vec) {
                            const int k_phys = k_offset + k_vec;
                            if (k_phys < nz_phys) {
                                pack[k_vec] = vvm_view(k_phys + halo_offset, 
                                                           iy_phys + halo_offset, 
                                                           ix_phys + halo_offset);
                            } else {
                                pack[k_vec] = 0.0; 
                            }
                        });
                });
        });
}

template<typename P3ViewType, typename VVMViewType>
void VVM_P3_Interface::unpack_2d_packed_to_3d(const P3ViewType& p3_view, const VVMViewType& vvm_view) {
    const int nz_phys = m_num_levs;
    const int ny_phys = grid_.get_local_physical_points_y();
    const int nx_phys = grid_.get_local_physical_points_x();
    const int nlev_packs = m_num_lev_packs;

    const int halo_offset = grid_.get_halo_cells(); 

    Kokkos::parallel_for("unpack_2d_to_3d", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();
            
            const int ix_phys = icol % nx_phys;
            const int iy_phys = icol / nx_phys;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs),
                [&](const int k_pack) {
                    const auto& pack = p3_view(icol, k_pack);
                    const int k_offset = k_pack * Spack::n;
                    
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, (int)Spack::n),
                        [&](const int k_vec) {
                            const int k_phys = k_offset + k_vec;
                            if (k_phys < nz_phys) {
                                vvm_view(k_phys + halo_offset, 
                                         iy_phys + halo_offset, 
                                         ix_phys + halo_offset) = pack[k_vec];
                            }
                        });
                });
        });
}

template<typename P3ViewType, typename VVMViewType>
void VVM_P3_Interface::unpack_1d_to_2d(const P3ViewType& p3_view, const VVMViewType& vvm_view) {
    const int ny_phys = grid_.get_local_physical_points_y();
    const int nx_phys = grid_.get_local_physical_points_x();
    const int ncol = m_num_cols;

    const int halo_offset = grid_.get_halo_cells(); //

    Kokkos::parallel_for("unpack_1d_to_2d", ncol,
        KOKKOS_LAMBDA(const int icol) {
            const int ix_phys = icol % nx_phys; 
            const int iy_phys = icol / nx_phys; 
            
            vvm_view(iy_phys + halo_offset, ix_phys + halo_offset) = p3_view(icol);
        });
}

template<typename VVMViewType, typename P3ViewType>
void VVM_P3_Interface::pack_2d_to_1d(const VVMViewType& vvm_view, const P3ViewType& p3_view) {
    const int ny_phys = grid_.get_local_physical_points_y();
    const int nx_phys = grid_.get_local_physical_points_x();
    const int ncol = m_num_cols;

    const int halo_offset = grid_.get_halo_cells(); //

    Kokkos::parallel_for("pack_2d_to_1d", ncol,
        KOKKOS_LAMBDA(const int icol) {
            const int ix_phys = icol % nx_phys; 
            const int iy_phys = icol / nx_phys; 

            p3_view(icol) = vvm_view(iy_phys + halo_offset, ix_phys + halo_offset);
        });
}



void VVM_P3_Interface::run(VVM::Core::State &state, const double dt) {

    m_infrastructure.dt = dt;
    m_infrastructure.it++;
    // 2. [Pack] 將資料從 VVM State (nz, ny, nx) 複製到 P3 內部緩衝區 (ncol, nlev_packs)
    
    // --- 預後狀態 (Prognostic State) ---
    // (我們假設 VVM 欄位名稱與 P3 內部視圖名稱對應)
    // (並假設 VVM 3D 欄位是 VVM::Core::View3D)
    /*
    pack_3d_to_2d_packed(state.get_field<3>("qc").get_device_data(), m_qc_view);
    pack_3d_to_2d_packed(state.get_field<3>("nc").get_device_data(), m_nc_view);
    pack_3d_to_2d_packed(state.get_field<3>("qr").get_device_data(), m_qr_view);
    pack_3d_to_2d_packed(state.get_field<3>("nr").get_device_data(), m_nr_view);
    pack_3d_to_2d_packed(state.get_field<3>("qi").get_device_data(), m_qi_view);
    pack_3d_to_2d_packed(state.get_field<3>("qm").get_device_data(), m_qm_view);
    pack_3d_to_2d_packed(state.get_field<3>("ni").get_device_data(), m_ni_view);
    pack_3d_to_2d_packed(state.get_field<3>("bm").get_device_data(), m_bm_view);
    pack_3d_to_2d_packed(state.get_field<3>("th").get_device_data(), m_th_view);
    pack_3d_to_2d_packed(state.get_field<3>("qv").get_device_data(), m_qv_view);


    // --- 診斷輸入 (Diagnostic Inputs) ---
    pack_3d_to_2d_packed(state.get_field("nc_nuceat_tend").get_view<VVM::Core::View3D>(), m_nc_nuceat_tend_view);
    pack_3d_to_2d_packed(state.get_field("nccn").get_view<VVM::Core::View3D>(), m_nccn_view);
    pack_3d_to_2d_packed(state.get_field("ni_activated").get_view<VVM::Core::View3D>(), m_ni_activated_view);
    pack_3d_to_2d_packed(state.get_field("inv_qc_relvar").get_view<VVM::Core::View3D>(), m_inv_qc_relvar_view);
    pack_3d_to_2d_packed(state.get_field("p_mid").get_view<VVM::Core::View3D>(), m_pres_view);
    pack_3d_to_2d_packed(state.get_field("pseudo_density_dry").get_view<VVM::Core::View3D>(), m_dpres_view);
    pack_3d_to_2d_packed(state.get_field("qv_prev_micro_step").get_view<VVM::Core::View3D>(), m_qv_prev_view);
    pack_3d_to_2d_packed(state.get_field("T_prev_micro_step").get_view<VVM::Core::View3D>(), m_t_prev_view);
    pack_3d_to_2d_packed(state.get_field("cld_frac_l").get_view<VVM::Core::View3D>(), m_cld_frac_l_view);
    pack_3d_to_2d_packed(state.get_field("cld_frac_i").get_view<VVM::Core::View3D>(), m_cld_frac_i_view);
    pack_3d_to_2d_packed(state.get_field("cld_frac_r").get_view<VVM::Core::View3D>(), m_cld_frac_r_view);
    pack_3d_to_2d_packed(state.get_field("dz").get_view<VVM::Core::View3D>(), m_dz_view);
    pack_3d_to_2d_packed(state.get_field("inv_exner").get_view<VVM::Core::View3D>(), m_inv_exner_view);
    */

    // 3. [Execute] 呼叫 P3 C++ 核心函式
    P3F::p3_main(
        m_runtime_options,
        m_prog_state, 
        m_diag_inputs, 
        m_diag_outputs, 
        m_infrastructure,
        m_history_only,
        m_lookup_tables,
#ifdef SCREAM_P3_SMALL_KERNELS
        temporaries,
#endif
        workspace_mgr, 
        m_num_cols, 
        m_num_levs, 
        m_p3constants
    );

    /*
    // 4. [Unpack] 將更新後的*預後變數*從 P3 緩衝區複製回 VVM State
    unpack_2d_packed_to_3d(m_qc_view, state.get_field("qc").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_nc_view, state.get_field("nc").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_qr_view, state.get_field("qr").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_nr_view, state.get_field("nr").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_qi_view, state.get_field("qi").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_qm_view, state.get_field("qm").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_ni_view, state.get_field("ni").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_bm_view, state.get_field("bm").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_th_view, state.get_field("th_atm").get_view<VVM::Core::View3D>());
    unpack_2d_packed_to_3d(m_qv_view, state.get_field("qv").get_view<VVM::Core::View3D>());

    // 5. [Unpack] 將 P3 產生的*診斷輸出*複製回 VVM State
    // (假設 VVM 中的降水欄位是 2D (ny, nx) -> VVM::Core::View2D)
    unpack_1d_to_2d(m_precip_liq_surf_view, state.get_field("precip_liq_surf_mass").get_view<VVM::Core::View2D>());
    unpack_1d_to_2d(m_precip_ice_surf_view, state.get_field("precip_ice_surf_mass").get_view<VVM::Core::View2D>());
    unpack_2d_packed_to_3d(m_qv2qi_depos_tend_view, state.get_field("qv2qi_depos_tend").get_view<VVM::Core::View3D>());

    // 6. 確保所有 GPU 核心執行完畢
    Kokkos::fence();
    */
}

} // namespace Physics
} // namespace VVM
