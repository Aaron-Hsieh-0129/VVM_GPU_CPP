#include "physics/p3/VVM_p3_process_interface.hpp"
#include "ekat/util/ekat_units.hpp"
#include "impl/p3_main_impl.hpp"
#include "core/Field.hpp"

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
    m_qv_view = view_2d("qv", m_num_cols, m_num_lev_packs);
    m_qc_view = view_2d("qc", m_num_cols, m_num_lev_packs);
    m_qr_view = view_2d("qr", m_num_cols, m_num_lev_packs);
    m_qi_view = view_2d("qi", m_num_cols, m_num_lev_packs);
    m_qm_view = view_2d("qm", m_num_cols, m_num_lev_packs);
    m_nc_view = view_2d("nc", m_num_cols, m_num_lev_packs);
    m_nr_view = view_2d("nr", m_num_cols, m_num_lev_packs);
    m_ni_view = view_2d("ni", m_num_cols, m_num_lev_packs);
    m_bm_view = view_2d("bm", m_num_cols, m_num_lev_packs);
    m_th_view = view_2d("th_atm", m_num_cols, m_num_lev_packs); // Corresponds to th_atm in EAMxx

    m_pmid_view      = view_2d("p_mid", m_num_cols, m_num_lev_packs);
    m_pmid_dry_view  = view_2d("p_mid_dry", m_num_cols, m_num_lev_packs);
    m_pseudo_density_view      = view_2d("pseudo_density", m_num_cols, m_num_lev_packs);
    m_pseudo_density_dry_view      = view_2d("pseudo_density_dry", m_num_cols, m_num_lev_packs);
    m_T_atm_view    = view_2d("T_atm", m_num_cols, m_num_lev_packs);
    m_cld_frac_t_view    = view_2d("cld_frac_t", m_num_cols, m_num_lev_packs);
    m_qv_prev_view   = view_2d("qv_prev_micro_step", m_num_cols, m_num_lev_packs);
    m_t_prev_view    = view_2d("T_prev_micro_step", m_num_cols, m_num_lev_packs);

    m_nc_nuceat_tend_view = view_2d("nc_nuceat_tend", m_num_cols, m_num_lev_packs);
    m_nccn_view           = view_2d("nccn", m_num_cols, m_num_lev_packs);
    m_ni_activated_view   = view_2d("ni_activated", m_num_cols, m_num_lev_packs);
    m_inv_qc_relvar_view  = view_2d("inv_qc_relvar", m_num_cols, m_num_lev_packs);

    m_dz_view = view_2d("dz", m_num_cols, m_num_lev_packs);
    m_inv_exner_view = view_2d("inv_exner", m_num_cols, m_num_lev_packs);
    m_cld_frac_i_view = view_2d("cld_frac_i", m_num_cols, m_num_lev_packs);
    m_cld_frac_l_view = view_2d("cld_frac_l", m_num_cols, m_num_lev_packs);
    m_cld_frac_r_view = view_2d("cld_frac_r", m_num_cols, m_num_lev_packs);

    // Diagnostic (OUT)
    m_qv2qi_depos_tend_view = view_2d("qv2qi_depos_tend", m_num_cols, m_num_lev_packs);
    m_precip_liq_flux_view = view_2d("precip_liq_flux", m_num_cols, m_num_lev_packs);
    m_precip_ice_flux_view = view_2d("precip_ice_flux", m_num_cols, m_num_lev_packs);
    m_precip_total_tend_view = view_2d("precip_total_tend", m_num_cols, m_num_lev_packs);
    m_nevapr_view = view_2d("nevapr", m_num_cols, m_num_lev_packs);
    m_rho_qi_view = view_2d("rho_qi", m_num_cols, m_num_lev_packs);

    m_diag_eff_radius_qc_view = view_2d("diag_eff_radius_qc", m_num_cols, m_num_lev_packs);
    m_diag_eff_radius_qi_view = view_2d("diag_eff_radius_qi", m_num_cols, m_num_lev_packs);
    m_diag_eff_radius_qr_view = view_2d("diag_eff_radius_qr", m_num_cols, m_num_lev_packs);

    m_liq_ice_exchange_view = view_2d("liq_ice_exchange", m_num_cols, m_num_lev_packs);
    m_vap_liq_exchange_view = view_2d("vap_liq_exchange", m_num_cols, m_num_lev_packs);
    m_vap_ice_exchange_view = view_2d("vap_ice_exchange", m_num_cols, m_num_lev_packs);

    // 1d scalar views
    m_precip_liq_surf_flux_view = view_1d("precip_liq_surf_flux", m_num_cols);
    m_precip_ice_surf_flux_view = view_1d("precip_ice_surf_flux", m_num_cols);
    m_precip_liq_surf_mass_view = view_1d("precip_liq_surf_mass_acc", m_num_cols);
    m_precip_ice_surf_mass_view = view_1d("precip_ice_surf_mass_acc", m_num_cols);

    const Int nk_pack = ekat::npack<Spack>(m_num_levs);
    const int nk_pack_p1 = ekat::npack<Spack>(m_num_levs+1);

    const auto policy = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, nk_pack);
    const int num_wsm_vars = 52;

    const size_t wsm_size_in_bytes = WSM::get_total_bytes_needed(nk_pack_p1, num_wsm_vars, policy);
    const size_t wsm_size_in_spacks = (wsm_size_in_bytes + sizeof(Spack) - 1) / sizeof(Spack);
    m_wsm_view_storage = Kokkos::View<Spack*>("P3 WSM Storage", wsm_size_in_spacks);
    m_wsm_data = m_wsm_view_storage.data();
    if (m_wsm_data == nullptr) {
        std::cerr << "ERROR: FAILED TO ALLOCATE WORKSPACE MANAGER MEMORY FOR P3." << std::endl;
    }
}

void VVM_P3_Interface::initialize(VVM::Core::State& state) {
    // Gather runtime options
    m_runtime_options.max_total_ni = config_.get_value<double>("physics.p3.max_total_ni"); 

    // Note: P3 can tune some constants from the namelist
    // VVM didn't implement this but one can find it in p3/share/physics_constants.hpp
    // m_p3constants.set_p3_from_namelist(m_params);
    // m_p3constants.print_p3constants(m_atm_logger);

    // Initialize p3
    bool is_root = (grid_.get_mpi_rank() == 0);
    scream::p3::p3_init(false, is_root);

    // This section ties the variables in m_prog_state/m_diag_inputs/m_diag_outputs with m_variables --Prognostic State Variables: m_prog_state.qc = m_qc_view; m_prog_state.nc = m_nc_view; m_prog_state.qr = m_qr_view;
    m_prog_state.nr = m_nr_view;
    m_prog_state.qi = m_qi_view;
    m_prog_state.qm = m_qm_view;
    m_prog_state.ni = m_ni_view;
    m_prog_state.bm = m_bm_view;
    m_prog_state.th = m_th_view;
    m_prog_state.qv = m_qv_view;

    // Diagnostic Input Variables:
    m_diag_inputs.nc_nuceat_tend = m_nc_nuceat_tend_view;
    m_diag_inputs.nccn           = m_nccn_view;
    m_diag_inputs.ni_activated   = m_ni_activated_view;
    m_diag_inputs.inv_qc_relvar  = m_inv_qc_relvar_view;

    // P3 will use dry pressure for dry qv_sat
    m_diag_inputs.pres           = m_pmid_dry_view;
    m_diag_inputs.dpres          = m_pseudo_density_dry_view; // This is pressure thickness but eamxx gives pseudo dry density
    m_diag_inputs.qv_prev        = m_qv_prev_view;
    m_diag_inputs.t_prev         = m_t_prev_view;
    m_diag_inputs.cld_frac_l     = m_cld_frac_l_view;
    m_diag_inputs.cld_frac_i     = m_cld_frac_i_view;
    m_diag_inputs.cld_frac_r     = m_cld_frac_r_view;
    m_diag_inputs.dz             = m_dz_view;
    m_diag_inputs.inv_exner      = m_inv_exner_view;

    // Diagnostic Outputs:
    m_diag_outputs.diag_eff_radius_qc = m_diag_eff_radius_qc_view;
    m_diag_outputs.diag_eff_radius_qi = m_diag_eff_radius_qi_view;
    m_diag_outputs.diag_eff_radius_qr = m_diag_eff_radius_qr_view;
    m_diag_outputs.precip_total_tend  = m_precip_total_tend_view;
    m_diag_outputs.nevapr             = m_nevapr_view;

    m_diag_outputs.precip_liq_surf  = m_precip_liq_surf_view;
    m_diag_outputs.precip_ice_surf  = m_precip_ice_surf_view;
    m_diag_outputs.qv2qi_depos_tend = m_qv2qi_depos_tend_view;
    m_diag_outputs.rho_qi           = m_rho_qi_view;
    m_diag_outputs.precip_liq_flux  = m_precip_liq_flux_view;
    m_diag_outputs.precip_ice_flux  = m_precip_ice_flux_view;

    // Load tables
    P3F::init_kokkos_ice_lookup_tables(m_lookup_tables.ice_table_vals, m_lookup_tables.collect_table_vals);
    P3F::init_kokkos_tables(m_lookup_tables.vn_table_vals, m_lookup_tables.vm_table_vals,
                            m_lookup_tables.revap_table_vals, m_lookup_tables.mu_r_table_vals,
                            m_lookup_tables.dnu_table_vals);

    const int nk_pack_p1 = ekat::npack<Spack>(m_num_levs+1);
    workspace_mgr.setup(m_wsm_data, nk_pack_p1, 52, m_policy);

    this->initialize_constant_buffers(state);
}

void VVM_P3_Interface::initialize_constant_buffers(VVM::Core::State& initial_state) {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    VVM::Core::Field<3> dz_mid_3d_field("dz_mid_3d", {nz, ny, nx});
    VVM::Core::Field<3> pbar_3d_field("pbar_3d", {nz, ny, nx});
    VVM::Core::Field<3> rhobar_3d_field("rhobar_3d", {nz, ny, nx});
    VVM::Core::Field<3> inv_pibar_3d_field("inv_pibar_3d", {nz, ny, nx});

    auto& dz_mid_3d = dz_mid_3d_field.get_mutable_device_data();
    auto& pbar_3d = pbar_3d_field.get_mutable_device_data();
    auto& rhobar_3d = rhobar_3d_field.get_mutable_device_data();
    auto& inv_pibar_3d = inv_pibar_3d_field.get_mutable_device_data();

    const auto& dz_mid = params_.dz_mid.get_device_data();
    const auto& pbar = initial_state.get_field<1>("pbar").get_device_data();
    const auto& rhobar = initial_state.get_field<1>("rhobar").get_device_data();
    const auto& pibar = initial_state.get_field<1>("pibar").get_device_data();
    const auto& th = initial_state.get_field<3>("th").get_device_data();
    auto& T = initial_state.get_field<3>("T").get_mutable_device_data();
    Kokkos::parallel_for("AssignValues",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            dz_mid_3d(k,j,i) = dz_mid(k);
            pbar_3d(k,j,i) = pbar(k);
            rhobar_3d(k,j,i) = rhobar(k);
            inv_pibar_3d(k,j,i) = 1. / pibar(k);
            T(k,j,i) = th(k,j,i) * pibar(k);
        }
    );
    pack_3d_to_2d_packed(dz_mid_3d, m_dz_view);
    pack_3d_to_2d_packed(pbar_3d, m_pmid_dry_view);
    pack_3d_to_2d_packed(rhobar_3d, m_pseudo_density_dry_view);
    pack_3d_to_2d_packed(inv_pibar_3d, m_inv_exner_view);
    pack_3d_to_2d_packed(initial_state.get_field<3>("T").get_device_data(), m_t_prev_view);
    pack_3d_to_2d_packed(initial_state.get_field<3>("qv").get_device_data(), m_qv_prev_view);
    
    const Real nccn_val     = 2.0e8; // #/kg
    const Real cld_frac_val = 1.0;   // 100%
    const Real inv_qc_r_val = 1.0;    
    const Real zero_val     = 0.0;

    auto& nccn_view     = m_nccn_view;
    auto& nc_nuceat_view= m_nc_nuceat_tend_view;
    auto& ni_act_view   = m_ni_activated_view;
    auto& cld_frac_l_view = m_cld_frac_l_view;
    auto& cld_frac_i_view = m_cld_frac_i_view;
    auto& cld_frac_r_view = m_cld_frac_r_view;
    auto& inv_qc_r_view = m_inv_qc_relvar_view;

    const int nlev_packs = m_num_lev_packs;
    Kokkos::parallel_for("fill_constant_buffers", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs),
                [&](const int k_pack) {
                    // Aerosol and Nucleation
                    nccn_view(icol, k_pack)      = nccn_val;
                    nc_nuceat_view(icol, k_pack) = zero_val;
                    ni_act_view(icol, k_pack)    = zero_val;

                    // Cloud fraction
                    cld_frac_l_view(icol, k_pack) = cld_frac_val;
                    cld_frac_i_view(icol, k_pack) = cld_frac_val;
                    cld_frac_r_view(icol, k_pack) = cld_frac_val;
                    inv_qc_r_view(icol, k_pack)   = inv_qc_r_val;
                });
        }
    );    
    Kokkos::fence();
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
void VVM_P3_Interface::unpack_2d_packed_to_3d(const P3ViewType& p3_view, VVMViewType& vvm_view) {
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
void VVM_P3_Interface::unpack_1d_to_2d(const P3ViewType& p3_view, VVMViewType& vvm_view) {
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
    // Prognostic State
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

    // Diagnostic Input
    // pack_3d_to_2d_packed(state.get_field<3>("p_mid").get_device_data(), m_pmid_view); // p_dry_mid -> m_pres_view
    // pack_3d_to_2d_packed(state.get_field<3>("qv_prev_micro_step").get_device_data(), m_qv_prev_view);
    // pack_3d_to_2d_packed(state.get_field<3>("T_prev_micro_step").get_device_data(), m_t_prev_view);


    // Assign values to local arrays used by P3, these are now stored in p3_loc.
    Kokkos::parallel_for("p3_main_local_vals", 
        Kokkos::RangePolicy<>(0,m_num_cols),
        m_p3_preproc
    ); // Kokkos::parallel_for(p3_main_local_vals)
    Kokkos::fence();

    workspace_mgr.reset_internals();

    m_infrastructure.dt = dt;
    m_infrastructure.it++;

    P3F::p3_main(
        m_runtime_options, m_prog_state, m_diag_inputs, m_diag_outputs, m_infrastructure,
        m_history_only, m_lookup_tables,
#ifdef SCREAM_P3_SMALL_KERNELS
        temporaries,
#endif
        workspace_mgr, m_num_cols, m_num_levs, m_p3constants
    );

    // Conduct the post-processing of the p3_main output.
    Kokkos::parallel_for("p3_main_local_vals",
        Kokkos::RangePolicy<>(0,m_num_cols),
        m_p3_postproc
    ); // Kokkos::parallel_for(p3_main_local_vals)
    Kokkos::fence();

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
