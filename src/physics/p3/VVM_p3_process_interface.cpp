#include "p3_functions.hpp"
#include "physics/p3/VVM_p3_process_interface.hpp"

#include <ekat_assert.hpp>
#include <ekat_units.hpp>

#include <array>
#include "utils/Timer.hpp"
#include "physics_functions.hpp" // for ETI only but harmless for GPU

using Real = scream::Real;
using Constants = scream::physics::Constants<Real>;
namespace VVM {
namespace Physics {

VVM_P3_Interface::VVM_P3_Interface(const VVM::Utils::ConfigurationManager &config, const VVM::Core::Grid &grid, const VVM::Core::Parameters &params, Core::HaloExchanger& halo_exchanger)
    : config_(config), grid_(grid), params_(params), halo_exchanger_(halo_exchanger), 
    m_num_cols(grid_.get_local_physical_points_x() * grid_.get_local_physical_points_y()),
    m_num_levs(grid_.get_local_physical_points_z()),
    m_num_lev_packs(ekat::npack<Spack>(m_num_levs))
{
    // Infrastructure initialization
    // dt is passed as an argument to run

    // Note that users can use runtime)options to load some configuration related to p3
    // runtime_options.load_runtime_options_from_file(m_params);

    m_infrastructure.it = 0;
    m_infrastructure.its = 0;
    m_infrastructure.ite = m_num_cols - 1;
    m_infrastructure.kts = 0;
    m_infrastructure.kte = m_num_levs - 1;
    // Get runtime options from config (mimicking m_params.get)
    m_infrastructure.predictNc = config_.get_value<bool>("physics.p3.do_predict_nc", true);
    m_infrastructure.prescribedCCN = config_.get_value<bool>("physics.p3.do_prescribed_ccn", true);

    // Set Kokkos execution policy
    using TPF = ekat::TeamPolicyFactory<KT::ExeSpace>;
    m_policy = TPF::get_default_team_policy(m_num_cols, m_num_lev_packs);
    m_team_size = m_policy.team_size();

    allocate_p3_buffers();
}

void VVM_P3_Interface::allocate_p3_buffers() {
    const Int nk_pack = ekat::npack<Spack>(m_num_levs);
    const int nk_pack_p1 = ekat::npack<Spack>(m_num_levs+1);
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
    m_th_view = view_2d("th_atm", m_num_cols, m_num_lev_packs);

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
    m_precip_liq_flux_view = view_2d("precip_liq_flux", m_num_cols, nk_pack_p1);
    m_precip_ice_flux_view = view_2d("precip_ice_flux", m_num_cols, nk_pack_p1);
    m_precip_total_tend_view = view_2d("precip_total_tend", m_num_cols, m_num_lev_packs);
    m_nevapr_view = view_2d("nevapr", m_num_cols, m_num_lev_packs);
    m_rho_qi_view = view_2d("rho_qi", m_num_cols, m_num_lev_packs);

    m_diag_eff_radius_qc_view = view_2d("diag_eff_radius_qc", m_num_cols, m_num_lev_packs);
    m_diag_eff_radius_qi_view = view_2d("diag_eff_radius_qi", m_num_cols, m_num_lev_packs);
    m_diag_eff_radius_qr_view = view_2d("diag_eff_radius_qr", m_num_cols, m_num_lev_packs);

    m_diag_equiv_reflectivity_view = view_2d("diag_equiv_reflectivity", m_num_cols, m_num_lev_packs);

    m_liq_ice_exchange_view = view_2d("liq_ice_exchange", m_num_cols, m_num_lev_packs);
    m_vap_liq_exchange_view = view_2d("vap_liq_exchange", m_num_cols, m_num_lev_packs);
    m_vap_ice_exchange_view = view_2d("vap_ice_exchange", m_num_cols, m_num_lev_packs);

    m_col_location_view = sview_2d("col_location", m_num_cols, 3);

    // 1d scalar views
    m_precip_liq_surf_flux_view = view_1d("precip_liq_surf_flux", m_num_cols);
    m_precip_ice_surf_flux_view = view_1d("precip_ice_surf_flux", m_num_cols);
    m_precip_liq_surf_mass_view = view_1d("precip_liq_surf_mass_acc", m_num_cols);
    m_precip_ice_surf_mass_view = view_1d("precip_ice_surf_mass_acc", m_num_cols);

    m_unused = view_2d("unused", m_num_cols, nk_pack_p1);
    m_unused = view_2d("unused", m_num_cols, nk_pack_p1);                                                            
    m_dummy_input = view_2d("dummy_input_zeros", m_num_cols, nk_pack_p1);                                            
    Kokkos::deep_copy(m_unused, 0.0);                                                                                
    Kokkos::deep_copy(m_dummy_input, 0.0);   

    const int num_wsm_vars = 64;

    const size_t wsm_size_in_bytes = WSM::get_total_bytes_needed(nk_pack_p1, num_wsm_vars, m_policy);
    const size_t wsm_size_in_spacks = (wsm_size_in_bytes + sizeof(Spack) - 1) / sizeof(Spack);
    m_wsm_view_storage = Kokkos::View<Spack*>("P3 WSM Storage", wsm_size_in_spacks);
    m_wsm_data = m_wsm_view_storage.data();
    if (m_wsm_data == nullptr) std::cerr << "ERROR: FAILED TO ALLOCATE WORKSPACE MANAGER MEMORY FOR P3." << std::endl;

}

void VVM_P3_Interface::initialize(VVM::Core::State& state) {
    int nx_total = grid_.get_local_total_points_x();
    int ny_total = grid_.get_local_total_points_y();
    int nz_total = grid_.get_local_total_points_z();
    
    if (!state.has_field("qc")) state.add_field<3>("qc", {nz_total, ny_total, nx_total});
    if (!state.has_field("qr")) state.add_field<3>("qr", {nz_total, ny_total, nx_total});
    if (!state.has_field("qi")) state.add_field<3>("qi", {nz_total, ny_total, nx_total});
    if (!state.has_field("qm")) state.add_field<3>("qm", {nz_total, ny_total, nx_total});
    if (!state.has_field("nc")) state.add_field<3>("nc", {nz_total, ny_total, nx_total});
    if (!state.has_field("nr")) state.add_field<3>("nr", {nz_total, ny_total, nx_total});
    if (!state.has_field("ni")) state.add_field<3>("ni", {nz_total, ny_total, nx_total});
    if (!state.has_field("bm")) state.add_field<3>("bm", {nz_total, ny_total, nx_total});
    if (!state.has_field("precip_liq_surf_mass")) state.add_field<2>("precip_liq_surf_mass", {ny_total, nx_total});
    if (!state.has_field("precip_ice_surf_mass")) state.add_field<2>("precip_ice_surf_mass", {ny_total, nx_total});
    if (!state.has_field("precip_liq_surf_flux")) state.add_field<2>("precip_liq_surf_flux", {ny_total, nx_total});
    if (!state.has_field("precip_ice_surf_flux")) state.add_field<2>("precip_ice_surf_flux", {ny_total, nx_total});
    if (!state.has_field("qp")) state.add_field<3>("qp", {nz_total, ny_total, nx_total}); // qc+qr+qi
    if (!state.has_field("diag_eff_radius_qc")) state.add_field<3>("diag_eff_radius_qc", {nz_total, ny_total, nx_total}); // qc+qr+qi
    if (!state.has_field("diag_eff_radius_qi")) state.add_field<3>("diag_eff_radius_qi", {nz_total, ny_total, nx_total}); // qc+qr+qi
    if (!state.has_field("diag_eff_radius_qr")) state.add_field<3>("diag_eff_radius_qr", {nz_total, ny_total, nx_total}); // qc+qr+qi
    if (!state.has_field("P_wet")) state.add_field<3>("P_wet", {nz_total, ny_total, nx_total});

    // Gather runtime options
    m_runtime_options.max_total_ni = config_.get_value<double>("physics.p3.max_total_ni"); 

    // Note: P3 can tune some constants from the namelist
    // VVM didn't implement this but one can find it in p3/share/physics_constants.hpp
    // m_p3constants.set_p3_from_namelist(m_params);
    // m_p3constants.print_p3constants(m_atm_logger);

    // Initialize p3
    bool is_root = (grid_.get_mpi_rank() == 0);
    m_lookup_tables = P3F::p3_init(/* write_tables = */ true, is_root);


    // This section ties the variables in m_prog_state/m_diag_inputs/m_diag_outputs with m_variables --Prognostic State Variables: m_prog_state.qc = m_qc_view; m_prog_state.nc = m_nc_view; m_prog_state.qr = m_qr_view;
    m_prog_state.qc = m_qc_view;
    m_prog_state.nc = m_nc_view;
    m_prog_state.qr = m_qr_view;
    m_prog_state.nr = m_nr_view;
    m_prog_state.qi = m_qi_view;
    m_prog_state.qm = m_qm_view;
    m_prog_state.ni = m_ni_view;
    m_prog_state.bm = m_bm_view;
    m_prog_state.qv = m_qv_view;
    m_prog_state.th = m_th_view;

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

    if (m_runtime_options.use_hetfrz_classnuc){
        // diag_inputs.hetfrz_immersion_nucleation_tend  = get_field_in("hetfrz_immersion_nucleation_tend").get_view<const Pack**>();
        // diag_inputs.hetfrz_contact_nucleation_tend    = get_field_in("hetfrz_contact_nucleation_tend").get_view<const Pack**>();
        // diag_inputs.hetfrz_deposition_nucleation_tend = get_field_in("hetfrz_deposition_nucleation_tend").get_view<const Pack**>();
    }
    else {
        // set to unused, double check if this has any side effects (testing should catch this)
        m_diag_inputs.hetfrz_immersion_nucleation_tend  = m_unused;
        m_diag_inputs.hetfrz_contact_nucleation_tend    = m_unused;
        m_diag_inputs.hetfrz_deposition_nucleation_tend = m_unused;
    }

    // Diagnostic Outputs:
    m_diag_outputs.diag_eff_radius_qc = m_diag_eff_radius_qc_view;
    m_diag_outputs.diag_eff_radius_qi = m_diag_eff_radius_qi_view;
    m_diag_outputs.diag_eff_radius_qr = m_diag_eff_radius_qr_view;
    m_diag_outputs.precip_total_tend  = m_precip_total_tend_view;
    m_diag_outputs.nevapr             = m_nevapr_view;
    m_diag_outputs.diag_equiv_reflectivity = m_diag_equiv_reflectivity_view;

    m_diag_outputs.precip_liq_surf  = m_precip_liq_surf_flux_view;
    m_diag_outputs.precip_ice_surf  = m_precip_ice_surf_flux_view;
    m_diag_outputs.qv2qi_depos_tend = m_qv2qi_depos_tend_view;
    m_diag_outputs.rho_qi           = m_rho_qi_view;
    m_diag_outputs.precip_liq_flux  = m_precip_liq_flux_view;
    m_diag_outputs.precip_ice_flux  = m_precip_ice_flux_view;

    m_infrastructure.col_location = m_col_location_view;

    m_history_only.liq_ice_exchange  = m_liq_ice_exchange_view;
    m_history_only.vap_liq_exchange  = m_vap_liq_exchange_view;
    m_history_only.vap_ice_exchange  = m_vap_ice_exchange_view;

    if (m_runtime_options.extra_p3_diags) {
        // if we are doing extra diagnostics, assign the fields to the history only struct
        // history_only.qr2qv_evap   = get_field_out("qr2qv_evap").get_view<Pack**>();
        // history_only.qi2qv_sublim = get_field_out("qi2qv_sublim").get_view<Pack**>();
        // history_only.qc2qr_accret = get_field_out("qc2qr_accret").get_view<Pack**>();
        // history_only.qc2qr_autoconv = get_field_out("qc2qr_autoconv").get_view<Pack**>();
        // history_only.qv2qi_vapdep = get_field_out("qv2qi_vapdep").get_view<Pack**>();
        // history_only.qc2qi_berg = get_field_out("qc2qi_berg").get_view<Pack**>();
        // history_only.qc2qr_ice_shed = get_field_out("qc2qr_ice_shed").get_view<Pack**>();
        // history_only.qc2qi_collect = get_field_out("qc2qi_collect").get_view<Pack**>();
        // history_only.qr2qi_collect = get_field_out("qr2qi_collect").get_view<Pack**>();
        // history_only.qc2qi_hetero_freeze = get_field_out("qc2qi_hetero_freeze").get_view<Pack**>();
        // history_only.qr2qi_immers_freeze = get_field_out("qr2qi_immers_freeze").get_view<Pack**>();
        // history_only.qi2qr_melt = get_field_out("qi2qr_melt").get_view<Pack**>();
        // history_only.qr_sed = get_field_out("qr_sed").get_view<Pack**>();
        // history_only.qc_sed = get_field_out("qc_sed").get_view<Pack**>();
        // history_only.qi_sed = get_field_out("qi_sed").get_view<Pack**>();
    } 
    else {
        // if not, let's use the unused buffer
        m_history_only.qr2qv_evap = m_unused;
        m_history_only.qi2qv_sublim = m_unused;
        m_history_only.qc2qr_accret = m_unused;
        m_history_only.qc2qr_autoconv = m_unused;
        m_history_only.qv2qi_vapdep = m_unused;
        m_history_only.qc2qi_berg = m_unused;
        m_history_only.qc2qr_ice_shed = m_unused;
        m_history_only.qc2qi_collect = m_unused;
        m_history_only.qr2qi_collect = m_unused;
        m_history_only.qc2qi_hetero_freeze = m_unused;
        m_history_only.qr2qi_immers_freeze = m_unused;
        m_history_only.qi2qr_melt = m_unused;
        m_history_only.qr_sed = m_unused;
        m_history_only.qc_sed = m_unused;
        m_history_only.qi_sed = m_unused;
    }
    
    auto m_cld_frac_l_in = m_cld_frac_t_view;
    auto m_cld_frac_i_in = m_cld_frac_t_view;
    if (m_runtime_options.use_separate_ice_liq_frac) {
        // cld_frac_l_in = get_field_in("cldfrac_liq").get_view<const Pack **>();
        // cld_frac_i_in = get_field_in("cldfrac_ice").get_view<const Pack **>();
    }
    m_p3_preproc.set_variables(m_num_cols, m_num_lev_packs, m_pmid_view, m_pmid_dry_view, m_pseudo_density_view, m_pseudo_density_dry_view,
        m_T_atm_view, m_cld_frac_t_view, m_cld_frac_l_in, m_cld_frac_i_in,
        m_qv_view, m_qc_view, m_nc_view, m_qr_view, m_nr_view, m_qi_view, m_qm_view, m_ni_view, m_bm_view, m_qv_prev_view,
        m_inv_exner_view, m_th_view, m_cld_frac_l_view, m_cld_frac_i_view, m_cld_frac_r_view, m_dz_view, m_runtime_options
    );
    
    m_p3_postproc.set_variables(m_num_cols, m_num_lev_packs,
        m_th_view, m_pmid_view, m_pmid_dry_view, m_T_atm_view, m_t_prev_view,
        m_pseudo_density_view, m_pseudo_density_dry_view,
        m_qv_view, m_qc_view, m_nc_view, m_qr_view, m_nr_view,
        m_qi_view, m_qm_view, m_ni_view, m_bm_view, m_qv_prev_view,
        m_diag_eff_radius_qc_view, m_diag_eff_radius_qi_view, 
        m_diag_eff_radius_qr_view,
        m_precip_liq_surf_flux_view, m_precip_ice_surf_flux_view,
        m_precip_liq_surf_mass_view, m_precip_ice_surf_mass_view
    );
    
    const int nk_pack_p1 = ekat::npack<Spack>(m_num_levs+1);
    workspace_mgr.setup(m_wsm_data, nk_pack_p1, 64, m_policy);

    this->initialize_constant_buffers(state);
}

void VVM_P3_Interface::initialize_constant_buffers(VVM::Core::State& initial_state) {
    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    m_pseudo_density = std::make_unique<VVM::Core::Field<3>>("pseudo_density", std::array<int, 3>{nz, ny, nx});

    VVM::Core::Field<3> dz_mid_3d_field("dz_mid_3d", {nz, ny, nx});
    // VVM::Core::Field<3> pbar_3d_field("pbar_3d", {nz, ny, nx});
    // VVM::Core::Field<3> dpbar_mid_3d_field("dpbar_mid_3d", {nz, ny, nx});
    VVM::Core::Field<3> p_dry_3d_field("p_dry_3d", {nz, ny, nx});
    VVM::Core::Field<3> dp_dry_3d_field("dp_dry_3d", {nz, ny, nx});
    VVM::Core::Field<3> rhobar_3d_field("rhobar_3d", {nz, ny, nx});
    VVM::Core::Field<3> inv_pibar_3d_field("inv_pibar_3d", {nz, ny, nx});

    auto& dz_mid_3d = dz_mid_3d_field.get_mutable_device_data();
    // auto& pbar_3d = pbar_3d_field.get_mutable_device_data();
    // auto& dpbar_mid_3d = dpbar_mid_3d_field.get_mutable_device_data();
    auto& p_dry_3d = p_dry_3d_field.get_mutable_device_data();
    auto& dp_dry_3d = dp_dry_3d_field.get_mutable_device_data();
    auto& rhobar_3d = rhobar_3d_field.get_mutable_device_data();
    auto& inv_pibar_3d = inv_pibar_3d_field.get_mutable_device_data();

    const auto& dz_mid = params_.dz_mid.get_device_data();
    const auto& pbar = initial_state.get_field<1>("pbar").get_device_data();
    const auto& dpbar_mid = initial_state.get_field<1>("dpbar_mid").get_device_data();
    const auto& rhobar = initial_state.get_field<1>("rhobar").get_device_data();
    const auto& pibar = initial_state.get_field<1>("pibar").get_device_data();
    const auto& th = initial_state.get_field<3>("th").get_device_data();
    auto& T = initial_state.get_field<3>("T").get_mutable_device_data();
    Kokkos::parallel_for("AssignValues",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            dz_mid_3d(k,j,i) = dz_mid(k);
            // pbar_3d(k,j,i) = pbar(k);
            // dpbar_mid_3d(k,j,i) = dpbar_mid(k);
            p_dry_3d(k,j,i) = pbar(k);
            dp_dry_3d(k,j,i) = dpbar_mid(k);
            rhobar_3d(k,j,i) = rhobar(k);
            inv_pibar_3d(k,j,i) = 1. / pibar(k);
            T(k,j,i) = th(k,j,i) * pibar(k);
        }
    );
    pack_3d_to_2d_packed(dz_mid_3d, m_dz_view);
    // pack_3d_to_2d_packed(pbar_3d, m_pmid_view);
    // pack_3d_to_2d_packed(dpbar_mid_3d, m_pseudo_density_dry_view);
    pack_3d_to_2d_packed(p_dry_3d, m_pmid_dry_view);
    pack_3d_to_2d_packed(dp_dry_3d, m_pseudo_density_dry_view);

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
    auto& cld_frac_t_view = m_cld_frac_t_view;
    auto& inv_qc_r_view = m_inv_qc_relvar_view;
    auto& col_loc_view = m_col_location_view;

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
                    cld_frac_t_view(icol, k_pack) = cld_frac_val;
                    inv_qc_r_view(icol, k_pack)   = inv_qc_r_val;

                    col_loc_view(icol, 0) = static_cast<Real>(icol); 
                    col_loc_view(icol, 1) = 0.0;
                    col_loc_view(icol, 2) = 0.0;
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

void VVM_P3_Interface::preprocessing_and_packing(VVM::Core::State& state) {
    const int nz_phys = m_num_levs;
    const int nx_phys = grid_.get_local_physical_points_x();
    // const int ny_phys = grid_.get_local_physical_points_y();
    const int nlev_packs = m_num_lev_packs;
    const int halo = grid_.get_halo_cells();

    auto qc_3d = state.get_field<3>("qc").get_mutable_device_data();
    auto nc_3d = state.get_field<3>("nc").get_mutable_device_data();
    auto qr_3d = state.get_field<3>("qr").get_mutable_device_data();
    auto nr_3d = state.get_field<3>("nr").get_mutable_device_data();
    auto qi_3d = state.get_field<3>("qi").get_mutable_device_data();
    auto qm_3d = state.get_field<3>("qm").get_mutable_device_data();
    auto ni_3d = state.get_field<3>("ni").get_mutable_device_data();
    auto bm_3d = state.get_field<3>("bm").get_mutable_device_data();
    auto th_3d = state.get_field<3>("th").get_device_data();
    auto qv_3d = state.get_field<3>("qv").get_mutable_device_data();

    auto th_m_3d = state.get_field<3>("th_m").get_device_data();
    auto qv_m_3d = state.get_field<3>("qv_m").get_device_data();

    
    auto pibar = state.get_field<1>("pibar").get_device_data();
    auto pbar = state.get_field<1>("pbar").get_device_data();
    auto dpbar_mid = state.get_field<1>("dpbar_mid").get_device_data();

    auto qc_p3 = m_qc_view;
    auto nc_p3 = m_nc_view;
    auto qr_p3 = m_qr_view;
    auto nr_p3 = m_nr_view;
    auto qi_p3 = m_qi_view;
    auto qm_p3 = m_qm_view;
    auto ni_p3 = m_ni_view;
    auto bm_p3 = m_bm_view;
    auto th_p3 = m_th_view;
    auto qv_p3 = m_qv_view;

    auto T_p3 = m_T_atm_view;
    auto Tm_p3 = m_t_prev_view;
    auto P_wet_p3 = m_pmid_view;
    auto P_dry_p3 = m_pmid_dry_view;
    auto Rho_wet_p3 = m_pseudo_density_view;
    auto Rho_dry_p3 = m_pseudo_density_dry_view;
    auto qv_prev_p3 = m_qv_prev_view;

    Kokkos::parallel_for("Fused_PreProc_Pack", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();
            const int ix = icol % nx_phys;
            const int iy = icol / nx_phys;
            
            const int ix_vvm = ix + halo;
            const int iy_vvm = iy + halo;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs), [&](const int k_pack) {
                Spack qc_val, nc_val, qr_val, nr_val, qi_val, qm_val, ni_val, bm_val, th_val, qv_val;
                Spack T_val, Tm_val, P_wet_val, P_dry_val, Rho_wet_val, qv_prev_val;

                const int k_offset = k_pack * Spack::n;

                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, (int)Spack::n), [&](const int k_vec) {
                    const int k_phys = k_offset + k_vec;
                    
                    if (k_phys < nz_phys) {
                        const int k_vvm = k_phys + halo;

                        Real qv_in = qv_3d(k_vvm, iy_vvm, ix_vvm);
                        if (qv_in < 0) { qv_in = 0; qv_3d(k_vvm, iy_vvm, ix_vvm) = 0; }
                        
                        Real qc_in = qc_3d(k_vvm, iy_vvm, ix_vvm);
                        if (qc_in < 0) { qc_in = 0; qc_3d(k_vvm, iy_vvm, ix_vvm) = 0; }

                        Real qr_in = qr_3d(k_vvm, iy_vvm, ix_vvm);
                        if (qr_in < 0) { qr_in = 0; qr_3d(k_vvm, iy_vvm, ix_vvm) = 0; }

                        Real qi_in = qi_3d(k_vvm, iy_vvm, ix_vvm);
                        if (qi_in < 0) { qi_in = 0; qi_3d(k_vvm, iy_vvm, ix_vvm) = 0; }

                        Real nc_in = nc_3d(k_vvm, iy_vvm, ix_vvm);
                        if (nc_in < 0) { nc_in = 0; nc_3d(k_vvm, iy_vvm, ix_vvm) = 0; }

                        Real ni_in = ni_3d(k_vvm, iy_vvm, ix_vvm);
                        if (ni_in < 0) { ni_in = 0; ni_3d(k_vvm, iy_vvm, ix_vvm) = 0; }

                        Real nr_in = nr_3d(k_vvm, iy_vvm, ix_vvm);
                        if (nr_in < 0) { nr_in = 0; nr_3d(k_vvm, iy_vvm, ix_vvm) = 0; }

                        Real qm_in = qm_3d(k_vvm, iy_vvm, ix_vvm);
                        Real bm_in = bm_3d(k_vvm, iy_vvm, ix_vvm);
                        Real th_in = th_3d(k_vvm, iy_vvm, ix_vvm);
                        Real th_m_in = th_m_3d(k_vvm, iy_vvm, ix_vvm);
                        Real qv_m_in = qv_m_3d(k_vvm, iy_vvm, ix_vvm);

                        Real pi_val = pibar(k_vvm);
                        Real pb_val = pbar(k_vvm);
                        Real dp_val = dpbar_mid(k_vvm);

                        Real T_calc = th_in * pi_val;
                        Real Tm_calc = th_m_in * pi_val;
                        
                        Real P_wet_calc = pb_val * (1.0 + qv_in);
                        Real Rho_wet_calc = dp_val * (1.0 + qv_in);
                        
                        Real P_dry_calc = pb_val;

                        qc_val[k_vec] = qc_in;
                        nc_val[k_vec] = nc_in;
                        qr_val[k_vec] = qr_in;
                        nr_val[k_vec] = nr_in;
                        qi_val[k_vec] = qi_in;
                        qm_val[k_vec] = qm_in;
                        ni_val[k_vec] = ni_in;
                        bm_val[k_vec] = bm_in;
                        th_val[k_vec] = th_in;
                        qv_val[k_vec] = qv_in;

                        T_val[k_vec] = T_calc;
                        Tm_val[k_vec] = Tm_calc;
                        P_wet_val[k_vec] = P_wet_calc;
                        P_dry_val[k_vec] = P_dry_calc;
                        Rho_wet_val[k_vec] = Rho_wet_calc;
                        qv_prev_val[k_vec] = qv_m_in;

                    } 
                    else {
                        qc_val[k_vec] = 0.0; nc_val[k_vec] = 0.0;
                        qr_val[k_vec] = 0.0; nr_val[k_vec] = 0.0;
                        qi_val[k_vec] = 0.0; qm_val[k_vec] = 0.0;
                        ni_val[k_vec] = 0.0; bm_val[k_vec] = 0.0;
                        th_val[k_vec] = 0.0; qv_val[k_vec] = 0.0;
                        T_val[k_vec] = 0.0;  Tm_val[k_vec] = 0.0;
                        P_wet_val[k_vec] = 0.0; P_dry_val[k_vec] = 0.0;
                        Rho_wet_val[k_vec] = 0.0;
                        qv_prev_val[k_vec] = 0.0;
                    }
                });

                qc_p3(icol, k_pack) = qc_val;
                nc_p3(icol, k_pack) = nc_val;
                qr_p3(icol, k_pack) = qr_val;
                nr_p3(icol, k_pack) = nr_val;
                qi_p3(icol, k_pack) = qi_val;
                qm_p3(icol, k_pack) = qm_val;
                ni_p3(icol, k_pack) = ni_val;
                bm_p3(icol, k_pack) = bm_val;
                th_p3(icol, k_pack) = th_val;
                qv_p3(icol, k_pack) = qv_val;

                T_p3(icol, k_pack) = T_val;
                Tm_p3(icol, k_pack) = Tm_val;
                P_wet_p3(icol, k_pack) = P_wet_val;
                P_dry_p3(icol, k_pack) = P_dry_val;
                Rho_wet_p3(icol, k_pack) = Rho_wet_val;
                qv_prev_p3(icol, k_pack) = qv_prev_val;
            });
        }
    );
}


void VVM_P3_Interface::postprocessing_and_unpacking(VVM::Core::State& state) {
    const int nz_phys = m_num_levs;
    const int nx_phys = grid_.get_local_physical_points_x();
    const int nlev_packs = m_num_lev_packs;
    const int halo = grid_.get_halo_cells();

    auto qc_3d = state.get_field<3>("qc").get_mutable_device_data();
    auto nc_3d = state.get_field<3>("nc").get_mutable_device_data();
    auto qr_3d = state.get_field<3>("qr").get_mutable_device_data();
    auto nr_3d = state.get_field<3>("nr").get_mutable_device_data();
    auto qi_3d = state.get_field<3>("qi").get_mutable_device_data();
    auto qm_3d = state.get_field<3>("qm").get_mutable_device_data();
    auto ni_3d = state.get_field<3>("ni").get_mutable_device_data();
    auto bm_3d = state.get_field<3>("bm").get_mutable_device_data();
    auto th_3d = state.get_field<3>("th").get_mutable_device_data();
    auto qv_3d = state.get_field<3>("qv").get_mutable_device_data();
    auto qp_3d = state.get_field<3>("qp").get_mutable_device_data(); 
    auto T_3d  = state.get_field<3>("T").get_mutable_device_data();
    auto qv_m_3d = state.get_field<3>("qv_m").get_mutable_device_data();

    auto eff_rad_qc_3d = state.get_field<3>("diag_eff_radius_qc").get_mutable_device_data();
    auto eff_rad_qi_3d = state.get_field<3>("diag_eff_radius_qi").get_mutable_device_data();
    auto eff_rad_qr_3d = state.get_field<3>("diag_eff_radius_qr").get_mutable_device_data();

    auto precip_liq_surf_2d = state.get_field<2>("precip_liq_surf_mass").get_mutable_device_data();
    auto precip_ice_surf_2d = state.get_field<2>("precip_ice_surf_mass").get_mutable_device_data();
    auto precip_liq_surf_flux_2d = state.get_field<2>("precip_liq_surf_flux").get_mutable_device_data();
    auto precip_ice_surf_flux_2d = state.get_field<2>("precip_ice_surf_flux").get_mutable_device_data();

    auto ITYPEW = state.get_field<3>("ITYPEW").get_device_data();
    auto thbar  = state.get_field<1>("thbar").get_device_data();
    auto pibar  = state.get_field<1>("pibar").get_device_data();

    auto qc_p3 = m_qc_view;
    auto nc_p3 = m_nc_view;
    auto qr_p3 = m_qr_view;
    auto nr_p3 = m_nr_view;
    auto qi_p3 = m_qi_view;
    auto qm_p3 = m_qm_view;
    auto ni_p3 = m_ni_view;
    auto bm_p3 = m_bm_view;
    auto th_p3 = m_th_view;
    auto qv_p3 = m_qv_view;
    auto T_p3  = m_T_atm_view;
    auto qv_prev_p3 = m_qv_prev_view;
    
    auto eff_rad_qc_p3 = m_diag_eff_radius_qc_view;
    auto eff_rad_qi_p3 = m_diag_eff_radius_qi_view;
    auto eff_rad_qr_p3 = m_diag_eff_radius_qr_view;

    auto precip_liq_surf_p3 = m_precip_liq_surf_mass_view;
    auto precip_ice_surf_p3 = m_precip_ice_surf_mass_view;
    auto precip_liq_surf_flux_p3 = m_precip_liq_surf_flux_view;
    auto precip_ice_surf_flux_p3 = m_precip_ice_surf_flux_view;

    Kokkos::parallel_for("Fused_PostProc_Unpack", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();
            const int ix = icol % nx_phys;
            const int iy = icol / nx_phys;
            
            const int ix_vvm = ix + halo;
            const int iy_vvm = iy + halo;

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                precip_liq_surf_2d(iy_vvm, ix_vvm) = precip_liq_surf_p3(icol);
                precip_ice_surf_2d(iy_vvm, ix_vvm) = precip_ice_surf_p3(icol);
                precip_liq_surf_flux_2d(iy_vvm, ix_vvm) = precip_liq_surf_flux_p3(icol);
                precip_ice_surf_flux_2d(iy_vvm, ix_vvm) = precip_ice_surf_flux_p3(icol);
            });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs), [&](const int k_pack) {
                const int k_offset = k_pack * Spack::n;

                Spack qc_val = qc_p3(icol, k_pack);
                Spack nc_val = nc_p3(icol, k_pack);
                Spack qr_val = qr_p3(icol, k_pack);
                Spack nr_val = nr_p3(icol, k_pack);
                Spack qi_val = qi_p3(icol, k_pack);
                Spack qm_val = qm_p3(icol, k_pack);
                Spack ni_val = ni_p3(icol, k_pack);
                Spack bm_val = bm_p3(icol, k_pack);
                Spack th_val = th_p3(icol, k_pack);
                Spack qv_val = qv_p3(icol, k_pack);
                Spack T_val  = T_p3(icol, k_pack);
                Spack qv_prev_val = qv_prev_p3(icol, k_pack);
                
                Spack eff_qc_val = eff_rad_qc_p3(icol, k_pack);
                Spack eff_qi_val = eff_rad_qi_p3(icol, k_pack);
                Spack eff_qr_val = eff_rad_qr_p3(icol, k_pack);

                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, (int)Spack::n), [&](const int k_vec) {
                    const int k_phys = k_offset + k_vec;
                    if (k_phys < nz_phys) {
                        const int k_vvm = k_phys + halo;
                        
                        if (ITYPEW(k_vvm, iy_vvm, ix_vvm) != 1) {
                            qc_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            nc_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            qr_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            nr_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            qi_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            qm_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            ni_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            bm_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            qv_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            th_3d(k_vvm, iy_vvm, ix_vvm) = thbar(k_vvm);
                            
                            T_3d(k_vvm, iy_vvm, ix_vvm) = thbar(k_vvm) * pibar(k_vvm);
                            qv_m_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            qp_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;

                            eff_rad_qc_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            eff_rad_qi_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;
                            eff_rad_qr_3d(k_vvm, iy_vvm, ix_vvm) = 0.0;

                        } 
                        else {
                            qc_3d(k_vvm, iy_vvm, ix_vvm) = qc_val[k_vec];
                            nc_3d(k_vvm, iy_vvm, ix_vvm) = nc_val[k_vec];
                            qr_3d(k_vvm, iy_vvm, ix_vvm) = qr_val[k_vec];
                            nr_3d(k_vvm, iy_vvm, ix_vvm) = nr_val[k_vec];
                            qi_3d(k_vvm, iy_vvm, ix_vvm) = qi_val[k_vec];
                            qm_3d(k_vvm, iy_vvm, ix_vvm) = qm_val[k_vec];
                            ni_3d(k_vvm, iy_vvm, ix_vvm) = ni_val[k_vec];
                            bm_3d(k_vvm, iy_vvm, ix_vvm) = bm_val[k_vec];
                            th_3d(k_vvm, iy_vvm, ix_vvm) = T_val[k_vec] / pibar(k_vvm);
                            qv_3d(k_vvm, iy_vvm, ix_vvm) = qv_val[k_vec];
                            
                            T_3d(k_vvm, iy_vvm, ix_vvm)  = T_val[k_vec]; 
                            qv_m_3d(k_vvm, iy_vvm, ix_vvm) = qv_prev_val[k_vec];

                            qp_3d(k_vvm, iy_vvm, ix_vvm) = qc_val[k_vec] + qr_val[k_vec] + qi_val[k_vec];

                            eff_rad_qc_3d(k_vvm, iy_vvm, ix_vvm) = eff_qc_val[k_vec];
                            eff_rad_qi_3d(k_vvm, iy_vvm, ix_vvm) = eff_qi_val[k_vec];
                            eff_rad_qr_3d(k_vvm, iy_vvm, ix_vvm) = eff_qr_val[k_vec];
                        }
                    }
                });
            });
        }
    );


    halo_exchanger_.exchange_halos(state.get_field<3>("qc"));
    halo_exchanger_.exchange_halos(state.get_field<3>("nc"));
    halo_exchanger_.exchange_halos(state.get_field<3>("qr"));
    halo_exchanger_.exchange_halos(state.get_field<3>("nr"));
    halo_exchanger_.exchange_halos(state.get_field<3>("qi"));
    halo_exchanger_.exchange_halos(state.get_field<3>("qm"));
    halo_exchanger_.exchange_halos(state.get_field<3>("ni"));
    halo_exchanger_.exchange_halos(state.get_field<3>("bm"));
    halo_exchanger_.exchange_halos(state.get_field<3>("th"));
    halo_exchanger_.exchange_halos(state.get_field<3>("qv"));
    halo_exchanger_.exchange_halos(state.get_field<3>("qp"));
    halo_exchanger_.exchange_halos(state.get_field<3>("T"));
    halo_exchanger_.exchange_halos(state.get_field<3>("qv_m"));
}


void VVM_P3_Interface::run(VVM::Core::State &state, const double dt) {
    VVM::Utils::Timer p3_timer("P3_timer");


    // FIXME: The pmid, pmid_try should be decided. The pmid from VVM is now dry pressure. 

    const int nz = grid_.get_local_total_points_z();
    const int ny = grid_.get_local_total_points_y();
    const int nx = grid_.get_local_total_points_x();
    const int h = grid_.get_halo_cells();

    preprocessing_and_packing(state);
    // Kokkos::fence();

    m_p3_postproc.m_dt = dt;

    // Assign values to local arrays used by P3, these are now stored in p3_loc.
    Kokkos::parallel_for("p3_main_local_vals", 
        m_policy,
        m_p3_preproc
    ); // Kokkos::parallel_for(p3_main_local_vals)
    // Kokkos::fence();

    m_infrastructure.dt = dt;
    m_infrastructure.it++;

    workspace_mgr.reset_internals();


    // Pre-P3 Saturation Adjustment (Emulating qcnuc + qccon)
    // This step creates cloud water (qc) and cloud number (nc) from supersaturation
    // BEFORE P3 runs, acting as the "activation" and "macrophysics" step.
    // Constants
    const Real nccn_val = 2.0e8; // Target CCN concentration (#/kg) or use m_nccn_view
    const Real min_qc = 1.0e-12; // Threshold for "new cloud"
    const Real Lv = 2.501e6;
    const Real Cp = Constants::Cpair;
    const Real Rv = Constants::RH2O;
    using Physics = scream::physics::Functions<Real, scream::DefaultDevice>;
    const int nlev_packs = m_num_lev_packs;

    auto qv_view = m_qv_view;
    auto qc_view = m_qc_view;
    auto qr_view = m_qr_view;
    auto nc_view = m_nc_view;
    auto nr_view = m_nr_view;
    auto th_view = m_th_view;
    auto p_view  = m_pmid_dry_view; 
    auto inv_exner_view = m_inv_exner_view;
    Kokkos::parallel_for("saturation_adjustment", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs), [&](const int k_pack) {
                auto& qv_pack = qv_view(icol, k_pack);
                auto& qc_pack = qc_view(icol, k_pack);
                auto& nc_pack = nc_view(icol, k_pack);
                auto& th_pack = th_view(icol, k_pack);
                const auto& p_pack = p_view(icol, k_pack);
                const auto& inv_exner_pack = inv_exner_view(icol, k_pack);

                Spack T_pack = th_pack / inv_exner_pack;

                // Calculate Saturation
                Smask range_mask(true);
                Spack qvs_pack = Physics::qv_sat_dry(
                    T_pack, p_pack, 
                    false, // is_ice = false 
                    range_mask, 
                    Physics::Polysvp1, 
                    "VVM_Sat_Adj_Opt"
                );

                // delta_q > 0: Supersaturated
                // delta_q < 0: Subsaturated
                Spack delta_q_potential = qv_pack - qvs_pack;

                // A: delta_q > 0
                // B: delta_q < 0 && qc > 1e-12
                auto need_condense = (delta_q_potential > 0.0);
                auto need_evaporate = (delta_q_potential < 0.0) && (qc_pack > 1.0e-12);
                auto need_adjustment = need_condense || need_evaporate;

                if (need_adjustment.any()) {
                    auto is_new_cloud = need_condense && (qc_pack < min_qc);
                    
                    // Thermal Inertia Factor
                    Spack denominator = 1.0 + (Lv * Lv * qvs_pack) / (Cp * Rv * T_pack * T_pack);
                    Spack adjustment = delta_q_potential / denominator;

                    // Evaporation Limiter
                    // adjustment < 0, only evaporates qc
                    // if adjustment < -qc, adjustment = -qc
                    auto limit_mask = (adjustment < -qc_pack);
                    adjustment.set(limit_mask, -qc_pack);

                    // Update State
                    qv_pack.set(need_adjustment, qv_pack - adjustment);
                    qc_pack.set(need_adjustment, qc_pack + adjustment);
                    
                    Spack d_th = inv_exner_pack * (Lv / Cp) * adjustment;
                    th_pack.set(need_adjustment, th_pack + d_th);

                    if (is_new_cloud.any()) {
                         nc_pack.set(is_new_cloud, nccn_val);
                    }

                    // Full Evaporation
                    auto is_cloud_gone = (qc_pack < 1.0e-12); 
                    if (is_cloud_gone.any()) {
                        nc_pack.set(is_cloud_gone, 0.0);
                    }
                }
            });
        }
    );
    // Kokkos::fence();

    P3F::p3_main(
        m_runtime_options, m_prog_state, m_diag_inputs, m_diag_outputs, m_infrastructure,
        m_history_only, m_lookup_tables,
#ifdef SCREAM_P3_SMALL_KERNELS
        temporaries,
#endif
        workspace_mgr, m_num_cols, m_num_levs
    );

    Kokkos::parallel_for("post_p3_saturation_adjustment", m_policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const int icol = team.league_rank();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev_packs), [&](const int k_pack) {
                auto& qv_pack = qv_view(icol, k_pack);
                auto& qc_pack = qc_view(icol, k_pack);
                auto& nc_pack = nc_view(icol, k_pack);
                auto& th_pack = th_view(icol, k_pack);
                const auto& p_pack = p_view(icol, k_pack);
                const auto& inv_exner_pack = inv_exner_view(icol, k_pack);

                Spack qc_post_p3 = qc_pack; 

                Spack T_pack = th_pack / inv_exner_pack;

                Smask range_mask(true);
                Spack qvs_pack = Physics::qv_sat_dry(
                    T_pack, p_pack, 
                    false, 
                    range_mask, 
                    Physics::Polysvp1, 
                    "Post_P3_Sat_Adj"
                );

                Spack delta_q = qv_pack - qvs_pack;
                auto need_condense = (delta_q > 0.0);
                
                auto need_evaporate = (delta_q < 0.0) && (qc_pack > min_qc);

                auto need_adj = need_condense || need_evaporate;

                if (need_adj.any()) {
                    Spack denominator = 1.0 + (Lv * Lv * qvs_pack) / (Cp * Rv * T_pack * T_pack);
                    Spack adj = delta_q / denominator;

                    auto limit_mask = (adj < -qc_pack);
                    adj.set(limit_mask, -qc_pack);

                    qv_pack.set(need_adj, qv_pack - adj);
                    qc_pack.set(need_adj, qc_pack + adj);
                    
                    Spack d_th = inv_exner_pack * (Lv / Cp) * adj;
                    th_pack.set(need_adj, th_pack + d_th);

                    auto is_numerical_creation = (adj > 0.0) && (qc_post_p3 < min_qc);

                    if (is_numerical_creation.any()) {
                        nc_pack.set(is_numerical_creation, nccn_val);
                    }
                    auto is_cloud_gone = (qc_pack < min_qc);
                    
                    if (is_cloud_gone.any()) {
                        nc_pack.set(is_cloud_gone, 0.0);
                    }
                }
            });
        }
    );

    // Conduct the post-processing of the p3_main output.
    Kokkos::parallel_for("p3_main_local_vals",
        m_policy,
        m_p3_postproc
    ); // Kokkos::parallel_for(p3_main_local_vals)
    Kokkos::fence();
    postprocessing_and_unpacking(state);
}

void VVM_P3_Interface::finalize() {
    m_wsm_view_storage = {}; 

    m_qv_view = {};
    m_qc_view = {};
    m_qr_view = {};
    m_qi_view = {};
    m_qm_view = {};
    m_nc_view = {};
    m_nr_view = {};
    m_ni_view = {};
    m_bm_view = {};
    m_th_view = {}; 

    m_pmid_view = {};
    m_pmid_dry_view = {};
    m_pseudo_density_view = {};
    m_pseudo_density_dry_view = {};
    m_T_atm_view = {};
    m_cld_frac_t_view = {};
    m_qv_prev_view = {};
    m_t_prev_view = {};

    m_nc_nuceat_tend_view = {};
    m_nccn_view = {};
    m_ni_activated_view = {};
    m_inv_qc_relvar_view = {};

    m_dz_view = {};
    m_inv_exner_view = {};
    m_cld_frac_i_view = {};
    m_cld_frac_l_view = {};
    m_cld_frac_r_view = {};

    m_qv2qi_depos_tend_view = {};
    m_precip_liq_flux_view = {};
    m_precip_ice_flux_view = {};
    m_precip_total_tend_view = {};
    m_nevapr_view = {};
    m_rho_qi_view = {};

    m_diag_eff_radius_qc_view = {};
    m_diag_eff_radius_qi_view = {};
    m_diag_eff_radius_qr_view = {};

    m_liq_ice_exchange_view = {};
    m_vap_liq_exchange_view = {};
    m_vap_ice_exchange_view = {};
    view_2d m_diag_equiv_reflectivity_view;

    m_precip_liq_surf_flux_view = {};
    m_precip_ice_surf_flux_view = {};
    m_precip_liq_surf_mass_view = {};
    m_precip_ice_surf_mass_view = {};

    m_unused = {};
    m_dummy_input = {};

    m_col_location_view = {};
}

} // namespace Physics
} // namespace VVM
