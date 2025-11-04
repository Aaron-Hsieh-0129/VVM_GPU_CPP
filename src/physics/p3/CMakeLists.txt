# add_library(vvm_p3_eamxx STATIC
set(P3_SRCS
    p3_f90.cpp
    p3_ic_cases.cpp
    # p3_iso_c.f90
    # "/work/aaron900129/E3SM/components/eam/src/physics/p3/scream/micro_p3.F90"
    # p3_functions_f90.cpp
    ../share/physics_saturation.cpp
    ../share/physics_test_data.cpp
    # ../share/physics_share_f2c.F90
    # physics_utils.F90
    # shr_kind_mod.F90
    # micro_p3_utils.F90
    # scream_types.F90
    # wv_sat_scream.F90
    # scream_abortutils.F90
    # debug_info.F90
)

if (NOT SCREAM_LIB_ONLY)
  list(APPEND P3_SRCS
    p3_functions_f90.cpp
    p3_main_wrap.cpp
    ) # Add f90 bridges needed for testing
endif()

if (NOT EAMXX_ENABLE_GPU OR Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OR Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
  list(APPEND P3_SRCS
    eti/p3_upwind.cpp
    eti/p3_cloud_sed.cpp
    eti/p3_ice_sed.cpp
    eti/p3_ice_collection.cpp
    eti/p3_ice_melting.cpp
    eti/p3_rain_sed.cpp
    eti/p3_table3.cpp
    eti/p3_table_ice.cpp
    eti/p3_dsd2.cpp
    eti/p3_find.cpp
    eti/p3_update_prognostics.cpp
    eti/p3_get_time_space_phys_variables.cpp
    eti/p3_autoconversion.cpp
    eti/p3_ice_deposition_sublimation.cpp
    eti/p3_conservation.cpp
    eti/p3_rain_self_collection.cpp
    eti/p3_back_to_cell_average.cpp
    eti/p3_cloud_rain_acc.cpp
    eti/p3_calc_rime_density.cpp
    eti/p3_cldliq_imm_freezing.cpp
    eti/p3_rain_imm_freezing.cpp
    eti/p3_droplet_self_coll.cpp
    eti/p3_evaporate_rain.cpp
    eti/p3_impose_max_total_ni.cpp
    eti/p3_calc_liq_relaxation_timescale.cpp
    eti/p3_ice_relaxation_timescale.cpp
    eti/p3_ice_nucleation.cpp
    eti/p3_ice_cldliq_wet_growth.cpp
    eti/p3_check_values.cpp
    eti/p3_incloud_mixingratios.cpp
    eti/p3_subgrid_variance_scaling.cpp
    eti/p3_main.cpp
    eti/p3_main_part1.cpp
    eti/p3_main_part2.cpp
    eti/p3_main_part3.cpp
    eti/p3_ice_supersat_conservation.cpp
    eti/p3_nc_conservation.cpp
    eti/p3_nr_conservation.cpp
    eti/p3_ni_conservation.cpp
    eti/p3_prevent_liq_supersaturation.cpp
  ) # P3 ETI SRCS
endif()

# List of dispatch source files if monolithic kernels are off
set(P3_SK_SRCS
    disp/p3_check_values_impl_disp.cpp
    disp/p3_ice_sed_impl_disp.cpp
    disp/p3_main_impl_part1_disp.cpp
    disp/p3_main_impl_part3_disp.cpp
    disp/p3_cloud_sed_impl_disp.cpp
    disp/p3_main_impl_disp.cpp
    disp/p3_main_impl_part2_disp.cpp
    disp/p3_rain_sed_impl_disp.cpp
)

set(P3_LIBS "p3")
if (SCREAM_P3_SMALL_KERNELS)
  add_library(p3 ${P3_SRCS} ${P3_SK_SRCS})
else()
  add_library(p3 ${P3_SRCS})
  if (NOT SCREAM_LIBS_ONLY AND NOT SCREAM_ONLY_GENERATE_BASELINES)
    add_library(p3_sk ${P3_SRCS} ${P3_SK_SRCS})
    # Always build p3_sk with SCREAM_P3_SMALL_KERNELS on
    target_compile_definitions(p3_sk PUBLIC "SCREAM_P3_SMALL_KERNELS")
    list(APPEND P3_LIBS "p3_sk")
  endif()
endif()


foreach (P3_LIB IN LISTS P3_LIBS)
  set_target_properties(${P3_LIB} PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${P3_LIB}/modules
  )
  target_compile_definitions(${P3_LIB} PUBLIC EAMXX_ENABLE_GPU SCREAM_DOUBLE_PRECISION)

  target_include_directories(${P3_LIB} PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/${P3_LIB}/modules
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    /work/aaron900129/E3SM/components/eam/src/physics/cam
    "${PROJECT_SOURCE_DIR}/externals/ekat/src/ekat"
    "${PROJECT_SOURCE_DIR}/externals/ekat/src/ekat/mpi"
    "${PROJECT_SOURCE_DIR}/externals/ekat/src/ekat/util"
    "${PROJECT_SOURCE_DIR}/externals/ekat/src/ekat/logging"

    "/work/aaron900129/spdlog_install/include"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/../share
    ${CMAKE_CURRENT_SOURCE_DIR}/../../share
    ${CMAKE_CURRENT_SOURCE_DIR}/../../share/util
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
  )
  target_link_libraries(${P3_LIB} Kokkos::kokkos ekat)
endforeach()

set(P3_TABLES
    ${PROJECT_SOURCE_DIR}/rundata/p3
    ${PROJECT_SOURCE_DIR}/tables/p3_lookup_table_1.dat-v4.1.1
    ${PROJECT_SOURCE_DIR}/tables/mu_r_table_vals.dat8
    ${PROJECT_SOURCE_DIR}/tables/revap_table_vals.dat8
    ${PROJECT_SOURCE_DIR}/tables/vm_table_vals.dat8
    ${PROJECT_SOURCE_DIR}/tables/vn_table_vals.dat8
)


# This executable can be used to re-generate tables in ${SCREAM_DATA_DIR}
add_executable(p3_tables_setup EXCLUDE_FROM_ALL p3_tables_setup.cpp)
target_link_libraries(p3_tables_setup p3)

#crusher change
if (Kokkos_ENABLE_HIP)
    set_source_files_properties(p3_functions_f90.cpp  PROPERTIES COMPILE_FLAGS -O0)
endif()

