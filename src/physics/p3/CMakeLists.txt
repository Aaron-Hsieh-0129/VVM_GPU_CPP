option(SCREAM_P3_SMALL_KERNELS "Enable small kernels for P3 microphysics to reduce memory usage" OFF)
option(SCREAM_DOUBLE_PRECISION "Use double precision (ON) or single (OFF)" ON)
option(EAMXX_ENABLE_GPU "Enable GPU kernels" ON)

set(SCREAM_PACK_SIZE 1 CACHE STRING "Size of SIMD packs")
set(SCREAM_SMALL_PACK_SIZE 1 CACHE STRING "Size of small SIMD packs")

set(P3_SRCS
  # eamxx_p3_process_interface.cpp
  # eamxx_p3_run.cpp
  VVM_p3_process_interface.cpp
)

# Add ETI source files if not on CUDA/HIP
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
    eti/p3_init.cpp
    eti/p3_update_prognostics.cpp
    eti/p3_get_time_space_phys_variables.cpp
    eti/p3_autoconversion.cpp
    eti/p3_ice_deposition_sublimation.cpp
    eti/p3_conservation.cpp
    eti/p3_rain_self_collection.cpp
    eti/p3_back_to_cell_average.cpp
    eti/p3_cloud_rain_acc.cpp
    eti/p3_calc_rime_density.cpp
    eti/p3_ice_classical_nucleation_impl.cpp
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
endif()

target_compile_definitions(p3 PUBLIC 
    EAMXX_HAS_P3
    $<$<BOOL:${SCREAM_DOUBLE_PRECISION}>:SCREAM_DOUBLE_PRECISION>
    $<$<BOOL:${EAMXX_ENABLE_GPU}>:EAMXX_ENABLE_GPU>
    $<$<BOOL:${SCREAM_P3_SMALL_KERNELS}>:SCREAM_P3_SMALL_KERNELS>
    SCREAM_PACK_SIZE=${SCREAM_PACK_SIZE}
    SCREAM_SMALL_PACK_SIZE=${SCREAM_SMALL_PACK_SIZE}
    SCREAM_POSSIBLY_NO_PACK_SIZE=1
)

target_compile_definitions(p3 PUBLIC EAMXX_HAS_P3)

set(SCREAM_BASE_DIR "/work/aaron900129/E3SM_master/components/eamxx")
foreach (P3_LIB IN LISTS P3_LIBS)
  target_include_directories(${P3_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${SCREAM_BASE_DIR}/../eam/src/physics/cam
    ${NCCL_INCLUDE_DIR}
  )
  target_link_libraries(${P3_LIB} eamxx_physics_share scream_share)
endforeach()

# Ensure tables are present in the data dir
set (PRECISION_SUFFIX "8")

target_link_libraries(eamxx_physics INTERFACE p3)
