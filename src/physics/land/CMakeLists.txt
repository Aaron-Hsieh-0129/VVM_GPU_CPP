set(LAND_SRCS
    mod_rank.f90
    mod_index.f90
    machine.f90
    mod_param.f90
    mod_const.f90
    physcons.f90
    namelist_soilveg.f90
    set_soilveg.f90
    fpvs_gpu.f90
    sfc_diff_gpu.f90
    sfc_ocean_gpu.f90
    sfc_sice_gpu.f90
    sfc_diag_gpu.f90
    sflx_gpu.f90
    sfc_drv_gpu.f90
    vvm_land_interface.f90
)

set(LAND_CXX_SRCS
    LandProcess.cpp
)

add_library(vvm_land SHARED ${LAND_SRCS} ${LAND_CXX_SRCS})

set_target_properties(vvm_land PROPERTIES LINKER_LANGUAGE Fortran)

set_target_properties(vvm_land PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options(vvm_land PRIVATE -fast -acc)

    target_compile_options(vvm_land PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-Mallocatable=03;-Mfreeform;-Mextend;-Mpreprocess;-r8>
    )
    
    target_link_options(vvm_land PRIVATE -acc -fortranlibs -c++libs)
endif()

target_link_libraries(vvm_land PUBLIC Kokkos::kokkos vvm_core)
target_link_libraries(eamxx_physics INTERFACE vvm_land)
