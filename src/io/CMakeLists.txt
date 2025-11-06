set(VVM_IO_SOURCES
    TxtReader.cpp
    OutputManager.cpp
    PnetcdfReader.cpp
)

set(VVM_IO_HEADERS
    Reader.hpp
    TxtReader.hpp
    OutputManager.hpp
    PnetcdfReader.hpp
)

add_library(vvm_io STATIC
    ${VVM_IO_SOURCES}
    ${VVM_IO_HEADERS}
)

target_link_libraries(vvm_io
    PUBLIC
        vvm_core
        Kokkos::kokkos
        adios2::adios2
        HDF5::HDF5
        PnetCDF::pnetcdf
        NetCDF::netcdf
        MPI::MPI_CXX
)
