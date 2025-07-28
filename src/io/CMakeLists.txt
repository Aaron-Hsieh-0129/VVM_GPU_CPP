set(VVM_IO_SOURCES
    TxtReader.cpp
    OutputManager.cpp
)

set(VVM_IO_HEADERS
    Reader.hpp
    TxtReader.hpp
    OutputManager.hpp
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
)
