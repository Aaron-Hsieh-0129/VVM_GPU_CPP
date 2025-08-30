set(VVM_UTILS_SOURCES
    ConfigurationManager.cpp
    TimingManager.cpp
    Timer.cpp
)

set(VVM_UTILS_HEADERS
    ConfigurationManager.hpp
    TimingManager.hpp
    Timer.hpp
)

add_library(vvm_utils STATIC
    ${VVM_UTILS_SOURCES}
    ${VVM_UTILS_HEADERS}
)

target_link_libraries(vvm_utils
    PUBLIC
        Kokkos::kokkos
)

