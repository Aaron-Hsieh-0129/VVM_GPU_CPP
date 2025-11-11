add_library(scream_share STATIC
    # scream_config.cpp
    # scream_session.cpp
    util/scream_utils.cpp
)

target_include_directories(scream_share PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${EKAT_INCLUDE_DIRS}
)

target_link_libraries(scream_share PUBLIC
    ekat
    Kokkos::kokkos
)
