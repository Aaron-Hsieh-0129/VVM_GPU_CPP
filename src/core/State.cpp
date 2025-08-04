#include "State.hpp"
#include "BoundaryConditionManager.hpp"

namespace VVM {
namespace Core {

State::State(const Utils::ConfigurationManager& config, const Parameters& params)
    : config_ref_(config), parameters_(params) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VVM::Core::Grid grid(config_ref_);

    // Get local points including halo cells
    int nx_total = grid.get_local_total_points_x();
    int ny_total = grid.get_local_total_points_y();
    int nz_total = grid.get_local_total_points_z();

    // 1D field
    add_field<1>("thbar", {nz_total});
    add_field<1>("rhobar", {nz_total});
    add_field<1>("rhobar_up", {nz_total});
    add_field<1>("pbar", {nz_total});
    add_field<1>("pibar", {nz_total});
    add_field<1>("U", {nz_total});

    // 2D field
    add_field<2>("htflx_sfc", {ny_total, nx_total});

    // 3D field
    add_field<3>("th", {nz_total, ny_total, nx_total});
    add_field<3>("xi", {nz_total, ny_total, nx_total});
    add_field<3>("eta", {nz_total, ny_total, nx_total});
    add_field<3>("zeta", {nz_total, ny_total, nx_total});
    add_field<3>("u", {nz_total, ny_total, nx_total});
    add_field<3>("v", {nz_total, ny_total, nx_total});
    add_field<3>("w", {nz_total, ny_total, nx_total});


    auto& u_data = get_field<3>("u").get_mutable_device_data();
    auto& v_data = get_field<3>("v").get_mutable_device_data();
    auto& w_data = get_field<3>("w").get_mutable_device_data();
    Kokkos::deep_copy(u_data, 0.0);
    Kokkos::deep_copy(v_data, 0.0);
    Kokkos::deep_copy(w_data, 10.0);

    // TODO: Add tracer auto loading from configuration file
}

// get_field is now a template in the header file.

} // namespace Core
} // namespace VVM
