#include "State.hpp"

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
    add_field<1>("z_mid", {nz_total});
    add_field<1>("z_up", {nz_total});
    add_field<1>("flex_height_coef_mid", {nz_total});
    add_field<1>("flex_height_coef_up", {nz_total});
    add_field<1>("dz_mid", {nz_total});
    add_field<1>("dz_up", {nz_total});
    add_field<1>("thbar", {nz_total});

    // 2D field
    add_field<2>("htflx_sfc", {ny_total, nx_total});


    // 3D field
    add_field<3>("etam", {nz_total, ny_total, nx_total});
    add_field<3>("etap", {nz_total, ny_total, nx_total});
    add_field<3>("thm", {nz_total, ny_total, nx_total});
    add_field<3>("thp", {nz_total, ny_total, nx_total});
    add_field<3>("u", {nz_total, ny_total, nx_total});
    add_field<3>("w", {nz_total, ny_total, nx_total});


    // 4D field
    add_field<4>("d_eta", {2, nz_total, ny_total, nx_total});
    add_field<4>("d_th", {2, nz_total, ny_total, nx_total});
}

// get_field is now a template in the header file.

} // namespace Core
} // namespace VVM
