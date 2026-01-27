#include "State.hpp"
#include "BoundaryConditionManager.hpp"

namespace VVM {
namespace Core {

#if defined(ENABLE_NCCL)
State::State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid, ncclComm_t nccl_comm, cudaStream_t nccl_stream)
    : config_ref_(config), parameters_(params), grid_(grid), nccl_comm_(nccl_comm), nccl_stream_(nccl_stream) {
#else
State::State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid)
    : config_ref_(config), parameters_(params), grid_(grid) {
#endif
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get local points including halo cells
    int nx_total = grid.get_local_total_points_x();
    int ny_total = grid.get_local_total_points_y();
    int nz_total = grid.get_local_total_points_z();
    // 0D field
    add_field<0>("utopmn", {});
    add_field<0>("vtopmn", {});

    // 1D field
    add_field<1>("Tbar", {nz_total});
    add_field<1>("Tvbar", {nz_total});
    add_field<1>("thbar", {nz_total});
    add_field<1>("rhobar", {nz_total});
    add_field<1>("rhobar_up", {nz_total});
    add_field<1>("pbar", {nz_total});
    add_field<1>("pbar_up", {nz_total});
    add_field<1>("dpbar_mid", {nz_total});
    add_field<1>("pibar", {nz_total});
    add_field<1>("qvbar", {nz_total});
    add_field<1>("U", {nz_total});
    add_field<1>("V", {nz_total});

    add_field<1>("lon", {nx_total});
    add_field<1>("lat", {ny_total});
    add_field<1>("f", {ny_total});

    // 2D field
    add_field<2>("psi", {ny_total, nx_total});
    add_field<2>("psinm1", {ny_total, nx_total});
    add_field<2>("chi", {ny_total, nx_total});
    add_field<2>("chinm1", {ny_total, nx_total});
    add_field<2>("utop", {ny_total, nx_total});
    add_field<2>("vtop", {ny_total, nx_total});
    add_field<2>("tempu", {ny_total, nx_total});
    add_field<2>("tempv", {ny_total, nx_total});
    add_field<2>("Tg", {ny_total, nx_total});

    // 3D field
    add_field<3>("th", {nz_total, ny_total, nx_total});
    add_field<3>("qv", {nz_total, ny_total, nx_total});
    add_field<3>("T", {nz_total, ny_total, nx_total});
    add_field<3>("T_m", {nz_total, ny_total, nx_total});
    add_field<3>("xi", {nz_total, ny_total, nx_total});
    add_field<3>("eta", {nz_total, ny_total, nx_total});
    add_field<3>("zeta", {nz_total, ny_total, nx_total});
    add_field<3>("u", {nz_total, ny_total, nx_total});
    add_field<3>("v", {nz_total, ny_total, nx_total});
    add_field<3>("w", {nz_total, ny_total, nx_total});
    add_field<3>("u_mean", {nz_total, ny_total, nx_total});
    add_field<3>("v_mean", {nz_total, ny_total, nx_total});
    add_field<3>("w_mean", {nz_total, ny_total, nx_total});
    add_field<3>("W3DNM1", {nz_total, ny_total, nx_total});
    add_field<3>("u_topo", {nz_total, ny_total, nx_total});
    add_field<3>("v_topo", {nz_total, ny_total, nx_total});
    add_field<3>("w_topo", {nz_total, ny_total, nx_total});
    add_field<3>("xi_topo", {nz_total, ny_total, nx_total});
    add_field<3>("eta_topo", {nz_total, ny_total, nx_total});

    // Rotation term
    add_field<3>("R_xi", {nz_total, ny_total, nx_total});
    add_field<3>("R_eta", {nz_total, ny_total, nx_total});
    add_field<3>("R_zeta", {nz_total, ny_total, nx_total});

    // Topography
    add_field<2>("topo", {ny_total, nx_total});
    add_field<3>("ITYPEU", {nz_total, ny_total, nx_total});
    add_field<3>("ITYPEV", {nz_total, ny_total, nx_total});
    add_field<3>("ITYPEW", {nz_total, ny_total, nx_total});

    // TODO: Add tracer auto loading from configuration file
}

} // namespace Core
} // namespace VVM
