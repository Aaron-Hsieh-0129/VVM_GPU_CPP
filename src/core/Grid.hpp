// Grid class defines the grid and domain decomposition for the simulation.
// Kokkos is used for parallel execution and MPI for inter-process communication.
// The class provides methods to access grid dimensions, local physical points,
// and halo cells, ensuring efficient data exchange between neighboring ranks.

#ifndef VVM_CORE_GRID_HPP
#define VVM_CORE_GRID_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Core {

// Define a structure to hold grid dimensions and related information
struct GridDimension {
    int global_size;    // Global grid size (total number of grid points)
    double d_coord;     // Grid spacing

    // MPI-related local fields
    int local_physical_start_idx; // Local physical region start index in global
    int local_physical_end_idx;   // Local physical region end index in global
    int local_physical_size;      // Local physical region grid point count
    int num_halo_cells;           // Halo cell thickness in each direction
};

class Grid {
public:
    // Constructor: Initialize grid dimensions based on configuration
    Grid(const VVM::Utils::ConfigurationManager& config);
    ~Grid();

    // Explicitly delete copy constructor and copy assignment operator
    // This prevents accidental copying of MPI communicators and Kokkos Views.
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    // print grid info (for debugging)
    void print_info() const;

    // Getters for grid information.
    // These are now regular host functions that access dims_host_mirror_.
    int get_local_total_points_z() const { return dims_host_mirror_(0).local_physical_size + 2 * dims_host_mirror_(0).num_halo_cells; }
    int get_local_total_points_y() const { return dims_host_mirror_(1).local_physical_size + 2 * dims_host_mirror_(1).num_halo_cells; }
    int get_local_total_points_x() const { return dims_host_mirror_(2).local_physical_size + 2 * dims_host_mirror_(2).num_halo_cells; }

    int get_local_physical_points_z() const { return dims_host_mirror_(0).local_physical_size; }
    int get_local_physical_points_y() const { return dims_host_mirror_(1).local_physical_size; }
    int get_local_physical_points_x() const { return dims_host_mirror_(2).local_physical_size; }

    int get_global_points_z() const { return dims_host_mirror_(0).global_size; }
    int get_global_points_y() const { return dims_host_mirror_(1).global_size; }
    int get_global_points_x() const { return dims_host_mirror_(2).global_size; }

    double get_dz() const { return dims_host_mirror_(0).d_coord; }
    double get_dy() const { return dims_host_mirror_(1).d_coord; }
    double get_dx() const { return dims_host_mirror_(2).d_coord; }

    int get_local_physical_start_z() const { return dims_host_mirror_(0).local_physical_start_idx; }
    int get_local_physical_end_z() const { return dims_host_mirror_(0).local_physical_end_idx; }
    int get_local_physical_start_y() const { return dims_host_mirror_(1).local_physical_start_idx; }
    int get_local_physical_end_y() const { return dims_host_mirror_(1).local_physical_end_idx; }
    int get_local_physical_start_x() const { return dims_host_mirror_(2).local_physical_start_idx; }
    int get_local_physical_end_x() const { return dims_host_mirror_(2).local_physical_end_idx; }

    int get_halo_cells() const { return dims_host_mirror_(0).num_halo_cells; }

    // MPI info
    int get_mpi_rank() const { return mpi_rank_; }
    int get_mpi_size() const { return mpi_size_; }
    KOKKOS_INLINE_FUNCTION
    MPI_Comm get_cart_comm() const { return cart_comm_; }

private:
    // Changed from std::vector to Kokkos::View to store grid dimensions.
    // This allows device access for KOKKOS_INLINE_FUNCTION getters.
    Kokkos::View<GridDimension*, Kokkos::DefaultExecutionSpace> dims_device_view_; // Stores dimensions for each axis (Z, Y, X)
    Kokkos::View<GridDimension*, Kokkos::HostSpace> dims_host_mirror_;

    int mpi_rank_;                      // Rank of the current MPI process
    int mpi_size_;                      // Total number of processes in MPI communicator
    MPI_Comm cart_comm_;                // MPI Cartesian communicator for halo exchange

    // Private helper function: Calculate local grid distribution based on global grid size and MPI process count
    void calculate_local_grid_distribution();
};

}
}


#endif // VVM_CORE_GRID_HPP
