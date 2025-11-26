#include "Grid.hpp"

namespace VVM {
namespace Core {

Grid::Grid(const VVM::Utils::ConfigurationManager& config)
    : dims_device_view_("GridDimensions", 3), // Initialize Kokkos::View for 3 dimensions (Z, Y, X)
      dims_host_mirror_("GridDimensions_Host", 3),   // Initialize host mirror
      cart_comm_(MPI_COMM_NULL)        // Initialize MPI_Comm to NULL for safety
{
    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

    // Read grid parameters from ConfigurationManager and populate the host mirror
    try {
        dims_host_mirror_(0).global_size = config.get_value<int>("grid.nz");
        dims_host_mirror_(1).global_size = config.get_value<int>("grid.ny");
        dims_host_mirror_(2).global_size = config.get_value<int>("grid.nx");

        dims_host_mirror_(0).d_coord = config.get_value<double>("grid.dz");
        dims_host_mirror_(1).d_coord = config.get_value<double>("grid.dy");
        dims_host_mirror_(2).d_coord = config.get_value<double>("grid.dx");

        if (dims_host_mirror_(0).global_size <= 0 || dims_host_mirror_(1).global_size <= 0 || dims_host_mirror_(2).global_size <= 0) {
            throw std::runtime_error("Grid dimensions must be positive integers.");
        }

        dims_host_mirror_(0).num_halo_cells = config.get_value<int>("grid.n_halo_cells");
        dims_host_mirror_(1).num_halo_cells = config.get_value<int>("grid.n_halo_cells");
        dims_host_mirror_(2).num_halo_cells = config.get_value<int>("grid.n_halo_cells");

        if (dims_host_mirror_(0).num_halo_cells < 0 || dims_host_mirror_(1).num_halo_cells < 0 || dims_host_mirror_(2).num_halo_cells < 0) {
            throw std::runtime_error("Number of halo cells cannot be negative.");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Grid initialization failed: " + std::string(e.what()));
    }

    // Copy the initialized data from host mirror to the device view
    Kokkos::deep_copy(dims_device_view_, dims_host_mirror_);
    Kokkos::fence(); // Ensure data is on device before proceeding with device-dependent operations

    // Calculate local grid distribution
    calculate_local_grid_distribution();

    if (mpi_rank_ == 0) {
        std::cout << "Grid initialized successfully." << std::endl;
    }
}

Grid::~Grid() {
    if (cart_comm_ != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_comm_);
    }
}

void Grid::calculate_local_grid_distribution() {
    // Assume Z dimension is not decomposed, each process has the full Z axis
    // Assume X and Y dimensions are decomposed

    // Z dimension (not decomposed, each process has the full Z axis)
    dims_host_mirror_(0).local_physical_size = dims_host_mirror_(0).global_size;
    dims_host_mirror_(0).local_physical_start_idx = 0;
    dims_host_mirror_(0).local_physical_end_idx = dims_host_mirror_(0).global_size - 1;

    // Decomposing the grid into a 2D topology (Y, X) using MPI Cartesian topology
    // 1. Determine process topology: Px * Py = mpi_size_
    // Try to find a process grid that is close to square
    int p_dims[2] = {1, 1}; // p_dims[0] for Y processes, p_dims[1] for X processes. p_dims represents the number of processes in each dimension
    int periods[2] = {1, 1}; // Periodic in Y and X 
                             // If boundary conditions are not periodic, this can be set to {0, 0}
    int reorder = 1;         // Allow MPI to reorder processes to optimize topology

    int ndims_cart = 2; // Default to 2D decomposition, even if we only decompose in X and Y

    // 2. Set p_dims based on global grid size
    if (dims_host_mirror_(1).global_size == 1 && dims_host_mirror_(2).global_size > 1) {
        // Only x direction has multiple points, perform 1D X decomposition
        p_dims[0] = 1; // Y direction only uses 1 process
        p_dims[1] = mpi_size_; // X direction uses all processes
        ndims_cart = 2; // Still create 2D topology, but one dimension has only 1 process

        periods[0] = 0; // No periodic b.c. along y-axis
        dims_host_mirror_(1).num_halo_cells = 0; // Make y-halo to be 0 
        if (mpi_rank_ == 0) {
            std::cout << "Detected 1D X-decomposition (ny=1, nx>1)" << std::endl;
        }
    } 
    else if (dims_host_mirror_(2).global_size == 1 && dims_host_mirror_(1).global_size > 1) {
        // Only y direction has multiple points, perform 1D Y decomposition
        p_dims[0] = mpi_size_; // Y direction uses all processes
        p_dims[1] = 1; // X direction only uses 1 process
        ndims_cart = 2; // Still create 2D topology, but one dimension has only 1 process

        periods[1] = 0; // No periodic b.c. along x-axis
        if (mpi_rank_ == 0) {
            std::cout << "Detected 1D Y-decomposition (nx=1, ny>1)" << std::endl;
        }
    } 
    else if (dims_host_mirror_(2).global_size > 1 && dims_host_mirror_(1).global_size > 1) {
        // Both X and Y directions have multiple points, perform 2D (X,Y) decomposition
        p_dims[0] = 0; // Let MPI decide the number of Y processes
        p_dims[1] = 0; // Let MPI decide the number of X processes
        MPI_Dims_create(mpi_size_, 2, p_dims); // Let MPI find the best 2D distribution
        ndims_cart = 2;
        if (mpi_rank_ == 0) {
            std::cout << "Detected 2D (X,Y) decomposition (nx>1, ny>1)" << std::endl;
        }
    } 
    else { // dims_[2].global_size == 1 && dims_[1].global_size == 1 (single point grid or full copy)
        // In this case, all processes will copy the entire microgrid
        p_dims[0] = 1;
        p_dims[1] = 1;
        if (mpi_rank_ == 0) {
            std::cout << "Detected no decomposition (nx=1, ny=1)" << std::endl;
            if (mpi_size_ > 1) {
                std::cout << "WARNING: With nx=1 and ny=1, all MPI ranks will have full copy of horizontal domain." << std::endl;
            }
        }
    }
    // If mpi_rank_ == 0, print decomposition information
    if (mpi_rank_ == 0) {
        std::cout << "MPI 2D Decomposition: Px=" << p_dims[1] << ", Py=" << p_dims[0] << std::endl;

        if (periods[1] == 1) std::cout << "Boundary Conditions: X-Periodic, ";
        else std::cout << "Boundary Conditions: X-Non Periodic, ";

        if (periods[0] == 1) std::cout << "Y-Periodic, ";
        else std::cout << "Y-Non Periodic, ";

        std::cout << "Z-NonPeriodic" << std::endl; // Added for clarity
    }

    std::cout << "Rank " << mpi_rank_ << " is ready to call MPI_Cart_create." << std::endl;
    
    // 3. Create MPI Cartesian Communicator
    MPI_Cart_create(MPI_COMM_WORLD, ndims_cart, p_dims, periods, reorder, &cart_comm_);

    std::cout << "Rank " << mpi_rank_ << " finished MPI_Cart_create." << std::endl;

    // 4. Get current process coordinates in 2D topology
    int coords[2]; // coords[0] for Y-coordinate, coords[1] for X-coordinate
    MPI_Cart_coords(cart_comm_, mpi_rank_, 2, coords);

    // 5. Calculate local grid range based on process coordinates
    // --- Y dimension decomposition (dims_[1]) ---
    if (p_dims[0] == 1) { // Only one process in Y direction, so no decomposition
        dims_host_mirror_(1).local_physical_size = dims_host_mirror_(1).global_size;
        dims_host_mirror_(1).local_physical_start_idx = 0;
        dims_host_mirror_(1).local_physical_end_idx = dims_host_mirror_(1).global_size - 1;
    } 
    else { // Multiple processes in Y direction, proceed with decomposition
        int base_local_N_y = dims_host_mirror_(1).global_size / p_dims[0];
        int remainder_y = dims_host_mirror_(1).global_size % p_dims[0];

        dims_host_mirror_(1).local_physical_start_idx = coords[0] * base_local_N_y + std::min(coords[0], remainder_y);
        dims_host_mirror_(1).local_physical_size = base_local_N_y + (coords[0] < remainder_y ? 1 : 0);
        dims_host_mirror_(1).local_physical_end_idx = dims_host_mirror_(1).local_physical_start_idx + dims_host_mirror_(1).local_physical_size - 1;
    }


    // --- X dimension decomposition (dims_[2]) ---
    if (dims_host_mirror_(2).global_size == 1) {
        dims_host_mirror_(2).local_physical_size = dims_host_mirror_(2).global_size;
        dims_host_mirror_(2).local_physical_start_idx = 0;
        dims_host_mirror_(2).local_physical_end_idx = dims_host_mirror_(2).global_size - 1;
    } 
    else { 
        int base_local_N_x = dims_host_mirror_(2).global_size / p_dims[1];
        int remainder_x = dims_host_mirror_(2).global_size % p_dims[1];

        dims_host_mirror_(2).local_physical_start_idx = coords[1] * base_local_N_x + std::min(coords[1], remainder_x);
        dims_host_mirror_(2).local_physical_size = base_local_N_x + (coords[1] < remainder_x ? 1 : 0);
        dims_host_mirror_(2).local_physical_end_idx = dims_host_mirror_(2).local_physical_start_idx + dims_host_mirror_(2).local_physical_size - 1;
    }

    // Copy the updated data from host mirror back to the device view
    Kokkos::deep_copy(dims_device_view_, dims_host_mirror_);
    Kokkos::fence(); // Ensure data is on device after modification
    
    if (mpi_rank_ == 0) {
        std::cout << "Grid initialized successfully with 2D decomposition." << std::endl;
    }
}


void Grid::print_info() const {
    // Get current process coordinates in the 2D topology
    // Use the stored cart_comm_ member, not create a temporary one, for consistency
    int p_dims_retrieved[2]; // To retrieve actual dimensions from the comm
    int periods_retrieved[2]; // To retrieve actual periods from the comm
    int coords[2];

    // Check if cart_comm_ is valid before using it
    if (cart_comm_ != MPI_COMM_NULL) {
        MPI_Cart_coords(cart_comm_, mpi_rank_, 2, coords);
        MPI_Cart_get(cart_comm_, 2, p_dims_retrieved, periods_retrieved, coords);
    } else {
        // Handle case where communicator is NULL (e.g., in a single-process run or error)
        coords[0] = -1; coords[1] = -1; // Indicate invalid coords
        p_dims_retrieved[0] = 0; p_dims_retrieved[1] = 0;
        periods_retrieved[0] = 0; periods_retrieved[1] = 0;
    }

    std::cout << "MPI Rank " << mpi_rank_ << " (Coords: Y=" << coords[0] << ", X=" << coords[1] << ") Local Grid Info:" << std::endl;
    std::cout << "  Periodic (Y,X): (" << (periods_retrieved[0] ? "Yes" : "No") << ", " << (periods_retrieved[1] ? "Yes" : "No") << ")" << std::endl;
    std::cout << "  Z: Global Start=" << dims_host_mirror_(0).local_physical_start_idx
              << ", Global End=" << dims_host_mirror_(0).local_physical_end_idx
              << ", Physical Size=" << dims_host_mirror_(0).local_physical_size
              << ", Total Size (incl. Halo)=" << (dims_host_mirror_(0).local_physical_size + 2 * dims_host_mirror_(0).num_halo_cells) << std::endl;
    std::cout << "  Y: Global Start=" << dims_host_mirror_(1).local_physical_start_idx
              << ", Global End=" << dims_host_mirror_(1).local_physical_end_idx
              << ", Physical Size=" << dims_host_mirror_(1).local_physical_size
              << ", Total Size (incl. Halo)=" << (dims_host_mirror_(1).local_physical_size + 2 * dims_host_mirror_(1).num_halo_cells) << std::endl;
    std::cout << "  X: Global Start=" << dims_host_mirror_(2).local_physical_start_idx
              << ", Global End=" << dims_host_mirror_(2).local_physical_end_idx
              << ", Physical Size=" << dims_host_mirror_(2).local_physical_size
              << ", Total Size (incl. Halo)=" << (dims_host_mirror_(2).local_physical_size + 2 * dims_host_mirror_(2).num_halo_cells) << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

} // namespace Core
} // namespace VVM
