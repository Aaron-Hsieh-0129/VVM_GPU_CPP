#ifndef VVM_CORE_FIELD_HPP
#define VVM_CORE_FIELD_HPP

#include <Kokkos_Core.hpp>
#include "Grid.hpp"        // Field depends on Grid information

namespace VVM {
namespace Core {

class Field {
public:
    // Kokkos::View to store the field data
    // The dimensions will be (total_z_points, total_y_points, total_x_points)
    // using LayoutRight for C/C++ style indexing (last dimension fastest changing)
    using DefaultDevice = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    Kokkos::View<double***, Kokkos::LayoutRight, DefaultDevice> data_;

    // Constructor: Takes a const reference to a Grid object
    explicit Field(const Grid& grid, const std::string& field_name = "UnnamedField");

    // Destructor (Kokkos::View manages its own memory, so usually empty here)
    ~Field() = default; // Or {} if you need to add custom cleanup

    // Delete copy constructor and assignment operator to prevent shallow copies
    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;

    // --- Initialization Methods (executed on device) ---

    // Initialize all field values to zero
    void initialize_to_zero();

    // Initialize field values based on global coordinates (example)
    // This example initializes u(x,y,z) = x + y + z for demonstration
    void initialize_with_coords(const Grid& grid);

    // --- Data Access / Transfer Methods ---

    // Get a const reference to the Kokkos::View (for device computations)
    KOKKOS_INLINE_FUNCTION
    const Kokkos::View<double***, Kokkos::LayoutRight, DefaultDevice> get_device_data() const {
        return data_;
    }
    
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<double***, Kokkos::LayoutRight, DefaultDevice>& get_mutable_device_data() {
        return data_;
    }

    // Get a Kokkos::View on the HostSpace (for CPU side access, involves deep_copy)
    Kokkos::View<double***, Kokkos::HostSpace> get_host_data() const;

    // --- Getters for Field Dimensions ---
    const std::string& get_name() const { return name_; }

    // --- Printing/Debugging Methods ---
    void print_field_info() const;
    void print_slice_z_at_k(const Grid& grid, int k_local_idx) const; // Print a Z-slice

private:
    std::string name_; // Name of the field for identification
    // Store grid dimensions for convenience in kernels, these are local to each rank
    int total_z_points_;
    int total_y_points_;
    int total_x_points_;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_FIELD_HPP