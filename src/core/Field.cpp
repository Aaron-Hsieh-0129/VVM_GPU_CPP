#include "Field.hpp"
#include <iostream>
#include <stdexcept> // For std::runtime_error

namespace VVM {
namespace Core {

Field::Field(const Grid& grid, const std::string& field_name)
    : name_(field_name)
{
    // Retrieve total local points from the Grid object, including halo cells
    // These values are already accessible on the device because Grid uses Kokkos::View
    total_z_points_ = grid.get_local_total_points_z();
    total_y_points_ = grid.get_local_total_points_y();
    total_x_points_ = grid.get_local_total_points_x();

    // Create the Kokkos::View with the determined dimensions
    // The label is useful for profiling and debugging

    data_ = Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::device_type>(
        name_, total_z_points_, total_y_points_, total_x_points_
    );

    // Perform an initial deep_copy to zero the memory. This is good practice.
    Kokkos::deep_copy(data_, 0.0);
    Kokkos::fence(); // Ensure initialization is complete before any other operations

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank " << rank << ": Field '" << name_ << "' constructed with dimensions Z="
              << total_z_points_ << ", Y=" << total_y_points_ << ", X=" << total_x_points_ << std::endl;

}

void Field::initialize_to_zero() {
    // Kokkos::deep_copy is an efficient way to set all elements to a scalar value
    Kokkos::deep_copy(data_, 0.0);
    Kokkos::fence(); // Synchronize after device operation
    // std::cout << "Field '" << name_ << "' initialized to zero." << std::endl;
}

void Field::initialize_with_coords(const Grid& grid) {
    // Capture necessary variables for the KOKKOS_LAMBDA
    // We pass `grid` by const reference to the lambda.
    // However, if `grid` itself contains a Kokkos::View (which it does, `dims_view_`),
    // the lambda needs to be able to access it. `[=]` or `[this, &grid]` usually works.
    // For `get_z_coord` etc., we need access to `grid.dims_view_`.
    // The KOKKOS_INLINE_FUNCTION getters of Grid make this easy.

    auto field_data = data_;

    Kokkos::parallel_for(name_ + "_initialize_with_coords",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {total_z_points_, total_y_points_, total_x_points_}),
        KOKKOS_LAMBDA(const int k_local, const int j_local, const int i_local) {
            field_data(k_local, j_local, i_local) = i_local + j_local + k_local;
        }
    );
    Kokkos::fence(); // Synchronize after device operation
    std::cout << "Field '" << name_ << "' initialized with coordinates." << std::endl;
}

Kokkos::View<double***, Kokkos::HostSpace> Field::get_host_data() const {
    // Correct way to get a host mirror from a const device View
    // create_mirror_view can take a const View and return a non-const HostMirror
    // The HostMirror will be allocated to match the dimensions of the device View.
    Kokkos::View<double***, Kokkos::HostSpace> host_data = Kokkos::create_mirror_view(data_);

    // Deep copy contents from device View to host View
    Kokkos::deep_copy(host_data, data_);
    Kokkos::fence(); // Ensure data is on host before returning
    return host_data;
}

void Field::update_device_from_host(Kokkos::View<const double***, Kokkos::LayoutRight, Kokkos::HostSpace> host_view) {
    Kokkos::deep_copy(data_, host_view);
    Kokkos::fence();
}

void Field::print_field_info() const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank " << rank << ": Field '" << name_ << "' info:" << std::endl;
    std::cout << "  Dimensions: Z=" << total_z_points_
              << ", Y=" << total_y_points_
              << ", X=" << total_x_points_ << std::endl;
    std::cout << "  Total points (local): " << total_z_points_ * total_y_points_ * total_x_points_ << std::endl;
    std::cout << "  Kokkos::View capacity: " << data_.size() << " elements." << std::endl;

    // Get a host copy to inspect values
    Kokkos::View<double***, Kokkos::HostSpace> host_data = get_host_data();

    // Print a few sample values (e.g., corner values)
    if (total_z_points_ > 0 && total_y_points_ > 0 && total_x_points_ > 0) {
        std::cout << "  Sample values:" << std::endl;
        std::cout << "    data_(0, 0, 0) = " << host_data(0, 0, 0) << std::endl;
        std::cout << "    data_(" << total_z_points_-1 << ", "
                  << total_y_points_-1 << ", "
                  << total_x_points_-1 << ") = "
                  << host_data(total_z_points_-1, total_y_points_-1, total_x_points_-1) << std::endl;
    }
}

void Field::print_slice_z_at_k(const Grid& grid, int k_local_idx) const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (k_local_idx < 0 || k_local_idx >= total_z_points_) {
        std::cerr << "Rank " << rank << ": Warning: Z-slice index " << k_local_idx << " out of bounds for field '" << name_ << "'." << std::endl;
        return;
    }

    Kokkos::View<double***, Kokkos::HostSpace> host_data = get_host_data();

    std::cout << "Rank " << rank << ": Field '" << name_ << "' Z-slice at local k_idx = " << k_local_idx << std::endl;

    // Print values for the slice (Y vs X)
    for (int j = 0; j < total_y_points_; ++j) {
        for (int i = 0; i < total_x_points_; ++i) {
            std::cout << host_data(k_local_idx, j, i) << "\t";
        }
        std::cout << std::endl; // New line after each row (Y-axis)
    }
    std::cout << "--------------------------------------" << std::endl;
}


} // namespace Core
} // namespace VVM