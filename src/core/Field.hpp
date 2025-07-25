// Field class is a Kokkos::View-based data structure
// that represents a 3D field in the simulation grid.
// It provides methods for initialization, data access, and printing field information.

#ifndef VVM_CORE_FIELD_HPP
#define VVM_CORE_FIELD_HPP

#include <Kokkos_Core.hpp>
#include "Grid.hpp"        // Field depends on Grid information

namespace VVM {
namespace Core {

// Helper to create Kokkos::View of varying dimensions
template<size_t Dim, typename ScalarType = double>
struct ViewTypeHelper;

template<typename ScalarType> struct ViewTypeHelper<1, ScalarType> { using type = Kokkos::View<ScalarType*>; };
template<typename ScalarType> struct ViewTypeHelper<2, ScalarType> { using type = Kokkos::View<ScalarType**>; };
template<typename ScalarType> struct ViewTypeHelper<3, ScalarType> { using type = Kokkos::View<ScalarType***>; };
template<typename ScalarType> struct ViewTypeHelper<4, ScalarType> { using type = Kokkos::View<ScalarType****>; };

template<size_t Dim>
class Field {
public:
    static constexpr size_t DimValue = Dim;

    // Kokkos::View to store the field data
    // The dimensions will be (total_z_points, total_y_points, total_x_points)
    // Here we don't specify the layout, because it can be determined by Kokkos default. 
    // LayoutRight is often used for CPU and LayoutLeft for GPU.
    using ViewType = typename ViewTypeHelper<Dim>::type;
    using HostMirrorType = typename ViewType::HostMirror;

    // Constructor
    explicit Field(const std::string& field_name, const std::array<int, Dim>& dims)
        : name_(field_name) {
        
        if constexpr (Dim == 1) data_ = ViewType(name_, dims[0]);
        else if constexpr (Dim == 2) data_ = ViewType(name_, dims[0], dims[1]);
        else if constexpr (Dim == 3) data_ = ViewType(name_, dims[0], dims[1], dims[2]);
        else if constexpr (Dim == 4) data_ = ViewType(name_, dims[0], dims[1], dims[2], dims[3]);

        Kokkos::deep_copy(data_, 0.0);
        Kokkos::fence();
    }

    // Destructor (Kokkos::View manages its own memory, so usually empty here)
    ~Field() = default; // Or {} if you need to add custom cleanup

    // Delete copy constructor and assignment operator to prevent shallow copies
    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;

    // --- Initialization Methods (executed on device) ---

    // Initialize all field values to zero
    void initialize_to_zero() {
        Kokkos::deep_copy(data_, 0.0);
        Kokkos::fence();
    }

    // Get a const reference to the Kokkos::View (for device computations)
    ViewType& get_mutable_device_data() { return data_; }
    const ViewType& get_device_data() const { return data_; }

    HostMirrorType get_host_data() const {
        HostMirrorType host_data = Kokkos::create_mirror_view(data_);
        Kokkos::deep_copy(host_data, data_);
        Kokkos::fence();
        return host_data;
    }

    const std::string& get_name() const { return name_; }

    // --- Printing/Debugging Methods ---
    void print_field_info() const;
    void print_slice_z_at_k(const Grid& grid, int N_idx, int k_local_idx) const;

private:
    std::string name_; // Name of the field for identification
    ViewType data_;
};

template<size_t Dim>
inline void Field<Dim>::print_slice_z_at_k(const Grid& grid, int N_idx, int k_local_idx) const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto host_data = get_host_data();

    std::cout << "Rank " << rank << ": Field '" << name_ << "' (" << Dim << "D)" << std::endl;

    // Use if constexpr to handle different dimensions at compile time
    if constexpr (Dim == 4) {
        if (N_idx < 0 || N_idx >= host_data.extent(0) ||
            k_local_idx < 0 || k_local_idx >= host_data.extent(1)) {
            std::cerr << "Warning: Slice index (" << N_idx << ", " << k_local_idx 
                      << ") out of bounds for field '" << name_ << "'." << std::endl;
            return;
        }
        std::cout << "  Slice at N=" << N_idx << ", k=" << k_local_idx << std::endl;
        for (int j = 0; j < host_data.extent(2); ++j) {
            for (int i = 0; i < host_data.extent(3); ++i) {
                std::cout << host_data(N_idx, k_local_idx, j, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else if constexpr (Dim == 3) {
        if (k_local_idx < 0 || k_local_idx >= host_data.extent(0)) {
            std::cerr << "Warning: Z-slice index " << k_local_idx << " out of bounds for field '" << name_ << "'." << std::endl;
            return;
        }
        std::cout << "  Z-slice at k=" << k_local_idx << std::endl;
        for (int j = 0; j < host_data.extent(1); ++j) {
            for (int i = 0; i < host_data.extent(2); ++i) {
                std::cout << host_data(k_local_idx, j, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else if constexpr (Dim == 2) {
        // For a 2D field, we ignore indices and print the whole field
        std::cout << "  Full 2D data:" << std::endl;
        for (int j = 0; j < host_data.extent(0); ++j) {
            for (int i = 0; i < host_data.extent(1); ++i) {
                std::cout << host_data(j, i) << "\t";
            }
            std::cout << std::endl;
        }
    } 
    else {
        // For other dimensions, this function is not applicable
        if (rank == 0) {
            std::cout << "  Printing is not implemented for " << Dim << "D fields." << std::endl;
        }
    }
    std::cout << "--------------------------------------" << std::endl;
}


} // namespace Core
} // namespace VVM

#endif // VVM_CORE_FIELD_HPP