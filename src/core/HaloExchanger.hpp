// HaloExchange processes halo data exchange between neighboring MPI ranks
// in a distributed grid simulation. It uses Kokkos for parallel execution
// and MPI for inter-process communication. The class provides methods to
// exchange halo data in the X, Y, and Z dimensions, ensuring that each rank
// has the necessary data from its neighbors for accurate computations.

#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "State.hpp"

namespace VVM {
namespace Core {

enum class HaloExchangeTags {
    X_LEFT_TO_RIGHT, // = 0
    X_RIGHT_TO_LEFT, // = 1
    Y_BOTTOM_TO_TOP, // = 2
    Y_TOP_TO_BOTTOM  // = 3
};

class HaloExchanger {
public:
    // Constructor
    explicit HaloExchanger(const Grid& grid);

    // Main public interface
    void exchange_halos(Field& field) const;

    // overload exchange halos for State
    void exchange_halos(State& state) const;

    // --- MOVED TO PUBLIC ---
    // NOTE: These are moved to public due to a nvcc compiler limitation
    // with __host__ __device__ lambdas in private/protected member functions.
    void exchange_halo_x(Field& field) const;
    void exchange_halo_y(Field& field) const;
    void exchange_halo_z(Field& field) const;


private:
    // Member variables
    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    int neighbors_x_[2];
    int neighbors_y_[2];
    int neighbors_z_[2];
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_HALOEXCHANGER_HPP