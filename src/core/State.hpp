// State class integrates various fields in the simulation.

#ifndef VVM_CORE_VVMSTATE_HPP
#define VVM_CORE_VVMSTATE_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "utils/ConfigurationManager.hpp"
#include "Parameters.hpp"
#include <map>
#include <string>
#include <memory>
#include <variant>

namespace VVM { namespace Dynamics { class AdamsBashforth2; } }

namespace VVM {
namespace Core {

// A variant that can hold a field of any supported dimension
using AnyField = std::variant<
    std::monostate, // default state
    Field<1>,
    Field<2>,
    Field<3>,
    Field<4>
>;

class State {
    friend class VVM::Dynamics::AdamsBashforth2;
public:
    // Constructor
    State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid);

    template<size_t Dim>
    void add_field(const std::string& name, std::initializer_list<int> dims_list) {
        if (dims_list.size() != Dim) {
            throw std::runtime_error("Dimension mismatch for field '" + name + "'");
        }
        std::array<int, Dim> dims;
        std::copy(dims_list.begin(), dims_list.end(), dims.begin());
        fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims);
    }

    template<size_t Dim>
    void add_field(const std::string& name, const std::array<int, Dim>& dims) {
        fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims);
    }

    // Get a field by name
    template<size_t Dim>
    Field<Dim>& get_field(const std::string& name) {
        try { 
            return std::get<Field<Dim>>(fields_.at(name));
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Field '" + name + "' not found in State.");
        }
        catch (const std::bad_variant_access& e) {
            throw std::runtime_error("Field '" + name + "' has incorrect dimension.");
        }
    }
    
    template<size_t Dim>
    const Field<Dim>& get_field(const std::string& name) const {
        try { 
            return std::get<Field<Dim>>(fields_.at(name));
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Field '" + name + "' not found in State.");
        }
        catch (const std::bad_variant_access& e) {
            throw std::runtime_error("Field '" + name + "' has incorrect dimension.");
        }
    }

    template<size_t Dim>
    double calculate_horizontal_mean(const Field<Dim>& field, int k_level = -1) const {
        auto view = field.get_device_data();

        const int ny_local = grid_.get_local_physical_points_y();
        const int nx_local = grid_.get_local_physical_points_x();
        const int h = grid_.get_halo_cells();
        const int gnx = grid_.get_global_points_x();
        const int gny = grid_.get_global_points_y();
        const double total_points_horizontal = static_cast<double>(gnx * gny);

        if (total_points_horizontal == 0) {
            return 0.0;
        }

        double local_sum = 0.0;

        if constexpr (Dim == 3) {
            if (k_level < h || k_level >= grid_.get_local_total_points_z() - h) {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                if (rank == 0) {
                    std::cerr << "Warning: k_level " << k_level << " is in the halo region for the 3D field '" << field.get_name() << "'. Returning 0." << std::endl;
                }
                return 0.0;
            }

            Kokkos::parallel_reduce("calculate_3d_local_sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny_local + h, nx_local + h}),
                KOKKOS_LAMBDA(const int j, const int i, double& update_sum) {
                    update_sum += view(k_level, j, i);
                }, local_sum);

        } 
        else if constexpr (Dim == 2) {
            Kokkos::parallel_reduce("calculate_2d_local_sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny_local + h, nx_local + h}),
                KOKKOS_LAMBDA(const int j, const int i, double& update_sum) {
                    update_sum += view(j, i);
                }, local_sum);
        } 
        else {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                std::cerr << "Warning: calculate_horizontal_mean does not support " << Dim << "D fields. Returning 0." << std::endl;
            }
            return 0.0;
        }

        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return global_sum / total_points_horizontal;
    }

    // Provide iterators to loop over all fields
    auto begin() { return fields_.begin(); } // First value
    auto end() { return fields_.end(); } // Last value
    auto begin() const { return fields_.cbegin(); } // First key
    auto end() const { return fields_.cend(); } // Last key

private:
    const Utils::ConfigurationManager& config_ref_;
    const Grid& grid_;
    const Parameters& parameters_;
    std::map<std::string, AnyField> fields_;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_VVMSTATE_HPP
