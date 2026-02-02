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
#include <cuda_runtime.h>
#if defined(ENABLE_NCCL)
    #include <nccl.h>
#endif

namespace VVM { namespace Dynamics { class AdamsBashforth2; } }

namespace VVM {
namespace Core {

// A variant that can hold a field of any supported dimension
using AnyField = std::variant<
    std::monostate, // default state
    Field<0>,
    Field<1>,
    Field<2>,
    Field<3>,
    Field<4>
>;

class State {
    friend class VVM::Dynamics::AdamsBashforth2;
public:
    // Constructor
#if defined(ENABLE_NCCL)
    State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid, ncclComm_t nccl_comm,
          cudaStream_t nccl_stream);
#else
    State(const Utils::ConfigurationManager& config, const Parameters& params, const Grid& grid);
#endif

    template<size_t Dim>
    void add_field(const std::string& name, std::initializer_list<int> dims_list) {
        if (dims_list.size() != Dim) {
            throw std::runtime_error("Dimension mismatch for field '" + name + "'");
        }
        std::array<int, Dim> dims;
        std::copy(dims_list.begin(), dims_list.end(), dims.begin());
        auto [it, inserted] = fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims);
        if (inserted) std::get<Field<Dim>>(it->second).set_to_zero();
    }

    template<size_t Dim>
    void add_field(const std::string& name, const std::array<int, Dim>& dims) {
        auto [it, inserted] = fields_.try_emplace(name, std::in_place_type_t<Field<Dim>>(), name, dims);
        if (inserted) std::get<Field<Dim>>(it->second).set_to_zero();
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


#if defined(ENABLE_NCCL)
    template<size_t Dim>
    void calculate_horizontal_mean(
        const Field<Dim>& field, 
        Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> d_mean_result, 
        int k_level = -1) const 
    {
        auto view = field.get_device_data();

        const int ny_local = grid_.get_local_physical_points_y();
        const int nx_local = grid_.get_local_physical_points_x();
        const int h = grid_.get_halo_cells();
        const int gnx = grid_.get_global_points_x();
        const int gny = grid_.get_global_points_y();
        const double total_points_horizontal = static_cast<double>(gnx * gny);

        const int nz = grid_.get_local_total_points_z();
        if (k_level == -1) k_level = nz-h-1;


        if (total_points_horizontal == 0.0) {
            Kokkos::parallel_for("set_zero_mean", 
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
                KOKKOS_LAMBDA(const int) {
                    d_mean_result() = 0.0;
                });
            Kokkos::fence();
            return;
        }

        Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> d_local_sum("local_sum");

        if constexpr (Dim == 3) {
            if (k_level < h || k_level >= grid_.get_local_total_points_z() - h) {
                Kokkos::parallel_for("set_halo_zero", 
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
                    KOKKOS_LAMBDA(const int) {
                        d_local_sum() = 0.0;
                    });
            } 
            else {
                Kokkos::parallel_reduce("calculate_3d_local_sum",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny_local + h, nx_local + h}),
                    KOKKOS_LAMBDA(const int j, const int i, double& update_sum) {
                        update_sum += view(k_level, j, i);
                    }, d_local_sum);
            }
        } 
        else if constexpr (Dim == 2) {
            Kokkos::parallel_reduce("calculate_2d_local_sum",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny_local + h, nx_local + h}),
                KOKKOS_LAMBDA(const int j, const int i, double& update_sum) {
                    update_sum += view(j, i);
                }, d_local_sum);
        } 
        else {
             Kokkos::parallel_for("set_unsupported_zero", 
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
                KOKKOS_LAMBDA(const int) {
                    d_local_sum() = 0.0;
                });
        }

        Kokkos::fence();

        ncclResult_t result = ncclAllReduce(
            d_local_sum.data(), 
            d_mean_result.data(), 
            1, 
            ncclDouble, 
            ncclSum, 
            nccl_comm_, 
            nccl_stream_
        );

        if (result != ncclSuccess) {
            printf("NCCL Error: %s\n", ncclGetErrorString(result));
        }

        cudaStreamSynchronize(nccl_stream_);

        Kokkos::parallel_for("scale_global_mean",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                d_mean_result() /= total_points_horizontal;
            });
    }
#endif

    template<size_t Dim>
    Kokkos::View<double> calculate_horizontal_mean(const Field<Dim>& field, int k_level = -1) const {
        Kokkos::View<double> ans("ans");
        Kokkos::deep_copy(ans, 0);
        auto view = field.get_device_data();

        const int ny_local = grid_.get_local_physical_points_y();
        const int nx_local = grid_.get_local_physical_points_x();
        const int h = grid_.get_halo_cells();
        const int gnx = grid_.get_global_points_x();
        const int gny = grid_.get_global_points_y();
        const double total_points_horizontal = static_cast<double>(gnx * gny);

        if (total_points_horizontal == 0) {
            return ans;
        }

        double local_sum = 0.0;

        if constexpr (Dim == 3) {
            if (k_level < h || k_level >= grid_.get_local_total_points_z() - h) {
                int rank;
                MPI_Comm_rank(grid_.get_comm(), &rank);
                if (rank == 0) {
                    std::cerr << "Warning: k_level " << k_level << " is in the halo region for the 3D field '" << field.get_name() << "'. Returning 0." << std::endl;
                }
                return ans;
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
            MPI_Comm_rank(grid_.get_comm(), &rank);
            if (rank == 0) {
                std::cerr << "Warning: calculate_horizontal_mean does not support " << Dim << "D fields. Returning 0." << std::endl;
            }
            return ans;
        }

        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, grid_.get_comm());

        Kokkos::deep_copy(ans, global_sum / total_points_horizontal);

        return ans;
    }

    // Provide iterators to loop over all fields
    auto begin() { return fields_.begin(); } // First value
    auto end() { return fields_.end(); } // Last value
    auto begin() const { return fields_.cbegin(); } // First key
    auto end() const { return fields_.cend(); } // Last key

    size_t get_step() const { return step_; }
    void increment_step() { step_++; }
    double get_time() const { return time_; }
    void advance_time(double dt) { time_ += dt; }

    bool has_field(const std::string& name) const {
        return fields_.find(name) != fields_.end();
    }

private:
    const Utils::ConfigurationManager& config_ref_;
    const Grid& grid_;
    const Parameters& parameters_;
    std::map<std::string, AnyField> fields_;

    size_t step_ = 0;
    double time_ = 0.0;

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm_;
    cudaStream_t nccl_stream_;
#endif
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_VVMSTATE_HPP
