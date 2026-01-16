#ifndef VVM_DYNAMICS_TIME_INTEGRATOR_HPP
#define VVM_DYNAMICS_TIME_INTEGRATOR_HPP

#include "TemporalScheme.hpp"
#include <string>
#include <vector>

namespace VVM {
namespace Dynamics {

class TimeIntegrator : public TemporalScheme {
public:
    explicit TimeIntegrator(std::string var_name, bool has_ab2, bool has_fe);
    ~TimeIntegrator() override;

    void step(
        Core::State& state,
        const Core::Grid& grid,
        const Core::Parameters& params,
        double dt
    ) const override;

    std::vector<std::string> get_required_state_suffixes() const override {
        // Only AB2 needs a previous state (_m)
        return has_ab2_terms_ ? std::vector<std::string>{"_m"} : std::vector<std::string>{};
    }

    // This is for sequential splitting due to the order of physics doesn't have to be the same with the dynamics.
    // This method is designed for update State in any physical process that can be called directly to do integration.
    // These process generally uses forward Euler.
    template<size_t Dim> 
    static void apply_forward_update(Core::State& state,const std::string var_name, const Core::Grid& grid, double dt, Core::Field<Dim>& tend_field) {
        const int nz = grid.get_local_total_points_z();
        const int ny = grid.get_local_total_points_y();
        const int nx = grid.get_local_total_points_x();
        const int h = grid.get_halo_cells();

        auto& field_new_view = state.get_field<3>(var_name).get_mutable_device_data();

        if constexpr (Dim == 2) {
            const auto& fe_tendency_data = tend_field.get_device_data();
            if (var_name == "zeta") {
                int NK2 = nz-h-1;
                Kokkos::parallel_for("Pure_Forward_Euler_Step",
                    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
                    KOKKOS_LAMBDA(const int j, const int i) {
                        field_new_view(NK2, j, i) += dt * fe_tendency_data(j, i);
                    }
                );
            }
        }
        else if constexpr(Dim == 3) {
            const auto& fe_tendency_data = tend_field.get_device_data();
            int k_start = h;
            int k_end = nz-h;
            if (var_name == "xi" || var_name == "eta") {
                k_end = nz-h-1;
            }
            Kokkos::parallel_for("Pure_Forward_Euler_Step",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
                KOKKOS_LAMBDA(const int k, const int j, const int i) {
                    field_new_view(k, j, i) += dt * fe_tendency_data(k, j, i);
                }
            );
        }
        return;
    }

private:
    std::string variable_name_;
    bool has_ab2_terms_;
    bool has_fe_terms_;
};

} // namespace Dynamics
} // namespace VVM
#endif
