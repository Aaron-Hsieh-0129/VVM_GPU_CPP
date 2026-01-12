#ifndef VVM_PHYSICS_TURBULENCE_PROCESS_HPP
#define VVM_PHYSICS_TURBULENCE_PROCESS_HPP

#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "core/HaloExchanger.hpp"
#include "core/Field.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Physics {

enum CoeffID {
    UU1 = 0, UU2,
    UV1, UV2,
    UW1, UW2,
    VU1, VU2,
    VV1, VV2,
    VW1, VW2,
    WU1, WU2,
    WV1, WV2,
    WW1, WW2,
    TOTAL_BITS
};

struct TerrainMasks {
    using MaskView = Kokkos::View<unsigned int***>;
    MaskView data;

    TerrainMasks() = default;

    TerrainMasks(int nz, int ny, int nx) 
        : data("Terrain_Masks", nz, ny, nx)
    {
        reset_to_ones();
    }

    void reset_to_ones() {
        unsigned int all_ones = (1u << TOTAL_BITS) - 1;
        Kokkos::deep_copy(data, all_ones);
    }

    KOKKOS_INLINE_FUNCTION
    void turn_off(int k, int j, int i, CoeffID id) const {
        Kokkos::atomic_fetch_and(&data(k, j, i), ~(1u << id));
    }

    KOKKOS_INLINE_FUNCTION
    void turn_off_all(int k, int j, int i) const {
        data(k, j, i) = 0u;
    }

    KOKKOS_INLINE_FUNCTION
    double val(int k, int j, int i, CoeffID id) const {
        return ((data(k, j, i) >> id) & 1) ? 1.0 : 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_active(int k, int j, int i, CoeffID id) const {
        return (data(k, j, i) >> id) & 1;
    }
    
    MaskView get_raw_view() { return data; }
};



class TurbulenceProcess {
public:
    TurbulenceProcess(const Utils::ConfigurationManager& config, 
                      const Core::Grid& grid, 
                      const Core::Parameters& params,
                      Core::HaloExchanger& halo_exchanger,
                      Core::State& state);

    void process_thermodynamics(Core::State& state, double dt);
    void process_dynamics(Core::State& state, double dt);

    void compute_coefficients(Core::State& state, double dt);

    template<size_t Dim>
    void calculate_tendencies(Core::State& state, 
                              const std::string& var_name, 
                              Core::Field<Dim>& out_tendency);

    void initialize(Core::State& state);
    void init_boundary_masks(Core::State& state);
    // void init_dh_coefficients(Core::State& state);

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;
    Core::HaloExchanger& halo_exchanger_;

    VVM::Core::Field<3> temp3d_tendency_;
    VVM::Core::Field<2> temp2d_tendency_;
    std::vector<std::string> dynamics_vars_;
    std::vector<std::string> thermodynamics_vars_;

    double dx_, dy_, dz_;
    double rdx_, rdy_, rdz_;
    double rdx2_, rdy2_, rdz2_;
    
    double deld_;    // Grid scale length
    double ramd0s_;  // Asymptotic mixing length squared
    double critmn_;  // Minimum viscosity
    
    double grav_;
    double vk_;

    // Kokkos::View<TopoMask***> mask_view_;
    TerrainMasks masks_;

    // VVM::Core::Field<3> DHUU1_, DHUU2_;
    // VVM::Core::Field<3> DHUV1_, DHUV2_;
    // VVM::Core::Field<3> DHUW1_, DHUW2_;
    // VVM::Core::Field<3> DHVU1_, DHVU2_;
    // VVM::Core::Field<3> DHVV1_, DHVV2_;
    // VVM::Core::Field<3> DHVW1_, DHVW2_;
    // VVM::Core::Field<3> DHWU1_, DHWU2_;
    // VVM::Core::Field<3> DHWV1_, DHWV2_;
    // VVM::Core::Field<3> DHWW1_, DHWW2_;
};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_TURBULENCE_PROCESS_HPP
