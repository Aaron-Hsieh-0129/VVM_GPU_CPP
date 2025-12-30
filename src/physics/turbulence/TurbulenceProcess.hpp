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

struct TopoMask {
    unsigned int bits = 0;

    static constexpr int SHIFT_U = 0;  // Bits 0-5
    static constexpr int SHIFT_V = 6;  // Bits 6-11
    static constexpr int SHIFT_W = 12; // Bits 12-17

    static constexpr int OFF_RIGHT = 0; // x+ 
    static constexpr int OFF_LEFT  = 1; // x- 
    static constexpr int OFF_FRONT = 2; // y+
    static constexpr int OFF_BACK  = 3; // y-
    static constexpr int OFF_TOP   = 4; // z+
    static constexpr int OFF_BOT   = 5; // z-

    KOKKOS_INLINE_FUNCTION void set_all_closed() { bits = 0; }
    KOKKOS_INLINE_FUNCTION void set_all_open() { bits = 0x0003FFFF; } // Set lower 18 bits to 1

    KOKKOS_INLINE_FUNCTION void open_bit(int shift, int off) {
        bits |= (1 << (shift + off));
    }

    KOKKOS_INLINE_FUNCTION void close_bit(int shift, int off) {
        bits &= ~(1 << (shift + off));
    }

    KOKKOS_INLINE_FUNCTION double get_bit(int shift, int off) const {
        return (bits & (1 << (shift + off))) ? 1.0 : 0.0;
    }

    // U-Channel Setters (DHUU, DHUV, DHUW)
    KOKKOS_INLINE_FUNCTION void open_u_right() { open_bit(SHIFT_U, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION void open_u_left()  { open_bit(SHIFT_U, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION void open_u_front() { open_bit(SHIFT_U, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION void open_u_back()  { open_bit(SHIFT_U, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION void open_u_top()   { open_bit(SHIFT_U, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION void open_u_bot()   { open_bit(SHIFT_U, OFF_BOT);   }

    // V-Channel Setters (DHVU, DHVV, DHVW)
    KOKKOS_INLINE_FUNCTION void open_v_right() { open_bit(SHIFT_V, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION void open_v_left()  { open_bit(SHIFT_V, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION void open_v_front() { open_bit(SHIFT_V, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION void open_v_back()  { open_bit(SHIFT_V, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION void open_v_top()   { open_bit(SHIFT_V, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION void open_v_bot()   { open_bit(SHIFT_V, OFF_BOT);   }

    // W-Channel Setters (DHWU, DHWV, DHWW)
    KOKKOS_INLINE_FUNCTION void open_w_right() { open_bit(SHIFT_W, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION void open_w_left()  { open_bit(SHIFT_W, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION void open_w_front() { open_bit(SHIFT_W, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION void open_w_back()  { open_bit(SHIFT_W, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION void open_w_top()   { open_bit(SHIFT_W, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION void open_w_bot()   { open_bit(SHIFT_W, OFF_BOT);   }

    // --- U-Channel Closers ---
    KOKKOS_INLINE_FUNCTION void close_u_right() { close_bit(SHIFT_U, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION void close_u_left()  { close_bit(SHIFT_U, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION void close_u_front() { close_bit(SHIFT_U, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION void close_u_back()  { close_bit(SHIFT_U, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION void close_u_top()   { close_bit(SHIFT_U, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION void close_u_bot()   { close_bit(SHIFT_U, OFF_BOT);   }

    // --- V-Channel Closers ---
    KOKKOS_INLINE_FUNCTION void close_v_right() { close_bit(SHIFT_V, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION void close_v_left()  { close_bit(SHIFT_V, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION void close_v_front() { close_bit(SHIFT_V, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION void close_v_back()  { close_bit(SHIFT_V, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION void close_v_top()   { close_bit(SHIFT_V, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION void close_v_bot()   { close_bit(SHIFT_V, OFF_BOT);   }

    // --- W-Channel Closers ---
    KOKKOS_INLINE_FUNCTION void close_w_right() { close_bit(SHIFT_W, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION void close_w_left()  { close_bit(SHIFT_W, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION void close_w_front() { close_bit(SHIFT_W, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION void close_w_back()  { close_bit(SHIFT_W, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION void close_w_top()   { close_bit(SHIFT_W, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION void close_w_bot()   { close_bit(SHIFT_W, OFF_BOT);   }

    // U-Channel (DHUU1, DHUU2...)
    KOKKOS_INLINE_FUNCTION double u_right() const { return get_bit(SHIFT_U, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION double u_left()  const { return get_bit(SHIFT_U, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION double u_front() const { return get_bit(SHIFT_U, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION double u_back()  const { return get_bit(SHIFT_U, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION double u_top()   const { return get_bit(SHIFT_U, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION double u_bot()   const { return get_bit(SHIFT_U, OFF_BOT);   }

    // V-Channel (DHVU1, DHVU2...)
    KOKKOS_INLINE_FUNCTION double v_right() const { return get_bit(SHIFT_V, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION double v_left()  const { return get_bit(SHIFT_V, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION double v_front() const { return get_bit(SHIFT_V, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION double v_back()  const { return get_bit(SHIFT_V, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION double v_top()   const { return get_bit(SHIFT_V, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION double v_bot()   const { return get_bit(SHIFT_V, OFF_BOT);   }

    // W-Channel (DHWU1, DHWU2...) - Used for Thermodynamics
    KOKKOS_INLINE_FUNCTION double w_right() const { return get_bit(SHIFT_W, OFF_RIGHT); }
    KOKKOS_INLINE_FUNCTION double w_left()  const { return get_bit(SHIFT_W, OFF_LEFT);  }
    KOKKOS_INLINE_FUNCTION double w_front() const { return get_bit(SHIFT_W, OFF_FRONT); }
    KOKKOS_INLINE_FUNCTION double w_back()  const { return get_bit(SHIFT_W, OFF_BACK);  }
    KOKKOS_INLINE_FUNCTION double w_top()   const { return get_bit(SHIFT_W, OFF_TOP);   }
    KOKKOS_INLINE_FUNCTION double w_bot()   const { return get_bit(SHIFT_W, OFF_BOT);   }
    
    KOKKOS_INLINE_FUNCTION bool is_all_closed() const { return bits == 0; }
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
    void init_dh_coefficients(Core::State& state);

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

    Kokkos::View<TopoMask***> mask_view_;


    VVM::Core::Field<3> DHUU1_, DHUU2_;
    VVM::Core::Field<3> DHUV1_, DHUV2_;
    VVM::Core::Field<3> DHUW1_, DHUW2_;
    VVM::Core::Field<3> DHVU1_, DHVU2_;
    VVM::Core::Field<3> DHVV1_, DHVV2_;
    VVM::Core::Field<3> DHVW1_, DHVW2_;

    VVM::Core::Field<3> DHWU1_, DHWU2_;
    VVM::Core::Field<3> DHWV1_, DHWV2_;
    VVM::Core::Field<3> DHWW1_, DHWW2_;

};

} // namespace Physics
} // namespace VVM

#endif // VVM_PHYSICS_TURBULENCE_PROCESS_HPP
