#ifndef VVM_DYNAMICS_TENDENCY_TERM_HPP
#define VVM_DYNAMICS_TENDENCY_TERM_HPP

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"

namespace VVM {
namespace Dynamics {

class TendencyTerm {
public:
    virtual ~TendencyTerm() = default;
    
    virtual void compute_tendency(
        Core::State& state, 
        const Core::Grid& grid,
        const Core::Parameters& params, 
        Core::Field<3>& out_tendency) const = 0;
};

} // namespace Dynamics
} // namespace VVM
#endif
