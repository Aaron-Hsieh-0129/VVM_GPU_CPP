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
    
    // 計算此過程產生的傾向，並「累加」到 tendencies 物件中
    virtual void compute_tendency(
        const Core::State& state, 
        Core::State& tendencies, 
        const Core::Grid& grid,
        const Core::Parameters& params) const = 0;
};

} // namespace Dynamics
} // namespace VVM
#endif
