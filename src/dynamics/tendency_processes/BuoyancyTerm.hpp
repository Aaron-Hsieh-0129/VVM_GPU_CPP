#ifndef VVM_DYNAMICS_BUOYANCY_TERM_HPP
#define VVM_DYNAMICS_BUOYANCY_TERM_HPP

#include "TendencyTerm.hpp"
#include "dynamics/spatial_schemes/SpatialScheme.hpp"
#include "core/HaloExchanger.hpp"
#include <memory>
#include <string>

namespace VVM {
namespace Dynamics {

class BuoyancyTerm : public TendencyTerm {
public:
    BuoyancyTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name, VVM::Core::HaloExchanger& halo_exchanger);
    ~BuoyancyTerm() override;

    void compute_tendency(
        Core::State& state, 
        const Core::Grid& grid,
        const Core::Parameters& params, 
        Core::Field<3>& out_tendency) const override;
private:
    std::unique_ptr<SpatialScheme> scheme_;
    std::string variable_name_;

    Core::HaloExchanger& halo_exchanger_;
};

} // namespace Dynamics
} // namespace VVM
#endif // VVM_DYNAMICS_BUOYANCY_TERM_HPP
