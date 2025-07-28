#ifndef VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP
#define VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "utils/ConfigurationManager.hpp"

namespace VVM {
namespace Core {

enum class ZBoundaryType {
    ZERO,           // Set to 0
    ZERO_GRADIENT   // Zero gradient (0 = 1, nz-1 = nz-2)
};

class BoundaryConditionManager {
public:
    explicit BoundaryConditionManager(const Grid& grid, const Utils::ConfigurationManager& config);

    void apply_z_bcs(State& state) const;

    template<size_t Dim>
    void apply_z_bcs_to_field(Field<Dim>& field) const;
private:

    const Grid& grid_ref_;
    ZBoundaryType top_bc_;
    ZBoundaryType bottom_bc_;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP
