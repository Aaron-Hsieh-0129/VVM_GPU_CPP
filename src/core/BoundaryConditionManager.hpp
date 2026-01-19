#ifndef VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP
#define VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP

#include "core/Grid.hpp"
#include "core/Field.hpp"

namespace VVM {
namespace Core {

class BoundaryConditionManager {
public:
    explicit BoundaryConditionManager(const Grid& grid);

    template<size_t Dim>
    void apply_dirichlet_zero(Field<Dim>& field) const;

    template<size_t Dim>
    void apply_vorticity_bc(Field<Dim>& field) const;

    template<size_t Dim>
    void apply_zero_gradient(Field<Dim>& field) const;

    template<size_t Dim>
    void apply_fixed_profile_z(Field<Dim>& field, const Field<1>& profile) const;

    template<size_t Dim>
    void apply_periodic(Field<Dim>& field) const;

    template<size_t Dim>
    void apply_zero_gradient_bottom_zero_top(Field<Dim>& field) const;

private:
    const Grid& grid_;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP
