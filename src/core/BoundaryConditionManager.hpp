#ifndef VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP
#define VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP

#include "core/Grid.hpp"
#include "core/Field.hpp"

namespace VVM {
namespace Core {

enum class HorizontalBCType {
    Periodic,
    ZeroGradient
};

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

    // Horizontal boundary
    void initialize_bc_types(const std::string& x_bc, const std::string& y_bc);
    
    template<size_t Dim>
    void apply_zero_gradient_x(Field<Dim>& field) const;

    template<size_t Dim>
    void apply_zero_gradient_y(Field<Dim>& field) const;

    template<size_t Dim>
    void apply_horizontal_bcs(Field<Dim>& field) const;

private:
    const Grid& grid_;

    HorizontalBCType x_bc_type_ = HorizontalBCType::Periodic;
    HorizontalBCType y_bc_type_ = HorizontalBCType::Periodic;
};

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_BOUNDARYCONDITIONMANAGER_HPP
