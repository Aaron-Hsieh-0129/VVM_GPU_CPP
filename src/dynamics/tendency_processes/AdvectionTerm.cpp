#include "AdvectionTerm.hpp"

namespace VVM {
namespace Dynamics {

AdvectionTerm::AdvectionTerm(std::unique_ptr<SpatialScheme> scheme, std::string var_name)
    : scheme_(std::move(scheme)), variable_name_(std::move(var_name)) {}

AdvectionTerm::~AdvectionTerm() = default;


void AdvectionTerm::compute_tendency(
    const Core::State& state, 
    const Core::Grid& grid,
    const Core::Parameters& params,
    Core::Field<3>& out_tendency) const {
    
    // 1. 從 state 中取得需要的場
    // 這裡以 variable_name_ (例如 "th") 作為要計算傾向的變數
    const auto& scalar_field = state.get_field<3>(variable_name_);
    const auto& u_field = state.get_field<3>("u");
    const auto& w_field = state.get_field<3>("w");

    std::cout << "Computed advection tendency for variable: " << variable_name_ << std::endl;

    const int nz = grid.get_local_physical_points_z();
    const int ny = grid.get_local_physical_points_y();
    const int nx = grid.get_local_physical_points_x();
    Core::Field<3> flux("flux", {nz, ny, nx});
    scheme_->calculate_flux_divergence_x(
        scalar_field, u_field, grid, params, flux
    );

    auto& tendency = out_tendency.get_mutable_device_data();
    auto& flux_data = flux.get_device_data();
    auto rdx_view = params.rdx;

    Kokkos::parallel_for("flux_convergence_tendency", 
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {nz-1, ny-1, nx-1}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            tendency(k,j,i) += -0.5*(flux_data(k,j,i) - flux_data(k,j,i-1)) * rdx_view();
        }
    );
}

} // namespace Dynamics
} // namespace VVM
