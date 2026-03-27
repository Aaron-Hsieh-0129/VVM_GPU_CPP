#include "LateralBoundaryNudging.hpp"
#include <iostream>

namespace VVM {
namespace Dynamics {

LateralBoundaryNudging::LateralBoundaryNudging(const Utils::ConfigurationManager& config, 
                                               const Core::Grid& grid, 
                                               const Core::Parameters& params,
                                               Core::State& state)
    : config_(config), grid_(grid), params_(params) 
{
    enable_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.enable", false);
    
    if (enable_) {
        nudge_W_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.west", true);
        nudge_E_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.east", true);
        nudge_S_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.south", true);
        nudge_N_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.north", true);

        tau_b_  = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.tau_b", 300.0);
        offset_ = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.offset", 2500.0);
        width_  = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.width", 600.0);
        radius_ = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.radius", 2500.0);
        
        inv_tau_b_ = 1.0 / tau_b_;

        target_vars_ = config_.get_value<std::vector<std::string>>(
            "dynamics.forcings.lateral_boundary_nudging.target_vars", 
            std::vector<std::string>{"th", "qv"}
        );
    }

}

void LateralBoundaryNudging::initialize(Core::State& state) {
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h  = grid_.get_halo_cells();
    double dx = grid_.get_dx();
    double dy = grid_.get_dy();

    int global_start_x = grid_.get_local_physical_start_x();
    int global_start_y = grid_.get_local_physical_start_y();
    double xsize = grid_.get_global_points_x() * dx;
    double ysize = grid_.get_global_points_y() * dy;

    if (!state.has_field("lbn_weight")) {
        state.add_field<2>("lbn_weight", {ny, nx});
    }
    for (const auto& var_name : target_vars_) {
        std::string ls_name = var_name + "_ls";
        if (!state.has_field(ls_name)) {
            state.add_field<3>(ls_name, {nz, ny, nx});
        }
        auto& var = state.get_field<3>(ls_name).get_mutable_device_data();
        if (var_name == "qv") {
            auto& qv = state.get_field<3>("qv").get_mutable_device_data();
            Kokkos::parallel_for("init_nudge", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
                KOKKOS_LAMBDA(int k, int j, int i) {
                    if (i < nx/2) {
                        var(k,j,i) = qv(k,j,i)*1.3;
                    }
                    else {
                        var(k,j,i) = qv(k,j,i)*0.7;
                    }
                }
            );
        }
    }
    
    auto& weight = state.get_field<2>("lbn_weight").get_mutable_device_data();
    
    double offset = offset_;
    double width  = width_;
    double radius = radius_;
    bool nW = nudge_W_, nE = nudge_E_, nS = nudge_S_, nN = nudge_N_;

    int rank = grid_.get_mpi_rank();
    if (rank == 0) {
        std::cout << "--- Initializing Lateral Boundary Nudging ---" << std::endl;
        std::cout << "  * Active boundaries - W: " << nW << ", E: " << nE 
                  << ", S: " << nS << ", N: " << nN << std::endl;
    }

    Kokkos::parallel_for("Init_LBN_Weight",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({h, h}, {ny-h, nx-h}),
        KOKKOS_LAMBDA(const int j, const int i) {
            
            double x = (global_start_x + (i - h)) * dx + 0.5 * dx;
            double y = (global_start_y + (j - h)) * dy + 0.5 * dy;

            double dc = offset + radius;
            double f = 0.0;

            bool in_W = (x < dc);
            bool in_E = (x > xsize - dc);
            bool in_S = (y < dc);
            bool in_N = (y > ysize - dc);

            if (in_W && in_S && nW && nS) {
                double D = Kokkos::sqrt((x - dc)*(x - dc) + (y - dc)*(y - dc)) - radius;
                f += Kokkos::exp(-0.5 * (D / width) * (D / width));
            }
            else if (in_E && in_S && nE && nS) {
                double D = Kokkos::sqrt((x - (xsize - dc))*(x - (xsize - dc)) + (y - dc)*(y - dc)) - radius;
                f += Kokkos::exp(-0.5 * (D / width) * (D / width));
            }
            else if (in_W && in_N && nW && nN) {
                double D = Kokkos::sqrt((x - dc)*(x - dc) + (y - (ysize - dc))*(y - (ysize - dc))) - radius;
                f += Kokkos::exp(-0.5 * (D / width) * (D / width));
            }
            else if (in_E && in_N && nE && nN) {
                double D = Kokkos::sqrt((x - (xsize - dc))*(x - (xsize - dc)) + (y - (ysize - dc))*(y - (ysize - dc))) - radius;
                f += Kokkos::exp(-0.5 * (D / width) * (D / width));
            }
            else {
                // If it's not corner or it's the corner withuout turning on boundary, using 1D linear line.
                if (nW) f += Kokkos::exp(-0.5 * ((x - offset) / width) * ((x - offset) / width));
                if (nE) f += Kokkos::exp(-0.5 * ((x - (xsize - offset)) / width) * ((x - (xsize - offset)) / width));
                if (nS) f += Kokkos::exp(-0.5 * ((y - offset) / width) * ((y - offset) / width));
                if (nN) f += Kokkos::exp(-0.5 * ((y - (ysize - offset)) / width) * ((y - (ysize - offset)) / width));
            }

            if (f < 1e-10) f = 0.0;
            if (f > 1.0) f = 1.0;

            weight(j, i) = f;
        }
    );
}

template<size_t Dim>
void LateralBoundaryNudging::calculate_tendencies(Core::State& state, 
                                                  const std::string& var_name, 
                                                  Core::Field<Dim>& out_tendency) const 
{
    const auto& weight = state.get_field<2>("lbn_weight").get_device_data();
    
    const auto& var = state.get_field<3>(var_name).get_device_data();
    
    const auto& var_ls = state.get_field<3>(var_name + "_ls").get_device_data();

    auto& tend = out_tendency.get_mutable_device_data();
    
    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h  = grid_.get_halo_cells();
    
    double inv_tau = inv_tau_b_;

    if constexpr (Dim == 3) {
        Kokkos::parallel_for("Sponge_Tendency_Lateral_" + var_name,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, h, h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                
                double fn = weight(j, i);
                
                if (fn > 0.0) {
                    tend(k, j, i) += fn * inv_tau * (var_ls(k, j, i) - var(k, j, i));
                }
            }
        );
    }
}

template void LateralBoundaryNudging::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency) const;

} // namespace Dynamics
} // namespace VVM
