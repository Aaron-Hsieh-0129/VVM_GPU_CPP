#include "LateralBoundaryNudging.hpp"
#include <iostream>
#include <pnetcdf.h>

namespace VVM {
namespace Dynamics {

void LateralBoundaryNudging::check_ncmpi_error(int status, const std::string& msg) const {
    if (status != NC_NOERR) {
        std::string err_msg = msg + ": " + ncmpi_strerror(status);
        int rank = grid_.get_mpi_rank();
        if (rank == 0) std::cerr << "PnetCDF Error in LBN: " << err_msg << std::endl;
        MPI_Abort(grid_.get_cart_comm(), status);
        throw std::runtime_error(err_msg);
    }
}

LateralBoundaryNudging::LateralBoundaryNudging(const Utils::ConfigurationManager& config, 
                                               const Core::Grid& grid, 
                                               const Core::Parameters& params,
                                               Core::State& state)
    : config_(config), grid_(grid), params_(params) 
{
    enable_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.enable", false);
    
    if (enable_) {
        nudge_W_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.west", false);
        nudge_E_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.east", false);
        nudge_S_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.south", false);
        nudge_N_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.boundaries.north", false);

        tau_b_  = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.tau_b", 300.0);
        offset_ = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.offset", 2500.0);
        width_  = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.width", 600.0);
        radius_ = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.radius", 2500.0);
        
        inv_tau_b_ = 1.0 / tau_b_;

        target_vars_ = config_.get_value<std::vector<std::string>>(
            "dynamics.forcings.lateral_boundary_nudging.target_vars", 
            std::vector<std::string>{"th", "qv"}
        );

        data_dir_ = config_.get_value<std::string>("dynamics.forcings.lateral_boundary_nudging.forcing_data.directory", "../rundata/LS_forcings/");
        time_varying_ = config_.get_value<bool>("dynamics.forcings.lateral_boundary_nudging.forcing_data.time_varying", false);

        if (time_varying_) {
            file_prefix_ = config_.get_value<std::string>("dynamics.forcings.lateral_boundary_nudging.forcing_data.file_prefix", "ls_forcing_");
            update_interval_ = config_.get_value<double>("dynamics.forcings.lateral_boundary_nudging.forcing_data.update_interval_s", 3600.0);
            time_T1_ = 0.0;
            time_T2_ = update_interval_;
        } 
        else {
            file_name_ = config_.get_value<std::string>("dynamics.forcings.lateral_boundary_nudging.forcing_data.file_name_for_not_varying", "ls_forcing_constant.nc");
        }
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
        state.add_field<3>(var_name + "_ls", {nz, ny, nx});
        
        if (time_varying_) {
            name_T1_[var_name] = var_name + "_ls_T1";
            name_T2_[var_name] = var_name + "_ls_T2";
            state.add_field<3>(name_T1_[var_name], {nz, ny, nx});
            state.add_field<3>(name_T2_[var_name], {nz, ny, nx});
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

    if (time_varying_) {
        std::ostringstream file_t1, file_t2;
        file_t1 << data_dir_ << file_prefix_ << std::setfill('0') << std::setw(6) << static_cast<int>(time_T1_) << ".nc";
        file_t2 << data_dir_ << file_prefix_ << std::setfill('0') << std::setw(6) << static_cast<int>(time_T2_) << ".nc";

        load_forcing_data(state, file_t1.str(), false); // load to T2
        for (const auto& var : target_vars_) { std::swap(name_T1_[var], name_T2_[var]); } // swap to T1
        load_forcing_data(state, file_t2.str(), false); // load to T2
    } 
    else {
        std::string full_filepath = data_dir_ + file_name_;
        load_forcing_data(state, full_filepath, true);
    }
}

void LateralBoundaryNudging::load_forcing_data(Core::State& state, const std::string& filepath, bool is_constant) {
    int ncid;
    int status = ncmpi_open(grid_.get_cart_comm(), filepath.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid);
    check_ncmpi_error(status, "Failed to open NetCDF file: " + filepath);

    if (grid_.get_mpi_rank() == 0) std::cout << "  - LBN Loaded Forcing Data: " << filepath << std::endl;

    MPI_Offset start[3] = {0, grid_.get_local_physical_start_y(), grid_.get_local_physical_start_x()};
    MPI_Offset count[3] = {grid_.get_global_points_z(), grid_.get_local_physical_points_y(), grid_.get_local_physical_points_x()};
    std::vector<double> host_buffer(count[0] * count[1] * count[2]);

    const int h = grid_.get_halo_cells();
    const int nz_in = count[0], ny_in = count[1], nx_in = count[2];

    for (const auto& var : target_vars_) {
        int varid;
        status = ncmpi_inq_varid(ncid, var.c_str(), &varid);
        check_ncmpi_error(status, "Cannot find 3D variable '" + var + "' in LBN file");
        check_ncmpi_error(ncmpi_get_vara_double_all(ncid, varid, start, count, host_buffer.data()), "Failed to read 3D variable");

        std::string target_field_name = is_constant ? (var + "_ls") : name_T2_[var];
        auto& field = state.get_field<3>(target_field_name);
        
        auto field_view_dev = field.get_mutable_device_data();
        auto field_view_host = Kokkos::create_mirror_view(field_view_dev);
        
        using HostExec = Kokkos::DefaultHostExecutionSpace;
        Kokkos::parallel_for("Init_LBN_Buffer_3D",
            Kokkos::MDRangePolicy<HostExec, Kokkos::Rank<3>>({0, 0, 0}, {nz_in, ny_in, nx_in}),
            [=](const int k, const int j, const int i) {
                size_t flat_idx = static_cast<size_t>(k) * ny_in * nx_in + static_cast<size_t>(j) * nx_in + static_cast<size_t>(i);
                field_view_host(k + h, j + h, i + h) = host_buffer[flat_idx];
            }
        );
        Kokkos::deep_copy(field_view_dev, field_view_host);
    }
    check_ncmpi_error(ncmpi_close(ncid), "Failed to close NetCDF file");
}

void LateralBoundaryNudging::update_large_scale_forcing(Core::State& state, double current_time) {
    if (!time_varying_) return;

    if (current_time >= time_T2_) {
        for (const auto& var : target_vars_) {
            auto& t1_view = state.get_field<3>(var + "_ls_T1").get_mutable_device_data();
            auto& t2_view = state.get_field<3>(var + "_ls_T2").get_mutable_device_data();
            auto temp = t1_view;
            t1_view = t2_view;
            t2_view = temp;
        }

        time_T1_ = time_T2_;
        time_T2_ += update_interval_;

        std::ostringstream filename_stream;
        filename_stream << data_dir_ << file_prefix_ 
                        << std::setfill('0') << std::setw(6) 
                        << static_cast<int>(time_T2_) << ".nc";
        
        load_forcing_data(state, filename_stream.str(), false);
    }

    double W = (current_time - time_T1_) / (time_T2_ - time_T1_);
    if (W < 0.0) W = 0.0;
    if (W > 1.0) W = 1.0;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();

    for (const auto& var : target_vars_) {
        const auto& t1_data = state.get_field<3>(var + "_ls_T1").get_device_data();
        const auto& t2_data = state.get_field<3>(var + "_ls_T2").get_device_data();
        auto& current_ls = state.get_field<3>(var + "_ls").get_mutable_device_data();

        Kokkos::parallel_for("Time_Interpolation_" + var,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nz, ny, nx}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                current_ls(k, j, i) = (1.0 - W) * t1_data(k, j, i) + W * t2_data(k, j, i);
            }
        );
    }
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
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({h, h, h}, {nz-h, ny-h, nx-h}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                
                double fn = weight(j, i);
                
                if (fn > 1e-6) {
                    tend(k, j, i) += fn * inv_tau * (var_ls(k, j, i) - var(k, j, i));
                }
            }
        );
    }
}

template void LateralBoundaryNudging::calculate_tendencies(Core::State& state, const std::string& var_name, Core::Field<3ul>& out_tendency) const;

} // namespace Dynamics
} // namespace VVM
