#include "OutputManager.hpp"
#include <filesystem>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace VVM {
namespace Core {

OutputManager::OutputManager(const Utils::ConfigurationManager& config, const Grid& grid, MPI_Comm comm) 
    : grid_(grid), comm_(comm) {
    enabled_ = config.get_value<bool>("output.enable_netcdf");
    if (!enabled_) {
        return;
    }

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);

    output_dir_ = config.get_value<std::string>("output.output_dir");
    filename_prefix_ = config.get_value<std::string>("output.output_filename_prefix");
    fields_to_output_ = config.get_value<std::vector<std::string>>("output.fields_to_output");
    
    g_x_start_ = config.get_value<int>("output.output_grid.x_start");
    g_y_start_ = config.get_value<int>("output.output_grid.y_start");
    g_z_start_ = config.get_value<int>("output.output_grid.z_start");
    x_stride_ = config.get_value<int>("output.output_grid.x_stride");
    y_stride_ = config.get_value<int>("output.output_grid.y_stride");
    z_stride_ = config.get_value<int>("output.output_grid.z_stride");
    
    g_x_end_ = config.get_value<int>("output.output_grid.x_end");
    if (g_x_end_ == -1) g_x_end_ = grid_.get_global_points_x() - 1;
    g_y_end_ = config.get_value<int>("output.output_grid.y_end");
    if (g_y_end_ == -1) g_y_end_ = grid_.get_global_points_y() - 1;
    g_z_end_ = config.get_value<int>("output.output_grid.z_end");
    if (g_z_end_ == -1) g_z_end_ = grid_.get_global_points_z() - 1;

    if (rank_ == 0) {
        if (!std::filesystem::exists(output_dir_)) {
            std::filesystem::create_directories(output_dir_);
        }
    }
    MPI_Barrier(comm_);

    std::string filename = output_dir_ + "/" + filename_prefix_ + ".nc";
    
    int ret = ncmpi_create(comm_, filename.c_str(), NC_CLOBBER | NC_64BIT_OFFSET, MPI_INFO_NULL, &ncid_);
    handle_pnetcdf_error(ret, "Failed to create NetCDF file: " + filename);

    // Background thread for I/O operations
    io_thread_ = std::thread(&OutputManager::io_thread_function, this);
}

OutputManager::~OutputManager() {
    if (enabled_) {
        // Send completion signal
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            finished_ = true;
        }
        cv_.notify_one(); // Notify I/O thread to check finished_ flag

        // Wait for I/O thread to finish
        if (io_thread_.joinable()) {
            io_thread_.join();
        }

        // Close NetCDF file
        if (ncid_ != -1) {
            ncmpi_close(ncid_);
        }
    }
}

void OutputManager::handle_pnetcdf_error(int status, const std::string& message) const {
    if (status != NC_NOERR) {
        if (rank_ == 0) {
            std::cerr << "PnetCDF Error on Rank " << rank_ << ": " << message << " (" << ncmpi_strerror(status) << ")" << std::endl;
        }
        MPI_Abort(comm_, status);
    }
}

void OutputManager::io_thread_function() {
    while (true) {
        OutputJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return !job_queue_.empty() || finished_; });

            if (job_queue_.empty() && finished_) {
                return;
            }

            job = std::move(job_queue_.front());
            job_queue_.pop();
        }

        write_job_to_netcdf(job);
    }
}

void OutputManager::write_output(const State& state, double time) {
    if (!enabled_) return;

    // Define dimensions and variables at the first write
    if (time_idx_ == 0) {
        define_dimensions_and_vars(state);
    }
    
    OutputJob job;
    job.time = time;

    for (const auto& field_name : fields_to_output_) {
        if (field_var_ids_.count(field_name)) {
            auto it = state.begin();
            while (it != state.end() && it->first != field_name) {
                ++it;
            }
            if (it != state.end()) {
                std::visit([&](const auto& field) {
                    using T = std::decay_t<decltype(field)>;
                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        // Copy device view to job (low cost)
                        job.fields_to_write[field_name] = field.get_device_data();
                    }
                }, it->second);
            }
        }
    }

    // Push job to the queue and notify the I/O thread
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        job_queue_.push(std::move(job));
    }
    cv_.notify_one();
}

void OutputManager::write_job_to_netcdf(const OutputJob& job) {
    ncmpi_begin_indep_data(ncid_);
    if (rank_ == 0) {
        MPI_Offset start[] = {time_idx_};
        MPI_Offset count[] = {1};
        handle_pnetcdf_error(ncmpi_put_vara_double(ncid_, time_var_id_, start, count, &job.time), "Failed to write time value");
    }
    ncmpi_end_indep_data(ncid_);

    for (const auto& field_pair : job.fields_to_write) {
        const std::string& field_name = field_pair.first;
        const AnyDeviceView& field_view_variant = field_pair.second;
        int var_id = field_var_ids_.at(field_name);

        std::visit([&](const auto& device_view) {
            using DeviceViewType = std::decay_t<decltype(device_view)>;

            if constexpr (!std::is_same_v<DeviceViewType, std::monostate>) {
                // This is a potentially expensive GPU -> CPU copy operation, now done in the I/O thread
                auto host_data = Kokkos::create_mirror_view(device_view);
                Kokkos::deep_copy(host_data, device_view);
                Kokkos::fence();

                int h = grid_.get_halo_cells();
                
                MPI_Offset local_x_start = grid_.get_local_physical_start_x();
                MPI_Offset local_y_start = grid_.get_local_physical_start_y();
                MPI_Offset local_z_start = grid_.get_local_physical_start_z();

                MPI_Offset intersect_x_start = std::max((MPI_Offset)local_x_start, g_x_start_);
                MPI_Offset intersect_y_start = std::max((MPI_Offset)local_y_start, g_y_start_);
                MPI_Offset intersect_z_start = std::max((MPI_Offset)local_z_start, g_z_start_);

                MPI_Offset intersect_x_end = std::min((MPI_Offset)grid_.get_local_physical_end_x(), g_x_end_);
                MPI_Offset intersect_y_end = std::min((MPI_Offset)grid_.get_local_physical_end_y(), g_y_end_);
                MPI_Offset intersect_z_end = std::min((MPI_Offset)grid_.get_local_physical_end_z(), g_z_end_);
                
                bool has_intersection = (intersect_x_end >= intersect_x_start && intersect_y_end >= intersect_y_start && intersect_z_end >= intersect_z_start);
                
                constexpr size_t Dim = DeviceViewType::rank;

                if constexpr (Dim == 2) {
                    MPI_Offset start[3] = {0, 0, 0};
                    MPI_Offset count[3] = {0, 0, 0};
                    std::vector<double> buffer;

                    if (has_intersection) {
                        MPI_Offset count_x = (intersect_x_end - intersect_x_start + x_stride_) / x_stride_;
                        MPI_Offset count_y = (intersect_y_end - intersect_y_start + y_stride_) / y_stride_;
                        
                        if (count_x > 0 && count_y > 0) {
                            buffer.resize(count_y * count_x);
                            for (MPI_Offset j=0; j<count_y; ++j) {
                                for (MPI_Offset i=0; i<count_x; ++i) {
                                    buffer[j * count_x + i] = host_data(h + (intersect_y_start - local_y_start) + j * y_stride_,
                                                                        h + (intersect_x_start - local_x_start) + i * x_stride_);
                                }
                            }
                            start[0] = time_idx_;
                            start[1] = (intersect_y_start - g_y_start_) / y_stride_;
                            start[2] = (intersect_x_start - g_x_start_) / x_stride_;
                            count[0] = 1;
                            count[1] = count_y;
                            count[2] = count_x;
                        }
                    }
                    handle_pnetcdf_error(ncmpi_put_vara_double_all(ncid_, var_id, start, count, buffer.data()), "Failed to write 2D field " + field_name);
                } 
                else if constexpr (Dim == 3) {
                    MPI_Offset start[4] = {0, 0, 0, 0};
                    MPI_Offset count[4] = {0, 0, 0, 0};
                    std::vector<double> buffer;

                    if (has_intersection) {
                        MPI_Offset count_x = (intersect_x_end - intersect_x_start + x_stride_) / x_stride_;
                        MPI_Offset count_y = (intersect_y_end - intersect_y_start + y_stride_) / y_stride_;
                        MPI_Offset count_z = (intersect_z_end - intersect_z_start + z_stride_) / z_stride_;

                        if (count_x > 0 && count_y > 0 && count_z > 0) {
                            buffer.resize(count_z * count_y * count_x);
                            for (MPI_Offset k=0; k<count_z; ++k) {
                                for (MPI_Offset j=0; j<count_y; ++j) {
                                    for (MPI_Offset i=0; i<count_x; ++i) {
                                        buffer[k*count_y*count_x + j*count_x + i] = 
                                            host_data(h + (intersect_z_start - local_z_start) + k*z_stride_,
                                                      h + (intersect_y_start - local_y_start) + j*y_stride_,
                                                      h + (intersect_x_start - local_x_start) + i*x_stride_);
                                    }
                                }
                            }
                            start[0] = time_idx_;
                            start[1] = (intersect_z_start - g_z_start_) / z_stride_;
                            start[2] = (intersect_y_start - g_y_start_) / y_stride_;
                            start[3] = (intersect_x_start - g_x_start_) / x_stride_;
                            count[0] = 1;
                            count[1] = count_z;
                            count[2] = count_y;
                            count[3] = count_x;
                        }
                    }
                    handle_pnetcdf_error(ncmpi_put_vara_double_all(ncid_, var_id, start, count, buffer.data()), "Failed to write 3D field " + field_name);
                } 
                else if constexpr (Dim == 4) {
                    MPI_Offset nv_dim_size = host_data.extent(0);
                    MPI_Offset start[5] = {0, 0, 0, 0, 0};
                    MPI_Offset count[5] = {0, 0, 0, 0, 0};
                    std::vector<double> buffer;
                    
                    if (has_intersection) {
                        MPI_Offset count_x = (intersect_x_end - intersect_x_start + x_stride_) / x_stride_;
                        MPI_Offset count_y = (intersect_y_end - intersect_y_start + y_stride_) / y_stride_;
                        MPI_Offset count_z = (intersect_z_end - intersect_z_start + z_stride_) / z_stride_;

                        if (count_x > 0 && count_y > 0 && count_z > 0) {
                            buffer.resize(nv_dim_size * count_z * count_y * count_x);
                            for(MPI_Offset n=0; n<nv_dim_size; ++n) {
                                for (MPI_Offset k=0; k<count_z; ++k) {
                                    for (MPI_Offset j=0; j<count_y; ++j) {
                                        for (MPI_Offset i=0; i<count_x; ++i) {
                                            buffer[n*count_z*count_y*count_x + k*count_y*count_x + j*count_x + i] = 
                                                host_data(n, 
                                                          h + (intersect_z_start - local_z_start) + k*z_stride_,
                                                          h + (intersect_y_start - local_y_start) + j*y_stride_,
                                                          h + (intersect_x_start - local_x_start) + i*x_stride_);
                                        }
                                    }
                                }
                            }
                            start[0] = time_idx_;
                            start[1] = 0;
                            start[2] = (intersect_z_start - g_z_start_) / z_stride_;
                            start[3] = (intersect_y_start - g_y_start_) / y_stride_;
                            start[4] = (intersect_x_start - g_x_start_) / x_stride_;
                            count[0] = 1;
                            count[1] = nv_dim_size;
                            count[2] = count_z;
                            count[3] = count_y;
                            count[4] = count_x;
                        }
                    }
                    handle_pnetcdf_error(ncmpi_put_vara_double_all(ncid_, var_id, start, count, buffer.data()), "Failed to write 4D field " + field_name);
                }
            }
        }, field_view_variant);
    }
    
    time_idx_++;
}

void OutputManager::define_dimensions_and_vars(const State& state) {
    int ret;

    MPI_Offset output_nx = (g_x_end_ >= g_x_start_) ? (g_x_end_ - g_x_start_ + x_stride_) / x_stride_ : 0;
    MPI_Offset output_ny = (g_y_end_ >= g_y_start_) ? (g_y_end_ - g_y_start_ + y_stride_) / y_stride_ : 0;
    MPI_Offset output_nz = (g_z_end_ >= g_z_start_) ? (g_z_end_ - g_z_start_ + z_stride_) / z_stride_ : 0;

    // --- Define Mode ---
    ret = ncmpi_def_dim(ncid_, "time", NC_UNLIMITED, &time_dim_id_); handle_pnetcdf_error(ret, "Failed to define time dimension");
    ret = ncmpi_def_dim(ncid_, "lev", output_nz, &z_dim_id_); handle_pnetcdf_error(ret, "Failed to define z dimension (lev)");
    ret = ncmpi_def_dim(ncid_, "lat", output_ny, &y_dim_id_); handle_pnetcdf_error(ret, "Failed to define y dimension (lat)");
    ret = ncmpi_def_dim(ncid_, "lon", output_nx, &x_dim_id_); handle_pnetcdf_error(ret, "Failed to define x dimension (lon)");
    ret = ncmpi_def_dim(ncid_, "nv", 2, &nv_dim_id_); handle_pnetcdf_error(ret, "Failed to define nv dimension for 4D vars");

    int time_dims[] = {time_dim_id_};
    ret = ncmpi_def_var(ncid_, "time", NC_DOUBLE, 1, time_dims, &time_var_id_); handle_pnetcdf_error(ret, "Failed to define time variable");

    int z_dims[] = {z_dim_id_};
    ret = ncmpi_def_var(ncid_, "lev", NC_DOUBLE, 1, z_dims, &z_var_id_); handle_pnetcdf_error(ret, "Failed to define z variable (lev)");

    int y_dims[] = {y_dim_id_};
    ret = ncmpi_def_var(ncid_, "lat", NC_DOUBLE, 1, y_dims, &y_var_id_); handle_pnetcdf_error(ret, "Failed to define y variable (lat)");

    int x_dims[] = {x_dim_id_};
    ret = ncmpi_def_var(ncid_, "lon", NC_DOUBLE, 1, x_dims, &x_var_id_); handle_pnetcdf_error(ret, "Failed to define x variable (lon)");
    
    // Long name and axis attributes
    ncmpi_put_att_text(ncid_, time_var_id_, "units", strlen("seconds since 0-0-0"), "seconds since 0-0-0");
    ncmpi_put_att_text(ncid_, time_var_id_, "long_name", 4, "time");
    ncmpi_put_att_text(ncid_, time_var_id_, "axis", 1, "T");

    ncmpi_put_att_text(ncid_, z_var_id_, "units", 5, "level");
    ncmpi_put_att_text(ncid_, z_var_id_, "long_name", 5, "level");
    ncmpi_put_att_text(ncid_, z_var_id_, "axis", 1, "Z");

    ncmpi_put_att_text(ncid_, y_var_id_, "units", 13, "degrees_north");
    ncmpi_put_att_text(ncid_, y_var_id_, "long_name", 8, "latitude");
    ncmpi_put_att_text(ncid_, y_var_id_, "axis", 1, "Y");

    ncmpi_put_att_text(ncid_, x_var_id_, "units", 12, "degrees_east");
    ncmpi_put_att_text(ncid_, x_var_id_, "long_name", 9, "longitude");
    ncmpi_put_att_text(ncid_, x_var_id_, "axis", 1, "X");

    for (const auto& field_name : fields_to_output_) {
        try {
            auto it = state.begin();
            while(it != state.end() && it->first != field_name) ++it;
            if (it == state.end()) throw std::runtime_error("Field not found in state.");

            std::visit([&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                int var_id = -1;
                int ret_val = NC_NOERR;

                if constexpr (std::is_same_v<T, Field<2>>) {
                    int var_dims[] = {time_dim_id_, y_dim_id_, x_dim_id_};
                    ret_val = ncmpi_def_var(ncid_, field_name.c_str(), NC_DOUBLE, 3, var_dims, &var_id);
                    if (var_id != -1) ncmpi_put_att_text(ncid_, var_id, "coordinates", 12, "time lat lon");
                } 
                else if constexpr (std::is_same_v<T, Field<3>>) {
                    int var_dims[] = {time_dim_id_, z_dim_id_, y_dim_id_, x_dim_id_};
                    ret_val = ncmpi_def_var(ncid_, field_name.c_str(), NC_DOUBLE, 4, var_dims, &var_id);
                    if (var_id != -1) ncmpi_put_att_text(ncid_, var_id, "coordinates", 16, "time lev lat lon");
                } 
                else if constexpr (std::is_same_v<T, Field<4>>) {
                    int var_dims[] = {time_dim_id_, nv_dim_id_, z_dim_id_, y_dim_id_, x_dim_id_};
                    ret_val = ncmpi_def_var(ncid_, field_name.c_str(), NC_DOUBLE, 5, var_dims, &var_id);
                    if (var_id != -1) ncmpi_put_att_text(ncid_, var_id, "coordinates", 19, "time nv lev lat lon");
                }
                
                if (var_id != -1) { 
                    handle_pnetcdf_error(ret_val, "Failed to define variable " + field_name);
                    field_var_ids_[field_name] = var_id;
                    double fill_value = -9.99e33;
                    ncmpi_put_att_double(ncid_, var_id, "missing_value", NC_DOUBLE, 1, &fill_value);
                    ncmpi_put_att_text(ncid_, var_id, "units", 7, "unknown");
                }
            }, it->second);

        } 
        catch (const std::runtime_error& e) {
             if (rank_ == 0) {
                std::cerr << "Warning: Skipping definition of field '" << field_name << "'. Reason: " << e.what() << std::endl;
            }
        }
    }
    
    ncmpi_put_att_text(ncid_, NC_GLOBAL, "Conventions", 6, "COARDS");
    ncmpi_put_att_text(ncid_, NC_GLOBAL, "title", 16, "VVM Model Output");

    ret = ncmpi_enddef(ncid_);
    handle_pnetcdf_error(ret, "Failed to end define mode");

    ncmpi_begin_indep_data(ncid_);
    if (rank_ == 0) {
        if (output_nz > 0) {
            std::vector<double> z_coords(output_nz);
            for(MPI_Offset i = 0; i < output_nz; ++i) z_coords[i] = (double)(g_z_start_ + i * z_stride_);
            ncmpi_put_var_double(ncid_, z_var_id_, z_coords.data());
        }
        if (output_ny > 0) {
            std::vector<double> y_coords(output_ny);
            for (MPI_Offset i = 0; i < output_ny; ++i) y_coords[i] = (double)(g_y_start_ + i * y_stride_);
            ncmpi_put_var_double(ncid_, y_var_id_, y_coords.data());
        }
        if (output_nx > 0) {
            std::vector<double> x_coords(output_nx);
            for (MPI_Offset i = 0; i < output_nx; ++i) x_coords[i] = (double)(g_x_start_ + i * x_stride_);
            ncmpi_put_var_double(ncid_, x_var_id_, x_coords.data());
        }
    }
    ncmpi_end_indep_data(ncid_);
}

} // namespace Core
} // namespace VVM