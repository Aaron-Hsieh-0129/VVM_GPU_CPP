#include "Hdf5RestartReader.hpp"

#include "core/vvm_types.hpp"

#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace VVM {
namespace IO {

namespace {
std::string join_variable_names(const std::vector<std::string>& names) {
    if (names.empty()) return "(none)";

    std::ostringstream oss;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << names[i];
    }
    return oss.str();
}

void checked_hdf5_call(herr_t status, const std::string& message) {
    if (status < 0) throw std::runtime_error(message);
}

std::vector<hsize_t> expected_shape_for_dim(size_t dim, const Core::Grid& grid) {
    if (dim == 1) return {static_cast<size_t>(grid.get_global_points_z())};
    if (dim == 2) return {static_cast<size_t>(grid.get_global_points_y()),
                          static_cast<size_t>(grid.get_global_points_x())};
    if (dim == 3) return {static_cast<size_t>(grid.get_global_points_z()),
                          static_cast<size_t>(grid.get_global_points_y()),
                          static_cast<size_t>(grid.get_global_points_x())};
    return {};
}

hid_t open_restart_dataset(hid_t file_id, const std::string& var_name) {
    const std::string step_path = "/Step0/" + var_name;
    hid_t dataset = -1;
    H5E_BEGIN_TRY {
        dataset = H5Dopen2(file_id, step_path.c_str(), H5P_DEFAULT);
    } H5E_END_TRY;
    if (dataset >= 0) return dataset;

    H5E_BEGIN_TRY {
        dataset = H5Dopen2(file_id, ("/" + var_name).c_str(), H5P_DEFAULT);
    } H5E_END_TRY;

    if (dataset < 0) {
        throw std::runtime_error("[Hdf5RestartReader] Variable '" + var_name +
                                 "' is missing from restart file.");
    }
    return dataset;
}

hid_t native_real_type() {
#ifdef VVM_USE_DOUBLE_PRECISION
    return H5T_NATIVE_DOUBLE;
#else
    return H5T_NATIVE_FLOAT;
#endif
}
} // namespace

Hdf5RestartReader::Hdf5RestartReader(const std::string& filepath,
                                     const Core::Grid& grid,
                                     const Core::Parameters& params,
                                     const Utils::ConfigurationManager& config,
                                     Core::HaloExchanger& halo_exchanger)
    : source_file_(filepath),
      grid_(grid),
      params_(params),
      config_(config),
      halo_exchanger_(halo_exchanger),
      comm_(grid.get_cart_comm()) {
    MPI_Comm_rank(comm_, &rank_);
}

void Hdf5RestartReader::read_and_initialize(Core::State& state) {
    std::vector<std::string> vars_1d;
    std::vector<std::string> vars_2d;
    std::vector<std::string> vars_3d;

    if (config_.has_key("restart.variables_to_read.1d")) {
        vars_1d = config_.get_value<std::vector<std::string>>("restart.variables_to_read.1d");
    }
    if (config_.has_key("restart.variables_to_read.2d")) {
        vars_2d = config_.get_value<std::vector<std::string>>("restart.variables_to_read.2d");
    }
    if (config_.has_key("restart.variables_to_read.3d")) {
        vars_3d = config_.get_value<std::vector<std::string>>("restart.variables_to_read.3d");
    } else {
        vars_3d = get_variables_to_read(state);
    }

    if (vars_1d.empty() && vars_2d.empty() && vars_3d.empty()) {
        throw std::runtime_error("[Hdf5RestartReader] No restart variables selected from " + source_file_);
    }

    print_variables_to_read(vars_1d, vars_2d, vars_3d);

    hid_t file_id = H5Fopen(source_file_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("[Hdf5RestartReader] Failed to open restart file " + source_file_);
    }

    try {
        for (const auto& var_name : vars_1d) read_field(file_id, var_name, state.get_field<1>(var_name));
        for (const auto& var_name : vars_2d) read_field(file_id, var_name, state.get_field<2>(var_name));
        for (const auto& var_name : vars_3d) read_field(file_id, var_name, state.get_field<3>(var_name));
    } catch (...) {
        H5Fclose(file_id);
        throw;
    }

    H5Fclose(file_id);

    // Use per-field template exchange (bypasses CUDA graph capture on stream_).
    // The State& overload triggers graph capture for every graph-enabled field,
    // leaving NCCL cross-stream associations that get baked into later solve_w /
    // relax_2d graphs.  Those graphs then reference the pre-resize buffer
    // pointers and crash when exchange_multiple_halos resizes the buffers in the
    // first run_step.
    for (const auto& var_name : vars_1d) {
        if (state.has_field(var_name))
            halo_exchanger_.exchange_halos(state.get_field<1>(var_name));
    }
    for (const auto& var_name : vars_2d) {
        if (state.has_field(var_name))
            halo_exchanger_.exchange_halos(state.get_field<2>(var_name));
    }
    for (const auto& var_name : vars_3d) {
        if (state.has_field(var_name))
            halo_exchanger_.exchange_halos(state.get_field<3>(var_name));
    }
}

std::vector<std::string> Hdf5RestartReader::get_variables_to_read(const Core::State& state) const {
    std::vector<std::string> result;
    std::unordered_set<std::string> output_fields;
    std::vector<std::string> skipped_output_fields;

    if (config_.has_key("output.fields_to_output")) {
        for (const auto& name : config_.get_value<std::vector<std::string>>("output.fields_to_output")) {
            output_fields.insert(name);
        }
    }

    auto prognostic_config = config_.get_value<nlohmann::json>("dynamics.prognostic_variables");
    for (const auto& item : prognostic_config.items()) {
        const std::string& var_name = item.key();
        if (!state.has_field(var_name)) continue;
        if (!output_fields.empty() && output_fields.count(var_name) == 0) {
            skipped_output_fields.push_back(var_name);
            continue;
        }
        result.push_back(var_name);
    }

    if (rank_ == 0 && !skipped_output_fields.empty()) {
        std::cout << "  [Hdf5RestartReader] Skipping prognostic variables not listed in output.fields_to_output: "
                  << join_variable_names(skipped_output_fields) << std::endl;
    }
    return result;
}

void Hdf5RestartReader::print_variables_to_read(const std::vector<std::string>& vars_1d,
                                                const std::vector<std::string>& vars_2d,
                                                const std::vector<std::string>& vars_3d) const {
    if (rank_ != 0) return;

    std::cout << "  [Hdf5RestartReader] Restart variables to read from " << source_file_ << ":" << std::endl;
    std::cout << "    1D: " << join_variable_names(vars_1d) << std::endl;
    std::cout << "    2D: " << join_variable_names(vars_2d) << std::endl;
    std::cout << "    3D: " << join_variable_names(vars_3d) << std::endl;
}

template<size_t Dim>
void Hdf5RestartReader::read_field(hid_t file_id,
                                   const std::string& var_name,
                                   Core::Field<Dim>& field) const {
    hid_t dataset = open_restart_dataset(file_id, var_name);
    hid_t filespace = H5Dget_space(dataset);
    if (filespace < 0) {
        H5Dclose(dataset);
        throw std::runtime_error("[Hdf5RestartReader] Failed to inspect variable '" + var_name + "'.");
    }

    const int ndims = H5Sget_simple_extent_ndims(filespace);
    const std::vector<hsize_t> expected_shape = expected_shape_for_dim(Dim, grid_);
    if (ndims != static_cast<int>(expected_shape.size())) {
        H5Sclose(filespace);
        H5Dclose(dataset);
        throw std::runtime_error("[Hdf5RestartReader] Variable '" + var_name + "' rank does not match current field.");
    }

    std::vector<hsize_t> shape(expected_shape.size());
    H5Sget_simple_extent_dims(filespace, shape.data(), nullptr);
    if (shape != expected_shape) {
        H5Sclose(filespace);
        H5Dclose(dataset);
        throw std::runtime_error("[Hdf5RestartReader] Variable '" + var_name + "' shape does not match current grid.");
    }

    const size_t h = static_cast<size_t>(grid_.get_halo_cells());
    auto field_view_dev = field.get_mutable_device_data();
    auto field_view_host = Kokkos::create_mirror_view(field_view_dev);
    Kokkos::deep_copy(field_view_host, field_view_dev);

    if constexpr (Dim == 1) {
        const size_t nz = static_cast<size_t>(grid_.get_global_points_z());
        std::vector<VVM::Real> buffer(nz);
        hsize_t start[1] = {0};
        hsize_t count[1] = {nz};
        checked_hdf5_call(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr),
                         "[Hdf5RestartReader] Failed to select hyperslab for '" + var_name + "'.");
        hid_t memspace = H5Screate_simple(1, count, nullptr);
        checked_hdf5_call(H5Dread(dataset, native_real_type(), memspace, filespace, H5P_DEFAULT, buffer.data()),
                         "[Hdf5RestartReader] Failed to read variable '" + var_name + "'.");
        H5Sclose(memspace);
        for (size_t k = 0; k < nz; ++k) field_view_host(k + h) = buffer[k];
    } else if constexpr (Dim == 2) {
        const size_t ny = static_cast<size_t>(grid_.get_local_physical_points_y());
        const size_t nx = static_cast<size_t>(grid_.get_local_physical_points_x());
        const size_t y0 = static_cast<size_t>(grid_.get_local_physical_start_y());
        const size_t x0 = static_cast<size_t>(grid_.get_local_physical_start_x());
        std::vector<VVM::Real> buffer(ny * nx);
        hsize_t start[2] = {y0, x0};
        hsize_t count[2] = {ny, nx};
        checked_hdf5_call(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr),
                         "[Hdf5RestartReader] Failed to select hyperslab for '" + var_name + "'.");
        hid_t memspace = H5Screate_simple(2, count, nullptr);
        checked_hdf5_call(H5Dread(dataset, native_real_type(), memspace, filespace, H5P_DEFAULT, buffer.data()),
                         "[Hdf5RestartReader] Failed to read variable '" + var_name + "'.");
        H5Sclose(memspace);

        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                field_view_host(j + h, i + h) = buffer[j * nx + i];
            }
        }
    } else if constexpr (Dim == 3) {
        const size_t nz = static_cast<size_t>(grid_.get_global_points_z());
        const size_t ny = static_cast<size_t>(grid_.get_local_physical_points_y());
        const size_t nx = static_cast<size_t>(grid_.get_local_physical_points_x());
        const size_t y0 = static_cast<size_t>(grid_.get_local_physical_start_y());
        const size_t x0 = static_cast<size_t>(grid_.get_local_physical_start_x());
        std::vector<VVM::Real> buffer(nz * ny * nx);
        hsize_t start[3] = {0, y0, x0};
        hsize_t count[3] = {nz, ny, nx};
        checked_hdf5_call(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr),
                         "[Hdf5RestartReader] Failed to select hyperslab for '" + var_name + "'.");
        hid_t memspace = H5Screate_simple(3, count, nullptr);
        checked_hdf5_call(H5Dread(dataset, native_real_type(), memspace, filespace, H5P_DEFAULT, buffer.data()),
                         "[Hdf5RestartReader] Failed to read variable '" + var_name + "'.");
        H5Sclose(memspace);

        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    field_view_host(k + h, j + h, i + h) = buffer[(k * ny + j) * nx + i];
                }
            }
        }
    }

    Kokkos::deep_copy(field_view_dev, field_view_host);
    H5Sclose(filespace);
    H5Dclose(dataset);
}

template void Hdf5RestartReader::read_field<1>(hid_t, const std::string&, Core::Field<1>&) const;
template void Hdf5RestartReader::read_field<2>(hid_t, const std::string&, Core::Field<2>&) const;
template void Hdf5RestartReader::read_field<3>(hid_t, const std::string&, Core::Field<3>&) const;

} // namespace IO
} // namespace VVM
