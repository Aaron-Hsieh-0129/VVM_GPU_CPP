#include "PnetcdfReader.hpp"
#include "core/Field.hpp"
#include <iostream>
#include <vector>

namespace VVM {
namespace IO {

void PnetcdfReader::check_ncmpi_error(int status, const std::string& msg) const {
    if (status != NC_NOERR) {
        std::string err_msg = msg + ": " + ncmpi_strerror(status);
        if (rank_ == 0) {
            std::cerr << "PnetCDF Error: " << err_msg << std::endl;
        }
        MPI_Abort(comm_, status);
        throw std::runtime_error(err_msg);
    }
}

PnetcdfReader::PnetcdfReader(const std::string& filepath, 
                             const VVM::Core::Grid& grid, 
                             const VVM::Core::Parameters& params, 
                             const VVM::Utils::ConfigurationManager& config, 
                             Core::HaloExchanger& halo_exchanger) 
    : source_file_(filepath), 
      grid_(grid), 
      params_(params), 
      config_(config),
      comm_(grid.get_cart_comm()),
      ncid_(-1), 
      halo_exchanger_(halo_exchanger) {
    MPI_Comm_rank(comm_, &rank_);
}

PnetcdfReader::~PnetcdfReader() {
    if (ncid_ != -1) {
        ncmpi_close(ncid_);
    }
}

std::map<std::string, MPI_Offset> PnetcdfReader::get_file_dimensions(int ncid) const {
    std::map<std::string, MPI_Offset> dims;
    int num_dims;
    check_ncmpi_error(ncmpi_inq_ndims(ncid, &num_dims), "Failed to get number of dimensions");

    for (int dimid = 0; dimid < num_dims; ++dimid) {
        char dim_name[NC_MAX_NAME + 1];
        MPI_Offset dim_len;
        check_ncmpi_error(ncmpi_inq_dim(ncid, dimid, dim_name, &dim_len), "Failed to inquire dimension");
        dims[std::string(dim_name)] = dim_len;
    }
    return dims;
}

void PnetcdfReader::validate_dimensions(const std::map<std::string, MPI_Offset>& file_dims) const {
    const auto& it_z = file_dims.find("nz");
    if (it_z == file_dims.end()) {
        throw std::runtime_error("NetCDF file is missing 'nz' dimension.");
    }
    if (it_z->second != grid_.get_global_points_z()) {
        throw std::runtime_error("NetCDF 'nz' dimension (" + std::to_string(it_z->second) + 
                               ") does not match grid configuration (" + std::to_string(grid_.get_global_points_z()) + ").");
    }

    const auto& it_y = file_dims.find("ny");
    if (it_y == file_dims.end()) {
        throw std::runtime_error("NetCDF file is missing 'ny' dimension.");
    }
    if (it_y->second != grid_.get_global_points_y()) {
        throw std::runtime_error("NetCDF 'ny' dimension (" + std::to_string(it_y->second) + 
                               ") does not match grid configuration (" + std::to_string(grid_.get_global_points_y()) + ").");
    }

    const auto& it_x = file_dims.find("nx");
    if (it_x == file_dims.end()) {
        throw std::runtime_error("NetCDF file is missing 'nx' dimension.");
    }
    if (it_x->second != grid_.get_global_points_x()) {
        throw std::runtime_error("NetCDF 'nx' dimension (" + std::to_string(it_x->second) + 
                               ") does not match grid configuration (" + std::to_string(grid_.get_global_points_x()) + ").");
    }
}

template<size_t Dim>
void PnetcdfReader::read_variable_1d(int ncid, const std::string& var_name, VVM::Core::Field<Dim>& field) {
    static_assert(Dim == 1, "read_variable_1d only works for 1D fields");

    int varid;
    int status = ncmpi_inq_varid(ncid, var_name.c_str(), &varid);
    if (status != NC_NOERR) {
        if (rank_ == 0) std::cerr << "Warning: Cannot find 1D variable '" << var_name << "' in NetCDF file. Skipping." << std::endl;
        return;
    }

    MPI_Offset start[1] = {0};
    MPI_Offset count[1] = { static_cast<MPI_Offset>(grid_.get_global_points_z()) };

    // Create buffers for halo
    std::vector<double> host_buffer(count[0]);

    // Every rank get complete 1D array
    check_ncmpi_error(ncmpi_get_vara_double_all(ncid, varid, start, count, host_buffer.data()),
                      "Failed to read 1D variable: " + var_name);

    // Copy to State Field
    auto field_view_dev = field.get_mutable_device_data();
    auto field_view_host = Kokkos::create_mirror_view(field_view_dev);
    
    const int h = grid_.get_halo_cells();
    for (size_t k = 0; k < count[0]; ++k) {
        field_view_host(k + h) = host_buffer[k];
    }
    
    Kokkos::deep_copy(field_view_dev, field_view_host);
}

template<size_t Dim>
void PnetcdfReader::read_variable_2d(int ncid, const std::string& var_name, VVM::Core::Field<Dim>& field) {
    static_assert(Dim == 2, "read_variable_2d only works for 2D fields");

    int varid;
    int status = ncmpi_inq_varid(ncid, var_name.c_str(), &varid);
    if (status != NC_NOERR) {
        if (rank_ == 0) std::cerr << "Warning: Cannot find 2D variable '" << var_name << "' in NetCDF file. Skipping." << std::endl;
        return;
    }

    MPI_Offset start[2];
    MPI_Offset count[2];

    start[0] = grid_.get_local_physical_start_y();
    count[0] = grid_.get_local_physical_points_y();

    start[1] = grid_.get_local_physical_start_x();
    count[1] = grid_.get_local_physical_points_x();

    size_t local_read_size = count[0] * count[1];
    std::vector<double> host_buffer(local_read_size);

    check_ncmpi_error(ncmpi_get_vara_double_all(ncid, varid, start, count, host_buffer.data()),
                      "Failed to read 2D variable: " + var_name);

    auto field_view_dev = field.get_mutable_device_data();
    auto field_view_host = Kokkos::create_mirror_view(field_view_dev);

    const int h = grid_.get_halo_cells();
    const int ny_in = static_cast<int>(count[0]);
    const int nx_in = static_cast<int>(count[1]);

    using HostExec = Kokkos::DefaultHostExecutionSpace;

    Kokkos::parallel_for("Init_Host_Buffer_2D",
        Kokkos::MDRangePolicy<HostExec, Kokkos::Rank<2>>({0, 0}, {ny_in, nx_in}),
        [=](const int j, const int i) {
            size_t flat_idx = static_cast<size_t>(j) * nx_in + static_cast<size_t>(i);
            field_view_host(j + h, i + h) = host_buffer[flat_idx];
        }
    );

    Kokkos::deep_copy(field_view_dev, field_view_host);
}


template<size_t Dim>
void PnetcdfReader::read_variable_3d(int ncid, const std::string& var_name, VVM::Core::Field<Dim>& field) {
    static_assert(Dim == 3, "read_variable_3d only works for 3D fields");

    int varid;
    int status = ncmpi_inq_varid(ncid, var_name.c_str(), &varid);
    if (status != NC_NOERR) {
        if (rank_ == 0) std::cerr << "Warning: Cannot find 3D variable '" << var_name << "' in NetCDF file. Skipping." << std::endl;
        return;
    }

    // (nz, ny, nx) input 
    MPI_Offset start[3];
    MPI_Offset count[3];

    start[0] = 0;
    count[0] = grid_.get_global_points_z();

    start[1] = grid_.get_local_physical_start_y();
    count[1] = grid_.get_local_physical_points_y();

    start[2] = grid_.get_local_physical_start_x();
    count[2] = grid_.get_local_physical_points_x();

    size_t local_read_size = count[0] * count[1] * count[2];
    std::vector<double> host_buffer(local_read_size);

    check_ncmpi_error(ncmpi_get_vara_double_all(ncid, varid, start, count, host_buffer.data()),
                      "Failed to read 3D variable: " + var_name);

    auto field_view_dev = field.get_mutable_device_data();
    auto field_view_host = Kokkos::create_mirror_view(field_view_dev);
    
    const int h = grid_.get_halo_cells();
    const int nz_in = static_cast<int>(count[0]);
    const int ny_in = static_cast<int>(count[1]);
    const int nx_in = static_cast<int>(count[2]);
    using HostExec = Kokkos::DefaultHostExecutionSpace;


    Kokkos::parallel_for("Init_Host_Buffer_3D",
        Kokkos::MDRangePolicy<HostExec, Kokkos::Rank<3>>({0, 0, 0}, {nz_in, ny_in, nx_in}),
        [=](const int k, const int j, const int i) {
            size_t flat_idx = static_cast<size_t>(k) * ny_in * nx_in + 
                              static_cast<size_t>(j) * nx_in + 
                              static_cast<size_t>(i);
            
            field_view_host(k + h, j + h, i + h) = host_buffer[flat_idx];
        }
    );

    Kokkos::deep_copy(field_view_dev, field_view_host);
}

void PnetcdfReader::read_and_initialize(VVM::Core::State& state) {
    if (rank_ == 0) {
        std::cout << "PnetcdfReader: Initializing state from NetCDF file: " << source_file_ << std::endl;
    }

    int status = ncmpi_open(comm_, source_file_.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid_);
    check_ncmpi_error(status, "Failed to open NetCDF file: " + source_file_);

    try {
        auto file_dims = get_file_dimensions(ncid_);
        validate_dimensions(file_dims);
    } 
    catch (const std::exception& e) {
        ncmpi_close(ncid_);
        ncid_ = -1;
        throw; 
    }

    std::vector<std::string> vars_1d;
    std::string key_1d = "netcdf_reader.variables_to_read.1d";
    if (config_.has_key(key_1d)) {
        vars_1d = config_.get_value<std::vector<std::string>>(key_1d);
        
        if (rank_ == 0) std::cout << "  - Attempting to load 1D variables from config key '" << key_1d << "'..." << std::endl;
        for (const auto& var_name : vars_1d) {
            try {
                read_variable_1d(ncid_, var_name, state.get_field<1>(var_name));
                if (rank_ == 0) std::cout << "    - Loaded 1D variable: " << var_name << std::endl;
            } 
            catch (const std::runtime_error& e) {
                if (rank_ == 0) std::cerr << "    - Warning for 1D var '" << var_name << "': " << e.what() << std::endl;
            }
        }
    } 
    else {
        if (rank_ == 0) std::cerr << "Warning: Config key '" << key_1d << "' not found. No 1D variables will be read by PnetcdfReader." << std::endl;
    }

    std::vector<std::string> vars_2d;
    std::string key_2d = "netcdf_reader.variables_to_read.2d";
    if (config_.has_key(key_2d)) {
        vars_2d = config_.get_value<std::vector<std::string>>(key_2d);
        
        if (rank_ == 0) std::cout << "  - Attempting to load 2D variables from config key '" << key_2d << "'..." << std::endl;
        for (const auto& var_name : vars_2d) {
            try {
                read_variable_2d(ncid_, var_name, state.get_field<2>(var_name));
                if (rank_ == 0) std::cout << "    - Loaded 2D variable: " << var_name << std::endl;
            } 
            catch (const std::runtime_error& e) {
                if (rank_ == 0) std::cerr << "    - Warning for 2D var '" << var_name << "': " << e.what() << std::endl;
            }
        }
    } 
    else {
        if (rank_ == 0) std::cerr << "Warning: Config key '" << key_2d << "' not found. No 2D variables will be read by PnetcdfReader." << std::endl;
    }


    std::vector<std::string> vars_3d;
    std::string key_3d = "netcdf_reader.variables_to_read.3d";
    if (config_.has_key(key_3d)) {
        vars_3d = config_.get_value<std::vector<std::string>>(key_3d);

        if (rank_ == 0) std::cout << "  - Attempting to load 3D variables from config key '" << key_3d << "'..." << std::endl;
        for (const auto& var_name : vars_3d) {
            try {
                read_variable_3d(ncid_, var_name, state.get_field<3>(var_name));
                if (rank_ == 0) std::cout << "    - Loaded 3D variable: " << var_name << std::endl;
            } 
            catch (const std::runtime_error& e) {
                if (rank_ == 0) std::cerr << "    - Warning for 3D var '" << var_name << "': " << e.what() << std::endl;
            }
        }
    } 
    else {
         if (rank_ == 0) std::cerr << "Warning: Config key '" << key_3d << "' not found. No 3D variables will be read by PnetcdfReader." << std::endl;
    }

    
    check_ncmpi_error(ncmpi_close(ncid_), "Failed to close NetCDF file");
    ncid_ = -1;

    if (rank_ == 0) {
        std::cout << "Finished loading from NetCDF file." << std::endl;
    }
    halo_exchanger_.exchange_halos(state);
}

} // namespace IO
} // namespace VVM
