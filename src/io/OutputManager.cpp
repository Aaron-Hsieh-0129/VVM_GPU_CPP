#include "OutputManager.hpp"
#include <sys/stat.h>
#include <cerrno>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp> 
#include <algorithm>

namespace VVM {
namespace IO {

OutputManager::OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, VVM::Core::State& state, MPI_Comm comm)
    : grid_(grid), params_(params), state_(state), comm_(comm), adios_(comm) {
    rank_ = 0;
    mpi_size_ = 1;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);

    output_dir_ = config.get_value<std::string>("output.output_dir");
    filename_prefix_ = config.get_value<std::string>("output.output_filename_prefix");
    fields_to_output_ = config.get_value<std::vector<std::string>>("output.fields_to_output");
    output_interval_s_ = config.get_value<double>("simulation.output_interval_s");
    total_time_ = config.get_value<double>("simulation.total_time_s");

    output_x_start_  = config.get_value<size_t>("output.output_grid.x_start");
    output_y_start_  = config.get_value<size_t>("output.output_grid.y_start");
    output_z_start_  = config.get_value<size_t>("output.output_grid.z_start");

    output_x_end_    = config.get_value<size_t>("output.output_grid.x_end");
    output_y_end_    = config.get_value<size_t>("output.output_grid.y_end");
    output_z_end_    = config.get_value<size_t>("output.output_grid.z_end");
    
    // Add strides, default to 1 if not specified in config
    // output_x_stride_ = config.get_value<size_t>("output.output_grid.x_stride");
    // output_y_stride_ = config.get_value<size_t>("output.output_grid.y_stride");
    // output_z_stride_ = config.get_value<size_t>("output.output_grid.z_stride");

    if (rank_ == 0) std::cout << "  [OutputManager] Config loaded. Creating directory..." << std::endl;

    if (rank_ == 0) {
        if (mkdir(output_dir_.c_str(), 0777) != 0 && errno != EEXIST) {
            perror(("Failed to create directory " + output_dir_).c_str());
        }
    }

    if (rank_ == 0) std::cout << "  [OutputManager] Directory created/checked. Waiting for MPI Barrier..." << std::endl;

    MPI_Barrier(comm_);

    if (rank_ == 0) std::cout << "  [OutputManager] Barrier passed. Writing ctl file..." << std::endl;
    if (rank_ == 0) grads_ctl_file();

    if (rank_ == 0) std::cout << "  [OutputManager] ctl file written. Initializing ADIOS2..." << std::endl;

    io_ = adios_.DeclareIO("VVM_IO");
    io_.SetEngine("HDF5");
    io_.SetParameter("IdleH5Writer",
                     "true"); // set this if not all ranks are writting
    io_.SetParameter("H5CollectiveMPIO", "yes");
    io_.SetParameter("H5_DRIVER", "MPIO");
    // io_.SetParameters({{"Threads", "4"}});

    if (rank_ == 0) std::cout << "  [OutputManager] ADIOS2 Initialized." << std::endl;
}

OutputManager::~OutputManager() {
    if (writer_) {
      writer_.Close();
    }
}

void OutputManager::define_variables() {
    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();

    std::string x_var_name = "coordinates/x";
    std::string y_var_name = "coordinates/y";
    std::string z_var_name = "coordinates/z_mid";

    // Define coordinate variables (these do not change over time)
    io_.DefineVariable<double>("time");
    io_.DefineVariable<double>(x_var_name, {gnx}, {0}, {rank_ == 0 ? gnx : 0});
    io_.DefineVariable<double>(y_var_name, {gny}, {0}, {rank_ == 0 ? gny : 0});
    io_.DefineVariable<double>(z_var_name, {gnz}, {0}, {rank_ == 0 ? gnz : 0});

    io_.DefineAttribute<std::string>("units", "hours since 2025-10-07 00:00:00", "time");
    io_.DefineAttribute<std::string>("long_name", "Time", "time");
    io_.DefineAttribute<std::string>("axis", "T", "time");

    io_.DefineAttribute<std::string>("units", "meter", z_var_name);
    io_.DefineAttribute<std::string>("long_name", "Height (grid center)", z_var_name);
    io_.DefineAttribute<std::string>("axis", "Z", z_var_name);
    
    io_.DefineAttribute<std::string>("units", "meter", y_var_name);
    io_.DefineAttribute<std::string>("long_name", "Y Direction", y_var_name);
    io_.DefineAttribute<std::string>("axis", "Y", y_var_name);
    
    io_.DefineAttribute<std::string>("units", "meter", x_var_name);
    io_.DefineAttribute<std::string>("long_name", "X direction", x_var_name);
    io_.DefineAttribute<std::string>("axis", "X", x_var_name);

    // Local physical points and offsets for current rank
    const size_t rank_lnx = grid_.get_local_physical_points_x();
    const size_t rank_lny = grid_.get_local_physical_points_y();
    const size_t rank_lnz = grid_.get_local_physical_points_z();

    const size_t rank_offset_x = grid_.get_local_physical_start_x();
    const size_t rank_offset_y = grid_.get_local_physical_start_y();
    const size_t rank_offset_z = grid_.get_local_physical_start_z();

    for (const auto& field_name : fields_to_output_) {
        auto it = state_.begin();
        while (it != state_.end() && it->first != field_name) ++it;

        if (it != state_.end()) {
            std::visit([&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    size_t actual_output_x_start = std::max(rank_offset_x, output_x_start_);
                    size_t actual_output_x_end = std::min(rank_offset_x + rank_lnx - 1, output_x_end_);
                    size_t local_output_nx = (actual_output_x_end >= actual_output_x_start) ? (actual_output_x_end - actual_output_x_start + 1) : 0;

                    size_t actual_output_y_start = std::max(rank_offset_y, output_y_start_);
                    size_t actual_output_y_end = std::min(rank_offset_y + rank_lny - 1, output_y_end_);
                    size_t local_output_ny = (actual_output_y_end >= actual_output_y_start) ? (actual_output_y_end - actual_output_y_start + 1) : 0;

                    size_t actual_output_z_start = std::max(rank_offset_z, output_z_start_);
                    size_t actual_output_z_end = std::min(rank_offset_z + rank_lnz - 1, output_z_end_);
                    size_t local_output_nz = (actual_output_z_end >= actual_output_z_start) ? (actual_output_z_end - actual_output_z_start + 1) : 0;
                    
                    if constexpr (T::DimValue == 1) {
                        size_t count = (rank_ == 0) ? local_output_nz : 0;
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz}, {actual_output_z_start}, {count});
                    }
                    else if constexpr (T::DimValue == 2) {
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gny, gnx}, {actual_output_y_start, actual_output_x_start}, {local_output_ny, local_output_nx});
                    }
                    else if constexpr (T::DimValue == 3) {
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz, gny, gnx}, {actual_output_z_start, actual_output_y_start, actual_output_x_start}, {local_output_nz, local_output_ny, local_output_nx});
                    }
                    else if constexpr (T::DimValue == 4) {
                        const size_t dim4 = field.get_device_data().extent(0);
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {dim4, gnz, gny, gnx}, {0, actual_output_z_start, actual_output_y_start, actual_output_x_start}, {dim4, local_output_nz, local_output_ny, local_output_nx});
                    }
                }
            }, it->second);
        }
    }
}

void OutputManager::write_static_data() {
    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();
    const size_t h = grid_.get_halo_cells();

    auto var_x = io_.InquireVariable<double>("coordinates/x");
    std::vector<double> x_coords;
    if (rank_ == 0) {
        x_coords.resize(gnx);
        for(size_t i = 0; i < gnx; ++i) x_coords[i] = i * grid_.get_dx();
    }
    writer_.Put<double>(var_x, x_coords.data(), adios2::Mode::Sync);

    auto var_y = io_.InquireVariable<double>("coordinates/y");
    std::vector<double> y_coords;
    if (rank_ == 0) {
        y_coords.resize(gny);
        for(size_t i = 0; i < gny; ++i) { y_coords[i] = i * grid_.get_dy(); }
    }
    writer_.Put<double>(var_y, y_coords.data(), adios2::Mode::Sync);

    auto var_z_mid = io_.InquireVariable<double>("coordinates/z_mid");
    auto z_mid_host = params_.z_mid.get_host_data();
    std::vector<double> z_mid_physical;
    if (rank_ == 0) {
        z_mid_physical.resize(gnz);
        for (size_t i = 0; i < gnz; ++i) {
            z_mid_physical[i] = z_mid_host(i + h);
        }
    }
    writer_.Put<double>(var_z_mid, z_mid_physical.data(), adios2::Mode::Sync);
}

std::string format_to_six_digits(int number) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << number;
    return ss.str();
}

void OutputManager::write(int step, double time) {
    VVM::Utils::Timer advection_x_timer("OUTPUT");

    if (rank_ == 0) std::cout << "  [OutputManager::write] Opening file for step " << step << "..." << std::endl;
    std::string filename = output_dir_ + "/" + filename_prefix_ + "_" + format_to_six_digits((int) (step/output_interval_s_)) + ".h5";
    writer_ = io_.Open(filename, adios2::Mode::Write, MPI_COMM_WORLD);

    if (rank_ == 0) std::cout << "  [OutputManager::write] File opened. Defining vars if needed..." << std::endl;

    if (!variables_defined_) {
        define_variables();
        variables_defined_ = true;
    }

    if (rank_ == 0) std::cout << "  [OutputManager::write] BeginStep..." << std::endl;
    writer_.BeginStep();
    
    adios2::Variable<double> var_time = io_.InquireVariable<double>("time");
    writer_.Put<double>(var_time, &time, adios2::Mode::Sync);
    write_static_data();

    const size_t h = grid_.get_halo_cells();

    // Local physical points and offsets for current rank
    const size_t rank_lnx = grid_.get_local_physical_points_x();
    const size_t rank_lny = grid_.get_local_physical_points_y();
    const size_t rank_lnz = grid_.get_local_physical_points_z();

    const size_t rank_offset_x = grid_.get_local_physical_start_x();
    const size_t rank_offset_y = grid_.get_local_physical_start_y();
    const size_t rank_offset_z = grid_.get_local_physical_start_z();

    // Determine actual output region for current rank based on configuration
    size_t actual_output_x_start = std::max(rank_offset_x, output_x_start_);
    size_t actual_output_x_end = std::min(rank_offset_x + rank_lnx - 1, output_x_end_);
    size_t local_output_nx = (actual_output_x_end >= actual_output_x_start) ? (actual_output_x_end - actual_output_x_start + 1) : 0;

    size_t actual_output_y_start = std::max(rank_offset_y, output_y_start_);
    size_t actual_output_y_end = std::min(rank_offset_y + rank_lny - 1, output_y_end_);
    size_t local_output_ny = (actual_output_y_end >= actual_output_y_start) ? (actual_output_y_end - actual_output_y_start + 1) : 0;

    size_t actual_output_z_start = std::max(rank_offset_z, output_z_start_);
    size_t actual_output_z_end = std::min(rank_offset_z + rank_lnz - 1, output_z_end_);
    size_t local_output_nz = (actual_output_z_end >= actual_output_z_start) ? (actual_output_z_end - actual_output_z_start + 1) : 0;


    for (const auto& field_name : fields_to_output_) {
        if (field_variables_.count(field_name)) {
            auto it = state_.begin();
            while (it != state_.end() && it->first != field_name) {
                ++it;
            }

            if (it != state_.end()) {
                auto& adios_var = field_variables_.at(field_name);

                std::visit([&](const auto& field) {
                    using T = std::decay_t<decltype(field)>;
                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        auto full_data_view = field.get_device_data();
                        
                        // Calculate local indices for Kokkos subview (relative to current rank's total view including halos)
                        // These are the starting indices within the current rank's Kokkos View where the output region begins.
                        size_t kokkos_subview_start_z = (actual_output_z_start - rank_offset_z) + h;
                        size_t kokkos_subview_start_y = (actual_output_y_start - rank_offset_y) + h;
                        size_t kokkos_subview_start_x = (actual_output_x_start - rank_offset_x) + h;

                        if constexpr (T::DimValue == 1) {
                            // Only rank 0 outputs 1D global variables
                            const double* data_ptr = nullptr;
                            Kokkos::View<double*, Kokkos::HostSpace> host_view;
                            if (rank_ == 0) {
                                // Create a host mirror for the subview data
                                auto subview_from_full = Kokkos::subview(full_data_view, 
                                                                         std::make_pair(kokkos_subview_start_z, kokkos_subview_start_z + local_output_nz));
                                host_view = Kokkos::create_mirror_view(subview_from_full);
                                Kokkos::deep_copy(host_view, subview_from_full);
                                data_ptr = host_view.data();
                                // auto phys_data_subview_cpu = Kokkos::create_mirror_view(subview_from_full);
                                // Kokkos::deep_copy(phys_data_subview_cpu, subview_from_full);
                                // writer_.Put<double>(adios_var, phys_data_subview_cpu.data());
                            }
                            writer_.Put<double>(adios_var, data_ptr, adios2::Mode::Sync);
                        }
                        else if constexpr (T::DimValue == 2) {
                            Kokkos::View<double**, Kokkos::LayoutRight> phys_data_subview("phys_data_2d_sub", local_output_ny, local_output_nx);
                            auto subview_from_full = Kokkos::subview(full_data_view, 
                                                           std::make_pair(kokkos_subview_start_y, kokkos_subview_start_y + local_output_ny), 
                                                           std::make_pair(kokkos_subview_start_x, kokkos_subview_start_x + local_output_nx));
                            Kokkos::deep_copy(phys_data_subview, subview_from_full);
                            auto phys_data_subview_host = Kokkos::create_mirror_view(phys_data_subview);
                            Kokkos::deep_copy(phys_data_subview_host, phys_data_subview);
                            writer_.Put<double>(adios_var, phys_data_subview_host.data());
                        } 
                        else if constexpr (T::DimValue == 3) {
                            Kokkos::View<double***, Kokkos::LayoutRight> phys_data_subview("phys_data_3d_sub", local_output_nz, local_output_ny, local_output_nx);
                            auto subview_from_full = Kokkos::subview(full_data_view,
                                                           std::make_pair(kokkos_subview_start_z, kokkos_subview_start_z + local_output_nz),
                                                           std::make_pair(kokkos_subview_start_y, kokkos_subview_start_y + local_output_ny),
                                                           std::make_pair(kokkos_subview_start_x, kokkos_subview_start_x + local_output_nx));
                            Kokkos::deep_copy(phys_data_subview, subview_from_full);
                            auto phys_data_subview_host = Kokkos::create_mirror_view(phys_data_subview);
                            Kokkos::deep_copy(phys_data_subview_host, phys_data_subview);
                            writer_.Put<double>(adios_var, phys_data_subview_host.data());
                        }
                        else if constexpr (T::DimValue == 4) {
                            const size_t dim4 = full_data_view.extent(0); // This dimension is not affected by spatial decomposition
                            Kokkos::View<double****, Kokkos::LayoutRight> phys_data_subview("phys_data_4d_sub", dim4, local_output_nz, local_output_ny, local_output_nx);
                            auto subview_from_full = Kokkos::subview(full_data_view,
                                                           Kokkos::ALL(), // All of the first dimension
                                                           std::make_pair(kokkos_subview_start_z, kokkos_subview_start_z + local_output_nz),
                                                           std::make_pair(kokkos_subview_start_y, kokkos_subview_start_y + local_output_ny),
                                                           std::make_pair(kokkos_subview_start_x, kokkos_subview_start_x + local_output_nx));
                            Kokkos::deep_copy(phys_data_subview, subview_from_full);
                            auto phys_data_subview_host = Kokkos::create_mirror_view(phys_data_subview);
                            Kokkos::deep_copy(phys_data_subview_host, phys_data_subview);
                            writer_.Put<double>(adios_var, phys_data_subview_host.data());
                        }
                    }
                }, it->second);
            }
        }
    }

    if (rank_ == 0) std::cout << "  [OutputManager::write] Puts done. EndStep (Wait for I/O)..." << std::endl;
    writer_.EndStep();
    writer_.Close();
    return;
}

void OutputManager::grads_ctl_file() {
    // Open output file
    std::ofstream outFile(output_dir_ + "/vvm.ctl");
    if (!outFile.is_open()) {
        std::cerr << "Error opening ctl file!" << std::endl;
        return;
    }

    auto z_mid_host = params_.z_mid.get_host_data();
    auto h = grid_.get_halo_cells();
    auto nz_phy = grid_.get_global_points_z();
    double dt = params_.get_value_host(params_.dt);

    // Write the .ctl file content
    outFile << "DSET ^" << filename_prefix_ << "_%tm6.h5\n";
    outFile << "DTYPE hdf5_grid\n";
    outFile << "OPTIONS template\n";
    outFile << "TITLE VVM_GPU_CPP\n";
    outFile << "UNDEF -9999.0\n";
    outFile << "XDEF " << grid_.get_global_points_x() << " LINEAR 0 1\n";
    outFile << "YDEF " << grid_.get_global_points_y() << " LINEAR 0 1\n";
    outFile << "ZDEF " << grid_.get_global_points_z() << " LEVELS ";
    for (int k = h; k < h+nz_phy; k++) {
        outFile << static_cast<int> (z_mid_host(k));
        if (k < nz_phy+h-1) outFile << ", ";
    }
    outFile << "\n";
    outFile << "TDEF " << (int) (total_time_ / (dt*output_interval_s_)+1) << " LINEAR 00:00Z01JAN2000 " << "1hr\n";
    outFile << "\n";

    // int outnum = 7; // xi,eta,zeta,u,v,w,th
    //
    // outFile << "VARS " << outnum << "\n";
    // outFile << "/Step0/th=>th " << nz_phy << " z,y,x theta\n";
    // outFile << "/Step0/u=>u " << nz_phy << " z,y,x u\n";
    // outFile << "/Step0/v=>v " << nz_phy << " z,y,x v\n";
    // outFile << "/Step0/w=>w " << nz_phy << " z,y,x w\n";
    // outFile << "/Step0/eta=>eta " << nz_phy << " z,y,x eta\n";
    // outFile << "/Step0/xi=>xi " << nz_phy << " z,y,x xi\n";
    // outFile << "/Step0/zeta=>zeta " << nz_phy << " z,y,x zeta\n";
    // outFile << "ubarTop=>ubarTop 1 t ubarTop\n";

    int valid_vars_count = 0;
    std::vector<std::string> lines_to_write;
    for (const auto& field_name : fields_to_output_) {
        auto it = state_.begin();
        while (it != state_.end() && it->first != field_name) ++it;
        if (it != state_.end()) {
            bool found_dim = false;
            std::visit([&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    std::stringstream ss;
                    if constexpr (T::DimValue == 3 || T::DimValue == 4) {
                        ss << "/Step0/" << field_name << "=>" << field_name << " " << nz_phy << " z,y,x " << field_name << "\n";
                        found_dim = true;
                    }
                    else if constexpr (T::DimValue == 2) {
                        ss << "/Step0/" << field_name << "=>" << field_name << " 0 y,x " << field_name << "\n";
                        found_dim = true;
                    }
                    else if constexpr (T::DimValue == 1) {
                        ss << "/Step0/" << field_name << "=>" << field_name << " " << nz_phy << " z " << field_name << "\n";
                        found_dim = true;
                    }
                    // else {
                    //     
                    // }

                    if (found_dim) {
                        lines_to_write.push_back(ss.str());
                        valid_vars_count++;
                    }
                }
            }, it->second);
        }
    }

    outFile << "VARS " << valid_vars_count << "\n";
    for(const auto& line : lines_to_write) {
        outFile << line;
    }
    outFile << "ENDVARS\n";
    // Close the file
    outFile.close();

    // Open topo output file
    std::ofstream outtopoFile(output_dir_ + "/topo.ctl");
    if (!outtopoFile.is_open()) {
        std::cerr << "Error opening topo ctl file!" << std::endl;
        return;
    }

    // Write the .ctl file content
    outtopoFile << "DSET ^" << "topo.h5\n";
    outtopoFile << "DTYPE hdf5_grid\n";
    outtopoFile << "OPTIONS template\n";
    outtopoFile << "TITLE VVM_GPU_CPP\n";
    outtopoFile << "UNDEF -9999.0\n";
    outtopoFile << "XDEF " << grid_.get_global_points_x() << " LINEAR 0 1\n";
    outtopoFile << "YDEF " << grid_.get_global_points_y() << " LINEAR 0 1\n";
    outtopoFile << "ZDEF " << grid_.get_global_points_z() << " LEVELS ";
    for (int k = h; k < h+nz_phy; k++) {
        outtopoFile << static_cast<int> (z_mid_host(k));
        if (k < nz_phy+h-1) outtopoFile << ", ";
    }
    outtopoFile << "\n";
    outtopoFile << "TDEF 1 LINEAR 00:00Z01JAN2000 1hr\n";

    outtopoFile << "VARS " << 1 << "\n";
    outtopoFile << "/Step0/topo=>topo 0 y,x topo\n";
    outtopoFile << "ENDVARS\n";

    // Close the file
    outtopoFile.close();
    return;
}

void OutputManager::write_static_topo_file() {
    if (rank_ == 0) {
        std::cout << "Writing static topography file..." << std::endl;
    }

    adios2::IO topo_io = adios_.DeclareIO("TOPO_IO");
    topo_io.SetEngine("HDF5");
    topo_io.SetParameter("IdleH5Writer", "true");
    topo_io.SetParameter("H5CollectiveMPIO", "yes");
    topo_io.SetParameter("H5_DRIVER", "MPIO");

    std::string filename = output_dir_ + "/topo.h5";
    adios2::Engine topo_writer = topo_io.Open(filename, adios2::Mode::Write, MPI_COMM_WORLD);

    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();
    const size_t h = grid_.get_halo_cells();

    const size_t rank_lnx = grid_.get_local_physical_points_x();
    const size_t rank_lny = grid_.get_local_physical_points_y();
    const size_t rank_offset_x = grid_.get_local_physical_start_x();
    const size_t rank_offset_y = grid_.get_local_physical_start_y();

    auto var_x = topo_io.DefineVariable<double>("coordinates/x", {gnx}, {0}, {rank_ == 0 ? gnx : 0});
    auto var_y = topo_io.DefineVariable<double>("coordinates/y", {gny}, {0}, {rank_ == 0 ? gny : 0});
    auto var_z = topo_io.DefineVariable<double>("coordinates/z_mid", {gnz}, {0}, {rank_ == 0 ? gnz : 0});

    auto var_topo = topo_io.DefineVariable<double>("topo", {gny, gnx},
                                                 {rank_offset_y, rank_offset_x},
                                                 {rank_lny, rank_lnx});
    topo_io.DefineAttribute<std::string>("units", "meter", var_topo.Name());
    topo_io.DefineAttribute<std::string>("long_name", "Topography Height", var_topo.Name());

    topo_writer.BeginStep();

    if (rank_ == 0) {
        std::vector<double> x_coords(gnx);
        for(size_t i = 0; i < gnx; ++i) { x_coords[i] = i * grid_.get_dx(); }
        topo_writer.Put<double>(var_x, x_coords.data(), adios2::Mode::Sync);

        std::vector<double> y_coords(gny);
        for(size_t i = 0; i < gny; ++i) { y_coords[i] = i * grid_.get_dy(); }
        topo_writer.Put<double>(var_y, y_coords.data(), adios2::Mode::Sync);

        auto z_mid_host = params_.z_mid.get_host_data();
        std::vector<double> z_mid_physical(gnz);
        for (size_t i = 0; i < gnz; ++i) {
            z_mid_physical[i] = z_mid_host(i + h);
        }
        topo_writer.Put<double>(var_z, z_mid_physical.data(), adios2::Mode::Sync);
    }

    try {
        const auto& topo_field = state_.get_field<2>("topo");
        auto topo_data_view = topo_field.get_device_data();

        Kokkos::View<double**, Kokkos::LayoutRight> topo_phys_subview("topo_phys_subview", rank_lny, rank_lnx);
        auto subview_from_full = Kokkos::subview(topo_data_view, 
                                               std::make_pair(h, h + rank_lny), 
                                               std::make_pair(h, h + rank_lnx));
        Kokkos::deep_copy(topo_phys_subview, subview_from_full);
        auto topo_phys_host = Kokkos::create_mirror_view(topo_phys_subview);
        Kokkos::deep_copy(topo_phys_host, topo_phys_subview);
        
        topo_writer.Put<double>(var_topo, topo_phys_host.data());
    } 
    catch (const std::exception& e) {
        if (rank_ == 0) {
            std::cerr << "Warning: Could not write 'topo' variable to static file. Reason: " << e.what() << std::endl;
        }
    }

    topo_writer.EndStep();
    topo_writer.Close();

    if (rank_ == 0) {
        std::cout << "Static topography file '" << filename << "' written successfully." << std::endl;
    }
}


} // namespace IO
} // namespace VVM
