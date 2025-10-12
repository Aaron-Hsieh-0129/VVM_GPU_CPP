#include "OutputManager.hpp"
#include <sys/stat.h>
#include <cerrno>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp> 
#include <algorithm>

namespace VVM {
namespace IO {

OutputManager::OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, MPI_Comm comm)
    : grid_(grid), params_(params), comm_(comm), adios_(comm) {
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

    if (rank_ == 0) {
        if (mkdir(output_dir_.c_str(), 0777) != 0 && errno != EEXIST) {
            perror(("Failed to create directory " + output_dir_).c_str());
        }
    }
    MPI_Barrier(comm_);
    grads_ctl_file();

    io_ = adios_.DeclareIO("VVM_IO");
    io_.SetEngine("HDF5");
    io_.SetParameter("IdleH5Writer",
                     "true"); // set this if not all ranks are writting
    io_.SetParameters({{"Threads", "4"}});
}

OutputManager::~OutputManager() {
    if (writer_) {
      writer_.Close();
    }
}

void OutputManager::define_variables(const VVM::Core::State& state) {
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
        auto it = state.begin();
        while (it != state.end() && it->first != field_name) ++it;

        if (it != state.end()) {
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
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz}, {actual_output_z_start}, {local_output_nz});
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
    std::vector<double> x_coords(gnx);
    for(size_t i = 0; i < gnx; ++i) { x_coords[i] = i * grid_.get_dx(); }
    writer_.Put<double>(var_x, x_coords.data(), adios2::Mode::Sync);

    auto var_y = io_.InquireVariable<double>("coordinates/y");
    std::vector<double> y_coords(gny);
    for(size_t i = 0; i < gny; ++i) { y_coords[i] = i * grid_.get_dy(); }
    writer_.Put<double>(var_y, y_coords.data(), adios2::Mode::Sync);

    auto var_z_mid = io_.InquireVariable<double>("coordinates/z_mid");
    auto z_mid_host = params_.z_mid.get_host_data();
    std::vector<double> z_mid_physical(gnz);
    for (size_t i = 0; i < gnz; ++i) {
        z_mid_physical[i] = z_mid_host(i + h);
    }
    writer_.Put<double>(var_z_mid, z_mid_physical.data(), adios2::Mode::Sync);
}

std::string format_to_six_digits(int number) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << number;
    return ss.str();
}

void OutputManager::write(const VVM::Core::State& state, int step, double time) {
    VVM::Utils::Timer advection_x_timer("OUTPUT");

    std::string filename = output_dir_ + "/" + filename_prefix_ + "_" + format_to_six_digits((int) (step/output_interval_s_)) + ".h5";
    writer_ = io_.Open(filename, adios2::Mode::Write);

    if (!variables_defined_) {
        define_variables(state);
        variables_defined_ = true;
    }

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
            auto it = state.begin();
            while (it != state.end() && it->first != field_name) {
                ++it;
            }

            if (it != state.end()) {
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
                            if (rank_ == 0) {
                                // Create a host mirror for the subview data
                                auto subview_from_full = Kokkos::subview(full_data_view, 
                                                                         std::make_pair(kokkos_subview_start_z, kokkos_subview_start_z + local_output_nz));
                                auto phys_data_subview_cpu = Kokkos::create_mirror_view(subview_from_full);
                                Kokkos::deep_copy(phys_data_subview_cpu, subview_from_full);
                                writer_.Put<double>(adios_var, phys_data_subview_cpu.data());
                            }
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

    int outnum = 7; // xi,eta,zeta,u,v,w,th

    outFile << "TDEF " << (int) (total_time_ / (dt*output_interval_s_)+1) << " LINEAR 00:00Z01JAN2000 " << "1hr\n";
    outFile << "\n";
    outFile << "VARS " << outnum << "\n";
    outFile << "/Step0/th=>th " << nz_phy << " z,y,x theta\n";
    outFile << "/Step0/u=>u " << nz_phy << " z,y,x u\n";
    outFile << "/Step0/v=>v " << nz_phy << " z,y,x v\n";
    outFile << "/Step0/w=>w " << nz_phy << " z,y,x w\n";
    outFile << "/Step0/eta=>eta " << nz_phy << " z,y,x eta\n";
    outFile << "/Step0/xi=>xi " << nz_phy << " z,y,x xi\n";
    outFile << "/Step0/zeta=>zeta " << nz_phy << " z,y,x zeta\n";
    // outFile << "ubarTop=>ubarTop 1 t ubarTop\n";
    outFile << "ENDVARS\n";

    // Close the file
    outFile.close();
    return;
}

} // namespace IO
} // namespace VVM
