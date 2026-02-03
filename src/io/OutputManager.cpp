#include "OutputManager.hpp"
#include <sys/stat.h>
#include <cerrno>
#include <algorithm>
#include <unordered_set>

namespace VVM {
namespace IO {

std::string OutputManager::format_to_six_digits(int number) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << number;
    return ss.str();
}

OutputManager::OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, VVM::Core::State& state, MPI_Comm comm)
    : grid_(grid), params_(params), state_(state), comm_(comm), adios_(comm) {
    
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);

    output_dir_ = config.get_value<std::string>("output.output_dir");
    filename_prefix_ = config.get_value<std::string>("output.output_filename_prefix");
    fields_to_output_ = config.get_value<std::vector<std::string>>("output.fields_to_output");
    output_interval_s_ = config.get_value<double>("simulation.output_interval_s");
    total_time_ = config.get_value<double>("simulation.total_time_s");

    // Default to HDF5 if not specified
    if (config.has_key("output.engine")) {
        engine_type_ = config.get_value<std::string>("output.engine");
    } 
    else {
        engine_type_ = "HDF5"; 
    }

    output_x_start_  = config.get_value<size_t>("output.output_grid.x_start");
    output_y_start_  = config.get_value<size_t>("output.output_grid.y_start");
    output_z_start_  = config.get_value<size_t>("output.output_grid.z_start");

    output_x_end_    = config.get_value<size_t>("output.output_grid.x_end");
    output_y_end_    = config.get_value<size_t>("output.output_grid.y_end");
    output_z_end_    = config.get_value<size_t>("output.output_grid.z_end");

    if (rank_ == 0) {
        if (!output_dir_.empty()) {
            std::string cmd = "mkdir -p " + output_dir_;
            system(cmd.c_str());
        }
    }

    MPI_Barrier(comm_);
    if (rank_ == 0) grads_ctl_file();

    io_ = adios_.DeclareIO("VVM_IO");
    io_.SetEngine(engine_type_);

    if (engine_type_ == "HDF5") {
        // Multi-node Collective Detection
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
        int node_rank;
        MPI_Comm_rank(nodeComm, &node_rank);
        int is_node_head = (node_rank == 0) ? 1 : 0;
        int total_nodes = 0;
        MPI_Allreduce(&is_node_head, &total_nodes, 1, MPI_INT, MPI_SUM, comm_);
        MPI_Comm_free(&nodeComm);

        std::string use_collective = (total_nodes > 1) ? "true" : "false";
        if (rank_ == 0) std::cout << "  [OutputManager] Engine: HDF5. Collective: " << use_collective << std::endl;
        
        io_.SetParameter("IdleH5Writer", "true");
        io_.SetParameter("H5CollectiveMPIO", use_collective);
    } 
    else if (engine_type_ == "SST") {
        int queue_limit = config.get_value<int>("output.queue_limit", 20);
        std::string data_transport = config.get_value<std::string>("output.data_transport", "RDMA");
        if (rank_ == 0) std::cout << "  [OutputManager] Engine: SST (BP5)." << std::endl;
        io_.SetParameter("MarshalMethod", "BP5"); 
        io_.SetParameter("DataTransport", data_transport);
        io_.SetParameter("QueueLimit", std::to_string(queue_limit)); 
    }
}

OutputManager::~OutputManager() {
    if (writer_) writer_.Close();
}

void OutputManager::define_variables() {
    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();

    io_.DefineVariable<double>("time");
    io_.DefineVariable<double>("coordinates/x", {gnx}, {0}, {rank_ == 0 ? gnx : 0});
    io_.DefineVariable<double>("coordinates/y", {gny}, {0}, {rank_ == 0 ? gny : 0});
    io_.DefineVariable<double>("coordinates/z_mid", {gnz}, {0}, {rank_ == 0 ? gnz : 0});

    io_.DefineAttribute<std::string>("units", "hours since 2025-10-07 00:00:00", "time");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/z_mid");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/y");
    io_.DefineAttribute<std::string>("units", "meter", "coordinates/x");

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
                    size_t actual_out_x_start = std::max(rank_offset_x, output_x_start_);
                    size_t actual_out_x_end = std::min(rank_offset_x + rank_lnx - 1, output_x_end_);
                    size_t local_nx = (actual_out_x_end >= actual_out_x_start) ? (actual_out_x_end - actual_out_x_start + 1) : 0;

                    size_t actual_out_y_start = std::max(rank_offset_y, output_y_start_);
                    size_t actual_out_y_end = std::min(rank_offset_y + rank_lny - 1, output_y_end_);
                    size_t local_ny = (actual_out_y_end >= actual_out_y_start) ? (actual_out_y_end - actual_out_y_start + 1) : 0;

                    size_t actual_out_z_start = std::max(rank_offset_z, output_z_start_);
                    size_t actual_out_z_end = std::min(rank_offset_z + rank_lnz - 1, output_z_end_);
                    size_t local_nz = (actual_out_z_end >= actual_out_z_start) ? (actual_out_z_end - actual_out_z_start + 1) : 0;
                    
                    if constexpr (T::DimValue == 1) {
                        size_t count = (rank_ == 0) ? local_nz : 0;
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz}, {actual_out_z_start}, {count});
                    }
                    else if constexpr (T::DimValue == 2) {
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gny, gnx}, {actual_out_y_start, actual_out_x_start}, {local_ny, local_nx});
                    }
                    else if constexpr (T::DimValue == 3) {
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz, gny, gnx}, {actual_out_z_start, actual_out_y_start, actual_out_x_start}, {local_nz, local_ny, local_nx});
                    }
                    else if constexpr (T::DimValue == 4) {
                        const size_t dim4 = field.get_device_data().extent(0);
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {dim4, gnz, gny, gnx}, {0, actual_out_z_start, actual_out_y_start, actual_out_x_start}, {dim4, local_nz, local_ny, local_nx});
                    }
                }
            }, it->second);
        }
    }
}

void OutputManager::write(int step, double time) {
    if (!variables_defined_) {
        define_variables();
        variables_defined_ = true;
    }

    if (engine_type_ == "HDF5") {
        std::string filename = output_dir_ + "/" + filename_prefix_ + "_" + format_to_six_digits((int) (time/output_interval_s_)) + ".h5";
        if (rank_ == 0) std::cout << "  [OutputManager] HDF5 Writing: " << filename << std::endl;
        writer_ = io_.Open(filename, adios2::Mode::Write, comm_);
    } 
    else if (engine_type_ == "SST") {
        if (!writer_) {
            if (rank_ == 0) std::cout << "  [OutputManager] SST Streaming: " << filename_prefix_ << std::endl;
            writer_ = io_.Open(filename_prefix_, adios2::Mode::Write, comm_);
        }
    }

    writer_.BeginStep();

    auto var_time = io_.InquireVariable<double>("time");
    writer_.Put<double>(var_time, &time, adios2::Mode::Sync);
    write_static_data();

    const size_t h = grid_.get_halo_cells();
    const size_t rank_off_x = grid_.get_local_physical_start_x();
    const size_t rank_off_y = grid_.get_local_physical_start_y();
    const size_t rank_off_z = grid_.get_local_physical_start_z();

    size_t out_x_start = std::max(rank_off_x, output_x_start_);
    size_t out_y_start = std::max(rank_off_y, output_y_start_);
    size_t out_z_start = std::max(rank_off_z, output_z_start_);

    for (const auto& field_name : fields_to_output_) {
        if (field_variables_.count(field_name)) {
            auto& adios_var = field_variables_.at(field_name);
            auto it = state_.begin();
            while (it != state_.end() && it->first != field_name) ++it;
             
            if (it != state_.end()) {
                std::visit([&](const auto& field) {
                    using T = std::decay_t<decltype(field)>;
                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        auto full_data_view = field.get_device_data();
                        using DevMemSpace = typename decltype(full_data_view)::memory_space;

                        size_t k_start = (out_z_start - rank_off_z) + h;
                        size_t j_start = (out_y_start - rank_off_y) + h;
                        size_t i_start = (out_x_start - rank_off_x) + h;

                        if constexpr (T::DimValue == 1) {
                            if (rank_ == 0) {
                                size_t count = adios_var.Count()[0];
                                auto subview = Kokkos::subview(full_data_view, std::make_pair(k_start, k_start + count));
                                
                                if (host_buffers_1d_.find(field_name) == host_buffers_1d_.end()) {
                                    host_buffers_1d_[field_name] = Kokkos::View<double*, Kokkos::HostSpace>(field_name + "_host", count);
                                }
                                auto& host_view = host_buffers_1d_[field_name];
                                Kokkos::deep_copy(host_view, subview);
                                writer_.Put(adios_var, host_view.data());
                            } 
                        }
                        else if constexpr (T::DimValue == 2) {
                            size_t ny = adios_var.Count()[0];
                            size_t nx = adios_var.Count()[1];
                            
                            // 2-Step Copy: Strided Device -> Contiguous Device -> Contiguous Host
                            Kokkos::View<double**, Kokkos::LayoutRight, DevMemSpace> dev_contig("temp_2d", ny, nx);
                            auto subview = Kokkos::subview(full_data_view, 
                                std::make_pair(j_start, j_start + ny), 
                                std::make_pair(i_start, i_start + nx));
                            Kokkos::deep_copy(dev_contig, subview);

                            if (host_buffers_2d_.find(field_name) == host_buffers_2d_.end()) {
                                host_buffers_2d_[field_name] = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>(field_name + "_host", ny, nx);
                            }
                            auto& host_view = host_buffers_2d_[field_name];
                            Kokkos::deep_copy(host_view, dev_contig);
                            
                            // UNCONDITIONAL PUT: Even if ny*nx is 0, we pass the pointer (which is valid/empty)
                            writer_.Put(adios_var, host_view.data());
                        }
                        else if constexpr (T::DimValue == 3) {
                            size_t nz = adios_var.Count()[0];
                            size_t ny = adios_var.Count()[1];
                            size_t nx = adios_var.Count()[2];

                            Kokkos::View<double***, Kokkos::LayoutRight, DevMemSpace> dev_contig("temp_3d", nz, ny, nx);
                            auto subview = Kokkos::subview(full_data_view,
                                std::make_pair(k_start, k_start + nz),
                                std::make_pair(j_start, j_start + ny),
                                std::make_pair(i_start, i_start + nx));
                            Kokkos::deep_copy(dev_contig, subview);

                            if (host_buffers_3d_.find(field_name) == host_buffers_3d_.end()) {
                                host_buffers_3d_[field_name] = Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>(
                                    field_name + "_host", nz, ny, nx);
                            }
                            auto& host_view = host_buffers_3d_[field_name];
                            Kokkos::deep_copy(host_view, dev_contig);
                            
                            writer_.Put(adios_var, host_view.data());
                        }
                        else if constexpr (T::DimValue == 4) {
                            size_t d4 = adios_var.Count()[0];
                            size_t nz = adios_var.Count()[1];
                            size_t ny = adios_var.Count()[2];
                            size_t nx = adios_var.Count()[3];

                            Kokkos::View<double****, Kokkos::LayoutRight, DevMemSpace> dev_contig("temp_4d", d4, nz, ny, nx);
                            auto subview = Kokkos::subview(full_data_view, Kokkos::ALL(),
                                std::make_pair(k_start, k_start + nz),
                                std::make_pair(j_start, j_start + ny),
                                std::make_pair(i_start, i_start + nx));
                            Kokkos::deep_copy(dev_contig, subview);

                            if (host_buffers_4d_.find(field_name) == host_buffers_4d_.end()) {
                                host_buffers_4d_[field_name] = Kokkos::View<double****, Kokkos::LayoutRight, Kokkos::HostSpace>(
                                    field_name + "_host", d4, nz, ny, nx);
                            }
                            auto& host_view = host_buffers_4d_[field_name];
                            Kokkos::deep_copy(host_view, dev_contig);
                            
                            writer_.Put(adios_var, host_view.data());
                        }
                    }
                }, it->second);
             }
        }
    }

    writer_.EndStep();
    
    if (engine_type_ == "HDF5") {
        writer_.Close();
    }
}

void OutputManager::write_static_data() {
    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();
    const size_t h = grid_.get_halo_cells();

    auto var_x = io_.InquireVariable<double>("coordinates/x");
    if (rank_ == 0) {
        std::vector<double> x_coords(gnx);
        for(size_t i = 0; i < gnx; ++i) x_coords[i] = i * grid_.get_dx();
        writer_.Put<double>(var_x, x_coords.data(), adios2::Mode::Sync);
    } 

    auto var_y = io_.InquireVariable<double>("coordinates/y");
    if (rank_ == 0) {
        std::vector<double> y_coords(gny);
        for(size_t i = 0; i < gny; ++i) y_coords[i] = i * grid_.get_dy();
        writer_.Put<double>(var_y, y_coords.data(), adios2::Mode::Sync);
    }

    auto var_z_mid = io_.InquireVariable<double>("coordinates/z_mid");
    if (rank_ == 0) {
        auto z_mid_host = params_.z_mid.get_host_data();
        std::vector<double> z_mid_physical(gnz);
        for (size_t i = 0; i < gnz; ++i) z_mid_physical[i] = z_mid_host(i + h);
        writer_.Put<double>(var_z_mid, z_mid_physical.data(), adios2::Mode::Sync);
    }
}

void OutputManager::grads_ctl_file() {
    std::ofstream outFile(output_dir_ + "/vvm.ctl");
    if (!outFile.is_open()) return;

    auto z_mid_host = params_.z_mid.get_host_data();
    auto h = grid_.get_halo_cells();
    auto nz_phy = grid_.get_global_points_z();
    
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
    outFile << "TDEF " << (int) (total_time_ / (output_interval_s_)+1) << " LINEAR 00:00Z01JAN2000 " << "1hr\n";
    outFile << "\n";

    int valid_vars_count = 0;
    std::vector<std::string> lines_to_write;
    for (const auto& field_name : fields_to_output_) {
        auto it = state_.begin();
        while (it != state_.end() && it->first != field_name) ++it;
        if (it != state_.end()) {
            std::visit([&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    std::stringstream ss;
                    if constexpr (T::DimValue == 3 || T::DimValue == 4) {
                        ss << "/Step0/" << field_name << "=>" << field_name << " " << nz_phy << " z,y,x " << field_name << "\n";
                        valid_vars_count++;
                        lines_to_write.push_back(ss.str());
                    } 
                    else if constexpr (T::DimValue == 2) {
                        ss << "/Step0/" << field_name << "=>" << field_name << " 0 y,x " << field_name << "\n";
                        valid_vars_count++;
                        lines_to_write.push_back(ss.str());
                    } 
                    else if constexpr (T::DimValue == 1) {
                        ss << "/Step0/" << field_name << "=>" << field_name << " " << nz_phy << " z " << field_name << "\n";
                        valid_vars_count++;
                        lines_to_write.push_back(ss.str());
                    }
                }
            }, it->second);
        }
    }
    outFile << "VARS " << valid_vars_count << "\n";
    for(const auto& line : lines_to_write) outFile << line;
    outFile << "ENDVARS\n";
    outFile.close();
}

void OutputManager::write_static_topo_file() {
    if (rank_ == 0) std::cout << "Writing static topography file..." << std::endl;

    adios2::IO topo_io = adios_.DeclareIO("TOPO_IO");
    topo_io.SetEngine("HDF5");
    topo_io.SetParameter("IdleH5Writer", "true");
    topo_io.SetParameter("H5CollectiveMPIO", "no");

    std::string filename = output_dir_ + "/topo.h5";
    adios2::Engine topo_writer = topo_io.Open(filename, adios2::Mode::Write, comm_);

    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();
    const size_t h = grid_.get_halo_cells();

    const size_t rank_lnx = grid_.get_local_physical_points_x();
    const size_t rank_lny = grid_.get_local_physical_points_y();
    const size_t rank_offset_x = grid_.get_local_physical_start_x();
    const size_t rank_offset_y = grid_.get_local_physical_start_y();

    auto var_topo = topo_io.DefineVariable<double>("topo", {gny, gnx}, {rank_offset_y, rank_offset_x}, {rank_lny, rank_lnx});
    topo_io.DefineAttribute<std::string>("units", "meter", var_topo.Name());

    topo_writer.BeginStep();

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
        if (rank_ == 0) std::cerr << "Warning: Could not write 'topo': " << e.what() << std::endl;
    }

    topo_writer.EndStep();
    topo_writer.Close();
}

} // namespace IO
} // namespace VVM
