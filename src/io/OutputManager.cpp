#include "OutputManager.hpp"
#include <sys/stat.h>
#include <cerrno>
#include <adios2/cxx11/KokkosView.h>
#include <Kokkos_Core.hpp> 


namespace VVM {
namespace IO {

OutputManager::OutputManager(const Utils::ConfigurationManager& config, const VVM::Core::Grid& grid, MPI_Comm comm)
    : grid_(grid), comm_(comm), adios_(comm) {
    rank_ = 0;
    mpi_size_ = 1;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);

    output_dir_ = config.get_value<std::string>("output.output_dir");
    filename_prefix_ = config.get_value<std::string>("output.output_filename_prefix");
    fields_to_output_ = config.get_value<std::vector<std::string>>("output.fields_to_output");

    output_x_start_  = config.get_value<size_t>("output.output_grid.x_start");
    output_y_start_  = config.get_value<size_t>("output.output_grid.y_start");
    output_z_start_  = config.get_value<size_t>("output.output_grid.z_start");

    output_x_end_    = config.get_value<size_t>("output.output_grid.x_end");
    output_y_end_    = config.get_value<size_t>("output.output_grid.y_end");
    output_z_end_    = config.get_value<size_t>("output.output_grid.z_end");

    if (rank_ == 0) {
        if (mkdir(output_dir_.c_str(), 0777) != 0 && errno != EEXIST) {
            perror(("Failed to create directory " + output_dir_).c_str());
        }
    }
    MPI_Barrier(comm_);

    io_ = adios_.DeclareIO("VVM_IO");
    io_.SetEngine("BP5");
    io_.SetParameters({{"Threads", "4"}});

    std::string filename = output_dir_ + "/" + filename_prefix_ + ".bp";
    writer_ = io_.Open(filename, adios2::Mode::Write);
}

OutputManager::~OutputManager() {
    if(writer_) {
      writer_.Close();
    }
}

void OutputManager::define_variables(const VVM::Core::State& state) {

    const size_t gnx = grid_.get_global_points_x();
    const size_t gny = grid_.get_global_points_y();
    const size_t gnz = grid_.get_global_points_z();

    const size_t lnx = grid_.get_local_physical_points_x();
    const size_t lny = grid_.get_local_physical_points_y();
    const size_t lnz = grid_.get_local_physical_points_z();

    const size_t offset_x = grid_.get_local_physical_start_x();
    const size_t offset_y = grid_.get_local_physical_start_y();
    const size_t offset_z = grid_.get_local_physical_start_z();

    for (const auto& field_name : fields_to_output_) {
        auto it = state.begin();
        while (it != state.end() && it->first != field_name) ++it;

        if (it != state.end()) {
            std::visit([&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    if constexpr (T::DimValue == 1) field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz}, {offset_z}, {lnz});
                    if constexpr (T::DimValue == 2) field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gny, gnx}, {offset_y, offset_x}, {lny, lnx});
                    else if constexpr (T::DimValue == 3) field_variables_[field_name] = io_.DefineVariable<double>(field_name, {gnz, gny, gnx}, {offset_z, offset_y, offset_x}, {lnz, lny, lnx});
                    else if constexpr (T::DimValue == 4) {
                        const size_t dim4 = field.get_device_data().extent(0);
                        field_variables_[field_name] = io_.DefineVariable<double>(field_name, {dim4, gnz, gny, gnx}, {0, offset_z, offset_y, offset_x}, {dim4, lnz, lny, lnx});
                    }
                }
            }, it->second);
        }
    }
}

void OutputManager::write(const VVM::Core::State& state, double time) {
    if (field_variables_.empty()) {
        define_variables(state);
    }

    writer_.BeginStep();
    
    adios2::Variable<double> var_time = io_.InquireVariable<double>("time");
    if (!var_time) {
        var_time = io_.DefineVariable<double>("time");
    }
    writer_.Put<double>(var_time, &time, adios2::Mode::Sync);

    const size_t h = grid_.get_halo_cells();

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
                        
                        const size_t lnx = grid_.get_local_physical_points_x();
                        const size_t lny = grid_.get_local_physical_points_y();
                        const size_t lnz = grid_.get_local_physical_points_z();

                        if constexpr (T::DimValue == 1) {
                            if (rank_ == 0) {
                                Kokkos::View<double*> phys_data("phys_data_1d", lnz);
                                auto subview = Kokkos::subview(full_data_view, std::make_pair(h, h + lnz));
                                Kokkos::deep_copy(phys_data, subview);
                                writer_.Put<double>(adios_var, phys_data.data());
                            }
                        }
                        else if constexpr (T::DimValue == 2) {
                            // const adios2::Box<adios2::Dims> mem_selection({h, h}, {lny, lnx});
                            // adios_var.SetMemorySelection(mem_selection);
                            // writer_.Put<double>(adios_var, full_data_view.data());

                            Kokkos::View<double**> phys_data("phys_data_2d", lny, lnx);
                            auto subview = Kokkos::subview(full_data_view, 
                                                           std::make_pair(h, h + lny), 
                                                           std::make_pair(h, h + lnx));
                            Kokkos::deep_copy(phys_data, subview);
                            writer_.Put<double>(adios_var, phys_data.data());
                        } 
                        else if constexpr (T::DimValue == 3) {
                            // const adios2::Box<adios2::Dims> mem_selection({h, h, h}, {lnz, lny, lnx});
                            // adios_var.SetMemorySelection(mem_selection);
                            // writer_.Put<double>(adios_var, full_data_view.data());

                            Kokkos::View<double***> phys_data("phys_data_3d", lnz, lny, lnx);
                            auto subview = Kokkos::subview(full_data_view,
                                                           std::make_pair(h, h + lnz),
                                                           std::make_pair(h, h + lny),
                                                           std::make_pair(h, h + lnx));
                            Kokkos::deep_copy(phys_data, subview);
                            writer_.Put<double>(adios_var, phys_data.data());
                        }
                        else if constexpr (T::DimValue == 4) {
                            // const size_t dim4 = full_data_view.extent(0);
                            // No halo points at the fourth dimension
                            // const adios2::Box<adios2::Dims> mem_selection({0, h, h, h}, {dim4, lnz, lny, lnx});
                            // adios_var.SetMemorySelection(mem_selection);
                            // writer_.Put<double>(adios_var, full_data_view.data());

                            const size_t dim4 = full_data_view.extent(0);
                            Kokkos::View<double****> phys_data("phys_data_4d", dim4, lnz, lny, lnx);
                            auto subview = Kokkos::subview(full_data_view,
                                                           Kokkos::ALL(),
                                                           std::make_pair(h, h + lnz),
                                                           std::make_pair(h, h + lny),
                                                           std::make_pair(h, h + lnx));
                            Kokkos::deep_copy(phys_data, subview);
                            writer_.Put<double>(adios_var, phys_data.data());
                        }
                    }
                }, it->second);
            }
        }
    }

    writer_.EndStep();
}

} // namespace IO
} // namespace VVM
