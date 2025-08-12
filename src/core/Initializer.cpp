#include "Initializer.hpp"
#include "BoundaryConditionManager.hpp"
#include "io/TxtReader.hpp"
#include <Kokkos_Core.hpp>

namespace VVM {
namespace Core {

Initializer::Initializer(const Utils::ConfigurationManager& config, const Grid& grid, Parameters& parameters, State &state) 
    : config_(config), grid_(grid), parameters_(parameters), state_(state) {
    initialize_grid();

    if (!config.has_key("initial_conditions")) {
        return;
    }

    std::string format = config.get_value<std::string>("initial_conditions.format");
    std::string source_file = config.get_value<std::string>("initial_conditions.source_file");

    if (format == "txt") {
        reader_ = std::make_unique<VVM::IO::TxtReader>(source_file, grid, parameters_);
    } 
    // else if (format == "netcdf") {
    //     // TODO: Netcdf input
    //     // reader_ = std::make_unique<Initializers::NetCDFReader>(source_file, grid);
    //     std::cerr << "Warning: NetCDF reader is not implemented yet. Skipping initialization." << std::endl;
    // } else {
    //     std::cerr << "Warning: Unsupported input format '" << format << "'. Skipping initialization." << std::endl;
    // }
}

void Initializer::initialize_state(State& state) const {
    if (reader_) {
        reader_->read_and_initialize(state);
    }
}

void Initializer::initialize_grid() const {
    VVM::Core::BoundaryConditionManager bc_manager(grid_);

    double DOMAIN = 15000.;
    double dz = config_.get_value<double>("grid.dz");
    double dz1 = config_.get_value<double>("grid.dz1");
    double CZ2 = (dz-dz1) / (dz * (DOMAIN-dz));
    double CZ1 = 1. - CZ2 * DOMAIN;

    const int nz_phys = grid_.get_local_physical_points_z();
    const int h = grid_.get_halo_cells();
    const int nz_total = grid_.get_local_total_points_z();

    auto z_mid_mutable = parameters_.z_mid.get_mutable_device_data();
    auto z_up_mutable = parameters_.z_up.get_mutable_device_data();
    auto flex_height_coef_mid_mutable = parameters_.flex_height_coef_mid.get_mutable_device_data();
    auto flex_height_coef_up_mutable = parameters_.flex_height_coef_up.get_mutable_device_data();

    Kokkos::parallel_for("Init_Z_flexZCoef", Kokkos::RangePolicy<>(0, grid_.get_local_total_points_z()),
        KOKKOS_LAMBDA(const int k) {
            z_mid_mutable(k) = (k-h-0.5) * dz;
            z_up_mutable(k) = (k-h) * dz;
            flex_height_coef_mid_mutable(k) = 1. / (CZ1 + 2 * CZ2 * z_mid_mutable(k));
            flex_height_coef_up_mutable(k) = 1. / (CZ1 + 2 * CZ2 * z_up_mutable(k));
            z_mid_mutable(k) = z_mid_mutable(k) * (CZ1 + CZ2 * z_mid_mutable(k));
            z_up_mutable(k) = z_up_mutable(k) * (CZ1 + CZ2 * z_up_mutable(k));
        }
    );
    
    auto dz_mid_mutable = parameters_.dz_mid.get_mutable_device_data();
    auto dz_up_mutable = parameters_.dz_up.get_mutable_device_data();
    Kokkos::parallel_for("Init_dz", Kokkos::RangePolicy<>(1, grid_.get_local_total_points_z()-1),
        KOKKOS_LAMBDA(const int k) {
            dz_mid_mutable(k) = z_up_mutable(k) - z_up_mutable(k-1);
            dz_up_mutable(k) = z_mid_mutable(k+1) - z_mid_mutable(k);
        }
    );

    auto fact1_xi_eta_mutable = parameters_.fact1_xi_eta.get_mutable_device_data();
    auto fact2_xi_eta_mutable = parameters_.fact2_xi_eta.get_mutable_device_data();
    Kokkos::parallel_for("Init_zflex_fact", Kokkos::RangePolicy<>(1, grid_.get_local_total_points_z()-1),
        KOKKOS_LAMBDA(const int k) {
            fact1_xi_eta_mutable(k) = flex_height_coef_up_mutable(k) / flex_height_coef_mid_mutable(k+1);
            fact2_xi_eta_mutable(k) = flex_height_coef_up_mutable(k) / flex_height_coef_mid_mutable(k);
        }
    );

    bc_manager.apply_z_bcs_to_field(parameters_.z_mid);
    bc_manager.apply_z_bcs_to_field(parameters_.z_up);
    bc_manager.apply_z_bcs_to_field(parameters_.flex_height_coef_mid);
    bc_manager.apply_z_bcs_to_field(parameters_.flex_height_coef_up);
    bc_manager.apply_z_bcs_to_field(parameters_.dz_mid);
    bc_manager.apply_z_bcs_to_field(parameters_.dz_up);

    return;
}

} // namespace Core
} // namespace VVM
