#include "RandomForcing.hpp"

namespace VVM {
namespace Dynamics {

RandomForcing::RandomForcing(const Utils::ConfigurationManager& config, 
                             const Core::Grid& grid,
                             const Core::Parameters& params)
    : config_(config), grid_(grid), params_(params) {
    
    enabled_ = config.get_value<bool>("dynamics.forcings.random_perturbation.enable", false);
    end_time_ = config.get_value<double>("dynamics.forcings.random_perturbation.time_s", 50.0);
    amplitude_ = config.get_value<double>("dynamics.forcings.random_perturbation.amplitude", 1.0);
    seed_ = config.get_value<int>("dynamics.forcings.random_perturbation.random_seed", 12345); 
}

void RandomForcing::initialize(Core::State& state) {
    double z_start_m = config_.get_value<double>("dynamics.forcings.random_perturbation.z_start_m", 0); 
    double z_end_m = config_.get_value<double>("dynamics.forcings.random_perturbation.z_end_m", 0); 
    auto z_mid_host = params_.z_mid.get_host_data();
    int nz = grid_.get_local_total_points_z();
    int h = grid_.get_halo_cells();
    
    for (int k = h; k < nz; ++k) {
        double z = z_mid_host(k);
        if (z >= z_start_m) {
            k_start_ = k;
            z_start_m = z;
            break;
        }
    }
    for (int k = nz-h; k > 0; k--) {
        double z = z_mid_host(k);
        if (z <= z_end_m) {
            k_end_ = k;
            z_end_m = z;
            break;
        }
    }

    if (grid_.get_mpi_rank() == 0 && enabled_) {
        if (k_start_ > k_end_) {
            std::cout << "[RandomForcing] WARNING: No vertical levels found between " 
                      << z_start_m << "m and " << z_end_m << "m." << std::endl;
        } 
        else {
            std::cout << "[RandomForcing] Initialized. Range: " << z_start_m << "m to " << z_end_m << "m "
                      << "(Indices k: " << k_start_ << " to " << k_end_ << ")" << std::endl;
        }
    }

}

void RandomForcing::apply(Core::State& state) {
    if (!enabled_) return;
    if (state.get_time() >= end_time_) return;
    if (k_start_ > k_end_) return;

    int nz = grid_.get_local_total_points_z();
    int ny = grid_.get_local_total_points_y();
    int nx = grid_.get_local_total_points_x();
    int h = grid_.get_halo_cells();

    Kokkos::Random_XorShift64_Pool<> rand_pool(seed_ + state.get_step() + 10000*grid_.get_mpi_rank());

    auto& th = state.get_field<3>("th").get_mutable_device_data();
    double amp = amplitude_;
    int k_start = k_start_;
    int k_end = k_end_ + 1;

    if (k_end > grid_.get_local_total_points_z()) k_end = grid_.get_local_total_points_z();

    Kokkos::parallel_for("RandomForcing_Apply",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({k_start, h, h}, {k_end, ny-h, nx-h}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto gen = rand_pool.get_state();
            double noise = gen.drand(-1.0, 1.0) * amp;
            th(k, j, i) += noise;
            
            rand_pool.free_state(gen);
        }
    );
    return;
}

} // namespace Dynamics
} // namespace VVM
