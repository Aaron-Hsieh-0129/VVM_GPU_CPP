#include "Grid.hpp"
#include "Field.hpp"
#include "State.hpp"
#include <algorithm>
#include <vector>
#include <Kokkos_Core.hpp>
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"


#if defined(ENABLE_NCCL)

#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

#include "utils/ConfigurationManager.hpp"
#include <nccl.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <set>

namespace VVM {
namespace Core {

using ExecSpace = Kokkos::Cuda;

class HaloExchanger {
public:
    explicit HaloExchanger(const Utils::ConfigurationManager& config, const Grid& grid, ncclComm_t nccl_comm, cudaStream_t stream);
    ~HaloExchanger();

    HaloExchanger(const HaloExchanger&) = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;

    void exchange_halos(State& state);

    template<size_t Dim>
    void exchange_halos(Field<Dim>& field, int depth = -1) const {
        VVM::Utils::Timer exchange_halos_timer("Exchnage_halos");
        exchange_halos_impl(field, depth);
        
        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(stream_, &capture_status);
        if (depth == -1 && capture_status == cudaStreamCaptureStatusNone && grid_ref_.get_mpi_size() > 1) {
             cudaStreamSynchronize(stream_);
        }
    }

    template<size_t Dim>
    void exchange_halos_impl(Field<Dim>& field, int depth = -1) const;

    void exchange_halos_slice(Field<3>& field, int k_layer) const;
    
    void exchange_halos_top_slice(Field<3>& field) const {
        const int nz = grid_ref_.get_local_total_points_z();
        const int h = grid_ref_.get_halo_cells();
        exchange_halos_slice(field, nz - h - 1);
    }

private:
    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    ncclComm_t nccl_comm_;
    cudaStream_t stream_;
    ExecSpace exec_space_; 

    int neighbor_left_, neighbor_right_;
    int neighbor_bottom_, neighbor_top_;

    bool is_single_rank_;

    std::set<std::string> enabled_graph_vars_;

    std::map<std::string, cudaGraphExec_t> graph_map_;

    mutable Kokkos::View<double*, ExecSpace> send_x_left_, recv_x_left_;
    mutable Kokkos::View<double*, ExecSpace> send_x_right_, recv_x_right_;
    mutable Kokkos::View<double*, ExecSpace> send_y_bottom_, recv_y_bottom_;
    mutable Kokkos::View<double*, ExecSpace> send_y_top_, recv_y_top_;

    size_t buffer_size_x_2d_, buffer_size_y_2d_;
    size_t buffer_size_x_3d_, buffer_size_y_3d_;
    size_t buffer_size_x_4d_, buffer_size_y_4d_;
    size_t buffer_size_slice_x_, buffer_size_slice_y_;
};

inline HaloExchanger::HaloExchanger(const Utils::ConfigurationManager& config, const Grid& grid, ncclComm_t nccl_comm, cudaStream_t stream)
    : grid_ref_(grid), 
      cart_comm_(grid.get_cart_comm()), 
      nccl_comm_(nccl_comm), 
      stream_(stream),
      exec_space_(stream)
{
    is_single_rank_ = (grid.get_mpi_size() == 1);

    if (config.has_key("optimization.cuda_graph_halo_exchange")) {
        auto vars = config.get_value<std::vector<std::string>>("optimization.cuda_graph_halo_exchange");
        enabled_graph_vars_.insert(vars.begin(), vars.end());
    }

    if (grid_ref_.get_mpi_rank() == 0 && !enabled_graph_vars_.empty()) {
        std::cout << "HaloExchanger: CUDA Graph enabled for fields: ";
        for (const auto& var : enabled_graph_vars_) std::cout << var << " ";
        std::cout << std::endl;
    }

    if (cart_comm_ != MPI_COMM_NULL) {
        int cart_left, cart_right, cart_bottom, cart_top;
        MPI_Cart_shift(cart_comm_, 1, 1, &cart_left, &cart_right);
        MPI_Cart_shift(cart_comm_, 0, 1, &cart_bottom, &cart_top);

        MPI_Group cart_group, world_group;
        MPI_Comm_group(cart_comm_, &cart_group);
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        int cart_ranks[4] = {cart_left, cart_right, cart_bottom, cart_top};
        int world_ranks[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

        for(int i=0; i<4; ++i) {
            if (cart_ranks[i] != MPI_PROC_NULL) {
                MPI_Group_translate_ranks(cart_group, 1, &cart_ranks[i], world_group, &world_ranks[i]);
            }
        }

        neighbor_left_   = world_ranks[0];
        neighbor_right_  = world_ranks[1];
        neighbor_bottom_ = world_ranks[2];
        neighbor_top_    = world_ranks[3];

        MPI_Group_free(&cart_group);
        MPI_Group_free(&world_group);
    } 
    else {
        neighbor_left_ = neighbor_right_ = neighbor_bottom_ = neighbor_top_ = MPI_PROC_NULL;
    }

    const int h = grid_ref_.get_halo_cells();
    if (h > 0) {
        const int nx = grid.get_local_total_points_x();
        const int ny = grid.get_local_total_points_y();
        const int nz = grid.get_local_total_points_z();
        const int nw_dummy = 4;

        buffer_size_x_2d_ = static_cast<size_t>(h) * ny;
        buffer_size_y_2d_ = static_cast<size_t>(h) * nx;
        buffer_size_x_3d_ = static_cast<size_t>(h) * ny * nz;
        buffer_size_y_3d_ = static_cast<size_t>(h) * nx * nz;
        buffer_size_x_4d_ = static_cast<size_t>(h) * ny * nz * nw_dummy;
        buffer_size_y_4d_ = static_cast<size_t>(h) * nx * nz * nw_dummy;
        
        buffer_size_slice_x_ = static_cast<size_t>(h) * ny;
        buffer_size_slice_y_ = static_cast<size_t>(h) * nx;

        size_t max_x = std::max({buffer_size_x_2d_, buffer_size_x_3d_, buffer_size_x_4d_, buffer_size_slice_x_});
        size_t max_y = std::max({buffer_size_y_2d_, buffer_size_y_3d_, buffer_size_y_4d_, buffer_size_slice_y_});

        if (max_x > 0) {
            send_x_left_  = Kokkos::View<double*, ExecSpace>("sxl", max_x);
            recv_x_left_  = Kokkos::View<double*, ExecSpace>("rxl", max_x);
            send_x_right_ = Kokkos::View<double*, ExecSpace>("sxr", max_x);
            recv_x_right_ = Kokkos::View<double*, ExecSpace>("rxr", max_x);
        }
        if (max_y > 0) {
            send_y_bottom_ = Kokkos::View<double*, ExecSpace>("syb", max_y);
            recv_y_bottom_ = Kokkos::View<double*, ExecSpace>("ryb", max_y);
            send_y_top_    = Kokkos::View<double*, ExecSpace>("syt", max_y);
            recv_y_top_    = Kokkos::View<double*, ExecSpace>("ryt", max_y);
        }
    }
}

inline HaloExchanger::~HaloExchanger() {
    for (auto& pair : graph_map_) {
        if (pair.second) {
            cudaGraphExecDestroy(pair.second);
        }
    }
}

inline void HaloExchanger::exchange_halos(State& state) {
    Kokkos::fence(); 

    for (auto& field_pair : state) {
        std::visit([this](auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                const std::string name = field.get_name();
                
                if (enabled_graph_vars_.count(name)) {
                    auto it = graph_map_.find(name);
                    
                    if (it == graph_map_.end()) {
                        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
                        
                        this->exchange_halos_impl(field);
                        
                        cudaGraph_t graph;
                        cudaStreamEndCapture(stream_, &graph);
                        
                        cudaGraphExec_t instance;
                        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
                        cudaGraphDestroy(graph);
                        
                        graph_map_[name] = instance;
                        it = graph_map_.find(name);
                    }
                    
                    cudaGraphLaunch(it->second, stream_);
                } 
                else {
                    this->exchange_halos_impl(field);
                }
            }
        }, field_pair.second);
    }
    
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream_, &capture_status);
}

template<size_t Dim>
void HaloExchanger::exchange_halos_impl(Field<Dim>& field, int depth) const {
    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    if (is_single_rank_) {
        // X-Direction Periodic Copy
        // Left Halo (start-h .. start) <== Right Phys (start+nx_phys-h .. start+nx_phys)
        // Right Halo (start+nx_phys .. end) <== Left Phys (start .. start+h)
        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("local_copy_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    // Left Halo gets Right Physical
                    data(j, halo_start_offset - h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
                    // Right Halo gets Left Physical
                    data(j, halo_start_offset + nx_phys + i_h) = data(j, halo_start_offset + i_h);
            });
        }
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int ny = data.extent(1);
            Kokkos::parallel_for("local_copy_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    // Left Halo gets Right Physical
                    data(k, j, halo_start_offset - h + i_h) = data(k, j, halo_start_offset + nx_phys - h + i_h);
                    // Right Halo gets Left Physical
                    data(k, j, halo_start_offset + nx_phys + i_h) = data(k, j, halo_start_offset + i_h);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int ny = data.extent(2);
            Kokkos::parallel_for("local_copy_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, ny, h}),
                KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                    data(w, k, j, halo_start_offset - h + i_h) = data(w, k, j, halo_start_offset + nx_phys - h + i_h);
                    data(w, k, j, halo_start_offset + nx_phys + i_h) = data(w, k, j, halo_start_offset + i_h);
            });
        }

        // Y-Direction Periodic Copy
        // Bottom Halo <== Top Phys
        // Top Halo <== Bottom Phys
        if constexpr (Dim == 2) {
            const int nx = data.extent(1);
            Kokkos::parallel_for("local_copy_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    // Bottom Halo gets Top Physical
                    data(halo_start_offset - h + j_h, i) = data(halo_start_offset + ny_phys - h + j_h, i);
                    // Top Halo gets Bottom Physical
                    data(halo_start_offset + ny_phys + j_h, i) = data(halo_start_offset + j_h, i);
            });
        } 
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int nx = data.extent(2);
            Kokkos::parallel_for("local_copy_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    data(k, halo_start_offset - h + j_h, i) = data(k, halo_start_offset + ny_phys - h + j_h, i);
                    data(k, halo_start_offset + ny_phys + j_h, i) = data(k, halo_start_offset + j_h, i);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int nx = data.extent(3);
            Kokkos::parallel_for("local_copy_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, nx, h}),
                KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                    data(w, k, halo_start_offset - h + j_h, i) = data(w, k, halo_start_offset + ny_phys - h + j_h, i);
                    data(w, k, halo_start_offset + ny_phys + j_h, i) = data(w, k, halo_start_offset + j_h, i);
            });
        }
        
        return; // Skip the rest of NCCL/MPI logic
    }
    
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;

    const int my_rank = grid_ref_.get_mpi_rank();

    size_t count_x = 0;
    size_t count_y = 0;
    
    if constexpr (Dim == 2) {
        count_x = static_cast<size_t>(h) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(1);
    } 
    else if constexpr (Dim == 3) {
        count_x = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(2) * data.extent(0);
    } 
    else if constexpr (Dim == 4) {
        count_x = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);
    }

    if (count_x > 0) {
        auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x));
        auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x));
        auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count_x));
        auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x));

        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    send_l(j * h + i_h) = data(j, halo_start_offset + i_h);
                    send_r(j * h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
            });
        } 
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int ny = data.extent(1);
            Kokkos::parallel_for("pack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    const size_t idx = k * (ny * h) + j * h + i_h;
                    send_l(idx) = data(k, j, halo_start_offset + i_h);
                    send_r(idx) = data(k, j, halo_start_offset + nx_phys - h + i_h);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int ny = data.extent(2);
            Kokkos::parallel_for("pack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, ny, h}),
                KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                    size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                    send_l(idx) = data(w, k, j, halo_start_offset + i_h);
                    send_r(idx) = data(w, k, j, halo_start_offset + nx_phys - h + i_h);
            });
        }

        if (neighbor_left != MPI_PROC_NULL && neighbor_left == my_rank && neighbor_right == my_rank) {
            Kokkos::deep_copy(exec_space_, recv_l, send_r);
            Kokkos::deep_copy(exec_space_, recv_r, send_l);
        }
        else {
            // NCCL Communication
            ncclGroupStart();
            if(neighbor_right != MPI_PROC_NULL) 
                ncclSend(send_r.data(), count_x, ncclDouble, neighbor_right, nccl_comm_, stream_);
            if(neighbor_left != MPI_PROC_NULL) 
                ncclRecv(recv_l.data(), count_x, ncclDouble, neighbor_left, nccl_comm_, stream_);

            if(neighbor_left != MPI_PROC_NULL) 
                ncclSend(send_l.data(), count_x, ncclDouble, neighbor_left, nccl_comm_, stream_);
            if(neighbor_right != MPI_PROC_NULL) 
                ncclRecv(recv_r.data(), count_x, ncclDouble, neighbor_right, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        // Unpack Data from Buffers
        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    if (neighbor_left != MPI_PROC_NULL) data(j, halo_start_offset - h + i_h) = recv_l(j * h + i_h);
                    if (neighbor_right != MPI_PROC_NULL) data(j, halo_start_offset + nx_phys + i_h) = recv_r(j * h + i_h);
            });
        } 
        else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int ny = data.extent(1);
            Kokkos::parallel_for("unpack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, ny, h}),
                KOKKOS_LAMBDA(int k, int j, int i_h) {
                    const size_t idx = k * (ny * h) + j * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(k, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int ny = data.extent(2);
            Kokkos::parallel_for("unpack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, ny, h}),
                KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                    size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(w, k, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(w, k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }

    if (count_y > 0) {
        auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y));
        auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y));
        auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count_y));
        auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y));

        // Pack
        if constexpr (Dim == 2) {
            const int nx = data.extent(1);
            Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = j_h * nx + i;
                    send_b(idx) = data(halo_start_offset + j_h, i);
                    send_t(idx) = data(halo_start_offset + ny_phys - h + j_h, i);
            });
        } else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int nx = data.extent(2);
            Kokkos::parallel_for("pack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    const size_t idx = k * (h * nx) + j_h * nx + i;
                    send_b(idx) = data(k, halo_start_offset + j_h, i);
                    send_t(idx) = data(k, halo_start_offset + ny_phys - h + j_h, i);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int nx = data.extent(3);
            Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, nx, h}),
                KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                    size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                    send_b(idx) = data(w, k, halo_start_offset + j_h, i);
                    send_t(idx) = data(w, k, halo_start_offset + ny_phys - h + j_h, i);
            });
        }

        if (neighbor_bottom != MPI_PROC_NULL && neighbor_bottom == my_rank && neighbor_top == my_rank) {
             Kokkos::deep_copy(exec_space_, recv_b, send_t);
             Kokkos::deep_copy(exec_space_, recv_t, send_b);
        }
        else {
            // NCCL
            ncclGroupStart();
            if(neighbor_top != MPI_PROC_NULL) 
                ncclSend(send_t.data(), count_y, ncclDouble, neighbor_top, nccl_comm_, stream_);
            if(neighbor_bottom != MPI_PROC_NULL) 
                ncclRecv(recv_b.data(), count_y, ncclDouble, neighbor_bottom, nccl_comm_, stream_);

            if(neighbor_bottom != MPI_PROC_NULL) 
                ncclSend(send_b.data(), count_y, ncclDouble, neighbor_bottom, nccl_comm_, stream_);
            if(neighbor_top != MPI_PROC_NULL) 
                ncclRecv(recv_t.data(), count_y, ncclDouble, neighbor_top, nccl_comm_, stream_);
            ncclGroupEnd();
        }

        // Unpack
        if constexpr (Dim == 2) {
            const int nx = data.extent(1);
            Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        } else if constexpr (Dim == 3) {
            const int nz = data.extent(0);
            const int nx = data.extent(2);
            Kokkos::parallel_for("unpack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(exec_space_, {0,0,0}, {nz, nx, h}),
                KOKKOS_LAMBDA(int k, int i, int j_h) {
                    const size_t idx = k * (h * nx) + j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(k, halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
        else if constexpr (Dim == 4) {
            const int nw = data.extent(0);
            const int nz = data.extent(1);
            const int nx = data.extent(3);
            Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(exec_space_, {0,0,0,0}, {nw, nz, nx, h}),
                KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                    size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(w, k, halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(w, k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
    }
    
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream_, &capture_status);
    if (depth == -1 && capture_status == cudaStreamCaptureStatusNone && grid_ref_.get_mpi_size() > 1) {
         cudaStreamSynchronize(stream_);
    }
}

inline void HaloExchanger::exchange_halos_slice(Field<3>& field, int k_layer) const {
    const int halo_start_offset = grid_ref_.get_halo_cells();
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny = data.extent(1);
    const int nx = data.extent(2);

    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;

    // Kokkos::fence(); 

    // --- Y-Direction Slice ---
    {
        size_t count = buffer_size_slice_y_;
        if (count > 0) {
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
            auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
            auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
            auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    send_b(idx) = data(k_layer, halo_start_offset + j_h, i);
                    send_t(idx) = data(k_layer, halo_start_offset + ny_phys - h + j_h, i);
            });

            ncclGroupStart();
            if(neighbor_top != MPI_PROC_NULL) ncclSend(send_t.data(), count, ncclDouble, neighbor_top, nccl_comm_, stream_);
            if(neighbor_bottom != MPI_PROC_NULL) ncclRecv(recv_b.data(), count, ncclDouble, neighbor_bottom, nccl_comm_, stream_);
            if(neighbor_bottom != MPI_PROC_NULL) ncclSend(send_b.data(), count, ncclDouble, neighbor_bottom, nccl_comm_, stream_);
            if(neighbor_top != MPI_PROC_NULL) ncclRecv(recv_t.data(), count, ncclDouble, neighbor_top, nccl_comm_, stream_);
            ncclGroupEnd();

            Kokkos::parallel_for("unpack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(k_layer, halo_start_offset - h + j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(k_layer, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
            });
        }
    }

    // --- X-Direction Slice ---
    {
        size_t count = buffer_size_slice_x_;
        if (count > 0) {
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
            auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
            auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    send_l(idx) = data(k_layer, j, halo_start_offset + i_h);
                    send_r(idx) = data(k_layer, j, halo_start_offset + nx_phys - h + i_h);
            });

            ncclGroupStart();
            if(neighbor_right != MPI_PROC_NULL) ncclSend(send_r.data(), count, ncclDouble, neighbor_right, nccl_comm_, stream_);
            if(neighbor_left != MPI_PROC_NULL) ncclRecv(recv_l.data(), count, ncclDouble, neighbor_left, nccl_comm_, stream_);
            if(neighbor_left != MPI_PROC_NULL) ncclSend(send_l.data(), count, ncclDouble, neighbor_left, nccl_comm_, stream_);
            if(neighbor_right != MPI_PROC_NULL) ncclRecv(recv_r.data(), count, ncclDouble, neighbor_right, nccl_comm_, stream_);
            ncclGroupEnd();

            Kokkos::parallel_for("unpack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(k_layer, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k_layer, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }
    
    cudaStreamSynchronize(stream_);
}

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_HALOEXCHANGER_HPP


#else

#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP


namespace VVM {
namespace Core {

enum class HaloExchangeTags {
    SEND_TO_RIGHT = 10, // X-direction
    SEND_TO_LEFT  = 11, // X-direction
    SEND_TO_TOP    = 20, // Y-direction
    SEND_TO_BOTTOM = 21,  // Y-direction
    SLICE_SEND_TO_RIGHT = 30,
    SLICE_SEND_TO_LEFT = 31,
    SLICE_SEND_TO_TOP = 40,
    SLICE_SEND_TO_BOTTOM = 41
};

// Struct to hold MPI requests for non-blocking communication
struct HaloExchangeRequests {
    std::vector<MPI_Request> requests;
    int count = 0;
};


class HaloExchanger {
public:
    explicit HaloExchanger(const Grid& grid);

    void exchange_halos(State& state) const;

    template<size_t Dim>
    void exchange_halos(Field<Dim>& field, int depth = -1) const {
        VVM::Utils::Timer exchange_halos_timer("Exchnage_halos");
        if constexpr (Dim >= 2) {
            auto reqs_y = post_exchange_halo_y(field, depth);
            wait_exchange_halo_y(field, reqs_y, depth);

            auto reqs_x = post_exchange_halo_x(field, depth);
            wait_exchange_halo_x(field, reqs_x, depth);
        }
    }

    // --- Asynchronous Halo Exchange Functions ---
    template<size_t Dim>
    HaloExchangeRequests post_exchange_halo_x(Field<Dim>& field, int depth = -1) const;

    template<size_t Dim>
    void wait_exchange_halo_x(Field<Dim>& field, HaloExchangeRequests& reqs, int depth = -1) const;

    template<size_t Dim>
    HaloExchangeRequests post_exchange_halo_y(Field<Dim>& field, int depth = -1) const;

    template<size_t Dim>
    void wait_exchange_halo_y(Field<Dim>& field, HaloExchangeRequests& reqs, int depth = -1) const;

    void exchange_halos_slice(Field<3>& field, int k_layer) const;
    void exchange_halos_top_slice(Field<3>& field) const {
        const int nz = grid_ref_.get_local_total_points_z();
        const int h = grid_ref_.get_halo_cells();
        exchange_halos_slice(field, nz - h - 1);
    }

private:
    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    int neighbor_left_, neighbor_right_;
    int neighbor_bottom_, neighbor_top_;

    mutable Kokkos::View<double*> send_x_left_, recv_x_left_;
    mutable Kokkos::View<double*> send_x_right_, recv_x_right_;
    mutable Kokkos::View<double*> send_y_bottom_, recv_y_bottom_;
    mutable Kokkos::View<double*> send_y_top_, recv_y_top_;

    // mutable Kokkos::View<double*>::HostMirror send_x_left_h_, recv_x_left_h_;
    // mutable Kokkos::View<double*>::HostMirror send_x_right_h_, recv_x_right_h_;
    // mutable Kokkos::View<double*>::HostMirror send_y_bottom_h_, recv_y_bottom_h_;
    // mutable Kokkos::View<double*>::HostMirror send_y_top_h_, recv_y_top_h_;

    size_t buffer_size_x_2d_, buffer_size_y_2d_;
    size_t buffer_size_x_3d_, buffer_size_y_3d_;
    size_t buffer_size_x_4d_, buffer_size_y_4d_;
    size_t buffer_size_slice_x_, buffer_size_slice_y_;
};


inline HaloExchanger::HaloExchanger(const Grid& grid)
    : grid_ref_(grid), cart_comm_(grid.get_cart_comm()) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (cart_comm_ == MPI_COMM_NULL) {
        std::cerr << "Rank " << rank << ": HaloExchanger initialized with NULL communicator!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    // Dim 1 is X (left-right), Dim 0 is y (up-down)
    MPI_Cart_shift(cart_comm_, 1, 1, &neighbor_left_, &neighbor_right_);
    MPI_Cart_shift(cart_comm_, 0, 1, &neighbor_bottom_, &neighbor_top_);

    const int h = grid_ref_.get_halo_cells();
    if (h > 0) {
        const int nx_total = grid.get_local_total_points_x();
        const int ny_total = grid.get_local_total_points_y();
        const int nz_total = grid.get_local_total_points_z();
        const int nw_dummy = 4; // Maximum for the 4th dimension is set to 4

        buffer_size_x_2d_ = static_cast<size_t>(h) * ny_total;
        buffer_size_y_2d_ = static_cast<size_t>(h) * nx_total;
        buffer_size_x_3d_ = static_cast<size_t>(h) * ny_total * nz_total;
        buffer_size_y_3d_ = static_cast<size_t>(h) * nx_total * nz_total;
        buffer_size_x_4d_ = static_cast<size_t>(h) * ny_total * nz_total * nw_dummy;
        buffer_size_y_4d_ = static_cast<size_t>(h) * nx_total * nz_total * nw_dummy;
        
        buffer_size_slice_x_ = static_cast<size_t>(h) * ny_total;
        buffer_size_slice_y_ = static_cast<size_t>(h) * nx_total;

        size_t max_buffer_size_x = std::max({buffer_size_x_2d_, buffer_size_x_3d_, buffer_size_x_4d_, buffer_size_slice_x_});
        size_t max_buffer_size_y = std::max({buffer_size_y_2d_, buffer_size_y_3d_, buffer_size_y_4d_, buffer_size_slice_y_});

        if (max_buffer_size_x > 0) {
            send_x_left_  = Kokkos::View<double*>("send_x_left_buf", max_buffer_size_x);
            recv_x_left_  = Kokkos::View<double*>("recv_x_left_buf", max_buffer_size_x);
            send_x_right_ = Kokkos::View<double*>("send_x_right_buf", max_buffer_size_x);
            recv_x_right_ = Kokkos::View<double*>("recv_x_right_buf", max_buffer_size_x);

            // send_x_left_h_  = Kokkos::create_mirror_view(send_x_left_);
            // recv_x_left_h_  = Kokkos::create_mirror_view(recv_x_left_);
            // send_x_right_h_ = Kokkos::create_mirror_view(send_x_right_);
            // recv_x_right_h_ = Kokkos::create_mirror_view(recv_x_right_);
        }
        if (max_buffer_size_y > 0) {
            send_y_bottom_ = Kokkos::View<double*>("send_y_bottom_buf", max_buffer_size_y);
            recv_y_bottom_ = Kokkos::View<double*>("recv_y_bottom_buf", max_buffer_size_y);
            send_y_top_    = Kokkos::View<double*>("send_y_top_buf", max_buffer_size_y);
            recv_y_top_    = Kokkos::View<double*>("recv_y_top_buf", max_buffer_size_y);

            // send_y_bottom_h_ = Kokkos::create_mirror_view(send_y_bottom_);
            // recv_y_bottom_h_ = Kokkos::create_mirror_view(recv_y_bottom_);
            // send_y_top_h_    = Kokkos::create_mirror_view(send_y_top_);
            // recv_y_top_h_    = Kokkos::create_mirror_view(recv_y_top_);
        }
    }
}

inline void HaloExchanger::exchange_halos(State& state) const {
    for (auto& field_pair : state) {
        std::visit([this](auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                this->exchange_halos(field);
            }
        }, field_pair.second);
    }
}

template<size_t Dim>
HaloExchangeRequests HaloExchanger::post_exchange_halo_x(Field<Dim>& field, int depth) const {
    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = grid_ref_.get_halo_cells();

    if (depth == -1) h = 1;
    else h = depth;

    if (h == 0) return {};

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    
    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(0);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);

    if (count == 0) return {};
    
    auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
    auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
    auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
    auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

    // auto send_l_h = Kokkos::subview(send_x_left_h_, std::make_pair((size_t)0, count));
    // auto recv_l_h = Kokkos::subview(recv_x_left_h_, std::make_pair((size_t)0, count));
    // auto send_r_h = Kokkos::subview(send_x_right_h_, std::make_pair((size_t)0, count));
    // auto recv_r_h = Kokkos::subview(recv_x_right_h_, std::make_pair((size_t)0, count));

    // Pack data
    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
            KOKKOS_LAMBDA(int j, int i_h) {
                send_l(j * h + i_h) = data(j, halo_start_offset + i_h);
                send_r(j * h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for("pack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                const size_t idx = k * (ny * h) + j * h + i_h;
                send_l(idx) = data(k, j, halo_start_offset + i_h);
                send_r(idx) = data(k, j, halo_start_offset + nx_phys - h + i_h);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for("pack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                send_l(idx) = data(w, k, j, halo_start_offset + i_h);
                send_r(idx) = data(w, k, j, halo_start_offset + nx_phys - h + i_h);
        });
    }
    Kokkos::fence();

    // Kokkos::deep_copy(send_l_h, send_l);
    // Kokkos::deep_copy(send_r_h, send_r);

    HaloExchangeRequests req_obj;
    req_obj.requests.resize(4);

    if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);

    // if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r_h.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l_h.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l_h.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r_h.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);

    return req_obj;
}

template<size_t Dim>
void HaloExchanger::wait_exchange_halo_x(Field<Dim>& field, HaloExchangeRequests& reqs, int depth) const {
    if (reqs.count == 0) return;

    MPI_Waitall(reqs.count, reqs.requests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();

    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = grid_ref_.get_halo_cells();
    if (depth == -1) h = 1;
    else h = depth;

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    
    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(0);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);

    auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
    auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

    // auto recv_l_h = Kokkos::subview(recv_x_left_h_, std::make_pair((size_t)0, count));
    // auto recv_r_h = Kokkos::subview(recv_x_right_h_, std::make_pair((size_t)0, count));

    // Kokkos::deep_copy(recv_l, recv_l_h);
    // Kokkos::deep_copy(recv_r, recv_r_h);
    // Kokkos::fence();

    // Unpack data
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
            KOKKOS_LAMBDA(int j, int i_h) {
                if (neighbor_left != MPI_PROC_NULL) data(j, halo_start_offset - h + i_h) = recv_l(j * h + i_h);
                if (neighbor_right != MPI_PROC_NULL) data(j, halo_start_offset + nx_phys + i_h) = recv_r(j * h + i_h);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for("unpack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                const size_t idx = k * (ny * h) + j * h + i_h;
                if (neighbor_left != MPI_PROC_NULL) data(k, j, halo_start_offset - h + i_h) = recv_l(idx);
                if (neighbor_right != MPI_PROC_NULL) data(k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for("unpack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                if (neighbor_left != MPI_PROC_NULL) data(w, k, j, halo_start_offset - h + i_h) = recv_l(idx);
                if (neighbor_right != MPI_PROC_NULL) data(w, k, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
        });
    }
}

template<size_t Dim>
HaloExchangeRequests HaloExchanger::post_exchange_halo_y(Field<Dim>& field, int depth) const {
    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = grid_ref_.get_halo_cells();
    if (depth == -1) h = 1;
    else h = depth;
    if (h == 0) return {};

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    size_t count = 0;
    if constexpr (Dim == 2) count = static_cast<size_t>(h) * data.extent(1);
    else if constexpr (Dim == 3) count = static_cast<size_t>(h) * data.extent(2) * data.extent(0);
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);
    
    if (count == 0) return {};

    auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
    auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
    auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
    auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

    // auto send_b_h = Kokkos::subview(send_y_bottom_h_, std::make_pair((size_t)0, count));
    // auto recv_b_h = Kokkos::subview(recv_y_bottom_h_, std::make_pair((size_t)0, count));
    // auto send_t_h = Kokkos::subview(send_y_top_h_, std::make_pair((size_t)0, count));
    // auto recv_t_h = Kokkos::subview(recv_y_top_h_, std::make_pair((size_t)0, count));

    // Pack data
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, h}),
            KOKKOS_LAMBDA(int i, int j_h) {
                const size_t idx = j_h * nx + i;
                send_b(idx) = data(halo_start_offset + j_h, i);
                send_t(idx) = data(halo_start_offset + ny_phys - h + j_h, i);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("pack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                send_b(idx) = data(k, halo_start_offset + j_h, i);
                send_t(idx) = data(k, halo_start_offset + ny_phys - h + j_h, i);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, nx, h}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                send_b(idx) = data(w, k, halo_start_offset + j_h, i);
                send_t(idx) = data(w, k, halo_start_offset + ny_phys - h + j_h, i);
        });
    }
    Kokkos::fence();

    // Kokkos::deep_copy(send_b_h, send_b);
    // Kokkos::deep_copy(send_t_h, send_t);

    HaloExchangeRequests req_obj;
    req_obj.requests.resize(4);
    
    if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);

    // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b_h.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t_h.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t_h.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b_h.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);

    return req_obj;
}

template<size_t Dim>
void HaloExchanger::wait_exchange_halo_y(Field<Dim>& field, HaloExchangeRequests& reqs, int depth) const {
    if (reqs.count == 0) return;

    MPI_Waitall(reqs.count, reqs.requests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();

    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = grid_ref_.get_halo_cells();
    if (depth == -1) h = 1;
    else h = depth;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    size_t count = 0;
    if constexpr (Dim == 2) count = buffer_size_y_2d_;
    else if constexpr (Dim == 3) count = buffer_size_y_3d_;
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);

    auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
    auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

    // auto recv_b_h = Kokkos::subview(recv_y_bottom_h_, std::make_pair((size_t)0, count));
    // auto recv_t_h = Kokkos::subview(recv_y_top_h_, std::make_pair((size_t)0, count));

    // Kokkos::deep_copy(recv_b, recv_b_h);
    // Kokkos::deep_copy(recv_t, recv_t_h);
    // Kokkos::fence();

    // Unpack data
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, h}),
            KOKKOS_LAMBDA(int i, int j_h) {
                const size_t idx = j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(halo_start_offset - h + j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("unpack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(k, halo_start_offset - h + j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, nx, h}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(w, k, halo_start_offset - h + j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(w, k, halo_start_offset + ny_phys + j_h, i) = recv_t(idx);
        });
    }
}


inline void HaloExchanger::exchange_halos_slice(Field<3>& field, int k_layer) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny = data.extent(1);
    const int nx = data.extent(2);

    // --- Y-direction exchange for the slice ---
    {
        size_t count = buffer_size_slice_y_;
        if (count > 0) {
            auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
            auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
            auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
            auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

            // auto send_b_h = Kokkos::subview(send_y_bottom_h_, std::make_pair((size_t)0, count));
            // auto recv_b_h = Kokkos::subview(recv_y_bottom_h_, std::make_pair((size_t)0, count));
            // auto send_t_h = Kokkos::subview(send_y_top_h_, std::make_pair((size_t)0, count));
            // auto recv_t_h = Kokkos::subview(recv_y_top_h_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    send_b(idx) = data(k_layer, h + j_h, i);
                    send_t(idx) = data(k_layer, h + ny_phys - h + j_h, i);
            });
            Kokkos::fence();

            // Kokkos::deep_copy(send_b_h, send_b);
            // Kokkos::deep_copy(send_t_h, send_t);

            MPI_Request reqs[4];
            int req_count = 0;
            if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);

            // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b_h.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            // if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t_h.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            // if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t_h.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            // if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b_h.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            
            if(req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            Kokkos::fence();

            // Kokkos::deep_copy(recv_b, recv_b_h);
            // Kokkos::deep_copy(recv_t, recv_t_h);
            // Kokkos::fence();

            const int neighbor_bottom = neighbor_bottom_;
            const int neighbor_top = neighbor_top_;
            Kokkos::parallel_for("unpack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    if (neighbor_bottom != MPI_PROC_NULL) data(k_layer, j_h, i) = recv_b(idx);
                    if (neighbor_top != MPI_PROC_NULL) data(k_layer, h + ny_phys + j_h, i) = recv_t(idx);
            });
            Kokkos::fence();
        }
    }

    // --- X-direction exchange for the slice ---
    {
        size_t count = buffer_size_slice_x_;
        if (count > 0) {
            auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
            auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
            auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
            auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

            // auto send_l_h = Kokkos::subview(send_x_left_h_, std::make_pair((size_t)0, count));
            // auto recv_l_h = Kokkos::subview(recv_x_left_h_, std::make_pair((size_t)0, count));
            // auto send_r_h = Kokkos::subview(send_x_right_h_, std::make_pair((size_t)0, count));
            // auto recv_r_h = Kokkos::subview(recv_x_right_h_, std::make_pair((size_t)0, count));

            Kokkos::parallel_for("pack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    send_l(idx) = data(k_layer, j, h + i_h);
                    send_r(idx) = data(k_layer, j, h + nx_phys - h + i_h);
            });
            Kokkos::fence();

            // Kokkos::deep_copy(send_l_h, send_l);
            // Kokkos::deep_copy(send_r_h, send_r);

            MPI_Request reqs[4];
            int req_count = 0;
            if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);

            // if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r_h.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            // if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l_h.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            // if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l_h.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            // if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r_h.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            
            if(req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            Kokkos::fence();

            // Kokkos::deep_copy(recv_l, recv_l_h);
            // Kokkos::deep_copy(recv_r, recv_r_h);
            // Kokkos::fence();

            const int neighbor_left = neighbor_left_;
            const int neighbor_right = neighbor_right_;
            Kokkos::parallel_for("unpack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    if (neighbor_left != MPI_PROC_NULL) data(k_layer, j, i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k_layer, j, h + nx_phys + i_h) = recv_r(idx);
            });
            Kokkos::fence();
        }
    }
}


} // namespace Core
} // namespace VVM

#endif // VVM_CORE_HALOEXCHANGER_HPP


#endif
