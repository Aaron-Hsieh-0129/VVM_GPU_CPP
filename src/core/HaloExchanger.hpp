// HaloExchange processes halo data exchange between neighboring MPI ranks
// in a distributed grid simulation. It uses Kokkos for parallel execution
// and MPI for inter-process communication. The class provides methods to
// exchange halo data in the X, Y, and Z dimensions, ensuring that each rank
// has the necessary data from its neighbors for accurate computations.

#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "State.hpp"
#include <nvtx3/nvToolsExt.h>

namespace VVM {
namespace Core {

enum class HaloExchangeTags {
    X_LEFT_TO_RIGHT, // = 0
    X_RIGHT_TO_LEFT, // = 1
    Y_BOTTOM_TO_TOP, // = 2
    Y_TOP_TO_BOTTOM  // = 3
};

class HaloExchanger {
public:
    // Constructor
    explicit HaloExchanger(const Grid& grid);

    void exchange_halos(State& state) const;

    template<size_t Dim>
    void exchange_halos(Field<Dim>& field) const {
        if constexpr (Dim == 2) {
            nvtxRangePushA("HaloExchange_Y_2D");
            exchange_halo_y_2d(field);
            nvtxRangePop();

            nvtxRangePushA("HaloExchange_X_2D");
            exchange_halo_x_2d(field);
            nvtxRangePop();
        }
        else if constexpr (Dim == 3) {
            nvtxRangePushA("HaloExchange_Y_3D");
            exchange_halo_y_3d(field);
            nvtxRangePop();
            
            nvtxRangePushA("HaloExchange_X_3D");
            exchange_halo_x_3d(field);
            nvtxRangePop();
        }
        else if constexpr (Dim == 4) {
            nvtxRangePushA("HaloExchange_Y_4D");
            exchange_halo_y_4d(field);
            nvtxRangePop();
            
            nvtxRangePushA("HaloExchange_X_4D");
            exchange_halo_x_4d(field);
            nvtxRangePop();
        }
    }

    void exchange_halo_x_2d(Field<2>& field) const;
    void exchange_halo_y_2d(Field<2>& field) const;
    void exchange_halo_x_3d(Field<3>& field) const;
    void exchange_halo_y_3d(Field<3>& field) const;
    void exchange_halo_x_4d(Field<4>& field) const;
    void exchange_halo_y_4d(Field<4>& field) const;

private:
    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    int neighbors_x_[2];
    int neighbors_y_[2];
    int neighbors_z_[2]; //Not used in 3D, but can be extended for 3D halo exchange

};


inline HaloExchanger::HaloExchanger(const Grid& grid)
    : grid_ref_(grid), cart_comm_(grid.get_cart_comm()) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (cart_comm_ == MPI_COMM_NULL) {
        std::cerr << "Rank " << rank << ": HaloExchanger initialized with NULL communicator!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    MPI_Cart_shift(cart_comm_, 1, 1, &neighbors_x_[0], &neighbors_x_[1]);
    MPI_Cart_shift(cart_comm_, 0, 1, &neighbors_y_[0], &neighbors_y_[1]);
    neighbors_z_[0] = MPI_PROC_NULL;
    neighbors_z_[1] = MPI_PROC_NULL;
}


inline void HaloExchanger::exchange_halos(State& state) const {
    for (auto& field_pair : state) {
        std::visit([this](auto& field) {
            using T = std::decay_t<decltype(field)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                nvtxRangePushA(field.get_name().c_str());
                this->exchange_halos(field);
                nvtxRangePop();
            }
        }, field_pair.second);
    }
}

inline void HaloExchanger::exchange_halo_x_2d(Field<2>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int ny = data.extent(0);
    const int nx_phys = grid_ref_.get_local_physical_points_x();

    const size_t halo_size = static_cast<size_t>(ny) * h;
    Kokkos::View<double*> send_buf_l("s_buf_l", halo_size), recv_buf_l("r_buf_l", halo_size);
    Kokkos::View<double*> send_buf_r("s_buf_r", halo_size), recv_buf_r("r_buf_r", halo_size);
    
    Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
        KOKKOS_LAMBDA(int j, int i_h) {
            // pack left physical boundary to send left
            send_buf_l(j * h + i_h) = data(j, h + i_h);
            // pack right physical boundary to send right
            send_buf_r(j * h + i_h) = data(j, h + nx_phys - h + i_h);
    });

    MPI_Request reqs[4];
    MPI_Irecv(recv_buf_l.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], 0, cart_comm_, &reqs[0]);
    MPI_Irecv(recv_buf_r.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], 1, cart_comm_, &reqs[1]);
    MPI_Isend(send_buf_r.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], 0, cart_comm_, &reqs[2]);
    MPI_Isend(send_buf_l.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], 1, cart_comm_, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
        KOKKOS_LAMBDA(int j, int i_h) {
            // unpack into left halo
            data(j, i_h) = recv_buf_l(j * h + i_h);
            // unpack into right halo
            data(j, h + nx_phys + i_h) = recv_buf_r(j * h + i_h);
    });
    Kokkos::fence();
}

inline void HaloExchanger::exchange_halo_y_2d(Field<2>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0 || (neighbors_y_[0] == MPI_PROC_NULL && neighbors_y_[1] == MPI_PROC_NULL)) return;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nx = data.extent(1);

    const size_t halo_size = static_cast<size_t>(h) * nx;
    Kokkos::View<double*> send_buf_b("s_buf_b", halo_size), recv_buf_b("r_buf_b", halo_size);
    Kokkos::View<double*> send_buf_t("s_buf_t", halo_size), recv_buf_t("r_buf_t", halo_size);

    Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {h, nx}),
        KOKKOS_LAMBDA(int j_h, int i) {
            send_buf_b(j_h * nx + i) = data(h + j_h, i);
            send_buf_t(j_h * nx + i) = data(h + ny_phys - h + j_h, i);
    });

    MPI_Request reqs[4];
    MPI_Irecv(recv_buf_b.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], 0, cart_comm_, &reqs[0]);
    MPI_Irecv(recv_buf_t.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], 1, cart_comm_, &reqs[1]);
    MPI_Isend(send_buf_t.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], 0, cart_comm_, &reqs[2]);
    MPI_Isend(send_buf_b.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], 1, cart_comm_, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {h, nx}),
        KOKKOS_LAMBDA(int j_h, int i) {
            data(j_h, i) = recv_buf_b(j_h * nx + i);
            data(h + ny_phys + j_h, i) = recv_buf_t(j_h * nx + i);
    });
    Kokkos::fence();
}


inline void HaloExchanger::exchange_halo_x_3d(Field<3>& field) const {
    const int num_halo = grid_ref_.get_halo_cells();
    if (num_halo == 0 || (neighbors_x_[0] == MPI_PROC_NULL && neighbors_x_[1] == MPI_PROC_NULL)) {
        return;
    }

    auto field_data = field.get_mutable_device_data();
    const int nz = grid_ref_.get_local_total_points_z();
    const int ny = grid_ref_.get_local_total_points_y();
    const int nx = grid_ref_.get_local_total_points_x();
    const int nx_phys = grid_ref_.get_local_physical_points_x();

    const size_t halo_size = static_cast<size_t>(nz) * ny * num_halo;

    Kokkos::View<double*> send_to_right_buf("send_to_right_buf", halo_size);
    Kokkos::View<double*> recv_from_left_buf("recv_from_left_buf", halo_size);
    Kokkos::View<double*> send_to_left_buf("send_to_left_buf", halo_size);
    Kokkos::View<double*> recv_from_right_buf("recv_from_right_buf", halo_size);

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {nz, ny, num_halo});
    
    nvtxRangePushA("Pack_X");
    // Step 1: Pack data
    // Pack right-most physical data to send to the RIGHT neighbor
    if (neighbors_x_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_to_send_right", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            send_to_right_buf(k*ny*num_halo + j*num_halo + i_h) = field_data(k, j, nx - 2*num_halo + i_h);
        });
    }
    // Pack left-most physical data to send to the LEFT neighbor
    if (neighbors_x_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_to_send_left", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            send_to_left_buf(k*ny*num_halo + j*num_halo + i_h) = field_data(k, j, num_halo + i_h);
        });
    }
    Kokkos::fence();
    nvtxRangePop();

    nvtxRangePushA("MPI_SendRecv_X");
    // Step 2: MPI Communication
    MPI_Request requests[4];
    int req_count = 0;
    
    // Post all receives first
    // Receive from LEFT neighbor (neighbors_x_[0])
    if (neighbors_x_[0] != MPI_PROC_NULL) {
        MPI_Irecv(recv_from_left_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], static_cast<int>(HaloExchangeTags::X_RIGHT_TO_LEFT), cart_comm_, &requests[req_count++]);
    }
    // Receive from RIGHT neighbor (neighbors_x_[1])
    if (neighbors_x_[1] != MPI_PROC_NULL) {
        MPI_Irecv(recv_from_right_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], static_cast<int>(HaloExchangeTags::X_LEFT_TO_RIGHT), cart_comm_, &requests[req_count++]);
    }

    // Then post all sends
    // Send LEFT data (send_to_left_buf) to RIGHT neighbor (neighbors_x_[1])
    if (neighbors_x_[1] != MPI_PROC_NULL) { 
        MPI_Isend(send_to_left_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], static_cast<int>(HaloExchangeTags::X_LEFT_TO_RIGHT), cart_comm_, &requests[req_count++]);
    }
    // Send RIGHT data (send_to_right_buf) to LEFT neighbor (neighbors_x_[0])
    if (neighbors_x_[0] != MPI_PROC_NULL) {
        MPI_Isend(send_to_right_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], static_cast<int>(HaloExchangeTags::X_RIGHT_TO_LEFT), cart_comm_, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    nvtxRangePop();

    nvtxRangePushA("Unpack_X");
    // Step 3: Unpack data
    // Data from LEFT neighbor goes into the LEFT halo
    if (neighbors_x_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_from_left", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            field_data(k, j, i_h) = recv_from_left_buf(k*ny*num_halo + j*num_halo + i_h);
        });
    }
    // Data from RIGHT neighbor goes into the RIGHT halo
    if (neighbors_x_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_from_right", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            field_data(k, j, nx - num_halo + i_h) = recv_from_right_buf(k*ny*num_halo + j*num_halo + i_h);
        });
    }
    Kokkos::fence();
    nvtxRangePop();
}

inline void HaloExchanger::exchange_halo_y_3d(Field<3>& field) const {
    const int num_halo = grid_ref_.get_halo_cells();
    if (num_halo == 0 || (neighbors_y_[0] == MPI_PROC_NULL && neighbors_y_[1] == MPI_PROC_NULL)) {
        return;
    }

    auto field_data = field.get_mutable_device_data();
    const int nz = grid_ref_.get_local_total_points_z();
    const int ny = grid_ref_.get_local_total_points_y();
    const int nx = grid_ref_.get_local_total_points_x();

    const size_t halo_size = static_cast<size_t>(nz) * num_halo * nx;

    Kokkos::View<double*> send_to_top_buf("send_top_buf", halo_size);
    Kokkos::View<double*> recv_from_bottom_buf("recv_from_bottom_buf", halo_size);
    Kokkos::View<double*> send_to_bottom_buf("send_bottom_buf", halo_size);
    Kokkos::View<double*> recv_from_top_buf("recv_top_buf", halo_size);

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {nz, num_halo, nx});
    
    nvtxRangePushA("Pack_Y");
    // Step 1: Pack data
    // Pack top-most physical data to send to the TOP neighbor
    if (neighbors_y_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_to_send_top", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            send_to_top_buf(k*num_halo*nx + j_h*nx + i) = field_data(k, ny - 2*num_halo + j_h, i);
        });
    }
    // Pack bottom-most physical data to send to the BOTTOM neighbor
    if (neighbors_y_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_to_send_bottom", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            send_to_bottom_buf(k*num_halo*nx + j_h*nx + i) = field_data(k, num_halo + j_h, i);
        });
    }
    Kokkos::fence();
    nvtxRangePop();

    nvtxRangePushA("MPI_SendRecv_Y");
    // Step 2: MPI Communication
    MPI_Request requests[4];
    int req_count = 0;

    // Post all receives first
    // Receive from BOTTOM neighbor (neighbors_y_[0])
    if (neighbors_y_[0] != MPI_PROC_NULL) {
        MPI_Irecv(recv_from_bottom_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], static_cast<int>(HaloExchangeTags::Y_TOP_TO_BOTTOM), cart_comm_, &requests[req_count++]);
    }
    // Receive from TOP neighbor (neighbors_y_[1])
    if (neighbors_y_[1] != MPI_PROC_NULL) {
        MPI_Irecv(recv_from_top_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], static_cast<int>(HaloExchangeTags::Y_BOTTOM_TO_TOP), cart_comm_, &requests[req_count++]);
    }

    // Then post all sends
    // Send TOP data (send_to_top_buf) to BOTTOM neighbor (neighbors_y_[0])
    if (neighbors_y_[0] != MPI_PROC_NULL) {
        MPI_Isend(send_to_top_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], static_cast<int>(HaloExchangeTags::Y_TOP_TO_BOTTOM), cart_comm_, &requests[req_count++]);
    }
    // Send BOTTOM data (send_to_bottom_buf) to TOP neighbor (neighbors_y_[1])
    if (neighbors_y_[1] != MPI_PROC_NULL) {
        MPI_Isend(send_to_bottom_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], static_cast<int>(HaloExchangeTags::Y_BOTTOM_TO_TOP), cart_comm_, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    nvtxRangePop();

    nvtxRangePushA("Unpack_Y");
    // Step 3: Unpack data
    // Data from BOTTOM neighbor goes into the BOTTOM halo
    if (neighbors_y_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_from_bottom", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            field_data(k, j_h, i) = recv_from_bottom_buf(k*num_halo*nx + j_h*nx + i);
        });
    }
    // Data from TOP neighbor goes into the TOP halo
    if (neighbors_y_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_from_top", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            field_data(k, ny - num_halo + j_h, i) = recv_from_top_buf(k*num_halo*nx + j_h*nx + i);
        });
    }
    Kokkos::fence();
    nvtxRangePop();
}


inline void HaloExchanger::exchange_halo_x_4d(Field<4>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nw = data.extent(0);
    const int nz = data.extent(1);
    const int ny = data.extent(2);
    const int nx_phys = grid_ref_.get_local_physical_points_x();

    const size_t halo_size = static_cast<size_t>(nw) * nz * ny * h;
    Kokkos::View<double*> send_buf_l("s_buf_l_4d", halo_size), recv_buf_l("r_buf_l_4d", halo_size);
    Kokkos::View<double*> send_buf_r("s_buf_r_4d", halo_size), recv_buf_r("r_buf_r_4d", halo_size);
    
    Kokkos::parallel_for("pack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
        KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
            size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
            send_buf_l(idx) = data(w, k, j, h + i_h);
            send_buf_r(idx) = data(w, k, j, h + nx_phys - h + i_h);
    });

    MPI_Request reqs[4];
    MPI_Irecv(recv_buf_l.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], 0, cart_comm_, &reqs[0]);
    MPI_Irecv(recv_buf_r.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], 1, cart_comm_, &reqs[1]);
    MPI_Isend(send_buf_r.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], 0, cart_comm_, &reqs[2]);
    MPI_Isend(send_buf_l.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], 1, cart_comm_, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    Kokkos::parallel_for("unpack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
        KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
            size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
            data(w, k, j, i_h) = recv_buf_l(idx);
            data(w, k, j, h + nx_phys + i_h) = recv_buf_r(idx);
    });
    Kokkos::fence();
}

inline void HaloExchanger::exchange_halo_y_4d(Field<4>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0 || (neighbors_y_[0] == MPI_PROC_NULL && neighbors_y_[1] == MPI_PROC_NULL)) return;
    
    auto data = field.get_mutable_device_data();
    const int nw = data.extent(0);
    const int nz = data.extent(1);
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    const int nx = data.extent(3);

    const size_t halo_size = static_cast<size_t>(nw) * nz * h * nx;
    Kokkos::View<double*> send_buf_b("s_buf_b_4d", halo_size), recv_buf_b("r_buf_b_4d", halo_size);
    Kokkos::View<double*> send_buf_t("s_buf_t_4d", halo_size), recv_buf_t("r_buf_t_4d", halo_size);

    Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, h, nx}),
        KOKKOS_LAMBDA(int w, int k, int j_h, int i) {
            size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
            send_buf_b(idx) = data(w, k, h + j_h, i);
            send_buf_t(idx) = data(w, k, h + ny_phys - h + j_h, i);
    });

    MPI_Request reqs[4];
    MPI_Irecv(recv_buf_b.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], 0, cart_comm_, &reqs[0]);
    MPI_Irecv(recv_buf_t.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], 1, cart_comm_, &reqs[1]);
    MPI_Isend(send_buf_t.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], 0, cart_comm_, &reqs[2]);
    MPI_Isend(send_buf_b.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], 1, cart_comm_, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, h, nx}),
        KOKKOS_LAMBDA(int w, int k, int j_h, int i) {
            size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
            data(w, k, j_h, i) = recv_buf_b(idx);
            data(w, k, h + ny_phys + j_h, i) = recv_buf_t(idx);
    });
    Kokkos::fence();
}


} // namespace Core
} // namespace VVM

#endif // VVM_CORE_HALOEXCHANGER_HPP