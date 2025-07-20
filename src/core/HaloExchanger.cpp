#include "HaloExchanger.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>
#include <nvtx3/nvToolsExt.h>

namespace VVM {
namespace Core {

// Constructor
HaloExchanger::HaloExchanger(const Grid& grid)
    : grid_ref_(grid),
      cart_comm_(MPI_COMM_NULL)
{
    cart_comm_ = grid_ref_.get_cart_comm();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (cart_comm_ == MPI_COMM_NULL) {
        std::cerr << "Rank " << rank << ": HaloExchanger initialized with NULL communicator from Grid!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // dimension 0 = Y-axis, dimension 1 = X-axis
    // For a positive shift (+1), source is the neighbor in the negative direction, dest is in the positive.
    // neighbors_x_[0] gets LEFT neighbor (source of rightward shift)
    // neighbors_x_[1] gets RIGHT neighbor (dest of rightward shift)
    MPI_Cart_shift(cart_comm_, 1, 1, &neighbors_x_[0], &neighbors_x_[1]); // X-shift

    // neighbors_y_[0] gets BOTTOM neighbor (source of upward shift)
    // neighbors_y_[1] gets TOP neighbor (dest of upward shift)
    MPI_Cart_shift(cart_comm_, 0, 1, &neighbors_y_[0], &neighbors_y_[1]); // Y-shift
    
    neighbors_z_[0] = MPI_PROC_NULL;
    neighbors_z_[1] = MPI_PROC_NULL;

    // This print statement is for debugging and can be removed in production.
    std::cout << "Rank " << rank << ": HaloExchanger initialized. Neighbors: "
              << "X- (Left): " << neighbors_x_[0] << ", X+ (Right): " << neighbors_x_[1]
              << ", Y- (Bottom): " << neighbors_y_[0] << ", Y+ (Top): " << neighbors_y_[1] << std::endl;
}

// Main method for a field
void HaloExchanger::exchange_halos(Field& field) const {
    nvtxRangePushA("HaloExchange_X");
    exchange_halo_x(field);
    nvtxRangePop();

    nvtxRangePushA("HaloExchange_Y");
    exchange_halo_y(field);
    nvtxRangePop();
}

// Main method for State
void HaloExchanger::exchange_halos(State& state) const {
    for (auto& field_pair : state) {
        nvtxRangePushA(field_pair.second.get_name().c_str());
        exchange_halos(field_pair.second);
        nvtxRangePop();
    }
}

void HaloExchanger::exchange_halo_x(Field& field) const {
    const int num_halo = grid_ref_.get_halo_cells();
    if (num_halo == 0 || (neighbors_x_[0] == MPI_PROC_NULL && neighbors_x_[1] == MPI_PROC_NULL)) {
        return;
    }

    auto field_data = field.get_mutable_device_data();
    const int nz = grid_ref_.get_local_total_points_z();
    const int ny = grid_ref_.get_local_total_points_y();
    const int nx = grid_ref_.get_local_total_points_x();
    
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

void HaloExchanger::exchange_halo_y(Field& field) const {
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

} // namespace Core
} // namespace VVM