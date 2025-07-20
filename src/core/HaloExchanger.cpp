#include "HaloExchanger.hpp"
#include <iostream>
#include <Kokkos_Core.hpp>
#include <nvtx3/nvToolsExt.h>

namespace VVM {
namespace Core {

// Constructor implementation
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
    MPI_Cart_shift(cart_comm_, 1, 1, &neighbors_x_[0], &neighbors_x_[1]); // X-shift
    MPI_Cart_shift(cart_comm_, 0, 1, &neighbors_y_[0], &neighbors_y_[1]); // Y-shift
    
    neighbors_z_[0] = MPI_PROC_NULL;
    neighbors_z_[1] = MPI_PROC_NULL;

    std::cout << "Rank " << rank << ": HaloExchanger initialized. Neighbors: "
              << "X- : " << neighbors_x_[0] << ", X+ : " << neighbors_x_[1]
              << ", Y- : " << neighbors_y_[0] << ", Y+ : " << neighbors_y_[1] << std::endl;
}

// Main method
void HaloExchanger::exchange_halos(Field& field) const {
    nvtxRangePushA("HaloExchange_X");
    exchange_halo_x(field);
    nvtxRangePop();

    nvtxRangePushA("HaloExchange_Y");
    exchange_halo_y(field);
    nvtxRangePop();
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
    if (neighbors_x_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_x_right", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            send_to_right_buf(k*ny*num_halo + j*num_halo + i_h) = field_data(k, j, nx - 2*num_halo + i_h);
        });
    }
    if (neighbors_x_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_x_left", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            send_to_left_buf(k*ny*num_halo + j*num_halo + i_h) = field_data(k, j, num_halo + i_h);
        });
    }
    Kokkos::fence();
    nvtxRangePop();

    nvtxRangePushA("MPI_Send_X");
    // Step 2: MPI Communication
    MPI_Request requests[4];
    int req_count = 0;
    
    // Post all receives first
    if (neighbors_x_[0] != MPI_PROC_NULL) { // Recv from Left
        MPI_Irecv(recv_from_left_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], 0, cart_comm_, &requests[req_count++]);
    }
    if (neighbors_x_[1] != MPI_PROC_NULL) { // Recv from Right
        MPI_Irecv(recv_from_right_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], 1, cart_comm_, &requests[req_count++]);
    }

    // Then post all sends
    if (neighbors_x_[1] != MPI_PROC_NULL) { // Send to Right
        MPI_Isend(send_to_right_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[1], 1, cart_comm_, &requests[req_count++]);
    }
    if (neighbors_x_[0] != MPI_PROC_NULL) { // Send to Left
        MPI_Isend(send_to_left_buf.data(), halo_size, MPI_DOUBLE, neighbors_x_[0], 0, cart_comm_, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    nvtxRangePop();

    nvtxRangePushA("Unpack_X");
    // Step 3: Unpack data
    if (neighbors_x_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_x_right", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            field_data(k, j, nx - num_halo + i_h) = recv_from_left_buf(k*ny*num_halo + j*num_halo + i_h);
        });
    }
    if (neighbors_x_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_x_left", policy, KOKKOS_LAMBDA(int k, int j, int i_h) {
            field_data(k, j, i_h) = recv_from_right_buf(k*ny*num_halo + j*num_halo + i_h);
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
    if (neighbors_y_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_y_top", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            send_to_top_buf(k*num_halo*nx + j_h*nx + i) = field_data(k, ny - 2*num_halo + j_h, i);
        });
    }
    if (neighbors_y_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("pack_y_bottom", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            send_to_bottom_buf(k*num_halo*nx + j_h*nx + i) = field_data(k, num_halo + j_h, i);
        });
    }
    Kokkos::fence();
    nvtxRangePop();

    nvtxRangePushA("MPI_Send_Y");
    // Step 2: MPI Communication
    MPI_Request requests[4];
    int req_count = 0;

    // Post all receives first
    if (neighbors_y_[0] != MPI_PROC_NULL) { // Recv from Bottom
        MPI_Irecv(recv_from_bottom_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], 2, cart_comm_, &requests[req_count++]);
    }
    if (neighbors_y_[1] != MPI_PROC_NULL) { // Recv from Top
        MPI_Irecv(recv_from_top_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], 3, cart_comm_, &requests[req_count++]);
    }

    // Then post all sends
    if (neighbors_y_[1] != MPI_PROC_NULL) { // Send to Top
        MPI_Isend(send_to_top_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[1], 3, cart_comm_, &requests[req_count++]);
    }
    if (neighbors_y_[0] != MPI_PROC_NULL) { // Send to Bottom
        MPI_Isend(send_to_bottom_buf.data(), halo_size, MPI_DOUBLE, neighbors_y_[0], 2, cart_comm_, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    nvtxRangePop();

    nvtxRangePushA("Unpack_Y");
    // Step 3: Unpack data
    if (neighbors_y_[0] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_y_bottom", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            field_data(k, j_h, i) = recv_from_bottom_buf(k*num_halo*nx + j_h*nx + i);
        });
    }
    if (neighbors_y_[1] != MPI_PROC_NULL) {
        Kokkos::parallel_for("unpack_y_top", policy, KOKKOS_LAMBDA(int k, int j_h, int i) {
            field_data(k, ny - num_halo + j_h, i) = recv_from_top_buf(k*num_halo*nx + j_h*nx + i);
        });
    }
    Kokkos::fence();
}

// Z-axis function remains unchanged
void HaloExchanger::exchange_halo_z(Field& field) const {}

} // namespace Core
} // namespace VVM