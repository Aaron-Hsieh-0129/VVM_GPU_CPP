#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "State.hpp"
#include "utils/Timer.hpp"
#include "utils/TimingManager.hpp"
#include <algorithm>
#include <vector>

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
    void exchange_halos(Field<Dim>& field) const {
        VVM::Utils::Timer exchange_halos_timer("Exchnage_halos");
        if constexpr (Dim >= 2) {
            auto reqs_y = post_exchange_halo_y(field);
            wait_exchange_halo_y(field, reqs_y);

            auto reqs_x = post_exchange_halo_x(field);
            wait_exchange_halo_x(field, reqs_x);
        }
    }

    // --- Asynchronous Halo Exchange Functions ---
    template<size_t Dim>
    HaloExchangeRequests post_exchange_halo_x(Field<Dim>& field) const;

    template<size_t Dim>
    void wait_exchange_halo_x(Field<Dim>& field, HaloExchangeRequests& reqs) const;

    template<size_t Dim>
    HaloExchangeRequests post_exchange_halo_y(Field<Dim>& field) const;

    template<size_t Dim>
    void wait_exchange_halo_y(Field<Dim>& field, HaloExchangeRequests& reqs) const;

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
        }
        if (max_buffer_size_y > 0) {
            send_y_bottom_ = Kokkos::View<double*>("send_y_bottom_buf", max_buffer_size_y);
            recv_y_bottom_ = Kokkos::View<double*>("recv_y_bottom_buf", max_buffer_size_y);
            send_y_top_    = Kokkos::View<double*>("send_y_top_buf", max_buffer_size_y);
            recv_y_top_    = Kokkos::View<double*>("recv_y_top_buf", max_buffer_size_y);
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
HaloExchangeRequests HaloExchanger::post_exchange_halo_x(Field<Dim>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return {};

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    
    size_t count = 0;
    if constexpr (Dim == 2) count = buffer_size_x_2d_;
    else if constexpr (Dim == 3) count = buffer_size_x_3d_;
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);

    if (count == 0) return {};
    
    auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count));
    auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
    auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count));
    auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

    // Pack data
    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
            KOKKOS_LAMBDA(int j, int i_h) {
                send_l(j * h + i_h) = data(j, h + i_h);
                send_r(j * h + i_h) = data(j, h + nx_phys - h + i_h);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for("pack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                const size_t idx = k * (ny * h) + j * h + i_h;
                send_l(idx) = data(k, j, h + i_h);
                send_r(idx) = data(k, j, h + nx_phys - h + i_h);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for("pack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                send_l(idx) = data(w, k, j, h + i_h);
                send_r(idx) = data(w, k, j, h + nx_phys - h + i_h);
        });
    }
    Kokkos::fence();

    HaloExchangeRequests req_obj;
    req_obj.requests.resize(4);

    if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT), cart_comm_, &req_obj.requests[req_obj.count++]);

    return req_obj;
}

template<size_t Dim>
void HaloExchanger::wait_exchange_halo_x(Field<Dim>& field, HaloExchangeRequests& reqs) const {
    if (reqs.count == 0) return;

    MPI_Waitall(reqs.count, reqs.requests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();

    const int h = grid_ref_.get_halo_cells();
    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    
    size_t count = 0;
    if constexpr (Dim == 2) count = buffer_size_x_2d_;
    else if constexpr (Dim == 3) count = buffer_size_x_3d_;
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);

    auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count));
    auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count));

    // Unpack data
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    if constexpr (Dim == 2) {
        const int ny = data.extent(0);
        Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {ny, h}),
            KOKKOS_LAMBDA(int j, int i_h) {
                if (neighbor_left != MPI_PROC_NULL) data(j, i_h) = recv_l(j * h + i_h);
                if (neighbor_right != MPI_PROC_NULL) data(j, h + nx_phys + i_h) = recv_r(j * h + i_h);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int ny = data.extent(1);
        Kokkos::parallel_for("unpack_x_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, ny, h}),
            KOKKOS_LAMBDA(int k, int j, int i_h) {
                const size_t idx = k * (ny * h) + j * h + i_h;
                if (neighbor_left != MPI_PROC_NULL) data(k, j, i_h) = recv_l(idx);
                if (neighbor_right != MPI_PROC_NULL) data(k, j, h + nx_phys + i_h) = recv_r(idx);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int ny = data.extent(2);
        Kokkos::parallel_for("unpack_x_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, ny, h}),
            KOKKOS_LAMBDA(int w, int k, int j, int i_h) {
                size_t idx = w * (nz*ny*h) + k * (ny*h) + j * h + i_h;
                if (neighbor_left != MPI_PROC_NULL) data(w, k, j, i_h) = recv_l(idx);
                if (neighbor_right != MPI_PROC_NULL) data(w, k, j, h + nx_phys + i_h) = recv_r(idx);
        });
    }
    Kokkos::fence();
}

template<size_t Dim>
HaloExchangeRequests HaloExchanger::post_exchange_halo_y(Field<Dim>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return {};

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    size_t count = 0;
    if constexpr (Dim == 2) count = buffer_size_y_2d_;
    else if constexpr (Dim == 3) count = buffer_size_y_3d_;
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);
    
    if (count == 0) return {};

    auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count));
    auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
    auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count));
    auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

    // Pack data
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, h}),
            KOKKOS_LAMBDA(int i, int j_h) {
                const size_t idx = j_h * nx + i;
                send_b(idx) = data(h + j_h, i);
                send_t(idx) = data(h + ny_phys - h + j_h, i);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("pack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                send_b(idx) = data(k, h + j_h, i);
                send_t(idx) = data(k, h + ny_phys - h + j_h, i);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, nx, h}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                send_b(idx) = data(w, k, h + j_h, i);
                send_t(idx) = data(w, k, h + ny_phys - h + j_h, i);
        });
    }
    Kokkos::fence();

    HaloExchangeRequests req_obj;
    req_obj.requests.resize(4);
    
    if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP), cart_comm_, &req_obj.requests[req_obj.count++]);
    if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM), cart_comm_, &req_obj.requests[req_obj.count++]);

    return req_obj;
}

template<size_t Dim>
void HaloExchanger::wait_exchange_halo_y(Field<Dim>& field, HaloExchangeRequests& reqs) const {
    if (reqs.count == 0) return;

    MPI_Waitall(reqs.count, reqs.requests.data(), MPI_STATUSES_IGNORE);
    Kokkos::fence();

    const int h = grid_ref_.get_halo_cells();
    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    size_t count = 0;
    if constexpr (Dim == 2) count = buffer_size_y_2d_;
    else if constexpr (Dim == 3) count = buffer_size_y_3d_;
    else if constexpr (Dim == 4) count = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);

    auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count));
    auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count));

    // Unpack data
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {nx, h}),
            KOKKOS_LAMBDA(int i, int j_h) {
                const size_t idx = j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(h + ny_phys + j_h, i) = recv_t(idx);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("unpack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, nx, h}),
            KOKKOS_LAMBDA(int k, int i, int j_h) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(k, j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(k, h + ny_phys + j_h, i) = recv_t(idx);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, nx, h}),
            KOKKOS_LAMBDA(int w, int k, int i, int j_h) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(w, k, j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(w, k, h + ny_phys + j_h, i) = recv_t(idx);
        });
    }
    Kokkos::fence();
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

            Kokkos::parallel_for("pack_y_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, h}),
                KOKKOS_LAMBDA(int i, int j_h) {
                    const size_t idx = static_cast<size_t>(j_h) * nx + i;
                    send_b(idx) = data(k_layer, h + j_h, i);
                    send_t(idx) = data(k_layer, h + ny_phys - h + j_h, i);
            });
            Kokkos::fence();

            MPI_Request reqs[4];
            int req_count = 0;
            if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Irecv(recv_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            if(neighbor_top_    != MPI_PROC_NULL) MPI_Irecv(recv_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            if(neighbor_top_    != MPI_PROC_NULL) MPI_Isend(send_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_TOP), cart_comm_, &reqs[req_count++]);
            if(neighbor_bottom_ != MPI_PROC_NULL) MPI_Isend(send_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_BOTTOM), cart_comm_, &reqs[req_count++]);
            
            if(req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            Kokkos::fence();

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

            Kokkos::parallel_for("pack_x_slice", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    const size_t idx = static_cast<size_t>(j) * h + i_h;
                    send_l(idx) = data(k_layer, j, h + i_h);
                    send_r(idx) = data(k_layer, j, h + nx_phys - h + i_h);
            });
            Kokkos::fence();

            MPI_Request reqs[4];
            int req_count = 0;
            if(neighbor_right_ != MPI_PROC_NULL) MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            if(neighbor_left_  != MPI_PROC_NULL) MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            if(neighbor_left_  != MPI_PROC_NULL) MPI_Isend(send_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_LEFT), cart_comm_, &reqs[req_count++]);
            if(neighbor_right_ != MPI_PROC_NULL) MPI_Isend(send_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SLICE_SEND_TO_RIGHT), cart_comm_, &reqs[req_count++]);
            
            if(req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            Kokkos::fence();

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
