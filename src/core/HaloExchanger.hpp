#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "State.hpp"

namespace VVM {
namespace Core {

enum class HaloExchangeTags {
    SEND_TO_RIGHT = 10, // X-direction
    SEND_TO_LEFT  = 11, // X-direction
    SEND_TO_TOP    = 20, // Y-direction
    SEND_TO_BOTTOM = 21  // Y-direction
};

class HaloExchanger {
public:
    explicit HaloExchanger(const Grid& grid);

    void exchange_halos(State& state) const;

    template<size_t Dim>
    void exchange_halos(Field<Dim>& field) const {
        if constexpr (Dim >= 2) {
            exchange_halo_y(field);
            exchange_halo_x(field);
        }
    }

    template<size_t Dim>
    void exchange_halo_x(Field<Dim>& field) const;

    template<size_t Dim>
    void exchange_halo_y(Field<Dim>& field) const;

private:
    const Grid& grid_ref_;
    MPI_Comm cart_comm_;
    int neighbor_left_, neighbor_right_;
    int neighbor_bottom_, neighbor_top_;
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
void HaloExchanger::exchange_halo_x(Field<Dim>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();

    // Packing size
    size_t count = h;
    for(int d = 0; d < Dim - 1; ++d) {
        count *= data.extent(d);
    }
    if (count == 0) return;

    Kokkos::View<double*> send_l("send_left_buf", count), recv_l("recv_left_buf", count);
    Kokkos::View<double*> send_r("send_right_buf", count), recv_r("recv_right_buf", count);

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

    // MPI Communication using Sendrecv to avoid deadlock
    MPI_Sendrecv(send_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT),
                 recv_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_LEFT),
                 cart_comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_r.data(), count, MPI_DOUBLE, neighbor_right_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT),
                 recv_l.data(), count, MPI_DOUBLE, neighbor_left_, static_cast<int>(HaloExchangeTags::SEND_TO_RIGHT),
                 cart_comm_, MPI_STATUS_IGNORE);

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
void HaloExchanger::exchange_halo_y(Field<Dim>& field) const {
    const int h = grid_ref_.get_halo_cells();
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int ny_phys = grid_ref_.get_local_physical_points_y();

    size_t count = h * data.extent(Dim-1); // h * nx
    for(int d = 0; d < Dim - 2; ++d) {
        count *= data.extent(d);
    }
    if (count == 0) return;

    Kokkos::View<double*> send_b("send_bottom_buf", count), recv_b("recv_bottom_buf", count);
    Kokkos::View<double*> send_t("send_top_buf", count), recv_t("recv_top_buf", count);

    // Pack data
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("pack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {h, nx}),
            KOKKOS_LAMBDA(int j_h, int i) {
                const size_t idx = j_h * nx + i;
                send_b(idx) = data(h + j_h, i);
                send_t(idx) = data(h + ny_phys - h + j_h, i);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("pack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, h, nx}),
            KOKKOS_LAMBDA(int k, int j_h, int i) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                send_b(idx) = data(k, h + j_h, i);
                send_t(idx) = data(k, h + ny_phys - h + j_h, i);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("pack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, h, nx}),
            KOKKOS_LAMBDA(int w, int k, int j_h, int i) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                send_b(idx) = data(w, k, h + j_h, i);
                send_t(idx) = data(w, k, h + ny_phys - h + j_h, i);
        });
    }
    Kokkos::fence();

    MPI_Sendrecv(send_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP),
                 recv_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_TOP),
                 cart_comm_, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_b.data(), count, MPI_DOUBLE, neighbor_bottom_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM),
                 recv_t.data(), count, MPI_DOUBLE, neighbor_top_, static_cast<int>(HaloExchangeTags::SEND_TO_BOTTOM),
                 cart_comm_, MPI_STATUS_IGNORE);

    // Unpack data
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;
    if constexpr (Dim == 2) {
        const int nx = data.extent(1);
        Kokkos::parallel_for("unpack_y_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {h, nx}),
            KOKKOS_LAMBDA(int j_h, int i) {
                const size_t idx = j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(h + ny_phys + j_h, i) = recv_t(idx);
        });
    } 
    else if constexpr (Dim == 3) {
        const int nz = data.extent(0);
        const int nx = data.extent(2);
        Kokkos::parallel_for("unpack_y_3d", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nz, h, nx}),
            KOKKOS_LAMBDA(int k, int j_h, int i) {
                const size_t idx = k * (h * nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(k, j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(k, h + ny_phys + j_h, i) = recv_t(idx);
        });
    } 
    else if constexpr (Dim == 4) {
        const int nw = data.extent(0);
        const int nz = data.extent(1);
        const int nx = data.extent(3);
        Kokkos::parallel_for("unpack_y_4d", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {nw, nz, h, nx}),
            KOKKOS_LAMBDA(int w, int k, int j_h, int i) {
                size_t idx = w * (nz*h*nx) + k * (h*nx) + j_h * nx + i;
                if (neighbor_bottom != MPI_PROC_NULL) data(w, k, j_h, i) = recv_b(idx);
                if (neighbor_top != MPI_PROC_NULL) data(w, k, h + ny_phys + j_h, i) = recv_t(idx);
        });
    }
    Kokkos::fence();
}

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_HALOEXCHANGER_HPP
