#ifndef VVM_CORE_HALOEXCHANGER_HPP
#define VVM_CORE_HALOEXCHANGER_HPP

#include "Grid.hpp"
#include "Field.hpp"
#include "State.hpp"
#include <nccl.h>
#include <cuda_runtime.h>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <vector>

namespace VVM {
namespace Core {

// 設定執行空間為 CUDA
using ExecSpace = Kokkos::Cuda;

class HaloExchanger {
public:
    explicit HaloExchanger(const Grid& grid, ncclComm_t nccl_comm, cudaStream_t stream);
    ~HaloExchanger();

    // 禁止複製 (資源管理類別不應被複製)
    HaloExchanger(const HaloExchanger&) = delete;
    HaloExchanger& operator=(const HaloExchanger&) = delete;

    // 主要入口：自動處理 Graph Capture 與 Launch
    void exchange_halos(State& state);

    // 針對單一 Field 的交換 (若在 Graph 外部呼叫會執行同步)
    template<size_t Dim>
    void exchange_halos(Field<Dim>& field, int depth = -1) const {
        exchange_halos_impl(field, depth);
        // 如果不是在 Graph 錄製模式下，手動呼叫通常需要同步以確保數據可用
        if (depth == -1 && !is_graph_created_) {
             cudaStreamSynchronize(stream_);
        }
    }

    // 內部實作函數 (被 Graph Capture 使用)
    template<size_t Dim>
    void exchange_halos_impl(Field<Dim>& field, int depth = -1) const;

    // 特定 Slice 交換 (通常用於 IO 或邊界條件，不放入主 Graph)
    void exchange_halos_slice(Field<3>& field, int k_layer) const;
    
    // 輔助函數：交換頂層 Slice
    void exchange_halos_top_slice(Field<3>& field) const {
        const int nz = grid_ref_.get_local_total_points_z();
        const int h = grid_ref_.get_halo_cells();
        // 確保取的是最後一層物理網格
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

    // CUDA Graph 相關狀態
    bool is_graph_created_ = false;
    cudaGraphExec_t graph_exec_ = nullptr;

    // Kokkos Views (Device Memory for Buffers)
    // 使用 mutable 讓 const 函數也能寫入 Buffer
    mutable Kokkos::View<double*, ExecSpace> send_x_left_, recv_x_left_;
    mutable Kokkos::View<double*, ExecSpace> send_x_right_, recv_x_right_;
    mutable Kokkos::View<double*, ExecSpace> send_y_bottom_, recv_y_bottom_;
    mutable Kokkos::View<double*, ExecSpace> send_y_top_, recv_y_top_;

    size_t buffer_size_x_2d_, buffer_size_y_2d_;
    size_t buffer_size_x_3d_, buffer_size_y_3d_;
    size_t buffer_size_x_4d_, buffer_size_y_4d_;
    size_t buffer_size_slice_x_, buffer_size_slice_y_;
};

// -------------------------------------------------------------------------
// 實作內容
// -------------------------------------------------------------------------

inline HaloExchanger::HaloExchanger(const Grid& grid, ncclComm_t nccl_comm, cudaStream_t stream)
    : grid_ref_(grid), 
      cart_comm_(grid.get_cart_comm()), 
      nccl_comm_(nccl_comm), 
      stream_(stream),
      exec_space_(stream)
{
    // 取得當前行程在 World 中的 Rank (除錯用)
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // ----------------------------------------------------------------
    // 關鍵修正：解決 MPI Reorder 導致的 Rank 不一致問題
    // ----------------------------------------------------------------
    if (cart_comm_ != MPI_COMM_NULL) {
        // 1. 先取得 cart_comm 拓撲中的鄰居 (這是 Local/Cart Rank)
        int cart_left, cart_right, cart_bottom, cart_top;
        MPI_Cart_shift(cart_comm_, 1, 1, &cart_left, &cart_right);  // X 方向
        MPI_Cart_shift(cart_comm_, 0, 1, &cart_bottom, &cart_top);  // Y 方向

        // 2. 準備翻譯：取得兩個 Communicator 的 Group
        MPI_Group cart_group, world_group;
        MPI_Comm_group(cart_comm_, &cart_group);
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);

        int cart_ranks[4] = {cart_left, cart_right, cart_bottom, cart_top};
        int world_ranks[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

        // 3. 逐一翻譯 Rank
        for(int i=0; i<4; ++i) {
            if (cart_ranks[i] != MPI_PROC_NULL) {
                // 將 Cart Rank 轉為 Global World Rank (NCCL 唯一認得的 ID)
                MPI_Group_translate_ranks(cart_group, 1, &cart_ranks[i], world_group, &world_ranks[i]);
            }
        }

        // 4. 儲存正確的 Global Ranks 給 NCCL 使用
        neighbor_left_   = world_ranks[0];
        neighbor_right_  = world_ranks[1];
        neighbor_bottom_ = world_ranks[2];
        neighbor_top_    = world_ranks[3];

        // 5. 釋放 Group 資源
        MPI_Group_free(&cart_group);
        MPI_Group_free(&world_group);
    } else {
        // 單核心或錯誤情況
        neighbor_left_ = neighbor_right_ = neighbor_bottom_ = neighbor_top_ = MPI_PROC_NULL;
    }

    const int h = grid_ref_.get_halo_cells();
    if (h > 0) {
        const int nx = grid.get_local_total_points_x();
        const int ny = grid.get_local_total_points_y();
        const int nz = grid.get_local_total_points_z();
        const int nw_dummy = 4; // 預留給 4D 變數

        // 計算各維度需要的 Buffer 大小
        buffer_size_x_2d_ = static_cast<size_t>(h) * ny;
        buffer_size_y_2d_ = static_cast<size_t>(h) * nx;
        buffer_size_x_3d_ = static_cast<size_t>(h) * ny * nz;
        buffer_size_y_3d_ = static_cast<size_t>(h) * nx * nz;
        buffer_size_x_4d_ = static_cast<size_t>(h) * ny * nz * nw_dummy;
        buffer_size_y_4d_ = static_cast<size_t>(h) * nx * nz * nw_dummy;
        
        buffer_size_slice_x_ = static_cast<size_t>(h) * ny;
        buffer_size_slice_y_ = static_cast<size_t>(h) * nx;

        // 取最大值分配一次記憶體，重複利用
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
    if (is_graph_created_ && graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
    }
}

inline void HaloExchanger::exchange_halos(State& state) {
    // 確保前面的計算已經完成，再開始通訊
    Kokkos::fence(); 

    if (!is_graph_created_) {
        // --- 錄製階段 ---
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        
        for (auto& field_pair : state) {
            std::visit([this](auto& field) {
                using T = std::decay_t<decltype(field)>;
                if constexpr (!std::is_same_v<T, std::monostate>) {
                    // 這裡不會真的執行，而是將操作加入 Graph
                    this->exchange_halos_impl(field);
                }
            }, field_pair.second);
        }
        
        cudaGraph_t graph;
        cudaStreamEndCapture(stream_, &graph);
        cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph); // Instantiate 後即可銷毀原始 Graph Template
        is_graph_created_ = true;
    }

    // --- 執行階段 ---
    // 發射整個通訊圖 (包含 Packing, NCCL, Unpacking)
    cudaGraphLaunch(graph_exec_, stream_);
    
    // 同步 Stream，確保此步驟完成 (視需求可移除，讓後續計算在同 Stream 上排隊)
    cudaStreamSynchronize(stream_);
}

template<size_t Dim>
void HaloExchanger::exchange_halos_impl(Field<Dim>& field, int depth) const {
    const int halo_start_offset = grid_ref_.get_halo_cells();
    int h = (depth == -1) ? grid_ref_.get_halo_cells() : depth;
    if (h == 0) return;

    auto data = field.get_mutable_device_data();
    const int nx_phys = grid_ref_.get_local_physical_points_x();
    const int ny_phys = grid_ref_.get_local_physical_points_y();
    
    // [重要修正] 將成員變數複製到 Local Const，避免 Lambda 捕捉 `this` 導致 Device SegFault
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;

    size_t count_x = 0;
    size_t count_y = 0;
    
    // 根據維度計算交換數量
    if constexpr (Dim == 2) {
        count_x = static_cast<size_t>(h) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(1);
    } else if constexpr (Dim == 3) {
        count_x = static_cast<size_t>(h) * data.extent(1) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(2) * data.extent(0);
    } else if constexpr (Dim == 4) {
        count_x = static_cast<size_t>(h) * data.extent(2) * data.extent(1) * data.extent(0);
        count_y = static_cast<size_t>(h) * data.extent(3) * data.extent(1) * data.extent(0);
    }

    // ================= X-Direction (左右交換) =================
    if (count_x > 0) {
        auto send_l = Kokkos::subview(send_x_left_, std::make_pair((size_t)0, count_x));
        auto recv_l = Kokkos::subview(recv_x_left_, std::make_pair((size_t)0, count_x));
        auto send_r = Kokkos::subview(send_x_right_, std::make_pair((size_t)0, count_x));
        auto recv_r = Kokkos::subview(recv_x_right_, std::make_pair((size_t)0, count_x));

        // 1. Pack Data to Buffers
        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("pack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    send_l(j * h + i_h) = data(j, halo_start_offset + i_h);
                    send_r(j * h + i_h) = data(j, halo_start_offset + nx_phys - h + i_h);
            });
        } else if constexpr (Dim == 3) {
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

        // 2. NCCL Communication
        ncclGroupStart();
        // 向右傳：送給右鄰居，從左鄰居收
        if(neighbor_right != MPI_PROC_NULL) 
            ncclSend(send_r.data(), count_x, ncclDouble, neighbor_right, nccl_comm_, stream_);
        if(neighbor_left != MPI_PROC_NULL) 
            ncclRecv(recv_l.data(), count_x, ncclDouble, neighbor_left, nccl_comm_, stream_);

        // 向左傳：送給左鄰居，從右鄰居收
        if(neighbor_left != MPI_PROC_NULL) 
            ncclSend(send_l.data(), count_x, ncclDouble, neighbor_left, nccl_comm_, stream_);
        if(neighbor_right != MPI_PROC_NULL) 
            ncclRecv(recv_r.data(), count_x, ncclDouble, neighbor_right, nccl_comm_, stream_);
        ncclGroupEnd();

        // 3. Unpack Data from Buffers
        if constexpr (Dim == 2) {
            const int ny = data.extent(0);
            Kokkos::parallel_for("unpack_x_2d", Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(exec_space_, {0,0}, {ny, h}),
                KOKKOS_LAMBDA(int j, int i_h) {
                    if (neighbor_left != MPI_PROC_NULL) data(j, halo_start_offset - h + i_h) = recv_l(j * h + i_h);
                    if (neighbor_right != MPI_PROC_NULL) data(j, halo_start_offset + nx_phys + i_h) = recv_r(j * h + i_h);
            });
        } else if constexpr (Dim == 3) {
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

    // ================= Y-Direction (上下交換) =================
    if (count_y > 0) {
        auto send_b = Kokkos::subview(send_y_bottom_, std::make_pair((size_t)0, count_y));
        auto recv_b = Kokkos::subview(recv_y_bottom_, std::make_pair((size_t)0, count_y));
        auto send_t = Kokkos::subview(send_y_top_, std::make_pair((size_t)0, count_y));
        auto recv_t = Kokkos::subview(recv_y_top_, std::make_pair((size_t)0, count_y));

        // 1. Pack
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

        // 2. NCCL
        ncclGroupStart();
        // 向上傳：送給上鄰居，從下鄰居收
        if(neighbor_top != MPI_PROC_NULL) 
            ncclSend(send_t.data(), count_y, ncclDouble, neighbor_top, nccl_comm_, stream_);
        if(neighbor_bottom != MPI_PROC_NULL) 
            ncclRecv(recv_b.data(), count_y, ncclDouble, neighbor_bottom, nccl_comm_, stream_);

        // 向下傳：送給下鄰居，從上鄰居收
        if(neighbor_bottom != MPI_PROC_NULL) 
            ncclSend(send_b.data(), count_y, ncclDouble, neighbor_bottom, nccl_comm_, stream_);
        if(neighbor_top != MPI_PROC_NULL) 
            ncclRecv(recv_t.data(), count_y, ncclDouble, neighbor_top, nccl_comm_, stream_);
        ncclGroupEnd();

        // 3. Unpack
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
    cudaStreamSynchronize(stream_);
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

    // [安全] 複製到區域變數
    const int neighbor_left = neighbor_left_;
    const int neighbor_right = neighbor_right_;
    const int neighbor_bottom = neighbor_bottom_;
    const int neighbor_top = neighbor_top_;

    // 手動交換 Slice 通常不在 Graph Capture 中，使用 fence 確保數據正確
    Kokkos::fence(); 

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
                    // [重要修正] 這裡修正了索引，確保寫入正確的 Ghost 區域
                    if (neighbor_left != MPI_PROC_NULL) data(k_layer, j, halo_start_offset - h + i_h) = recv_l(idx);
                    if (neighbor_right != MPI_PROC_NULL) data(k_layer, j, halo_start_offset + nx_phys + i_h) = recv_r(idx);
            });
        }
    }
    
    // Slice 交換後進行同步
    cudaStreamSynchronize(stream_);
}

} // namespace Core
} // namespace VVM

#endif // VVM_CORE_HALOEXCHANGER_HPP
