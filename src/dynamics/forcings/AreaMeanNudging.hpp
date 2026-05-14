#pragma once

#include "core/State.hpp"
#include "core/Grid.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include <string>

#if defined(ENABLE_NCCL)
#include <nccl.h>
#else
#include <mpi.h>
#endif

namespace VVM {
namespace Dynamics {

class AreaMeanNudging {
public:
    AreaMeanNudging(const Utils::ConfigurationManager& config, 
                    const Core::Grid& grid, 
                    const Core::Parameters& params);
    ~AreaMeanNudging() = default;

    void initialize(Core::State& state);

    void apply_vorticity(Core::State& state, VVM::Real dt);
    void apply_uvtopmn(Core::State& state, VVM::Real dt);

private:
    const Utils::ConfigurationManager& config_;
    const Core::Grid& grid_;
    const Core::Parameters& params_;

    bool enable_;
    VVM::Real uvtau_;
    VVM::Real nudgelim_;
    
    VVM::Real inv_total_xy_pts_;

#if defined(ENABLE_NCCL)
    ncclComm_t nccl_comm_; 
    cudaStream_t stream_;
#endif
};

} // namespace Dynamics
} // namespace VVM
