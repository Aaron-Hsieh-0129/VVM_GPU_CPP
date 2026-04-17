# System architecture

GVVM is a **C++17** cloud-resolving model using **Kokkos** for on-device parallelism (CUDA backend enabled in CMake), **MPI** for distributed memory, and optional **NCCL** for collectives on NVIDIA GPUs when `ENABLE_NCCL` is on.

## Repository layout (application code)

| Path | Role |
| ---- | ---- |
| `src/main.cpp` | MPI init, Kokkos init, optional NCCL, config load, split communicators for I/O servers, `Grid` / `Parameters` / `State` / `HaloExchanger` / `Model` / `OutputManager`, time loop |
| `src/driver/` | `Model`: orchestrates dynamical core, physics, and tendencies; implements `init`, `run_step`, `finalize` |
| `src/core/` | `Grid`, `State`, `Field`, `Parameters`, `HaloExchanger`, initializer, boundary helpers |
| `src/dynamics/` | Vector-vorticity dynamical core, time integration, forcings (sponge, nudging, random), idealized tests |
| `src/physics/` | P3 (`p3/`), RRTMGP (`rrtmgp/`), turbulence, surface, land (Noah); CMake aggregates as `eamxx_physics` interface + static libs |
| `src/io/` | `OutputManager` (ADIOS2), `IOServer` (SST consumer to HDF5) |
| `src/utils/` | `ConfigurationManager` (JSON via nlohmann), timing and timers |
| `src/share/` | Shared EAMxx-derived utilities, constants, physics helpers |
| `externals/ekat/` | EKAT submodule: logging, YAML, testing utilities, Kokkos integration |
| `rundata/` | Sample `default_config.json`, initial profiles, P3 lookup tables |

Fortran pieces (e.g. Noah OpenACC) are linked through the physics/land subtree as required by CMake.

## Execution model

1. **MPI:** `MPI_Init`, then optional **shared-memory** communicator sizing to set OpenMP threads per rank (`omp_set_num_threads`).
2. **GPU:** `cudaGetDeviceCount` / `cudaGetDevice`; Kokkos is initialized with `set_device_id` from the node-local rank modulo GPU count.
3. **Configuration:** `ConfigurationManager` loads JSON (default `../rundata/input_configs/default_config.json` or CLI path).
4. **I/O split:** If `--io-tasks N` > 0, ranks are colored into simulation vs I/O; I/O ranks call `run_io_server` and exit; simulation ranks continue.
5. **Simulation ranks:** NCCL communicator is created when enabled; `Grid` builds a Cartesian MPI decomposition; `State` and `HaloExchanger` are constructed; `Model::init` runs initializer and optional physics `initialize`/`init`; `OutputManager` writes initial step; the loop calls `model.run_step(dt)` until `simulation.total_time_s` is reached.

## Major libraries (CMake targets)

- `vvm_driver` — `Model`
- `vvm_core` — grid, state, halos, parameters
- `vvm_dynamics` — dynamical core and related forcings
- `vvm_io` — ADIOS2 output and I/O server
- `vvm_utils` — configuration and timing
- `scream_share` — shared EAMxx code
- `eamxx_physics` — interface aggregating P3, RRTMGP, turbulence, surface, land

The executable links `MPI::MPI_CXX`, `Kokkos::kokkos`, and the targets above.

## Communication

- **Halo exchange:** `HaloExchanger` exchanges halos for listed fields; CUDA graph optimization is configurable via `optimization.cuda_graph_halo_exchange` in JSON.
- **NCCL:** Used when `ENABLE_NCCL` is defined; `HaloExchanger` and `State` constructors take `ncclComm_t` and a CUDA stream in that build.

## Configuration

JSON keys are resolved with dotted paths (e.g. `physics.p3.enable_p3`) in `ConfigurationManager::find_node`. No separate YAML runtime config is required for the main executable; EKAT may use YAML for its own tooling in subprojects.
