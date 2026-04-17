# Developer guides

These pages summarize how GVVM is structured in this repository and how major subsystems fit together.

| Guide | Contents |
| ------ | -------- |
| [System architecture](architecture.md) | Directories, libraries, main program flow, MPI and Kokkos |
| [Physics implementation](physics.md) | P3, RRTMGP, turbulence, surface, land, and the driver time step |

**Build system:** The root `CMakeLists.txt` configures C++17, CUDA/Kokkos, MPI, EKAT, NetCDF, PnetCDF, HDF5, ADIOS2, and optional NCCL. The main executable target is `vvm`, built from `src/main.cpp` and linked against `vvm_driver`, `vvm_core`, `vvm_io`, `vvm_dynamics`, `scream_share`, and `eamxx_physics`.

**Tests:** With `-DBUILD_TESTS=ON`, CMake registers integration tests that run `vvm` with JSON files under `tests/configs/` and compare HDF5 output to baselines using `tests/scripts/verify_output.py`.
