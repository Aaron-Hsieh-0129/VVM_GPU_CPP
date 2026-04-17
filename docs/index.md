# GVVM (GPU-accelerated Vector Vorticity Model)

**GVVM** is a GPU-accelerated, [Kokkos](https://kokkos.org/)-based C++ implementation of the **Vector Vorticity equation cloud-resolving Model (VVM)**. It uses **MPI** for domain decomposition, **CUDA** (via Kokkos) on NVIDIA GPUs in current configurations, and **ADIOS2** for I/O.

## Features

- **3D cloud-resolving dynamics** — Vector vorticity formulation with configurable tendencies, sponge layer, and idealized test modes.
- **Microphysics** — P3 scheme (E3SM EAMxx lineage).
- **Radiation** — RRTMGP (E3SM EAMxx lineage, Kokkos-enabled build).
- **Turbulence and surface** — Subgrid turbulence and surface flux options.
- **Land** — Noah land surface model (Fortran OpenACC), with contributions from the Central Weather Administration (CWA) of Taiwan.
- **Taiwan-oriented workflows** — NetCDF-based topography and land fields; see [TaiwanVVM](examples/taiwan-vvm.md).

## Documentation map

| Section | Description |
| -------- | ----------- |
| [Quick Start](quick-start.md) | Dependencies, build, first run |
| [User guides](user-guides/index.md) | JSON configuration and I/O |
| [Developer guides](developer-guides/index.md) | Architecture and physics hooks |
| [Examples](examples/index.md) | Idealized tests and Taiwan setups |

## Source repository

Clone and build instructions use the project root layout (`CMakePresets.json`, `src/`, `rundata/`). The upstream repository is [`VVM_GPU_CPP`](https://github.com/Aaron-Hsieh-0129/VVM_GPU_CPP) on GitHub.

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

## Contact

For bugs, feature requests, or contributions, open an issue on the GitHub repository. For questions about the model or usage: **B08209006@ntu.edu.tw**.
