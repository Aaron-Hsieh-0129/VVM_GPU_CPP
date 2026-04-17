# Model configuration

Runtime settings are loaded from a **single JSON file**. The executable searches for `../rundata/input_configs/default_config.json` relative to the current working directory when you run from the `build` directory, unless you pass another path as the first non-option argument (see [Command-line options](#command-line-options)).

The repository ships an example at `rundata/input_configs/default_config.json`. Keys prefixed with `__` are documentation-only and are not read by the code.

## Command-line options

| Argument | Meaning |
| -------- | ------- |
| `path/to/config.json` | Optional. If present as the first non-flag argument, selects the configuration file. Flags such as `--io-tasks` are skipped when resolving this path. |
| `--io-tasks N` | Reserve **N** MPI ranks for asynchronous I/O (see [I/O management](io-management.md)). |

Example:

```bash
cd build
mpirun -np 1 ./vvm
mpirun -np 1 ./vvm /path/to/my_run.json
```

## Top-level sections

The configuration is grouped into nested objects. The following sections mirror the structure used in `default_config.json`.

### `grid`

| Key | Role |
| --- | ---- |
| `nx`, `ny`, `nz` | Global horizontal and vertical grid sizes. |
| `n_halo_cells` | Halo width for MPI decomposition and boundary stencils. |
| `dx`, `dy`, `dz`, `dz1` | Grid spacing (meters); `dz1` is used in vertical stretching together with `dz`. |
| `boundary_condition.x`, `boundary_condition.y` | Lateral boundary types, e.g. `periodic` or `zero_gradient` (see inline `__bc_option` in the sample file). |
| `fix_lonlat` | When true, longitude/latitude handling follows the fixed lon/lat mode used in Taiwan-oriented setups. |

### `simulation`

| Key | Role |
| --- | ---- |
| `total_time_s` | Total simulated time (seconds). |
| `dt_s` | Model time step (seconds); mapped to internal `dt` after initialization. |
| `output_interval_s` | Interval for calling the output writer (seconds of **simulated** time). |
| `idealized_test` | Dynamics test mode: `none`, or one of `advection_u`, `advection_v`, `advection_w`, `stretching`, `twisting`, `2dbubble` (disables the vertical wind solver for selected modes). |

Integration tests under `tests/configs/` set `idealized_test` to match each case name where applicable.

### `initial_conditions`

| Key | Role |
| --- | ---- |
| `format` | Input profile format (e.g. `txt`). |
| `source_file` | Path to the 1D profile used to initialize the column. |
| `perturbation` | Optional perturbation preset (e.g. `none`). |

Place shared profiles under `rundata/initial_conditions/profiles/` and point `source_file` at the file you need.

### `netcdf_reader`

Used when spatial fields (topography, land masks, surface fields) are read from NetCDF.

| Key | Role |
| --- | ---- |
| `source_file` | Path to the NetCDF file (often generated or prepared under `rundata/initial_conditions/spatial/`). |
| `variables_to_read` | Lists of variable names, e.g. `2d` arrays: `lon`, `lat`, `topo`, land-surface fields for Noah, etc. |

The exact variable set should match what your preprocessing script wrote (see [TaiwanVVM example](../examples/taiwan-vvm.md)).

### `output`

| Key | Role |
| --- | ---- |
| `output_dir` | Directory for run output (created as needed). |
| `engine` | ADIOS2 engine, e.g. `HDF5` for direct files or `SST` for streaming to dedicated I/O ranks. |
| `queue_limit` | Buffering limit for queued steps (asynchronous / streaming setups). |
| `data_transport` | Optional ADIOS2 transport hint (`RDMA`, `WAN`, or empty for local default). |
| `enable_netcdf` | Whether to enable NetCDF-related output paths where applicable. |
| `output_filename_prefix` | Prefix for output files and streams. |
| `fields_to_output` | Names of model fields to write (must match registered state fields). |
| `output_grid` | Subset of the domain: `x_start`, `x_end`, etc.; use `-1` for “through end”. |

For engine `SST`, rank 0 removes any stale `.sst` directory with the same prefix before the run starts.

### `dynamics`

Contains the vertical velocity solver (`w_solver_method`: `tridiagonal` or `jacobi`), sponge and perturbation forcings, lateral nudging, and **per-variable tendency settings** under `prognostic_variables`. Each prognostic field (e.g. `th`, `xi`, `eta`, `zeta`, hydrometeors) lists `tendency_terms` with `enable` flags and scheme names (`temporal_scheme`, `spatial_scheme`).

### `physics`

| Block | Purpose |
| ----- | -------- |
| `p3` | P3 microphysics: `enable_p3`, lookup table options, `do_predict_nc`, CCN behavior, `max_total_ni`. |
| `rrtmgp` | RRTMGP radiation: `enable_rrtmgp`, `column_chunk_size`, `rad_frequency_step`, gases, solar/orbital settings. |
| `turbulence` | Subgrid turbulence: `enable_turbulence`. |
| `surface` | Surface fluxes: `enable_surface`, `frequency_step`, `mode` (e.g. `sflux_tc_2d`). |
| `land` | Noah land surface: `enable_land`, `frequency_step`. |

### `optimization`

`cuda_graph_halo_exchange` lists field names for which CUDA graph capture is used in halo exchange (when supported).

### `constants` and `add_model_fields`

`constants` holds physical constants (gravity, gas constants, etc.). `add_model_fields` can declare extra 3D tracers (e.g. `tracer1`) allocated on the state.

## CMake and environment

Library paths and compilers are **not** set in this JSON file. Use `CMakePresets.json` and CMake cache variables for HDF5, NetCDF, PnetCDF, NVHPC, Kokkos, and MPI (see [Quick Start](../quick-start.md)).
