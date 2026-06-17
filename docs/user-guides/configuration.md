# Model configuration

Runtime settings are loaded from a **single JSON file**. The recommended `submit.py` workflow passes this path to the executable for you. If you run the executable directly, it searches for `../rundata/input_configs/default_config.json` relative to the current working directory when you run from the `build` directory, unless you pass another path as the first non-option argument (see [Command-line options](#command-line-options)).

The repository ships an example at `rundata/input_configs/default_config.json`. Keys prefixed with `__` are documentation-only and are not read by the code.

## Command-line options

| Argument | Meaning |
| -------- | ------- |
| `path/to/config.json` | Optional. If present as the first non-flag argument, selects the configuration file. Flags such as `--io-tasks` are skipped when resolving this path. |
| `--io-tasks N` | Reserve **N** MPI ranks for asynchronous I/O (see [I/O management](io-management.md)). |

Example:

```bash
cd $VVM_ROOT
./submit.py -c /path/to/my_run.json --compute 1 --local

# Advanced direct MPI only
mpirun -np 1 ./build/vvm /path/to/my_run.json
```

## How to edit a run

Most experiments follow the same editing order:

1. Set the grid and time controls (`grid`, `simulation`).
2. Choose the sounding profile and spatial NetCDF inputs (`initial_conditions`, `netcdf_reader`).
3. Choose restart behavior, if any (`restart`).
4. Choose output engine, fields, and subdomain (`output`).
5. Enable dynamics forcings and tendency terms (`dynamics`).
6. Enable physics packages and their call frequencies (`physics`).
7. Tune acceleration options (`optimization`) only after the run is physically configured.

Keep JSON paths relative to the directory where `submit.py` starts the model, normally the project root. Keys starting with `__` or `_` are inline comments for humans; they are not intended as runtime controls.

## Top-level sections

The configuration is grouped into nested objects. The sections below mirror the landfix-era `default_config.json` layout and the attached example.

### `grid`

`grid` defines the global mesh, halo width, horizontal boundary behavior, and vertical coordinate construction.

| Key | Role |
| --- | ---- |
| `nx`, `ny`, `nz` | Global domain size in x, y, and z. MPI decomposes this domain across compute ranks. |
| `n_halo_cells` | Halo width used by stencil operations and MPI halo exchange. The common value is `2`. |
| `dx`, `dy` | Horizontal grid spacing in meters. |
| `dz`, `dz1` | Vertical spacing controls in meters. `dz1` is used by stretched-coordinate logic together with `dz`. |
| `boundary_condition.x`, `boundary_condition.y` | Lateral boundary type. Current examples use `periodic`. |
| `fix_lonlat` | Enables fixed longitude/latitude handling for Taiwan-oriented real-case inputs. |
| `vertical_coordinate_type` | Vertical grid mode: `default`, `taiwanvvm`, or `rcemip`. |
| `rcemip_grid_data_path` | Profile path used when `vertical_coordinate_type` is `rcemip`. |

Use `taiwanvvm` when you need the TaiwanVVM-style vertical coordinate and output coordinate handling. Use `default` for ordinary idealized or simple real-case tests unless the case explicitly requires another coordinate.

### `simulation`

`simulation` controls how long the model advances and how often output is written.

| Key | Role |
| --- | ---- |
| `total_time_s` | Total simulated time in seconds. |
| `dt_s` | Model time step in seconds. Several physics frequencies must be divisible by this value. |
| `output_interval_s` | Output cadence in simulated seconds. |
| `idealized_test` | Selects a built-in dynamics test: `none`, `advection_u`, `advection_v`, `advection_w`, `stretching`, `twisting`, or `2dbubble`. |

Integration tests under `tests/configs/` set `idealized_test` to match each case name where applicable. For production-like runs, keep `idealized_test` as `none`.

### `initial_conditions`

`initial_conditions` selects the one-dimensional sounding/profile and optional initial perturbations.

| Key | Role |
| --- | ---- |
| `format` | Input profile format. The current path is normally `txt`. |
| `source_file` | Path to the profile used to initialize pressure, thermodynamics, and winds. |
| `constant_upper_wind.enable` | If true, upper-level winds above the pressure threshold are held constant while reading the text profile. |
| `constant_upper_wind.pressure_threshold_Pa` | Pressure threshold for constant upper-wind handling. |
| `perturbation` | Optional perturbation preset: `none`, `2dbubble`, or `3dbubble`. |

Place shared profiles under `rundata/initial_conditions/profiles/`. The profile must be consistent with the vertical coordinate choice and the physics you enable.

### `netcdf_reader`

`netcdf_reader` loads spatial two-dimensional fields such as longitude, latitude, topography, land mask, vegetation, soil, and surface parameters.

| Key | Role |
| --- | ---- |
| `source_file` | Spatial NetCDF file, usually under `rundata/initial_conditions/spatial/`. |
| `Tg_source` | Surface/ground temperature source. `atmosphere` initializes `Tg` from the atmospheric state; `netcdf` reads it from the NetCDF field. |
| `variables_to_read.2d` | List of 2D fields to read. Common fields include `lon`, `lat`, `topo`, `sea_land_ice_mask`, `vegtype`, `soiltype`, `slopetype`, `Tg`, `albedo`, `gvf`, `lai`, `shdmax`, and `shdmin`. |

The exact variable set must match what your preprocessing tool wrote. For Taiwan-style cases, generate or prepare the file before submission; see [TaiwanVVM example](../examples/taiwan-vvm.md).

### `restart`

`restart` controls whether the model initializes prognostic fields from an existing output file.

| Key | Role |
| --- | ---- |
| `enable` | Enables restart loading when true. |
| `source_file` | Restart source. Landfix supports `.h5` through the HDF5 restart reader and `.nc` through the PnetCDF reader. |
| `file_interval_s` | Time represented by one numbered output index. For example, `vvm_output_000144.h5` with `file_interval_s = 600` starts at `144 * 600 = 86400 s`. |
| `variables_to_read.1d`, `variables_to_read.2d`, `variables_to_read.3d` | Optional explicit restart variable lists. |

If no explicit `restart.variables_to_read.3d` is supplied for HDF5 restart, the reader selects prognostic variables from `dynamics.prognostic_variables` and filters them through `output.fields_to_output`. That means fields needed after restart should either be listed explicitly under `restart.variables_to_read` or be included in `output.fields_to_output`.

When restart is enabled, the restart state replaces the normal perturbation initialization. The restart filename must contain a trailing number before the extension so the model can recover the restart time.

### `output`

`output` configures ADIOS2 output, field selection, and optional subsetting.

| Key | Role |
| --- | ---- |
| `output_dir` | Directory for run output. `submit.py` creates it before launching the job. |
| `engine` | ADIOS2 engine. Use `HDF5` for direct file output or `SST` for asynchronous streaming to I/O ranks. |
| `queue_limit` | ADIOS2 queue depth for buffered/asynchronous writes. |
| `data_transport` | Optional ADIOS2 transport hint, e.g. `RDMA`, `WAN`, or empty for default behavior. |
| `enable_netcdf` | Enables NetCDF-related output paths where supported. |
| `output_filename_prefix` | Prefix for generated output files and streams, commonly `vvm_output`. |
| `fields_to_output` | Ordered list of state fields to write. Each name must exist in the model state. |
| `output_grid` | Output subset: `x_start`, `x_end`, `y_start`, `y_end`, `z_start`, `z_end`. Use `-1` for "through the end" in example configs. |

For `engine = "SST"`, submit with dedicated I/O ranks:

```bash
./submit.py -c my_config.json --compute 16 --io 4 --nodes 4 --gpus 5
```

Use `fields_to_output` deliberately. A large list is convenient for diagnostics but increases file size and I/O cost. Common field groups are:

| Field group | Examples |
| --- | --- |
| Base state | `thbar`, `pibar`, `rhobar`, `rhobar_up` |
| Dynamics | `u`, `v`, `w`, `th`, `xi`, `eta`, `zeta` |
| Moisture/P3 | `qv`, `qc`, `qr`, `qi`, `qm`, `nc`, `nr`, `ni`, `bm` |
| Radiation | `sw_heating`, `lw_heating`, `swdn`, `lwdn`, `lwup`, `swup_toa`, `swdn_toa`, `lwup_toa`, `lwdn_toa`, `swup_sfc`, `swdn_sfc`, `lwup_sfc`, `lwdn_sfc` |
| Surface/land | `Tg`, `sfc_flux_th`, `sfc_flux_qv`, `sfc_flux_u`, `sfc_flux_v`, `le`, `hfx`, `st1`, `st2`, `st3`, `st4`, `gfx`, `topo` |

### `dynamics`

`dynamics` contains the vertical wind solver, external forcings, and per-field tendency term settings.

#### `dynamics.solver`

| Key | Role |
| --- | ---- |
| `w_solver_method` | Vertical velocity solver: `tridiagonal` for the original method or `jacobi` for a 3D-parallel iteration path. |
| `iteration` | Iteration count used by iterative solver paths. |
| `WRXMU` | Solver relaxation/control parameter used by the wind solver. |

#### `dynamics.forcings.sponge_layer`

The sponge layer damps selected fields above `sponge_layer_base`.

| Key | Role |
| --- | ---- |
| `enable` | Turns sponge-layer forcing on/off. |
| `damp_thermo` | Damps thermodynamic variables. |
| `damp_vort` | Damps vorticity variables. |
| `sponge_layer_base` | Height in meters where damping begins. |
| `inv_CRAD` | Damping timescale/control value used to compute the damping coefficient. |

#### `dynamics.forcings.random_perturbation`

Random perturbation is useful for triggering convection in otherwise smooth initial states.

| Key | Role |
| --- | ---- |
| `enable` | Turns random forcing on/off. |
| `time_s` | Applies perturbations until this simulated time. |
| `amplitude` | Perturbation magnitude. |
| `z_start_m`, `z_end_m` | Vertical layer where perturbations are applied. |
| `random_seed` | Seed for repeatable perturbations. |

#### `dynamics.forcings.lateral_boundary_nudging`

Lateral boundary nudging relaxes selected variables toward large-scale forcing near selected boundaries.

| Key | Role |
| --- | ---- |
| `enable` | Turns nudging on/off. |
| `boundaries.west/east/south/north` | Selects which boundaries are nudged. |
| `tau_b` | Nudging timescale in seconds. |
| `offset`, `width`, `radius` | Geometric controls for the nudging zone and taper. |
| `target_vars` | Variables to nudge, e.g. `qv`. |
| `forcing_data.time_varying` | If true, forcing files are selected by time using `file_prefix` and `update_interval_s`. |
| `forcing_data.directory` | Directory containing large-scale forcing files. |
| `forcing_data.file_name_for_not_varying` | File used when forcing is constant in time. |
| `forcing_data.file_prefix` | Prefix for time-varying forcing files. |
| `forcing_data.update_interval_s` | Time spacing between time-varying forcing files. |

#### `dynamics.forcings.areamn`

Area-mean nudging is an optional landfix forcing that stores and relaxes area-mean vorticity/wind reference quantities.

| Key | Role |
| --- | ---- |
| `enable` | Turns area-mean nudging on/off. |
| `uvtau` | Relaxation timescale/control value. |
| `nudge_start_m` | Height in meters where nudging starts. |

#### `dynamics.prognostic_variables`

Each prognostic variable lists enabled tendency terms and the numerical schemes used to advance them.

```json
"th": {
  "tendency_terms": {
    "advection": {
      "enable": true,
      "temporal_scheme": "AdamsBashforth2",
      "spatial_scheme": "Takacs"
    }
  }
}
```

Common prognostic variables include `th`, `xi`, `eta`, `zeta`, `qv`, `qc`, `qr`, `qi`, `qm`, `nc`, `nr`, `ni`, and `bm`. P3-related fields should be present when P3 is enabled. Typical terms are `advection`, `buoyancy`, `stretching`, `twisting`, and `coriolis`; unsupported combinations should not be invented in the JSON because the dynamical core constructs tendency objects from these names.

### `physics`

`physics` enables packages and sets their call frequencies.

| Block | Purpose |
| ----- | -------- |
| `p3` | Microphysics. Controls `enable_p3`, lookup table generation, cloud droplet number prediction, prescribed CCN behavior, and `max_total_ni`. |
| `rrtmgp` | Radiation. Controls `enable_rrtmgp`, chunking, call frequency, active gases, fixed solar options, pool size, and start time. |
| `turbulence` | Subgrid turbulence. |
| `surface_process` | Combined surface/land/ocean interface used in landfix. |

#### `physics.rrtmgp`

| Key | Role |
| --- | ---- |
| `enable_rrtmgp` | Enables RRTMGP radiation. |
| `column_chunk_size` | Number of columns processed per chunk. Larger values can be faster but use more memory; tune with domain size and rank count. |
| `rad_frequency_s` | Radiation call frequency in seconds. Must divide evenly by `simulation.dt_s`. |
| `active_gases` | Gas list passed to RRTMGP, e.g. `h2o`, `co2`, `o3`, `n2o`, `co`, `ch4`, `o2`, `n2`. |
| `fixed_total_solar_irradiance` | Use a fixed solar irradiance when positive; `-1` means default/orbital behavior. |
| `fixed_solar_zenith_angle` | Use a fixed zenith angle when positive; `-1` means default/orbital behavior. |
| `pool_size_multiplier` | Memory-pool sizing control. |
| `time` | Calendar start time used by radiation and some output metadata. |

#### `physics.surface_process`

| Key | Role |
| --- | ---- |
| `enable` | Enables the surface process wrapper. |
| `frequency_s` | Surface/land/ocean call frequency in seconds. Must divide evenly by `simulation.dt_s`. |
| `land_scheme` | `none` or `noahlsm`. `noahlsm` calls the Fortran Noah land model. |
| `ocean_scheme` | `none`, `sflux_2d`, `sflux_tc_2d`, or `tco_ocean`. |

When `land_scheme = "noahlsm"`, land points are handled by the Noah land model. Choosing `sflux_2d` or `sflux_tc_2d` for `ocean_scheme` uses the C++ surface-flux implementation for ocean points and disables the ocean part inside the land process.

### `optimization`

| Key | Role |
| --- | ---- |
| `cuda_graph_solver` | Enables CUDA graph acceleration for solver paths that support it. |
| `cuda_graph_halo_exchange` | Field list for CUDA graph halo exchange capture, e.g. `u`, `w`, `xi`, `eta`, `zeta`, `th`, and hydrometeors. |

Only include fields that exist in the current state and are exercised by the run. If you disable a physics package, remove its fields from graph lists unless the code still allocates them for your case.

### `constants`

`constants` holds physical constants used across dynamics and physics.

| Key | Typical meaning |
| --- | --- |
| `gravity` | Gravitational acceleration. |
| `Rd` | Dry-air gas constant. |
| `PSFC`, `P0` | Reference surface/reference pressure. |
| `Cp` | Heat capacity at constant pressure. |
| `Lv` | Latent heat of vaporization. |
| `OMEGA` | Earth rotation rate. |
| `PI` | Pi. |

Change constants only for controlled sensitivity experiments.


Adding a field allocates it, but it does not automatically add dynamics, physics tendencies, initialization, or output. Add corresponding tendency configuration and include it in `output.fields_to_output` if you want it advanced and written.

## Consistency checklist

Before submitting a long run, check:

- `physics.rrtmgp.rad_frequency_s`, `physics.surface_process.frequency_s`, and `simulation.output_interval_s` are sensible multiples of `simulation.dt_s`.
- Every `output.fields_to_output` name is allocated by the selected dynamics/physics configuration.
- If `output.engine` is `SST`, submit with `submit.py --io N`.
- If `restart.enable` is true, the restart filename contains a numeric output index and `restart.file_interval_s` matches the original output interval.
- The NetCDF variables listed in `netcdf_reader.variables_to_read.2d` exist in `netcdf_reader.source_file`.
- P3 hydrometeor variables are present in `dynamics.prognostic_variables` when `physics.p3.enable_p3` is true.
- CUDA graph field lists match the fields actually allocated in the run.

## CMake and environment

Library paths and compilers are **not** set in this JSON file. Use `CMakePresets.json` and CMake cache variables for HDF5, NetCDF, PnetCDF, NVHPC, Kokkos, and MPI (see [Quick Start](../quick-start.md)).
