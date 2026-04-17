# TaiwanVVM-style simulations

High-resolution simulations over **Taiwan** use **realistic topography** and **land-surface fields** read from NetCDF, together with the **Noah** land model when enabled. This workflow is often called **TaiwanVVM** in project documentation.

## Configuration overview

1. **Grid and domain** — Set `grid.nx`, `grid.ny`, `grid.nz`, and spacing to match your experiment. `fix_lonlat: true` is typical for lon/lat-consistent setups (see sample `default_config.json`).

2. **Spatial input file** — `netcdf_reader.source_file` must point to a NetCDF file containing the variables you need. The sample `default_config.json` lists 2D variables such as `lon`, `lat`, `topo`, `sea_land_ice_mask`, vegetation and soil types, slopes, ground temperature, albedo, greenness, LAI, etc. Your file must provide the same names (or you must change the list and preprocessing to match).

3. **Land and surface** — Enable `physics.land.enable_land` for Noah; tune `frequency_step`. Enable `physics.surface.enable_surface` and choose `mode` (e.g. `sflux_tc_2d`) when you need surface flux coupling. These interact with the 2D fields read from NetCDF.

4. **Output** — Set `output.output_dir` to a writable path; list diagnostics you need in `output.fields_to_output` (including `topo`, `lon`, `lat` if you want them in the archive).

## Generating spatial NetCDF

The script `scripts/generate_init_nc.py` reads **`rundata/input_configs/default_config.json`** (same path as the main config used in the script) and writes:

- `netcdf_reader.source_file` from the config (directories are created if missing).

It supports:

- **`USE_TAIWAN_TOPO = True`** — Reads high-resolution Taiwan data from `SOURCE_TW_DATA` (default `../rundata/land/topolsm_TW.nc`) and coarsens to the configured grid.
- **`USE_TAIWAN_TOPO = False`** — Idealized ridge and land-type patterns for synthetic experiments.

Edit the constants at the top of the script and run it from the `scripts/` directory with Python dependencies (`netCDF4`, `numpy`, `scipy`) installed. Align paths with your machine.

## Large-scale forcing (optional)

For lateral boundary forcing, `scripts/generate_ls_forcing.py` can prepare forcing files; `dynamics.forcings.lateral_boundary_nudging` in JSON points to directories and file naming under `rundata/LS_forcings/` when enabled.

## Operational notes

- Ensure **NetCDF** and **PnetCDF** libraries used at build time match the files you read and write.
- Taiwan runs are often **large**; adjust MPI ranks, `physics.rrtmgp.column_chunk_size`, and I/O strategy (`output.engine`, `--io-tasks`) for your cluster.
- The repository README credits **Noah** land GPU work to the Central Weather Administration (CWA) of Taiwan.

For a minimal first run, start from `rundata/input_configs/default_config.json`, replace `output.output_dir` and `initial_conditions.source_file` with local paths, and verify `netcdf_reader.source_file` exists after preprocessing.
