# I/O management

GVVM writes simulation fields through **ADIOS2**. The configuration file (`output` section) selects the engine, directory, filename prefix, and which fields to write.

## Synchronous I/O (typical)

With `output.engine` set to `HDF5` and **no** dedicated I/O ranks, the simulation ranks write directly using ADIOS2’s HDF5 engine. Set `output_dir` and `output_filename_prefix` to a location your job can create.

`OutputManager` (see `src/io/OutputManager.cpp`) defines variables from the configured field list and writes them at intervals driven by `simulation.output_interval_s` in `main.cpp`.

## Asynchronous I/O with SST

When `output.engine` is `SST`, the model streams data over an ADIOS2 **SST** stream. You can run dedicated **I/O server** ranks that read SST and write HDF5 to disk, overlapping compute and I/O.

### Rank layout

- Total MPI ranks = **simulation ranks** + **I/O ranks**.
- Pass `--io-tasks N` so that **N** ranks join the I/O communicator; the remainder run the dynamical core.

Examples (from the project README):

```bash
# 1 simulation rank, 1 I/O rank
mpirun -np 2 ./vvm --io-tasks 1

# 2 simulation ranks, 2 I/O ranks
mpirun -np 4 ./vvm --io-tasks 2
```

The first `world_size - N` ranks run `Grid`, `Model`, and `OutputManager` on the simulation communicator. The last `N` ranks execute `VVM::IO::run_io_server()` (`src/io/IOServer.cpp`), which opens the SST stream named from `output_dir` and `output_filename_prefix`, then writes collective HDF5 on the I/O side.

### Requirements

- ADIOS2 built with MPI and the engines you use (SST + HDF5).
- Consistent `output_dir` / `output_filename_prefix` so writers and readers resolve the same stream.

### Stale SST cleanup

On global rank 0, if the engine is `SST`, the main program removes an existing directory `output_dir/output_filename_prefix.sst` before starting, to avoid reusing stale stream data.

## Field selection and subsets

- `output.fields_to_output` lists **field names** that must exist on the model state (e.g. `u`, `v`, `w`, `th`, `xi`, `eta`, `zeta`, hydrometeors, radiation diagnostics, surface fields).
- `output.output_grid` restricts the written region using start/end indices per direction; `-1` means “to the end” of that dimension.

## NetCDF and preprocessing

Reading spatial inputs (topography, land) uses **NetCDF** via the `netcdf_reader` block in the JSON file. Writing may include NetCDF-related options depending on build flags (`enable_netcdf` in `output`). Preprocessing scripts under `scripts/` (for example `generate_init_nc.py`) produce NetCDF aligned with `netcdf_reader` and land-surface needs.
