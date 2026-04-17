# Quick Start

This guide covers dependencies, building GVVM, and running the `vvm` executable.

## Requirements

### Compilers and runtime

| Component | Minimum | Notes |
| --------- | ------- | ----- |
| C++ compiler | GCC 11+ | C++17 |
| CUDA | 11.4+ | NVIDIA GPUs are the tested target |
| MPI | OpenMPI 4.x+ | Use `mpic++` / `mpicc` / `mpifort` consistent with your toolchain |

**NVHPC 24.9+** is recommended on NVIDIA systems: it bundles CUDA, OpenMPI, and math libraries that align with the CMake hints used in `CMakePresets.json`.

### Libraries

| Library | Minimum (tested) | Role |
| ------- | ------------------ | ---- |
| CMake | 3.20 | Build |
| Kokkos | 4.7+ | If not found, CMake may fetch Kokkos 4.5.x via `FetchContent` |
| HDF5 | 1.14.5+ | NetCDF / ADIOS2 stack |
| NetCDF-C | 4.4+ | I/O |
| NetCDF-Fortran | 4.4+ | Fortran interfaces |
| PnetCDF | 1.14+ | Parallel I/O |
| ADIOS2 | 2.11+ | Model output (`HDF5` / `SST` engines) |

The root `CMakeLists.txt` also expects **NVIDIA CPU Math Library** (`libnvcpumath`) and, when `ENABLE_NCCL` is ON (default), **NCCL** under `NVHPC_DIR`. Turn off NCCL with `-DENABLE_NCCL=OFF` only if you have a matching build and know the implications for halo exchange.

## Build

### 1. Clone the repository

```bash
git clone https://github.com/Aaron-Hsieh-0129/VVM_GPU_CPP.git
cd VVM_GPU_CPP
```

### 2. Configure CMake presets

Edit `CMakePresets.json` (or pass cache variables on the command line) so that:

- `CMAKE_CXX_COMPILER`, `CMAKE_C_COMPILER`, `CMAKE_Fortran_COMPILER` point to MPI wrappers.
- `NVHPC_DIR` matches your installation.
- `HDF5_DIR`, `NETCDF_C_DIR`, `NETCDF_Fortran_DIR`, `PNETCDF_DIR` point to your dependency prefixes.

`find_package(ADIOS2 REQUIRED CXX MPI)` must succeed using your `CMAKE_PREFIX_PATH` or install layout.

### 3. Configure and compile

```bash
cmake --preset <your_preset_name> -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

The main binary is **`build/vvm`** (`RUNTIME_OUTPUT_DIRECTORY` is the build root).

## Configure a run

1. Copy or edit **`rundata/input_configs/default_config.json`**. This is the default path the executable uses when run from `build/` (see `src/main.cpp`).

2. Set **`output.output_dir`** to a directory you can write.

3. Point **`initial_conditions.source_file`** at a profile under `rundata/initial_conditions/profiles/` (or your own file).

4. For spatial NetCDF (topography, land), set **`netcdf_reader.source_file`** and run `scripts/generate_init_nc.py` if you need to generate that file.

5. Optional: pass a different config path as the **first non-option argument**:

   ```bash
   mpirun -np 1 ./vvm /path/to/my_config.json
   ```

Full key reference: [Model configuration](user-guides/configuration.md).

## Run

From the **build directory** (so relative paths like `../rundata/...` resolve):

```bash
cd build
mpirun -np 1 ./vvm
```

### Asynchronous I/O (optional)

Reserve ranks for dedicated I/O servers that consume an ADIOS2 **SST** stream and write HDF5 (`output.engine` must be `SST`). Example:

```bash
# 1 simulation rank + 1 I/O rank
mpirun -np 2 ./vvm --io-tasks 1

# 2 simulation ranks + 2 I/O ranks
mpirun -np 4 ./vvm --io-tasks 2
```

Details: [I/O management](user-guides/io-management.md).

## Documentation site

To preview this documentation locally (requires [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/)):

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Use `requirements-docs.txt` so your local MkDocs/Material versions match GitHub Actions. Then open the served URL in your browser.

