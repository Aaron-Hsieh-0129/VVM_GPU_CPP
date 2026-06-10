# GPUVVM (GPU-accelerated Vector Vorticity Model)

[![C++](https://img.shields.io/badge/C++-17%2B-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-Performance_Portability-blueviolet.svg)](https://kokkos.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A GPU-accelerated (Kokkos-based) C++ implementation of the **Vector Vorticity equation cloud-resolving Model (VVM)**.

Full documentation is available through the MkDocs site: <https://aaron-hsieh-0129.github.io/VVM_GPU_CPP/>.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [License](#license)
- [Contact & Support](#contact--support)


## Features

- **3D Cloud-Resolving Dynamics**: High-performance atmospheric dynamics utilizing the Vector Vorticity formulation.
- **Advanced Physics Schemes**:
  - **Microphysics**: P3 microphysics scheme adapted from E3SM EAMxx, with restored vapor-cloud water (qv $\leftrightarrow$ qc) conversion processes based on the original Fortran P3 formulation.
  - **Radiation**: RRTMGP radiation scheme adapted from E3SM EAMxx.
  - **Land Surface Model**: Noah land surface model with GPU acceleration (Fortran OpenACC), provided by the Central Weather Administration (CWA) of Taiwan.
- **TaiwanVVM Support**: Capable of simulating high-resolution Taiwan topography using generated terrain datasets (example scripts provided).


## Requirements

### Compiler Dependencies

| Dependency       | Minimum Version | Note                                         |
| :--------------- | :-------------- | :------------------------------------------- |
| **C++ Compiler** | `≥ gcc 11`      |                                              |
| **CUDA Toolkit** | `≥ 11.4`        | Currently tested exclusively on NVIDIA GPUs. |
| **OpenMPI**      | `≥ 11.4`        |                                              |

**Recommendation:** For running on NVIDIA GPUs, installing **NVHPC (≥ 24.9)** is highly recommended, as it includes the required OpenMPI and CUDA-related packages out of the box. Development for other GPU platforms and CPU-only architectures is planned for the future.

### Library Dependencies

*(Lower versions might work but have not been formally tested.)*

| Library | Minimum Version | Library | Minimum Version |
| :--- | :--- | :--- | :--- |
| **CMake** | `≥ 3.20` | **netcdf-c** | `≥ 4.4.1.1` |
| **Kokkos** | `≥ 4.7.01` | **netcdf-fortran**| `≥ 4.4.1` |
| **HDF5** | `≥ 1.14.5` | **pnetcdf** | `≥ 1.14.1` |
| **ADIOS2** | `≥ 2.11.0` | | |

*Please refer to our [Installation Tutorial](https://aaron-hsieh-0129.github.io/VVM_GPU_CPP/user-guides/environment/) for detailed instructions on setting up these libraries.*


## Quick Start

### Step 1: Clone the Repository
After installing the required libraries, clone the project from GitHub:
```bash
git clone https://github.com/Aaron-Hsieh-0129/VVM_GPU_CPP.git
cd VVM_GPU_CPP
```

### Step 2: Environment Setup

```bash
export VVM_ROOT=/absolute/path/to/your/VVM_GPU_CPP
cd $VVM_ROOT
```

`submit.py` can detect the project root when it is launched from the repository, but `VVM_ROOT` keeps build and run examples copy-paste friendly.

### Step 3: Configure CMake Presets

Open `CMakePresets.json` and configure or add a preset matching your machine cluster. Update the environment paths (`NVHPC_DIR`, `CMAKE_CXX_COMPILER`, `HDF5_DIR`, etc.) to match your build prefix.

At runtime, the submission wrapper (`submit.py`) reads this file, extracts the selected preset, and prepares the execution environment, including MPI and library paths.


### Step 4: Build the Project

Compile the project from the root directory. Replace `<your_preset_name>` with your configured preset and `<core_number>` with the number of CPU cores for parallel building:

```bash
cmake --preset <your_preset_name> -DBUILD_TESTS=ON
cmake --build build -j<core_number>
```

### Step 5: Configure the Experiment

- **Main Settings**: Modify `rundata/input_configs/default_config.json` to design your experiment. Each physical process has its own toggle switch.
    
- **Initial Conditions**:
    
    - Generate initial input files using the Python tools located under `tools/`.
        
    - Specify your generated spatial file path (typically placed in `rundata/initial_conditions/spatial/`) within `default_config.json`.
        
    - Initial profiles should be placed under `rundata/initial_conditions/profiles/`.

### Step 6: Submit and Execute

Use the root-level `submit.py` wrapper for normal runs. It handles local execution and SLURM submission, sets the runtime environment from `CMakePresets.json`, manages MPI task counts, creates output directories, and helps keep CPU/GPU allocation consistent.

> **Important:** Direct `mpirun` commands are advanced/debugging commands. For production runs, use `submit.py`; incorrect CPU/GPU assignment and I/O-rank allocation can substantially slow the model.

#### Interactive Mode

Simply run the script without any arguments and follow the guided prompts:

```bash
./submit.py
```

The interactive phase shows which fields you need to fill in, explains the run options, and prints an equivalent command-line invocation at the end so you can reuse it for future runs.

#### Command-Line Mode

For automated workflows or quick executions, you can pass arguments directly.

- Local Execution (HDF5 Engine):

```bash
./submit.py --local --preset <your_preset_name> -c ./rundata/input_configs/default_config.json --compute 4
```

- SLURM Submission (SST Engine with Asynchronous I/O):

```bash
./submit.py --preset <your_preset_name> -c ./rundata/input_configs/default_config.json --compute 16 --io 4 --nodes 4 --gpus 5 -t 24:00:00
```

For more options and resource-layout guidance, see the [Job Submission guide](https://aaron-hsieh-0129.github.io/VVM_GPU_CPP/user-guides/job-submission/).

#### Manual MPI Execution (Advanced)

Manual MPI is useful for small debug runs after your environment is already loaded. It bypasses the wrapper's resource checks, so verify CPU binding, rank placement, GPU visibility, and I/O ranks yourself.

```bash
mpirun -np 1 ./build/vvm ./rundata/input_configs/default_config.json
```

##### Asynchronous I/O (Optional)

To use asynchronous output, specify the SST engine in `default_config.json`. You can then allocate dedicated tasks for I/O.

For example, to use **1 GPU/CPU for the model** and **1 CPU for I/O**:

```bash
mpirun -np 2 ./build/vvm ./rundata/input_configs/default_config.json --io-tasks 1
```

To use **2 GPUs/CPUs for the model** and **2 CPUs for I/O**:

```bash
mpirun -np 4 ./build/vvm ./rundata/input_configs/default_config.json --io-tasks 2
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Acknowledgments & References

- **E3SM EAMxx**: The base implementation of the P3 microphysics and RRTMGP radiation schemes were adapted from the E3SM project.
- **P3 Microphysics**: The explicit condensation and evaporation processes between water vapor and cloud water, which are absent in the EAMxx version, have been re-implemented according to the original P3 formulation (e.g., *Morrison and Milbrandt, 2015*).
- **CWA Noah LSM**: The GPU-accelerated Noah land surface model is generously provided by the Central Weather Administration (CWA) of Taiwan.


## Contact & Support
GitHub Issues: For bugs, feature requests, or code contributions, please open an issue on the GitHub repository.

Email: Users can contact us for more questions regarding the model or its usage at B08209006@ntu.edu.tw.
