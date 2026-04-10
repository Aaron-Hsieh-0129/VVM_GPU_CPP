# GVVM (GPU-accelerated Vector Vorticity Model)

[![C++](https://img.shields.io/badge/C++-17%2B-blue.svg)](https://isocpp.org/)
[![Kokkos](https://img.shields.io/badge/Kokkos-Performance_Portability-blueviolet.svg)](https://kokkos.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A GPU-accelerated (Kokkos-based) C++ implementation of the **Vector Vorticity equation cloud-resolving Model (VVM)**.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [License](#license)
- [Contact & Support](#contact--support)

---

## Features

- **3D Cloud-Resolving Dynamics**: High-performance atmospheric dynamics utilizing the Vector Vorticity formulation.
- **Advanced Physics Schemes**:
  - **Microphysics**: P3 microphysics scheme adapted from E3SM EAMxx.
  - **Radiation**: RRTMGP radiation scheme adapted from E3SM EAMxx.
  - **Land Surface Model**: Noah land surface model with GPU acceleration (Fortran OpenACC), provided by the Central Weather Administration (CWA) of Taiwan.
- **TaiwanVVM Support**: Capable of simulating high-resolution Taiwan topography using generated terrain datasets (example scripts provided).

---

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

*Please refer to our [Installation Tutorial](link-to-tutorial-here) for detailed instructions on setting up these libraries.*

---

## Quick Start

### Step 1: Clone the Repository
After installing the required libraries, clone the project from GitHub:
```bash
git clone https://github.com/Aaron-Hsieh-0129/VVM_GPU_CPP.git
cd VVM_GPU_CPP
```

### Step 2: Configure CMake Presets

Add your library installation paths to `CMakePresets.json`. You can follow the existing configurations inside the file to specify your local environment paths.

### Step 3: Build the Project

Compile the project from the root directory. Replace `<your_preset_name>` with your configured preset and `<core_number>` with the number of CPU cores for parallel building:

```bash
cmake --preset <your_preset_name> -DBUILD_TESTS=ON
cmake --build build -j<core_number>
```

### Step 4: Configure the Experiment

- **Main Settings**: Modify `rundata/input_configs/default_configs.json` to design your experiment. Each physical process has its own toggle switch.
    
- **Initial Conditions**:
    
    - Generate initial input files using the Python scripts located under `scripts/`.
        
    - Specify your generated spatial file path (typically placed in `rundata/initial_conditions/spatial/`) within the `default_configs.json`.
        
    - Initial profiles should be placed under `rundata/initial_conditions/profiles/`.

### Step 5: Execute

Run the model from the `build` directory:

```bash
cd build
mpirun -np 1 ./vvm
```


#### Asynchronous I/O (Optional)

To use asynchronous output, specify the SST engine in your `default_configs.json`. You can then allocate dedicated tasks for I/O.

For example, to use **1 GPU/CPU for the model** and **1 CPU for I/O**:

```bash
mpirun -np 2 ./vvm --io-tasks 1
```

To use **2 GPUs/CPUs for the model** and **2 CPUs for I/O**:

```bash
mpirun -np 4 ./vvm --io-tasks 2
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


## Contact & Support
GitHub Issues: For bugs, feature requests, or code contributions, please open an issue on the GitHub repository.

Email: Users can contact us for more questions regarding the model or its usage at B08209006@ntu.edu.tw.
