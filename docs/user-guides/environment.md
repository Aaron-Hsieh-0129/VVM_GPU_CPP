# GVVM Installation Guide
This guide provides step-by-step instructions for building the dependencies required for GVVM from source.

## 0. Preparation
To make this guide easy to copy and paste, please define your target installation directory first. All libraries will be installed under this directory.

```bash
# Replace this with your desired installation path
export INSTALL_DIR=/path/to/your/custom/libs
mkdir -p $INSTALL_DIR

```
Before compiling the base compiler (GCC), it is recommended to clear your library paths to avoid linking conflicts with system libraries:

```bash
unset LIBRARY_PATH LD_LIBRARY_PATH

```

---

## 1. Compiler & Core Tools

### GCC 11.4
GVVM requires C++17 support. If your system GCC is too old, build GCC 11.4:

```bash
wget https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/gcc-11.4.0.tar.gz
tar -zxvf gcc-11.4.0.tar.gz
cd gcc-11.4.0
./contrib/download_prerequisites

mkdir build && cd build
../configure --prefix=$INSTALL_DIR/gcc11 \
             --enable-languages=c,c++,fortran \
             --disable-multilib \
             --disable-bootstrap
make -j$(nproc)
make install
cd ../..

# Add to current session for subsequent builds
export PATH=$INSTALL_DIR/gcc11/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/gcc11/lib64:$INSTALL_DIR/gcc11/lib:$LD_LIBRARY_PATH

```
*(Note: If your system has a higher version of GCC but you still want to force CMake to use this GCC 11, you will need to add `-DCMAKE_C_FLAGS="--gcc-toolchain=$INSTALL_DIR/gcc11"` and `-DCMAKE_CXX_FLAGS="--gcc-toolchain=$INSTALL_DIR/gcc11"` during the GVVM CMake configuration step later.)*

### NVIDIA HPC SDK (NVHPC 24.9)
Download and install NVHPC 24.9 from the official NVIDIA website. Assuming it is installed at `/path/to/nvhpc_24_9`, export its path so we can use its MPI and CUDA wrappers for the rest of the installation:

```bash
export NVHPC_ROOT=/path/to/nvhpc_24_9/Linux_x86_64/24.9
export CUDA_HOME=$NVHPC_ROOT/cuda/12.6
export PATH=$NVHPC_ROOT/comm_libs/openmpi/openmpi-3.1.5/bin:$CUDA_HOME/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/comm_libs/openmpi/openmpi-3.1.5/lib:$CUDA_HOME/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$LD_LIBRARY_PATH

```

### CMake 4.2.0

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.2.0/cmake-4.2.0.tar.gz
tar -zxvf cmake-4.2.0.tar.gz
cd cmake-4.2.0
./configure --prefix=$INSTALL_DIR/cmake
make -j$(nproc)
make install
cd ..

export PATH=$INSTALL_DIR/cmake/bin:$PATH

```

---

## 2. I/O Libraries

### HDF5 1.14.5
Must be compiled with MPI wrappers for parallel I/O.

```bash
tar -zxvf hdf5-1.14.5.tar.gz
cd hdf5-1.14.5
./configure --prefix=$INSTALL_DIR/hdf5 \
            --enable-parallel --enable-shared --enable-cxx --enable-unsupported \
            CC="mpicc" CXX="mpic++" FC="mpifort" LIBS="-lm"
make -j$(nproc)
make install
cd ..

```

### ZLIB 1.3.1 (Optional)
*(Skip this step if zlib is already available on your system.)*

```bash
tar -zxvf zlib-1.3.1.tar.gz
cd zlib-1.3.1
./configure --prefix=$INSTALL_DIR/zlib
make -j$(nproc)
make install
cd ..

```

### NetCDF-C 4.4.1.1
IMPORTANT WARNING: Do NOT use MPI wrappers (mpicc, mpic++) to compile NetCDF-C and NetCDF-Fortran. Using MPI wrappers here will cause errors in the RRTMGP NetCDF reader. Use standard serial compilers (gcc, g++, gfortran) instead.
```bash
tar -zxvf netcdf-c-4.4.1.1.tar.gz
cd netcdf-c-4.4.1.1
./configure --prefix=$INSTALL_DIR/netcdf-c \
            --enable-netcdf-4 \
            CC=gcc CXX=g++ FC=gfortran \
            CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" FCFLAGS="-fPIC -O2"
make -j$(nproc)
make install
cd ..

```

### PnetCDF 1.14.1
PnetCDF provides parallel I/O for classic NetCDF files. This **must** be compiled with MPI wrappers.

```bash
tar -zxvf pnetcdf-1.14.1.tar.gz
cd pnetcdf-1.14.1
./configure --prefix=$INSTALL_DIR/pnetcdf \
            --with-netcdf4=$INSTALL_DIR/netcdf-c \
            --disable-shared \
            CC=mpicc CXX=mpic++ FC=mpifort \
            CFLAGS="-fPIC -O2" CXXFLAGS="-fPIC -O2" FFLAGS="-fPIC -O2" FCFLAGS="-fPIC -O2"
make -j$(nproc)
make install
cd ..

```

### NetCDF-Fortran 4.4.1
IMPORTANT WARNING: Again, use standard serial compilers (gcc, gfortran), NOT MPI wrappers. The -fallow-argument-mismatch flag is required for modern GCC.
```bash
tar -zxvf netcdf-fortran-4.4.1.tar.gz
cd netcdf-fortran-4.4.1
export FFLAGS="-g -O2 -fallow-argument-mismatch" 
export FCFLAGS="-g -O2 -fallow-argument-mismatch"

./configure --prefix=$INSTALL_DIR/netcdf-fortran \
            --enable-shared \
            CC=gcc FC=gfortran
make -j$(nproc)
make install
cd ..

```

### ADIOS2 2.10.0

```bash
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/adios2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_HDF5=ON \
    -DADIOS2_USE_Kokkos=ON \
    -DKokkos_ROOT=$INSTALL_DIR/kokkos \
    -DHDF5_ROOT=$INSTALL_DIR/hdf5
make -j$(nproc)
make install
cd ../..

```

---

## 3. Core Frameworks (Kokkos & ADIOS2)

### Kokkos 4.7.01
Note: Replace -DKokkos_ARCH_HOPPER90=ON with the appropriate architecture flag for your GPU (e.g., AMPERE80, VOLTA70, ADA89).
```bash
tar -zxvf kokkos-4.7.01.tar.gz
cd kokkos-4.7.01
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/kokkos \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ARCH_HOPPER90=ON \
    -DBUILD_SHARED_LIBS=TRUE \
    -DKokkos_ENABLE_MPI=ON
make -j$(nproc)
make install
cd ../..

```

---

## 4. Environment Setup Script
To avoid cluttering your `.bashrc`, create a file named `env_setup.sh` in your workspace. Source this file (`source env_setup.sh`) every time before compiling or running GVVM.
**env_setup.sh:**

```bash
#!/bin/bash

# --- 1. Base Paths ---
export INSTALL_DIR=/path/to/your/custom/libs
export NVHPC_ROOT=/path/to/nvhpc_24_9/Linux_x86_64/24.9

# --- 2. NVHPC & MPI & CUDA ---
export CUDA_HOME=$NVHPC_ROOT/cuda/12.6
export OPAL_PREFIX=$NVHPC_ROOT/comm_libs/openmpi/openmpi-3.1.5

export PATH=$CUDA_HOME/bin:$OPAL_PREFIX/bin:$NVHPC_ROOT/compilers/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$OPAL_PREFIX/include:$NVHPC_ROOT/math_libs/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$OPAL_PREFIX/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$OPAL_PREFIX/lib:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/math_libs/lib64:$LD_LIBRARY_PATH

# --- 3. GCC 11 & CMake ---
export PATH=$INSTALL_DIR/gcc11/bin:$INSTALL_DIR/cmake/bin:$PATH
export C_INCLUDE_PATH=$INSTALL_DIR/gcc11/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$INSTALL_DIR/gcc11/lib64:$INSTALL_DIR/gcc11/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/gcc11/lib64:$INSTALL_DIR/gcc11/lib:$LD_LIBRARY_PATH

# --- 4. I/O & Framework Libraries ---
for LIB in hdf5 netcdf-c pnetcdf netcdf-fortran kokkos adios2; do
    export PATH=$INSTALL_DIR/$LIB/bin:$PATH
    export C_INCLUDE_PATH=$INSTALL_DIR/$LIB/include:$C_INCLUDE_PATH
    export LIBRARY_PATH=$INSTALL_DIR/$LIB/lib64:$INSTALL_DIR/$LIB/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$INSTALL_DIR/$LIB/lib64:$INSTALL_DIR/$LIB/lib:$LD_LIBRARY_PATH
done

echo "GVVM Environment Loaded Successfully!"

```

---

## 5. Running GVVM
Once compiled, you can run GVVM using MPI. Here are some examples of execution commands:
**Standard run (binding to cores):**

```bash
mpirun -np 4 ./vvm

```
*(Note: If you are using asynchronous I/O via ADIOS2 SST engine, remember to append the --io-tasks flag as described in the Quick Start guide).*
