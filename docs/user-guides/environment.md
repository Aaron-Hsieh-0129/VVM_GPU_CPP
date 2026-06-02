# GPUVVM Installation Guide
This guide provides step-by-step instructions for building the dependencies required for GPUVVM from source.

## 0. Preparation
To make this guide easy to copy and paste, please define your target installation directory first. All libraries will be installed under this directory.

```bash
# Replace this with your desired installation path
export INSTALL_DIR=/path/to/your/custom/libs
mkdir -p $INSTALL_DIR

export PATH=$INSTALL_DIR/bin:$PATH
export C_INCLUDE_PATH=$INSTALL_DIR/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/$LIB/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```
Before compiling the base compiler (GCC), it is recommended to clear your library paths to avoid linking conflicts with system libraries:

```bash
unset LIBRARY_PATH LD_LIBRARY_PATH
```

---

## 1. Compiler & Core Tools

### GCC 11.4
GPUVVM requires C++17 support. If your system GCC is too old, build GCC 11.4:

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
*(Note: If your system has a higher version of GCC but you still want to force CMake to use this GCC 11, you will need to add `-DCMAKE_C_FLAGS="--gcc-toolchain=$INSTALL_DIR/gcc11"` and `-DCMAKE_CXX_FLAGS="--gcc-toolchain=$INSTALL_DIR/gcc11"` during the configuration.)*

### NVIDIA HPC SDK (NVHPC 24.9)
Download and install NVHPC 24.9 from the official NVIDIA website. Assuming it is installed at `/path/to/nvhpc_24_9`, export its path so we can use its MPI and CUDA wrappers for the rest of the installation:

```bash
export NVHPC_DEFAULT_CUDA=13.0
export NVHPC_VERSION=25.9
export NVHPC_HPCX_VERSION=2.24
export NVHPC_ROOT=/path/to/nvhpc/Linux_x86_64/${NVHPC_VERSION}

# Just give info above

export CUDA_HOME=${NVHPC_ROOT}/cuda/${NVHPC_DEFAULT_CUDA}
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=${NVHPC_ROOT}/compilers/bin:$PATH
export LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${NVHPC_ROOT}/math_libs/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LD_LIBRARY_PATH

export HPCX_HOME=${NVHPC_ROOT}/comm_libs/${NVHPC_DEFAULT_CUDA}/hpcx/hpcx-${NVHPC_HPX_VERSION}
export PATH=${HPCX_HOME}/ompi/bin:${HPCX_HOME}/ucx/bin:${PATH}
export LD_LIBRARY_PATH=${HPCX_HOME}/ompi/lib:${HPCX_HOME}/ucx/lib:${HPCX_HOME}/sharp/lib:${HPCX_HOME}/nccl_rdma_sharp_plugin/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${HPCX_HOME}/ompi/lib:${HPCX_HOME}/ucx/lib:${HPCX_HOME}/sharp/lib:${HPCX_HOME}/nccl_rdma_sharp_plugin/lib:${LIBRARY_PATH}
export C_INCLUDE_PATH=${HPCX_HOME}/ompi/include:${HPCX_HOME}/ucx/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${HPCX_HOME}/ompi/include:${HPCX_HOME}/ucx/include:${CPLUS_INCLUDE_PATH}
export OPAL_PREFIX=${HPCX_HOME}/ompi
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
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz
tar -zxvf hdf5-1.14.5.tar.gz
cd hdf5-1.14.5
./configure --prefix=$INSTALL_DIR \
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
./configure --prefix=$INSTALL_DIR
make -j$(nproc)
make install
cd ..
```

### NetCDF-C 4.4.1.1
IMPORTANT WARNING: Do NOT use MPI wrappers (mpicc, mpic++) to compile NetCDF-C and NetCDF-Fortran. Using MPI wrappers here will cause errors in the RRTMGP NetCDF reader. Use standard serial compilers (gcc, g++, gfortran) instead.
```bash
wget https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.4.1.1.tar.gz
tar -zxvf v4.4.1.1.tar.gz
cd netcdf-c-4.4.1.1
./configure --prefix=$INSTALL_DIR \
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
wget https://parallel-netcdf.github.io/Release/pnetcdf-1.14.1.tar.gz
tar -zxvf pnetcdf-1.14.1.tar.gz
cd pnetcdf-1.14.1
./configure --prefix=$INSTALL_DIR \
            --with-netcdf4=$INSTALL_DIR \
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
wget https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.4.1.tar.gz
tar -zxvf v4.4.1.tar.gz
cd netcdf-fortran-4.4.1
export FFLAGS="-g -O2 -fallow-argument-mismatch" 
export FCFLAGS="-g -O2 -fallow-argument-mismatch"

./configure --prefix=$INSTALL_DIR \
            --enable-shared \
            CC=gcc FC=gfortran
make -j$(nproc)
make install
cd ..

```

### Kokkos 4.7.01
Note: Replace -DKokkos_ARCH_HOPPER90=ON with the appropriate architecture flag for your GPU (e.g., AMPERE80, VOLTA70, ADA89).
```bash
wget https://github.com/kokkos/kokkos/releases/download/4.7.01/kokkos-4.7.01.tar.gz
tar -zxvf kokkos-4.7.01.tar.gz
cd kokkos-4.7.01
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ARCH_HOPPER90=ON \
    -DBUILD_SHARED_LIBS=TRUE
make -j$(nproc)
make install
cd ../..

```


### ADIOS2 2.11.0

```bash
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
git checkout tags/v2.11.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpic++ \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_HDF5=ON \
    -DADIOS2_USE_Kokkos=ON \
    -DKokkos_ROOT=$INSTALL_DIR \
    -DHDF5_ROOT=$INSTALL_DIR
make -j$(nproc)
make install
cd ../..

```

Warning: If using less than 2.11.0, some errors may appear when compiling VVM. You need to modify adios2/cxx/KokkosView.h to adios2/cxx11/KokkosView.h in the code and cmakelist.

---

## 4. Environment Setup Script
To avoid cluttering your `.bashrc`, create a file named `env_setup.sh` in your workspace. Source this file (`source env_setup.sh`) every time before compiling or running GPUVVM.
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
export PATH=$INSTALL_DIR/bin:$PATH
export C_INCLUDE_PATH=$INSTALL_DIR/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/$LIB/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$INSTALL_DIR/lib64:$INSTALL_DIR/lib:$LD_LIBRARY_PATH

echo "GPUVVM Environment Loaded Successfully!"

```