export PATH=/work/aaron900129/local/bin:$PATH

export GCC_DIR=/work/aaron900129/gcc9
export PATH=$GCC_DIR/bin:$PATH
export C_INCLUDE_PATH=$GCC_DIR/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$GCC_DIR/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GCC_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC_DIR/lib:$LD_LIBRARY_PATH

export NVHPC_DEFAULT_CUDA=12.6
export NVHPC_VERSION=24.9
export NVHPC_ROOT=/work/aaron900129/nvhpc_24_9/Linux_x86_64/${NVHPC_VERSION}
export C_INCLUDE_PATH=${NVHPC_ROOT}/comm_libs/openmpi/openmpi-3.1.5/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/comm_libs/openmpi/openmpi-3.1.5/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${NVHPC_ROOT}/comm_libs/openmpi/openmpi-3.1.5/lib:$LIBRARY_PATH
export PATH=$NVHPC_ROOT/compilers/bin:$PATH

export CUDA_HOME=${NVHPC_ROOT}/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export PATH=$NVHPC_ROOT/comm_libs/openmpi/openmpi-3.1.5/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/compilers/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=${NVHPC_ROOT}/math_libs/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${NVHPC_ROOT}/math_libs/lib64:$LD_LIBRARY_PATH

export NVCOMPILER_COMM_LIBS_HOME=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/openmpi/openmpi-3.1.5

export Kokkos_DIR=/work/aaron900129/kokkos_install
export PATH=${Kokkos_DIR}/bin:$PATH
export C_INCLUDE_PATH=${Kokkos_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${Kokkos_DIR}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${Kokkos_DIR}/lib64:$LD_LIBRARY_PATH

export HDF5_DIR=/work/aaron900129/hdf5_install
export PATH=${HDF5_DIR}/bin:$PATH
export C_INCLUDE_PATH=${HDF5_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${HDF5_DIR}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${HDF5_DIR}/lib:$LD_LIBRARY_PATH

export ADIOS2_DIR=/work/aaron900129/adios2_install
export PATH=${ADIOS2_DIR}/bin:$PATH
export C_INCLUDE_PATH=${ADIOS2_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${ADIOS2_DIR}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${ADIOS2_DIR}/lib64:$LD_LIBRARY_PATH

export AMGX_DIR=/work/aaron900129/amgx_install
export PATH=${AMGX_DIR}/bin:$PATH
export C_INCLUDE_PATH=${AMGX_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${AMGX_DIR}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${AMGX_DIR}/lib:$LD_LIBRARY_PATH

export SZ_DIR=/work/aaron900129/sz_install
export PATH=${SZ_DIR}/bin:$PATH
export C_INCLUDE_PATH=${SZ_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${SZ_DIR}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${SZ_DIR}/lib64:$LD_LIBRARY_PATH

export ZFP_DIR=/work/aaron900129/zfp_install/
export PATH=${ZFP_DIR}/bin:$PATH
export C_INCLUDE_PATH=${ZFP_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${ZFP_DIR}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${ZFP_DIR}/lib64:$LD_LIBRARY_PATH

export ZSTD_DIR=/work/aaron900129/zstd_install
export PATH=${ZSTD_DIR}/bin:$PATH
export C_INCLUDE_PATH=${ZSTD_DIR}/include:$C_INCLUDE_PATH
export LIBRARY_PATH=${ZSTD_DIR}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${ZSTD_DIR}/lib64:$LD_LIBRARY_PATH
