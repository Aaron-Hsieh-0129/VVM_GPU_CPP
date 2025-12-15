#!/bin/bash
#SBATCH --account=MST114418
#SBATCH --partition=normal
#SBATCH --job-name=VVM_GPU_CPP
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=log//%j.out
#SBATCH --error=log/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz85721@gmail.com


cmake --list-presets
CMAKE_PRESET_NAME="blaze"
BUILD_DIR="build"


if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "Configuring project..."
    cmake --preset $CMAKE_PRESET_NAME
    
    if [ $? -ne 0 ]; then
        echo "Configuration failed."
        exit 1
    fi
else
    echo "Configuration found. Skipping configure step."
fi


export MY_PLUGIN_PATH=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/nccl_rdma_sharp_plugin/lib
export SHARP_LIB_PATH=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/sharp/lib

export HPCX_HOME=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20
source $HPCX_HOME/hpcx-init.sh
hpcx_load
export PATH=$HPCX_HOME/ompi/bin:$PATH
export LD_LIBRARY_PATH=$HPCX_HOME/ompi/lib:$HPCX_HOME/ucx/lib:$HPCX_HOME/sharp/lib:$HPCX_HOME/nccl_rdma_sharp_plugin/lib:$LD_LIBRARY_PATH

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    omp_threads=$SLURM_CPUS_PER_TASK
else
    omp_threads=1
fi
echo $OMP_NUM_THREADS

# Compile
cmake --build build -j32

# Run
cd build
mpirun -np 4 \
  -x NCCL_DEBUG=INFO \
  -x HDF5_USE_FILE_LOCKING=FALSE \
  --mca io ompio \
  --mca sharedfp ^lockedfile,individual \
  --bind-to socket --map-by socket \
  ./vvm
