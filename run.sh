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

source ./scripts/env_twnia2_nano5.sh
export MY_PLUGIN_PATH=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/nccl_rdma_sharp_plugin/lib
export SHARP_LIB_PATH=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/sharp/lib

export HPCX_HOME=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20
source $HPCX_HOME/hpcx-init.sh
hpcx_load
export PATH=$HPCX_HOME/ompi/bin:$PATH
export LD_LIBRARY_PATH=$HPCX_HOME/ompi/lib:$HPCX_HOME/ucx/lib:$HPCX_HOME/sharp/lib:$HPCX_HOME/nccl_rdma_sharp_plugin/lib:$LD_LIBRARY_PATH

# if [ -n "$SLURM_CPUS_PER_TASK" ]; then
#     omp_threads=$SLURM_CPUS_PER_TASK
# else
#     omp_threads=1
# fi
export OMP_NUM_THREADS=32
echo $OMP_NUM_THREADS

# Compile and run
cd build/
# cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_Fortran_COMPILER=mpifort
# && mpirun -np 4 --bind-to core ./vvm
make -j64


mpirun -np 4 \
  -x NCCL_DEBUG=INFO \
  -x HDF5_USE_FILE_LOCKING=FALSE \
  --mca io ompio \
  --mca sharedfp ^lockedfile,individual \
  --bind-to socket --map-by socket \
  ./vvm
