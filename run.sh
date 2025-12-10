#!/bin/bash
#SBATCH --account=MST114418
#SBATCH --partition=normal
#SBATCH --job-name=VVM_GPU_CPP
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=log//%j.out
#SBATCH --error=log/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz85721@gmail.com

source ./scripts/env_twnia2_nano5.sh

# export UCX_NET_DEVICES="mlx5_2"
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL


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
make -j16
# mpirun -np 2 --bind-to none ./vvm

# mpirun -np 2 -x NCCL_SOCKET_IFNAME=ib0 -x NCCL_DEBUG=INFO --mca btl_tcp_if_include ib0 --mca btl_base_verbose 30 ./vvm

export MY_PLUGIN_PATH=/work/aaron900129/nvhpc_24_9/Linux_x86_64/24.9/comm_libs/12.6/hpcx/hpcx-2.20/nccl_rdma_sharp_plugin/lib

mpirun -np 2 \
  -x LD_LIBRARY_PATH=$MY_PLUGIN_PATH:$LD_LIBRARY_PATH \
  -x NCCL_SOCKET_IFNAME=ib0 \
  -x NCCL_IB_HCA=mlx5_0,mlx5_2 \
  -x NCCL_DEBUG=INFO \
  -x OMP_PROC_BIND=spread \
  -x OMP_PLACES=threads \
  -x HDF5_USE_FILE_LOCKING=FALSE \
  --mca io ompio \
  --mca sharedfp ^lockedfile,individual \
  --mca btl_tcp_if_include ib0 \
  --mca oob_tcp_if_include ib0 \
  --mca btl_base_warn_component_unused 0 \
  ./vvm


# mpirun -np 16 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ib0 --bind-to core ./vvm
