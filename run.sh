#!/bin/bash
#SBATCH --account=MST114418
#SBATCH --partition=gp4d
#SBATCH --job-name=VVM_GPU_CPP
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=log//%j.out
#SBATCH --error=log/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz85721@gmail.com

source ./scripts/env_twnia2_nano5.sh

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
# make -j32
# && mpirun -np 4 --bind-to core ./vvm

mpirun -np 16 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ib0 --bind-to core ./vvm
