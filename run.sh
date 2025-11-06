#!/bin/bash
#SBATCH --account=MST114049
#SBATCH --partition=gp2d
#SBATCH --job-name=VVM_GPU_CPP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=log//%j.out
#SBATCH --error=log/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz85721@gmail.com

source ./scripts/env_twnia2_nano5.sh

# rm -rf build
# mkdir build

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    omp_threads=$SLURM_CPUS_PER_TASK
else
    omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads
echo $OMP_NUM_THREADS

# Compile and run
cd build/ && /work/aaron900129/local/bin/cmake .. && make -j 4 && mpirun -np 4 --bind-to core ./vvm

