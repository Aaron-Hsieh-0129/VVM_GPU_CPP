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

# rm -rf build
# mkdir build
#
# if [ -n "$SLURM_CPUS_PER_TASK" ]; then
#     omp_threads=$SLURM_CPUS_PER_TASK
# else
#     omp_threads=1
# fi
# export OMP_NUM_THREADS=$omp_threads
# echo $OMP_NUM_THREADS
#
# # Compile and run
# cd build/ && /work/aaron900129/local/bin/cmake .. && make -j 4 && mpirun -np 4 --bind-to core ./vvm

cd build

# Find output path and convert files
CONFIG_FILE="../data/input_configs/default_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: cannot find configuration file, '$CONFIG_FILE'"
    exit 1
fi

OUTPUT_DIR=$(grep -oE '"output_dir": ".*"' "$CONFIG_FILE" | sed -E 's/"output_dir": "(.*)",?/\1/')
FILENAME_PREFIX=$(grep -oE '"output_filename_prefix": ".*"' "$CONFIG_FILE" | sed -E 's/"output_filename_prefix": "(.*)",?/\1/')

FULL_OUTPUT_PATH="${OUTPUT_DIR}/${FILENAME_PREFIX}.bp"

echo "Output file path: "
echo "$FULL_OUTPUT_PATH"

cp ../scripts/convert_multiprocess.py ${OUTPUT_DIR}/.
cd ${OUTPUT_DIR}

/work/aaron900129/adios2_install/bin/bp2h5 ${FILENAME_PREFIX}.bp ${FILENAME_PREFIX}.nc
python convert_multiprocess.py

