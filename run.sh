#!/bin/bash
#SBATCH --account=MST114418
#SBATCH --partition=normal
#SBATCH --job-name=VVM_GPU_CPP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz85721@gmail.com

CONFIG_FILE="/home/mog/VVM_GPU_CPP_twnia3/rundata/input_configs/sea_grass_mountain.json"
export nvhpc_path=/home/mog/nvhpc_24_9/Linux_x86_64/24.9
export HPCX_HOME=$nvhpc_path/comm_libs/12.6/hpcx/hpcx-2.20
export MY_PLUGIN_PATH=$nvhpc_path/comm_libs/12.6/hpcx/hpcx-2.20/nccl_rdma_sharp_plugin/lib
export SHARP_LIB_PATH=$nvhpc_path/comm_libs/12.6/hpcx/hpcx-2.20/sharp/lib


# NOTE: User needs to specify things above this line
# ==========================================================

set -e

# Environment Setup & Compilation Phase
cmake --list-presets
CMAKE_PRESET_NAME="nano5"
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

source $HPCX_HOME/hpcx-init.sh
export PATH=$HPCX_HOME/ompi/bin:$PATH
export LD_LIBRARY_PATH=$HPCX_HOME/ompi/lib:$HPCX_HOME/ucx/lib:$HPCX_HOME/sharp/lib:$HPCX_HOME/nccl_rdma_sharp_plugin/lib:$LD_LIBRARY_PATH

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    omp_threads=$SLURM_CPUS_PER_TASK
else
    omp_threads=1
fi
echo "OMP_NUM_THREADS=" $OMP_NUM_THREADS

# Compile
echo "Building project..."
# cmake --build build -j32

# Minimalist Git Snapshot Phase
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

echo "Reading configuration and parsing internal paths..."
CURRENT_DIR=$(pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)

PARSE_CMD=$(python3 -c "
import json, sys, os
try:
    with open('$CONFIG_FILE') as f:
        d = json.load(f)
        
    out_dir = d.get('output', {}).get('output_dir', '')
    prof = d.get('initial_conditions', {}).get('source_file', '')
    spat = d.get('netcdf_reader', {}).get('source_file', '')
    
    def resolve(p):
        return os.path.abspath(os.path.join('$CURRENT_DIR', 'build', p)) if p else ''
        
    print(f\"OUTPUT_DIR='{out_dir}'\")
    print(f\"PROF_FILE='{resolve(prof)}'\")
    print(f\"SPAT_FILE='{resolve(spat)}'\")
except Exception as e:
    print(f\"echo 'Error parsing JSON: {e}'\")
    sys.exit(1)
")
eval "$PARSE_CMD"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Failed to read output.output_dir from $CONFIG_FILE"
    exit 1
fi

echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

SNAPSHOT_DIR="$OUTPUT_DIR/code_snapshot"

rm -rf "$SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"

echo "Creating a pristine, lightweight Git snapshot at $SNAPSHOT_DIR ..."

# Extract without ignored folders
INCLUDE_PATHS=$(git -C "$REPO_ROOT" ls-tree --name-only HEAD | grep -vE "^(rundata|tests|docs|externals)$" | tr '\n' ' ')
git -C "$REPO_ROOT" archive HEAD $INCLUDE_PATHS | tar -x -C "$SNAPSHOT_DIR"

# Pre-configure .gitignore
cd "$SNAPSHOT_DIR"
echo "rundata/" >> .gitignore
echo "tests/" >> .gitignore
echo "docs/" >> .gitignore
echo "externals/" >> .gitignore
echo "build/" >> .gitignore
echo "log/" >> .gitignore

# Initialize Git
git init -q
git add .
git -c user.name="VVM-Snapshot" -c user.email="snapshot@local" commit -q -m "Base commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
cd "$CURRENT_DIR"

# Rsync uncommitted modifications
rsync -a \
    --exclude='.git' \
    --exclude='build' \
    --exclude='log' \
    --exclude='rundata' \
    --exclude='tests' \
    --exclude='docs' \
    --exclude='externals' \
    --exclude='tags' \
    "$REPO_ROOT/" "$SNAPSHOT_DIR/"

# Copy config and inputs (Keeping original filenames)
echo "Copying configuration and specific input files..."
cp "$CONFIG_FILE" "$SNAPSHOT_DIR/"
echo " -> Copied $(basename "$CONFIG_FILE")"

if [ -f "$PROF_FILE" ]; then 
    cp "$PROF_FILE" "$SNAPSHOT_DIR/"
    echo " -> Copied $(basename "$PROF_FILE")"
fi

if [ -f "$SPAT_FILE" ]; then 
    cp "$SPAT_FILE" "$SNAPSHOT_DIR/"
    echo " -> Copied $(basename "$SPAT_FILE")"
fi

echo "Snapshot ready! It contains only source code and input files."
echo "---------------------------------------------------"

# Execution Phase
cd build

echo "Starting VVM model execution..."
mpirun -np 1 \
  -x NCCL_DEBUG=INFO \
  -x HDF5_USE_FILE_LOCKING=FALSE \
  --mca io ompio \
  --mca sharedfp ^lockedfile,individual \
  --bind-to socket --map-by socket \
  ./vvm "$CONFIG_FILE"
