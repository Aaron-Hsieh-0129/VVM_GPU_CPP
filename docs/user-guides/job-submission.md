# Job Submission

GPUVVM jobs should normally be launched with the root-level `submit.py` wrapper. The wrapper is the supported path because it reads `CMakePresets.json`, prepares library paths, creates run directories, separates compute and I/O ranks, and requests CPU/GPU resources consistently for local or SLURM execution.

**Use `submit.py` first:** Direct `mpirun` commands are kept only for advanced debugging. For performance runs, incorrect CPU/GPU assignment can make ranks share GPUs, starve I/O tasks, or slow the model substantially.

## Interactive mode

Run the wrapper without arguments and answer the prompts:

```bash
cd $VVM_ROOT
./submit.py
```

The wizard detects available CMake presets, checks the configured output engine, and recommends a GPU count so the usual mapping is one MPI task per GPU.

It also shows which fields you need to fill in, explains the important run options, and prints an equivalent command-line invocation at the end. Save that command for future runs when you want to skip the interactive phase.

## Command-line mode

Use command-line mode for repeatable local tests and batch submissions.

### Local HDF5 test

```bash
cd $VVM_ROOT
./submit.py \
  --local \
  --preset <your_preset_name> \
  -c ./rundata/input_configs/default_config.json \
  --compute 4
```

### SLURM compute run

```bash
cd $VVM_ROOT
./submit.py \
  --preset <your_preset_name> \
  -c ./rundata/input_configs/default_config.json \
  --compute 16 \
  --nodes 1 \
  --gpus 16 \
  --cpus 1 \
  -t 24:00:00
```

### SLURM with SST I/O

```bash
cd $VVM_ROOT
./submit.py \
  --preset <your_preset_name> \
  -c ./rundata/input_configs/default_config.json \
  --compute 16 \
  --io 4 \
  --nodes 4 \
  --gpus 5 \
  --cpus 1 \
  -t 24:00:00
```

## Choosing resources

`--compute` is the number of simulation MPI ranks. `--io` is the number of dedicated I/O ranks used when `output.engine` is `SST`. The total MPI size is:

```text
total MPI tasks = compute tasks + I/O tasks
```

For GPU runs, request enough GPUs so compute ranks do not unexpectedly share devices. A common starting point is:

```text
GPUs per node >= ceil((compute tasks + I/O tasks) / nodes)
```

I/O ranks are CPU-side SST readers/writers, but they still consume MPI ranks and CPU cores. If you use fewer GPUs than tasks per node, the wrapper will warn that MPI ranks may share GPUs.

## What the wrapper does

- Sets `VVM_ROOT` from the script location.
- Loads compilers and library paths from the selected `CMakePresets.json` entry.
- Creates the configured output and log directories.
- Requires `--io` when the JSON uses `output.engine = "SST"`.
- Runs `tools/core_run.sh` locally or submits it through `sbatch`.
- Maps local MPI ranks to visible GPU IDs and exports runtime variables used by the executable.

## Direct MPI commands

Direct MPI remains useful for small debugging sessions after the environment has already been prepared.

**Advanced only:** These commands bypass the wrapper's allocation checks. On multi-node or multi-GPU runs, make sure rank placement, GPU visibility, CPU binding, OpenMP threads, and I/O ranks are correct before trusting performance numbers.

```bash
cd $VVM_ROOT
mpirun -np 1 ./build/vvm ./rundata/input_configs/default_config.json
```

For asynchronous I/O, reserve the final ranks for I/O servers:

```bash
cd $VVM_ROOT

# 1 simulation rank + 1 I/O rank
mpirun -np 2 ./build/vvm ./rundata/input_configs/default_config.json --io-tasks 1

# 2 simulation ranks + 2 I/O ranks
mpirun -np 4 ./build/vvm ./rundata/input_configs/default_config.json --io-tasks 2
```
