#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys

# Pre-defined Defaults
DEFAULT_CONFIG = "./rundata/input_configs/taiwanvvm.json"
DEFAULT_COMPUTE = 1
DEFAULT_IO = 0
DEFAULT_NODES = 1
DEFAULT_CPUS = 8
DEFAULT_TIME = "24:00:00"
DEFAULT_OUT = "./log/%j.out"
DEFAULT_ERR = "./log/%j.err"
DEFAULT_JOB_NAME = "VVM_GPU_CPP"
DEFAULT_ACCOUNT = "MST114418"
DEFAULT_PARTITION = "normal"

def ask(prompt_text, default_val):
    ans = input(f"{prompt_text} [{default_val}]: ").strip()
    return ans if ans else default_val

def peek_io_engine(config_path):
    """Quickly read the JSON to determine the IO engine before proceeding."""
    try:
        with open(os.path.abspath(config_path), 'r') as f:
            return json.load(f).get("output", {}).get("engine", "HDF5")
    except Exception:
        return "HDF5"

def interactive_wizard():
    print("====================================================================")
    print("                   VVM GPU C++ Interactive Setup")
    print("====================================================================")
    print(" Tip: Skip this wizard using command line arguments.")
    print("   [Quick SLURM + SST]  ./submit.py -c sst_run.json --compute 16 --io 2")
    print("   [Full SLURM Config]  ./submit.py -c large_run.json --compute 16 --io 4 \\")
    print("                        --nodes 4 --gpus 5 --cpus 8 -t 48:00:00 \\")
    print("                        --job-name VVM_TYPHOON -A MST114418 -p normal")
    print("====================================================================\n")
    print(" (Press Enter at any prompt to use the default value)")

    class Args: pass
    args = Args()

    args.local = ask("Run locally without SLURM? (y/N)", "N").upper() == "Y"
    args.config = ask("Configuration file", DEFAULT_CONFIG)
    
    io_engine = peek_io_engine(args.config)

    args.compute = int(ask("Compute Tasks (MPI ranks)", DEFAULT_COMPUTE))
    
    if io_engine == "SST":
        print("\n[Notice] SST engine detected in config. ADIOS2 SST requires dedicated IO tasks.")
        args.io = int(ask("IO Tasks (MPI ranks)", "1"))
    else:
        args.io = 0
        
    total_tasks = args.compute + args.io
    args.cpus = int(ask("CPUs per task (OpenMP threads)", DEFAULT_CPUS))
    
    if not args.local:
        args.nodes = int(ask("\nNumber of Nodes", DEFAULT_NODES))
    else:
        args.nodes = 1
        
    # --- Smart Resource Recommendation (Applies to both SLURM and Local) ---
    tasks_per_node = math.ceil(total_tasks / args.nodes)
    print(f"\n[Recommendation] You have {total_tasks} total task(s) across {args.nodes} node(s).")
    print(f"                 To ensure 1 Task = 1 GPU (Best Performance),")
    print(f"                 you can request {tasks_per_node} GPU(s) per node. If not, IO and compute may share the same GPU.")
    
    args.gpus = int(ask("GPUs per Node", tasks_per_node))
    
    if args.gpus < tasks_per_node:
        print(f"  -> [WARNING] You requested {args.gpus} GPU(s) but will have up to {tasks_per_node} task(s) per node.")
        print(f"               Some MPI ranks will SHARE a GPU. This may degrade performance!\n")
    elif args.gpus > tasks_per_node:
        print(f"  -> [NOTICE] You requested more GPUs than tasks. Some GPUs will be idle.\n")
    # ---------------------------------------------------------------------

    if not args.local:
        args.time = ask("Time Limit", DEFAULT_TIME)
        args.job_name = ask("Job Name", DEFAULT_JOB_NAME)
        args.account = ask("SLURM Account", DEFAULT_ACCOUNT)
        args.partition = ask("SLURM Partition", DEFAULT_PARTITION)
        args.out = ask("Standard Output Log", DEFAULT_OUT)
        args.err = ask("Standard Error Log", DEFAULT_ERR)
    else:
        args.time = None
        args.job_name = DEFAULT_JOB_NAME
        args.account = None
        args.partition = None
        args.out = DEFAULT_OUT
        args.err = DEFAULT_ERR

    print("\n--- Setup Complete ---\n")
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="VVM GPU C++ Job Submission Wrapper")
    parser.add_argument("-c", "--config", help="Path to JSON configuration file")
    parser.add_argument("--local", action="store_true", help="Run locally without SLURM")
    parser.add_argument("--compute", type=int, default=DEFAULT_COMPUTE, help="Compute tasks")
    parser.add_argument("--io", type=int, default=0, help="IO tasks (Required for SST)")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node")
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS, help="CPUs per task (OpenMP threads)")
    parser.add_argument("-t", "--time", type=str, default=DEFAULT_TIME, help="Wall time limit")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Standard output file")
    parser.add_argument("--err", type=str, default=DEFAULT_ERR, help="Standard error file")
    parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME, help="SLURM job name")
    parser.add_argument("-A", "--account", type=str, default=DEFAULT_ACCOUNT, help="SLURM account")
    parser.add_argument("-p", "--partition", type=str, default=DEFAULT_PARTITION, help="SLURM partition")

    if len(sys.argv) == 1:
        return interactive_wizard()

    args = parser.parse_args()
    if not args.config:
        print("[Error] --config is required in command-line mode.")
        sys.exit(1)
        
    return args

def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)

    if not os.path.isfile(config_path):
        print(f"[Error] Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to parse JSON: {e}")
        sys.exit(1)

    output_info = config_data.get("output", {})
    io_engine = output_info.get("engine", "HDF5")
    out_dir = output_info.get("output_dir", "")
    
    prof = config_data.get("initial_conditions", {}).get("source_file", "")
    spat = config_data.get("netcdf_reader", {}).get("source_file", "")

    if not out_dir:
        print("[Error] output.output_dir missing in JSON.")
        sys.exit(1)

    if io_engine == "SST" and args.io == 0:
        print("[Error] SST engine requires at least 1 IO task (--io).")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    
    log_out_dir = os.path.dirname(args.out)
    if log_out_dir:
        os.makedirs(log_out_dir, exist_ok=True)
        
    log_err_dir = os.path.dirname(args.err)
    if log_err_dir:
        os.makedirs(log_err_dir, exist_ok=True)

    total_tasks = args.compute + args.io
    curr_dir = os.getcwd()
    env = os.environ.copy()
    
    env["VVM_CONFIG_FILE"] = config_path
    env["VVM_COMPUTE_TASKS"] = str(args.compute)
    env["VVM_IO_TASKS"] = str(args.io)
    env["VVM_TOTAL_TASKS"] = str(total_tasks)
    env["VVM_IO_ENGINE"] = io_engine
    env["VVM_OUTPUT_DIR"] = out_dir
    env["VVM_PROF_FILE"] = os.path.join(curr_dir, "build", prof) if prof else ""
    env["VVM_SPAT_FILE"] = os.path.join(curr_dir, "build", spat) if spat else ""
    env["OMP_NUM_THREADS"] = str(args.cpus)

    script_path = "tools/core_run.sh"
    if not os.path.isfile(script_path):
        print(f"[Error] Core script missing at {script_path}.")
        sys.exit(1)

    if args.local:
        print("=========================================")
        print(" Executing: LOCAL MODE (Bypassing SLURM)")
        print(f" Compute: {args.compute} | IO: {args.io} | Total MPI: {total_tasks}")
        print(f" GPUs/node: {args.gpus} | OpenMP Threads: {args.cpus}")
        print("=========================================")
        cmd = ["bash", script_path]
    else:
        print("=========================================")
        print(f" Executing: SLURM SUBMISSION")
        print(f" Job: {args.job_name} | Nodes: {args.nodes}")
        print(f" Tasks: {total_tasks} | GPUs/node: {args.gpus} | CPUs/task: {args.cpus}")
        print(f" Account: {args.account} | Partition: {args.partition}")
        print("=========================================")
        
        cmd = [
            "sbatch",
            f"--job-name={args.job_name}",
            f"--nodes={args.nodes}",
            f"--ntasks={total_tasks}",
            f"--gpus-per-node={args.gpus}",
            f"--cpus-per-task={args.cpus}",
            f"--time={args.time}",
            f"--output={args.out}",
            f"--error={args.err}"
        ]

        if args.account:
            cmd.append(f"--account={args.account}")
        if args.partition:
            cmd.append(f"--partition={args.partition}")

        cmd.append(script_path)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Process failed with code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
