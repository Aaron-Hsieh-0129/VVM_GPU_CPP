#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
import sys


def get_vvm_root():
    """Automatically detect the project root where submit.py is located."""
    return os.path.dirname(os.path.abspath(__file__))

def get_available_presets(vvm_root):
    """Parse CMakePresets.json to get a list of available preset names."""
    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    if not os.path.exists(preset_file):
        return []
    try:
        with open(preset_file, 'r') as f:
            presets_data = json.load(f)
        return [p.get("name") for p in presets_data.get("configurePresets", []) if p.get("name")]
    except Exception:
        return []

def setup_environment(preset_name):
    """Parse CMakePresets.json and inject paths into the environment natively."""
    env = os.environ.copy()
    vvm_root = get_vvm_root()
    env["VVM_ROOT"] = vvm_root
     
    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    if not os.path.exists(preset_file):
        print(f"[Warning] {preset_file} not found. Environment may be incomplete.")
        return env

    try:
        with open(preset_file, 'r') as f:
            presets_data = json.load(f)
         
        cache_vars = {}
        for p in presets_data.get("configurePresets", []):
            if p.get("name") == preset_name:
                cache_vars = p.get("cacheVariables", {})
                break
                 
        if not cache_vars:
            print(f"[Warning] Preset '{preset_name}' not found in CMakePresets.json.")
            return env
             
        print(f"[Info] Successfully loaded environment from CMake Preset: '{preset_name}'")

        # 1. Dynamically extract HPCX_HOME from the CXX Compiler path
        cxx_compiler = cache_vars.get("CMAKE_CXX_COMPILER", "")
        if "/ompi/bin/" in cxx_compiler:
            hpcx_home = cxx_compiler.split("/ompi/bin/")[0]
            my_plugin_path = f"{hpcx_home}/nccl_rdma_sharp_plugin/lib"
            sharp_lib_path = f"{hpcx_home}/sharp/lib"
             
            env["HPCX_HOME"] = hpcx_home
            env["MY_PLUGIN_PATH"] = my_plugin_path
            env["SHARP_LIB_PATH"] = sharp_lib_path
             
            env["VVM_PRE_RUN_CMD"] = f"source {hpcx_home}/hpcx-init.sh"
            env["PATH"] = f"{hpcx_home}/ompi/bin:" + env.get("PATH", "")

        # 2. Extract I/O Library Paths
        lib_dirs = []
        for key in ["HDF5_DIR", "NETCDF_C_DIR", "NETCDF_Fortran_DIR", "PNETCDF_DIR"]:
            val = cache_vars.get(key, "")
            if val:
                lib_dirs.append(f"{val}/lib")
                lib_dirs.append(f"{val}/lib64")
         
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    except Exception as e:
        print(f"[Warning] Error parsing CMakePresets.json: {e}")
         
    return env

# Pre-defined Defaults
DEFAULT_CONFIG = "rundata/input_configs/taiwanvvm.json"
DEFAULT_COMPUTE = 1
DEFAULT_IO = 0
DEFAULT_NODES = 1
DEFAULT_CPUS = 8
DEFAULT_TIME = "24:00:00"
DEFAULT_OUT = "log/%j.out"
DEFAULT_ERR = "log/%j.err"
DEFAULT_JOB_NAME = "VVM_GPU_CPP"
DEFAULT_ACCOUNT = "MST114418"
DEFAULT_PARTITION = "normal"

def ask(prompt_text, default_val):
    ans = input(f"{prompt_text} [{default_val}]: ").strip()
    return ans if ans else default_val

def peek_io_engine(config_path):
    try:
        with open(os.path.abspath(config_path), 'r') as f:
            return json.load(f).get("output", {}).get("engine", "HDF5")
    except Exception:
        return "HDF5"

def create_code_snapshot(repo_root, snapshot_dir, config_path, prof_path, spat_path, out_dir_raw):
    print(f"\n[Info] Creating code snapshot at: {snapshot_dir}")
     
    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)
         
    out_base = os.path.normpath(out_dir_raw).split(os.sep)[0]
    if out_base in [".", "..", ""]:  
        out_base = "output"   
         
    ignore_patterns = shutil.ignore_patterns(
        '.git', 'build', 'log', 'rundata', 'tests', 'docs', 'externals', 'tags', '*.o', 'output', out_base
    )
    shutil.copytree(repo_root, snapshot_dir, ignore=ignore_patterns)

    shutil.copy2(config_path, snapshot_dir)
    if prof_path and os.path.isfile(prof_path):
        shutil.copy2(prof_path, snapshot_dir)
    if spat_path and os.path.isfile(spat_path):
        shutil.copy2(spat_path, snapshot_dir)

    gitignore_content = f"rundata/\ntests/\ndocs/\nexternals/\nbuild/\nlog/\noutput/\n{out_base}/\n"
    with open(os.path.join(snapshot_dir, ".gitignore"), "w") as f:
        f.write(gitignore_content)

    try:
        subprocess.run(["git", "init", "-q"], cwd=snapshot_dir, check=True)
        subprocess.run(["git", "add", "."], cwd=snapshot_dir, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Snapshot", "-c", "user.email=snap@local", "commit", "-q", "-m", "Auto Snapshot"],
            cwd=snapshot_dir, check=True
        )
    except Exception as e:
        print(f"[Warning] Git snapshot commit failed (Ignored): {e}")

def interactive_wizard():
    vvm_root = get_vvm_root()
    presets = get_available_presets(vvm_root)
    default_preset = presets[0] if presets else "unknown"

    print("====================================================================")
    print("                   GVVM Interactive Setup Wizard")
    print("====================================================================")
    print("  Note: All relative paths are based on the auto-detected $VVM_ROOT.")
    print("--------------------------------------------------------------------")
    print(" Configurable Options (Default Values):")
    print(f"   --local        : Run locally without SLURM (False)")
    print(f"   -c, --config   : Configuration file ({DEFAULT_CONFIG})")
    print(f"   --preset       : CMake Preset to load environment from")
    print(f"   --compute      : Compute Tasks / MPI ranks ({DEFAULT_COMPUTE})")
    print(f"   --io           : IO Tasks / MPI ranks ({DEFAULT_IO})")
    print(f"   --nodes        : Number of Nodes ({DEFAULT_NODES})")
    print(f"   --gpus         : GPUs per Node (Auto-calculated/Recommended 1:1)")
    print(f"   --cpus         : CPUs per task / OpenMP threads ({DEFAULT_CPUS})")
    print(f"   -t, --time     : Wall time limit ({DEFAULT_TIME})")
    print(f"   --out          : Standard output log file ({DEFAULT_OUT})")
    print(f"   --err          : Standard error log file ({DEFAULT_ERR})")
    print(f"   --job-name     : SLURM Job Name ({DEFAULT_JOB_NAME})")
    print(f"   -A, --account  : SLURM Account ({DEFAULT_ACCOUNT})")
    print(f"   -p, --partition: SLURM Partition ({DEFAULT_PARTITION})")
    print("====================================================================")

    class Args: pass
    args = Args()

    args.local = ask("Run locally without SLURM? (y/N)", "N").upper() == "Y"
    args.config = ask("Configuration file", DEFAULT_CONFIG)
     
    if presets:
        print("\n Available CMake Presets:")
        for i, p in enumerate(presets):
            print(f"   [{i+1}] {p}")
        preset_choice = ask(f"Select Preset (1-{len(presets)} or type name)", "1")
        if preset_choice.isdigit() and 1 <= int(preset_choice) <= len(presets):
            args.preset = presets[int(preset_choice)-1]
        else:
            args.preset = preset_choice
    else:
        args.preset = ask("CMake Preset Environment", default_preset)
    # -----------------------

    io_engine = peek_io_engine(args.config)
    args.compute = int(ask("\nCompute Tasks (MPI ranks)", DEFAULT_COMPUTE))
     
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
         
    tasks_per_node = math.ceil(total_tasks / args.nodes)
    print(f"\n[Recommendation] You have {total_tasks} total task(s) across {args.nodes} node(s).")
    print(f"                 To ensure 1 Task = 1 GPU (Best Performance), request {tasks_per_node} GPU(s) per node.")
     
    args.gpus = int(ask("GPUs per Node", tasks_per_node))
     
    if args.gpus < tasks_per_node:
        print(f"  -> [WARNING] Some MPI ranks will SHARE a GPU. This may degrade performance!\n")

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

    # --- NEW: Construct the equivalent full command line string ---
    cmd_parts = [sys.argv[0]]
    if args.local:
        cmd_parts.append("--local")
    
    cmd_parts.append(f'-c "{args.config}"')
    cmd_parts.append(f'--preset "{args.preset}"')
    cmd_parts.append(f'--compute {args.compute}')
    cmd_parts.append(f'--io {args.io}')
    cmd_parts.append(f'--nodes {args.nodes}')
    cmd_parts.append(f'--gpus {args.gpus}')
    cmd_parts.append(f'--cpus {args.cpus}')
    
    if not args.local:
        if args.time: cmd_parts.append(f'-t "{args.time}"')
        if args.out: cmd_parts.append(f'--out "{args.out}"')
        if args.err: cmd_parts.append(f'--err "{args.err}"')
        if args.job_name: cmd_parts.append(f'--job-name "{args.job_name}"')
        if args.account: cmd_parts.append(f'-A "{args.account}"')
        if args.partition: cmd_parts.append(f'-p "{args.partition}"')

    full_cmd = " ".join(cmd_parts)

    print("\n--- Setup Complete ---")
    print("\n[Tip] You can skip this wizard in the future by running the following command directly:\n")
    print(f"  {full_cmd}\n")
    print("----------------------\n")
    
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="VVM GPU C++ Job Submission Wrapper")
    parser.add_argument("-c", "--config", help="Path to JSON configuration file")
    parser.add_argument("--preset", type=str, help="CMake Preset name to load environment from")
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
         
    if not args.preset:
        presets = get_available_presets(get_vvm_root())
        args.preset = presets[0] if presets else "unknown"
         
    return args

def main():
    args = parse_args()
     
    env = setup_environment(args.preset)
    VVM_ROOT = env.get("VVM_ROOT")
     
    if not VVM_ROOT:
        print("\n[Error] Failed to detect VVM_ROOT. Ensure script is run from project root.\n")
        sys.exit(1)

    os.chdir(VVM_ROOT)

    config_path_user = os.path.abspath(args.config)
    if not os.path.isfile(config_path_user):
        print(f"[Error] Configuration file not found: {config_path_user}")
        sys.exit(1)

    try:
        with open(config_path_user, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to parse JSON: {e}")
        sys.exit(1)

    output_info = config_data.get("output", {})
    io_engine = output_info.get("engine", "HDF5")
    out_dir_raw = output_info.get("output_dir", "")
     
    prof_file = config_data.get("initial_conditions", {}).get("source_file", "")
    spat_file = config_data.get("netcdf_reader", {}).get("source_file", "")

    if not out_dir_raw:
        print("[Error] output.output_dir missing in JSON.")
        sys.exit(1)

    if io_engine == "SST" and args.io == 0:
        print("[Error] SST engine requires at least 1 IO task (--io).")
        sys.exit(1)

    out_dir_abs = os.path.abspath(out_dir_raw)
    prof_path = os.path.abspath(prof_file) if prof_file else ""
    spat_path = os.path.abspath(spat_file) if spat_file else ""

    os.makedirs(out_dir_abs, exist_ok=True)
     
    log_out_dir = os.path.dirname(os.path.abspath(args.out))
    if log_out_dir: os.makedirs(log_out_dir, exist_ok=True)
    log_err_dir = os.path.dirname(os.path.abspath(args.err))
    if log_err_dir: os.makedirs(log_err_dir, exist_ok=True)

    snapshot_dir = os.path.join(out_dir_abs, "code_snapshot")
    create_code_snapshot(VVM_ROOT, snapshot_dir, config_path_user, prof_path, spat_path, out_dir_raw)

    total_tasks = args.compute + args.io
     
    env["VVM_CONFIG_FILE"] = config_path_user
    env["VVM_COMPUTE_TASKS"] = str(args.compute)
    env["VVM_IO_TASKS"] = str(args.io)
    env["VVM_TOTAL_TASKS"] = str(total_tasks)
    env["VVM_IO_ENGINE"] = io_engine
    env["VVM_OUTPUT_DIR"] = out_dir_abs
    env["OMP_NUM_THREADS"] = str(args.cpus)

    script_path = os.path.join(VVM_ROOT, "tools", "core_run.sh")
    if not os.path.isfile(script_path):
        print(f"[Error] Core script missing at {script_path}.")
        sys.exit(1)

    if args.local:
        print("\n=========================================")
        print(" Executing: LOCAL MODE (Bypassing SLURM)")
        print(f" Preset : {args.preset}")
        print(f" Compute: {args.compute} | IO: {args.io} | Total MPI: {total_tasks}")
        print("=========================================")
        cmd = ["bash", script_path]
    else:
        print("\n=========================================")
        print(f" Executing: SLURM SUBMISSION")
        print(f" Job: {args.job_name} | Nodes: {args.nodes}")
        print(f" Preset : {args.preset}")
        print(f" Tasks: {total_tasks} | GPUs/node: {args.gpus} | CPUs/task: {args.cpus}")
        print("=========================================")
        tasks_per_node = math.ceil(total_tasks / args.nodes)
        cmd = [
            "sbatch",
            f"--job-name={args.job_name}",
            f"--nodes={args.nodes}",
            f"--ntasks={total_tasks}",
            f"--ntasks-per-node={tasks_per_node}",
            f"--gpus-per-node={args.gpus}",
            f"--cpus-per-task={args.cpus}",
            f"--time={args.time}",
            f"--output={os.path.abspath(args.out)}",
            f"--error={os.path.abspath(args.err)}"
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
