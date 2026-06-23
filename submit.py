#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

try:
    import readline
except ImportError:
    readline = None


# ==============================================================================
# Defaults
# ==============================================================================

DEFAULT_CONFIG = "rundata/input_configs/taiwanvvm.json"
DEFAULT_COMPUTE = 1
DEFAULT_IO = None                 # None means infer from output.engine
DEFAULT_NODES = 1
DEFAULT_GPUS = None               # None means infer from compute ranks per node
DEFAULT_CPUS = 1                  # CLI option only; not asked in wizard
DEFAULT_TIME = "24:00:00"
DEFAULT_OUT = "log/%j.out"
DEFAULT_ERR = "log/%j.err"
DEFAULT_JOB_NAME = "VVM_GPU_CPP"
DEFAULT_ACCOUNT = "MST114418"
DEFAULT_PARTITION = "normal"
DEFAULT_EXPORT = "ALL"
DEFAULT_EXCLUSIVE = True


# ==============================================================================
# Path / environment helpers
# ==============================================================================

def get_vvm_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_available_presets(vvm_root):
    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    if not os.path.exists(preset_file):
        return []

    try:
        with open(preset_file, "r") as f:
            presets_data = json.load(f)

        return [
            p.get("name")
            for p in presets_data.get("configurePresets", [])
            if p.get("name")
        ]
    except Exception:
        return []


def append_unique(paths, path):
    if not path:
        return

    if not os.path.isdir(path):
        return

    if path not in paths:
        paths.append(path)


def setup_environment(preset_name):
    """
    Load controlled build-time dependency paths from CMakePresets.json.

    Important:
    - Do not stage/copy libfabric.
    - Do not prepend .vvm_runtime_libs.
    - Do not manually add /usr/lib*, /lib*, etc. into LD_LIBRARY_PATH.

    Forcing broad system paths into LD_LIBRARY_PATH can mix incompatible
    MPI / UCX / UCC / libfabric / HDF5 / ADIOS2 runtime stacks.
    """
    env = os.environ.copy()
    vvm_root = get_vvm_root()
    env["VVM_ROOT"] = vvm_root

    preset_file = os.path.join(vvm_root, "CMakePresets.json")
    if not os.path.exists(preset_file):
        print(f"[Warning] {preset_file} not found. Environment may be incomplete.")
        return env

    try:
        with open(preset_file, "r") as f:
            presets_data = json.load(f)

        cache_vars = {}
        for p in presets_data.get("configurePresets", []):
            if p.get("name") == preset_name:
                cache_vars = p.get("cacheVariables", {})
                break

        if not cache_vars:
            print(f"[Warning] Preset '{preset_name}' not found in CMakePresets.json.")
            return env

        print(f"[Info] Loaded environment from CMake preset: '{preset_name}'")

        # ----------------------------------------------------------------------
        # HPCX / MPI environment metadata
        # ----------------------------------------------------------------------
        # Only set metadata and PATH here. Do not force libfabric / UCX / UCC
        # paths through submit.py. core_run.sh may source hpcx-init.sh, and the
        # runtime stack should remain internally consistent.
        # ----------------------------------------------------------------------
        cxx_compiler = cache_vars.get("CMAKE_CXX_COMPILER", "")
        if "/ompi/bin/" in cxx_compiler:
            hpcx_home = cxx_compiler.split("/ompi/bin/")[0]
            my_plugin_path = f"{hpcx_home}/nccl_rdma_sharp_plugin/lib"
            sharp_lib_path = f"{hpcx_home}/sharp/lib"

            env["HPCX_HOME"] = hpcx_home
            env["MY_PLUGIN_PATH"] = my_plugin_path
            env["SHARP_LIB_PATH"] = sharp_lib_path
            env["VVM_ENV_SCRIPT"] = f"{hpcx_home}/hpcx-init.sh"
            env["VVM_PRE_RUN_CMD"] = f"source {hpcx_home}/hpcx-init.sh"

            # Ensure mpirun from the intended HPCX tree is first.
            env["PATH"] = f"{hpcx_home}/ompi/bin:" + env.get("PATH", "")

        # ----------------------------------------------------------------------
        # Explicitly configured project dependencies
        # ----------------------------------------------------------------------
        # These paths come from the same CMake preset used to build the code.
        # Avoid adding broad system library paths here.
        # ----------------------------------------------------------------------
        lib_dirs = []

        for key in [
            "HDF5_DIR",
            "NETCDF_C_DIR",
            "NETCDF_Fortran_DIR",
            "PNETCDF_DIR",
            "ADIOS2_DIR",
        ]:
            val = cache_vars.get(key, "")
            if not val:
                continue

            append_unique(lib_dirs, os.path.join(val, "lib"))
            append_unique(lib_dirs, os.path.join(val, "lib64"))

        if lib_dirs:
            extra_ld = ":".join(lib_dirs)
            env["VVM_EXTRA_LD_LIBRARY_PATH"] = extra_ld

            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = extra_ld + (":" + old_ld if old_ld else "")

    except Exception as e:
        print(f"[Warning] Error parsing CMakePresets.json: {e}")

    return env


# ==============================================================================
# Config helpers
# ==============================================================================

def ask(prompt_text, default_val):
    ans = input(f"{prompt_text} [{default_val}]: ").strip()
    return ans if ans else default_val


def ask_path(prompt_text, default_val):
    """
    Prompt for a file/path with TAB completion.

    Works in normal Linux terminals. If readline is unavailable, falls back to
    normal input().
    """
    if readline is None:
        return ask(prompt_text, default_val)

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()

    def path_completer(text, state):
        expanded_text = os.path.expanduser(text)
        matches = glob.glob(expanded_text + "*")

        results = []
        for match in matches:
            if os.path.isdir(match):
                match += os.sep
            results.append(match)

        results = sorted(results)

        try:
            return results[state]
        except IndexError:
            return None

    try:
        readline.set_completer(path_completer)

        # Do not treat "/" as a word separator.
        readline.set_completer_delims(" \t\n")

        readline.parse_and_bind("tab: complete")

        ans = input(f"{prompt_text} [{default_val}]: ").strip()
        return ans if ans else default_val

    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def read_config(config_path):
    config_path_abs = os.path.abspath(os.path.expanduser(config_path))

    if not os.path.isfile(config_path_abs):
        raise FileNotFoundError(f"Configuration file not found: {config_path_abs}")

    with open(config_path_abs, "r") as f:
        return json.load(f), config_path_abs


def peek_io_engine(config_path):
    try:
        config_data, _ = read_config(config_path)
        return config_data.get("output", {}).get("engine", "HDF5")
    except Exception:
        return "HDF5"


def infer_io_tasks(io_engine, compute_tasks, io_tasks):
    """
    IO rank default policy:
    - If user explicitly gives --io N, use it.
    - If SST and --io is omitted, use io = compute.
    - Otherwise use io = 0.
    """
    if io_tasks is not None:
        return io_tasks

    if io_engine == "SST":
        return compute_tasks

    return 0


def infer_gpus_per_node(compute_tasks, nodes, gpus):
    """
    GPU request policy:
    - GPUs are requested for compute ranks only.
    - IO ranks do not increase the GPU request by default.

    Example:
      compute = 64, io = 64, nodes = 8
      compute_per_node = 8
      total_tasks_per_node = 16
      gpus_per_node = 8
    """
    if gpus is not None:
        return gpus

    return max(1, math.ceil(compute_tasks / nodes))


# ==============================================================================
# Snapshot
# ==============================================================================

def create_code_snapshot(repo_root, snapshot_dir, config_path, prof_path, spat_path, out_dir_raw):
    print(f"\n[Info] Creating code snapshot at: {snapshot_dir}")

    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)

    out_base = os.path.normpath(out_dir_raw).split(os.sep)[0]
    if out_base in [".", "..", ""]:
        out_base = "output"

    ignore_patterns = shutil.ignore_patterns(
        ".git",
        ".vvm_runtime_libs",
        "build",
        "log",
        "rundata",
        "tests",
        "docs",
        "externals",
        "tags",
        "*.o",
        "output",
        out_base,
    )

    shutil.copytree(repo_root, snapshot_dir, ignore=ignore_patterns)
    shutil.copy2(config_path, snapshot_dir)

    if prof_path and os.path.isfile(prof_path):
        shutil.copy2(prof_path, snapshot_dir)

    if spat_path and os.path.isfile(spat_path):
        shutil.copy2(spat_path, snapshot_dir)

    gitignore_content = (
        ".vvm_runtime_libs/\n"
        "rundata/\n"
        "tests/\n"
        "docs/\n"
        "externals/\n"
        "build/\n"
        "log/\n"
        "output/\n"
        f"{out_base}/\n"
    )

    with open(os.path.join(snapshot_dir, ".gitignore"), "w") as f:
        f.write(gitignore_content)

    try:
        subprocess.run(["git", "init", "-q"], cwd=snapshot_dir, check=True)
        subprocess.run(["git", "add", "."], cwd=snapshot_dir, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Snapshot",
                "-c",
                "user.email=snap@local",
                "commit",
                "-q",
                "-m",
                "Auto Snapshot",
            ],
            cwd=snapshot_dir,
            check=True,
        )
    except Exception as e:
        print(f"[Warning] Git snapshot commit failed; ignored: {e}")


# ==============================================================================
# Interactive wizard
# ==============================================================================

def interactive_wizard():
    vvm_root = get_vvm_root()
    presets = get_available_presets(vvm_root)
    default_preset = presets[0] if presets else "unknown"

    print("====================================================================")
    print(" GVVM Interactive Setup Wizard")
    print("====================================================================")
    print(" Note: All relative paths are based on the auto-detected $VVM_ROOT.")
    print("")
    print(" This wizard asks only the common options needed for a normal run.")
    print(" Other options are still available from the command line; see below.")
    print("--------------------------------------------------------------------")
    print(" Prompted Options")
    print("--------------------------------------------------------------------")
    print(f" --local          : Run locally without SLURM (default: False)")
    print(f" -c, --config     : Configuration file (default: {DEFAULT_CONFIG})")
    print("                    TAB completion is enabled for this path prompt.")
    print(f" --preset         : CMake preset to load environment from")
    print(f" --compute        : Compute MPI ranks (default: {DEFAULT_COMPUTE})")
    print(f" --nodes          : Number of SLURM nodes (default: {DEFAULT_NODES})")
    print(f" -t, --time       : Wall time limit (default: {DEFAULT_TIME})")
    print(f" --out            : Standard output log file (default: {DEFAULT_OUT})")
    print(f" --err            : Standard error log file (default: {DEFAULT_ERR})")
    print(f" --job-name       : SLURM job name (default: {DEFAULT_JOB_NAME})")
    print(f" -A, --account    : SLURM account (default: {DEFAULT_ACCOUNT})")
    print(f" -p, --partition  : SLURM partition (default: {DEFAULT_PARTITION})")
    print("--------------------------------------------------------------------")
    print(" Derived Defaults")
    print("--------------------------------------------------------------------")
    print(f" --cpus           : CPUs per task / OMP_NUM_THREADS (default: {DEFAULT_CPUS})")
    print("                    This is not asked in the wizard. Override with --cpus N.")
    print("")
    print(" --io             : IO MPI ranks")
    print("                    If output.engine == SST, default is --io = --compute.")
    print("                    Otherwise, default is --io = 0.")
    print("                    This is not asked in the wizard. Override with --io N.")
    print("")
    print(" --gpus           : GPUs per node")
    print("                    Default is ceil(compute ranks / nodes).")
    print("                    This is not asked in the wizard. Override with --gpus N.")
    print("                    This is not based on compute + IO ranks.")
    print("                    IO ranks do not increase the GPU request by default.")
    print("")
    print(" --ntasks         : compute ranks + IO ranks")
    print(" --ntasks-per-node: ceil((compute ranks + IO ranks) / nodes)")
    print(" --gpus-per-node  : ceil(compute ranks / nodes), unless --gpus is given")
    print("--------------------------------------------------------------------")
    print(" Advanced SLURM Options: command-line only")
    print("--------------------------------------------------------------------")
    print(f" --export SPEC    : SLURM export option (default: {DEFAULT_EXPORT})")
    print(" --exclusive      : Request exclusive nodes. This is the default.")
    print(" --no-exclusive   : Do not request exclusive nodes.")
    print(" --exclude LIST   : Exclude specific nodes.")
    print("                    Example: --exclude 25a-hgpn062")
    print(" --nodelist LIST  : Request specific nodes.")
    print("                    Example: --nodelist 25a-hgpn001,25a-hgpn002")
    print(" --contiguous     : Request contiguous nodes.")
    print(" --slurm-arg ARG  : Append raw sbatch argument. Can be repeated.")
    print("                    Example: --slurm-arg '--qos=debug'")
    print("====================================================================")

    class Args:
        pass

    args = Args()

    args.local = ask("Run locally without SLURM? (y/N)", "N").upper() == "Y"
    args.config = ask_path("Configuration file", DEFAULT_CONFIG)

    if presets:
        print("\nAvailable CMake presets:")
        for i, p in enumerate(presets):
            print(f" [{i + 1}] {p}")

        preset_choice = ask(f"Select preset (1-{len(presets)} or type name)", "1")
        if preset_choice.isdigit() and 1 <= int(preset_choice) <= len(presets):
            args.preset = presets[int(preset_choice) - 1]
        else:
            args.preset = preset_choice
    else:
        args.preset = ask("CMake preset environment", default_preset)

    args.compute = int(ask("\nCompute tasks / MPI ranks", DEFAULT_COMPUTE))

    # Not prompted in wizard.
    args.io = DEFAULT_IO
    args.cpus = DEFAULT_CPUS
    args.gpus = DEFAULT_GPUS

    if not args.local:
        args.nodes = int(ask("\nNumber of nodes", DEFAULT_NODES))
    else:
        args.nodes = 1

    io_engine = peek_io_engine(args.config)
    inferred_io = infer_io_tasks(io_engine, args.compute, args.io)
    inferred_gpus = infer_gpus_per_node(args.compute, args.nodes, args.gpus)

    print("")
    print(f"[Info] Detected output engine: {io_engine}")
    print(f"[Info] IO ranks default: {inferred_io}")
    print(f"[Info] CPUs per task default: {args.cpus}")
    print(f"[Info] GPUs per node default: {inferred_gpus}")
    print("       This is derived from compute ranks per node.")
    print("       IO ranks do not increase the GPU request by default.")
    print("       Override with --gpus N if needed.")

    if not args.local:
        args.time = ask("\nTime limit", DEFAULT_TIME)
        args.job_name = ask("Job name", DEFAULT_JOB_NAME)
        args.account = ask("SLURM account", DEFAULT_ACCOUNT)
        args.partition = ask("SLURM partition", DEFAULT_PARTITION)
        args.out = ask("Standard output log", DEFAULT_OUT)
        args.err = ask("Standard error log", DEFAULT_ERR)
    else:
        args.time = None
        args.job_name = DEFAULT_JOB_NAME
        args.account = None
        args.partition = None
        args.out = DEFAULT_OUT
        args.err = DEFAULT_ERR

    # Advanced command-line-only options.
    args.exclude = None
    args.nodelist = None
    args.contiguous = False
    args.export = DEFAULT_EXPORT
    args.exclusive = DEFAULT_EXCLUSIVE
    args.slurm_arg = []

    cmd_parts = [sys.argv[0]]

    if args.local:
        cmd_parts.append("--local")

    cmd_parts.append(f'-c "{args.config}"')
    cmd_parts.append(f'--preset "{args.preset}"')
    cmd_parts.append(f"--compute {args.compute}")
    cmd_parts.append(f"--nodes {args.nodes}")

    if not args.local:
        cmd_parts.append(f'-t "{args.time}"')
        cmd_parts.append(f'--out "{args.out}"')
        cmd_parts.append(f'--err "{args.err}"')
        cmd_parts.append(f'--job-name "{args.job_name}"')
        cmd_parts.append(f'-A "{args.account}"')
        cmd_parts.append(f'-p "{args.partition}"')

    print("\n--- Setup Complete ---")
    print("\nEquivalent command:\n")
    print(" " + " ".join(cmd_parts))
    print("\nUse --help to see advanced command-line-only options.\n")

    return args


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VVM GPU C++ job submission wrapper"
    )

    parser.add_argument("-c", "--config", help="Path to JSON configuration file")
    parser.add_argument("--preset", type=str, help="CMake preset name to load environment from")
    parser.add_argument("--local", action="store_true", help="Run locally without SLURM")

    parser.add_argument("--compute", type=int, default=DEFAULT_COMPUTE, help="Compute MPI ranks")
    parser.add_argument(
        "--io",
        type=int,
        default=DEFAULT_IO,
        help="IO MPI ranks. Default: same as compute for SST; 0 otherwise.",
    )
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES, help="Number of nodes")
    parser.add_argument(
        "--gpus",
        type=int,
        default=DEFAULT_GPUS,
        help="GPUs per node. Default: ceil(compute ranks / nodes).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=DEFAULT_CPUS,
        help="CPUs per task / OMP_NUM_THREADS. Default: 1.",
    )

    parser.add_argument("-t", "--time", type=str, default=DEFAULT_TIME, help="Wall time limit")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Standard output log")
    parser.add_argument("--err", type=str, default=DEFAULT_ERR, help="Standard error log")
    parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME, help="SLURM job name")
    parser.add_argument("-A", "--account", type=str, default=DEFAULT_ACCOUNT, help="SLURM account")
    parser.add_argument("-p", "--partition", type=str, default=DEFAULT_PARTITION, help="SLURM partition")

    # Advanced SLURM options. CLI only; not prompted in wizard.
    parser.add_argument("--exclude", type=str, default=None, help="SLURM --exclude node list")
    parser.add_argument("--nodelist", type=str, default=None, help="SLURM --nodelist node list")
    parser.add_argument("--contiguous", action="store_true", help="Request contiguous SLURM nodes")
    parser.add_argument(
        "--export",
        type=str,
        default=DEFAULT_EXPORT,
        help="SLURM --export value. Default: ALL.",
    )
    parser.add_argument(
        "--exclusive",
        dest="exclusive",
        action="store_true",
        help="Request exclusive node allocation. Default.",
    )
    parser.add_argument(
        "--no-exclusive",
        dest="exclusive",
        action="store_false",
        help="Do not request exclusive node allocation.",
    )
    parser.set_defaults(exclusive=DEFAULT_EXCLUSIVE)

    parser.add_argument(
        "--slurm-arg",
        action="append",
        default=[],
        help="Append a raw sbatch argument. Repeatable. Example: --slurm-arg='--qos=debug'",
    )

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


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    env = setup_environment(args.preset)

    vvm_root = env.get("VVM_ROOT")
    if not vvm_root:
        print("[Error] Failed to detect VVM_ROOT.")
        sys.exit(1)

    os.chdir(vvm_root)

    try:
        config_data, config_path_user = read_config(args.config)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    output_info = config_data.get("output", {})
    io_engine = output_info.get("engine", "HDF5")
    out_dir_raw = output_info.get("output_dir", "")

    if not out_dir_raw:
        print("[Error] output.output_dir missing in JSON.")
        sys.exit(1)

    if args.nodes <= 0:
        print("[Error] --nodes must be positive.")
        sys.exit(1)

    if args.compute <= 0:
        print("[Error] --compute must be positive.")
        sys.exit(1)

    if args.io is not None and args.io < 0:
        print("[Error] --io cannot be negative.")
        sys.exit(1)

    if args.cpus <= 0:
        print("[Error] --cpus must be positive.")
        sys.exit(1)

    # Derived defaults.
    args.io = infer_io_tasks(io_engine, args.compute, args.io)
    args.gpus = infer_gpus_per_node(args.compute, args.nodes, args.gpus)

    if args.gpus <= 0:
        print("[Error] --gpus must be positive.")
        sys.exit(1)

    total_tasks = args.compute + args.io

    if io_engine == "SST" and args.io == 0:
        print("[Error] SST engine requires IO ranks. Use --io N or omit --io to default to compute ranks.")
        sys.exit(1)

    prof_file = config_data.get("initial_conditions", {}).get("source_file", "")
    spat_file = config_data.get("netcdf_reader", {}).get("source_file", "")

    out_dir_abs = os.path.abspath(out_dir_raw)
    prof_path = os.path.abspath(prof_file) if prof_file else ""
    spat_path = os.path.abspath(spat_file) if spat_file else ""

    os.makedirs(out_dir_abs, exist_ok=True)

    if args.out:
        out_log_dir = os.path.dirname(os.path.abspath(args.out))
        if out_log_dir:
            os.makedirs(out_log_dir, exist_ok=True)

    if args.err:
        err_log_dir = os.path.dirname(os.path.abspath(args.err))
        if err_log_dir:
            os.makedirs(err_log_dir, exist_ok=True)

    snapshot_dir = os.path.join(out_dir_abs, "code_snapshot")
    create_code_snapshot(
        vvm_root,
        snapshot_dir,
        config_path_user,
        prof_path,
        spat_path,
        out_dir_raw,
    )

    compute_per_node = math.ceil(args.compute / args.nodes)
    io_per_node = math.ceil(args.io / args.nodes) if args.io > 0 else 0

    # Important:
    # - total tasks include compute + IO ranks
    # - GPU request is based only on compute ranks
    tasks_per_node = math.ceil(total_tasks / args.nodes)

    env["VVM_CONFIG_FILE"] = config_path_user
    env["VVM_COMPUTE_TASKS"] = str(args.compute)
    env["VVM_IO_TASKS"] = str(args.io)
    env["VVM_TOTAL_TASKS"] = str(total_tasks)
    env["VVM_COMPUTE_PER_NODE"] = str(compute_per_node)
    env["VVM_IO_PER_NODE"] = str(io_per_node)
    env["VVM_IO_ENGINE"] = io_engine
    env["VVM_OUTPUT_DIR"] = out_dir_abs
    env["OMP_NUM_THREADS"] = str(args.cpus)
    env["VVM_GPUS"] = str(args.gpus)

    script_path = os.path.join(vvm_root, "tools", "core_run.sh")
    if not os.path.isfile(script_path):
        print(f"[Error] Core script missing at {script_path}.")
        sys.exit(1)

    print("\n=========================================")
    print(" VVM SUBMISSION SUMMARY")
    print("=========================================")
    print(f" Mode              : {'LOCAL' if args.local else 'SLURM'}")
    print(f" Config            : {config_path_user}")
    print(f" Preset            : {args.preset}")
    print(f" Output engine     : {io_engine}")
    print(f" Compute ranks     : {args.compute}")
    print(f" IO ranks          : {args.io}")
    print(f" Total ranks       : {total_tasks}")
    print(f" Nodes             : {args.nodes}")
    print(f" Compute/node      : {compute_per_node}")
    print(f" IO/node           : {io_per_node}")
    print(f" Total tasks/node  : {tasks_per_node}")
    print(f" GPUs/node         : {args.gpus}")
    print(f" CPUs/task         : {args.cpus}")
    if not args.local:
        print(f" Exclusive         : {args.exclusive}")
        print(f" Export            : {args.export}")
        if args.exclude:
            print(f" Exclude           : {args.exclude}")
        if args.nodelist:
            print(f" Nodelist          : {args.nodelist}")
        if args.contiguous:
            print(" Contiguous        : True")
        if args.slurm_arg:
            print(f" Extra sbatch args : {' '.join(args.slurm_arg)}")
    print("=========================================\n")

    if args.local:
        cmd = ["bash", script_path]

    else:
        cmd = [
            "sbatch",
            f"--job-name={args.job_name}",
            f"--nodes={args.nodes}",
            f"--ntasks={total_tasks}",
            f"--ntasks-per-node={tasks_per_node}",
            f"--gpus-per-node={args.gpus}",
            "--gpu-bind=none",
            f"--cpus-per-task={args.cpus}",
            f"--export={args.export}",
            f"--time={args.time}",
            f"--output={os.path.abspath(args.out)}",
            f"--error={os.path.abspath(args.err)}",
        ]

        if args.exclusive:
            cmd.append("--exclusive")

        if args.account:
            cmd.append(f"--account={args.account}")

        if args.partition:
            cmd.append(f"--partition={args.partition}")

        if args.exclude:
            cmd.append(f"--exclude={args.exclude}")

        if args.nodelist:
            cmd.append(f"--nodelist={args.nodelist}")

        if args.contiguous:
            cmd.append("--contiguous")

        for extra_arg in args.slurm_arg:
            if extra_arg:
                cmd.append(extra_arg)

        cmd.append(script_path)

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Process failed with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
