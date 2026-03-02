#!/usr/bin/env python3
"""
Run all 21 evaluation experiments in parallel across 3 GPUs.
Runs 3 commands at a time (one per GPU), waits for all to finish before starting next batch.
Stops immediately if any command fails.
"""
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Change to repo root
repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

# Results directory
RESULTS_DIR = repo_root / "notebooks" / "outputs" / "eval_results"

# Backend name mapping (matches the code)
BACKEND_MAP = {
    "exact_k": "exact_k",
    "cagra": "cuvs_cagra",
    "cagra_only": "cuvs_cagra_only",
    "ivf": "ivf",
    "ivf_only": "ivf_only"
}

def get_expected_filename(cmd_info):
    """Get the expected output filename for a command"""
    desc = cmd_info["desc"]
    cmd = cmd_info["cmd"]
    
    # Checkpoint shootout experiments
    if any("ckpt_shootout_comprehensive" in str(arg) for arg in cmd):
        try:
            dataset_idx = cmd.index("--dataset") + 1
            size_idx = cmd.index("--size") + 1
            backend_idx = cmd.index("--train_backend") + 1
            
            dataset = cmd[dataset_idx]
            size = cmd[size_idx]
            backend = cmd[backend_idx]
            backend_name = BACKEND_MAP.get(backend, backend)
            
            return f"{dataset}-{size}_{backend_name}_up_to_e3_comprehensive_results.json"
        except (ValueError, IndexError) as e:
            print(f"Error parsing checkpoint shootout command: {e}")
            return None
    
    # Cross-dataset experiments
    elif any("dataset_shootout" in str(arg) for arg in cmd):
        try:
            train_ds_idx = cmd.index("--dataset_trained") + 1
            test_ds_idx = cmd.index("--dataset_test") + 1
            train_size_idx = cmd.index("--size_trained") + 1
            test_size_idx = cmd.index("--size_test") + 1
            backend_idx = cmd.index("--train_backend") + 1
            
            train_ds = cmd[train_ds_idx]
            test_ds = cmd[test_ds_idx]
            train_size = cmd[train_size_idx]
            test_size = cmd[test_size_idx]
            backend = cmd[backend_idx]
            backend_name = BACKEND_MAP.get(backend, backend)
            
            return f"cross-dataset_{test_ds}-{test_size}_{backend_name}_from-{train_ds}-{train_size}_ckpt_up_to_e3_comprehensive_results.json"
        except (ValueError, IndexError) as e:
            print(f"Error parsing dataset shootout command: {e}")
            return None
    
    return None

def is_experiment_complete(cmd_info):
    """Check if experiment result file already exists"""
    filename = get_expected_filename(cmd_info)
    if filename is None:
        return False
    result_path = RESULTS_DIR / filename
    return result_path.exists()

# All 21 commands
COMMANDS = [
    # Checkpoint shootout - Datacomp (8.2m) - 5 backends
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "datacomp", "--full_dataset", "--size", "8.2m", "--train_backend", "cagra_only",
            "--at_k", "10,50,100",
        ],
        "desc": "datacomp-8.2m_cuvs_cagra_only"
    },
    {
        "gpu": 1,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "datacomp", "--full_dataset", "--size", "8.2m", "--train_backend", "cagra",
            "--at_k", "10,50,100",
        ],
        "desc": "datacomp-8.2m_cuvs_cagra"
    },
    {
        "gpu": 2,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "datacomp", "--full_dataset", "--size", "8.2m", "--train_backend", "exact_k",
            "--at_k", "10,50,100",
        ],
        "desc": "datacomp-8.2m_exact_k"
    },
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "datacomp", "--full_dataset", "--size", "8.2m", "--train_backend", "ivf_only",
            "--at_k", "10,50,100",
        ],
        "desc": "datacomp-8.2m_ivf_only"
    },
    {
        "gpu": 1,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "datacomp", "--full_dataset", "--size", "8.2m", "--train_backend", "ivf",
            "--at_k", "10,50,100",
        ],
        "desc": "datacomp-8.2m_ivf"
    },
    # Checkpoint shootout - Laion (10m) - 5 backends
    {
        "gpu": 2,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "laion", "--full_dataset", "--size", "10m", "--train_backend", "cagra_only",
            "--at_k", "10,50,100",
        ],
        "desc": "laion-10m_cuvs_cagra_only"
    },
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "laion", "--full_dataset", "--size", "10m", "--train_backend", "cagra",
            "--at_k", "10,50,100",
        ],
        "desc": "laion-10m_cuvs_cagra"
    },
    {
        "gpu": 1,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "laion", "--full_dataset", "--size", "10m", "--train_backend", "exact_k",
            "--at_k", "10,50,100",
        ],
        "desc": "laion-10m_exact_k"
    },
    {
        "gpu": 2,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "laion", "--full_dataset", "--size", "10m", "--train_backend", "ivf_only",
            "--at_k", "10,50,100",
        ],
        "desc": "laion-10m_ivf_only"
    },
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "laion", "--full_dataset", "--size", "10m", "--train_backend", "ivf",
            "--at_k", "10,50,100",
        ],
        "desc": "laion-10m_ivf"
    },
    # Checkpoint shootout - T2I (10m) - 5 backends
    # T2I experiments are very memory-intensive, assign them to separate GPUs
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "t2i", "--full_dataset", "--size", "10m", "--train_backend", "cagra_only",
            "--at_k", "10,50,100",
        ],
        "desc": "t2i-10m_cagra_only"
    },
    {
        "gpu": 1,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "t2i", "--full_dataset", "--size", "10m", "--train_backend", "cagra",
            "--at_k", "10,50,100",
        ],
        "desc": "t2i-10m_cuvs_cagra"
    },
    {
        "gpu": 2,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "t2i", "--full_dataset", "--size", "10m", "--train_backend", "exact_k",
            "--at_k", "10,50,100",
        ],
        "desc": "t2i-10m_exact_k"
    },
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "t2i", "--full_dataset", "--size", "10m", "--train_backend", "ivf_only",
            "--at_k", "10,50,100",
        ],
        "desc": "t2i-10m_ivf_only"
    },
    {
        "gpu": 1,
        "cmd": [
            "python", "scripts/ckpt_shootout_comprehensive.py",
            "--dataset", "t2i", "--full_dataset", "--size", "10m", "--train_backend", "ivf",
            "--at_k", "10,50,100",
        ],
        "desc": "t2i-10m_ivf"
    },
    # Cross-dataset - Datacomp-8.2m from Laion-10m
    # These are also large, so assign to GPU 2 to avoid conflicts with t2i
    {
        "gpu": 2,
        "cmd": [
            "python", "scripts/dataset_shootout.py",
            "--dataset_trained", "laion",
            "--dataset_test", "datacomp",
            "--train_backend", "cagra",
            "--size_trained", "10m",
            "--size_test", "8.2m",
            "--pca_regime", "ckpt",
            "--shared_pca",
            "--full_dataset"
        ],
        "desc": "cross-dataset_datacomp-8.2m_cuvs_cagra_from-laion-10m"
    },
    {
        "gpu": 0,
        "cmd": [
            "python", "scripts/dataset_shootout.py",
            "--dataset_trained", "laion",
            "--dataset_test", "datacomp",
            "--train_backend", "ivf",
            "--size_trained", "10m",
            "--size_test", "8.2m",
            "--pca_regime", "ckpt",
            "--shared_pca",
            "--full_dataset"
        ],
        "desc": "cross-dataset_datacomp-8.2m_ivf_from-laion-10m"
    },
    # Cross-dataset - Laion-10m from Datacomp-8.2m
    {
        "gpu": 1,
        "cmd": [
            "python", "scripts/dataset_shootout.py",
            "--dataset_trained", "datacomp",
            "--dataset_test", "laion",
            "--train_backend", "cagra",
            "--size_trained", "8.2m",
            "--size_test", "10m",
            "--pca_regime", "ckpt",
            "--shared_pca",
            "--full_dataset"
        ],
        "desc": "cross-dataset_laion-10m_cuvs_cagra_from-datacomp-8.2m"
    },
    {
        "gpu": 2,
        "cmd": [
            "python", "scripts/dataset_shootout.py",
            "--dataset_trained", "datacomp",
            "--dataset_test", "laion",
            "--train_backend", "ivf",
            "--size_trained", "8.2m",
            "--size_test", "10m",
            "--pca_regime", "ckpt",
            "--shared_pca",
            "--full_dataset"
        ],
        "desc": "cross-dataset_laion-10m_ivf_from-datacomp-8.2m"
    },
]


def run_command(cmd_info):
    """Run a single command and return (success, desc, error_msg)"""
    gpu = cmd_info["gpu"]
    cmd = cmd_info["cmd"]
    desc = cmd_info["desc"]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"[GPU {gpu}] Starting: {desc}")
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"[GPU {gpu}] ✓ Completed: {desc}")
        return (True, desc, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with exit code {e.returncode}"
        print(f"[GPU {gpu}] ✗ FAILED: {desc} - {error_msg}")
        return (False, desc, error_msg)
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        print(f"[GPU {gpu}] ✗ FAILED: {desc} - {error_msg}")
        return (False, desc, error_msg)


def main():
    # FIRST: Check existing files FIRST before doing anything else
    print("=" * 80)
    print("CHECKING EXISTING RESULT FILES")
    print("=" * 80)
    
    remaining_commands = []
    skipped = []
    
    for cmd_info in COMMANDS:
        filename = get_expected_filename(cmd_info)
        if filename:
            result_path = RESULTS_DIR / filename
            if result_path.exists():
                skipped.append((cmd_info["desc"], filename))
                continue
        # Only add to remaining if file doesn't exist
        # Remove pre-assigned GPU - will be assigned dynamically
        cmd_info_copy = cmd_info.copy()
        if "gpu" in cmd_info_copy:
            del cmd_info_copy["gpu"]
        remaining_commands.append(cmd_info_copy)
    
    if skipped:
        print(f"\n✓ SKIPPING {len(skipped)} already completed experiments:")
        for desc, filename in skipped:
            print(f"  ✓ {desc}")
            print(f"    → {filename}")
        print()
    
    if not remaining_commands:
        print("\n✅ All 21 experiments are already complete! Nothing to run.")
        sys.exit(0)
    
    print(f"\n📋 RUNNING {len(remaining_commands)} remaining experiments:")
    for cmd_info in remaining_commands:
        filename = get_expected_filename(cmd_info)
        print(f"  → {cmd_info['desc']} (will create: {filename})")
    print()
    print("=" * 80)
    
    # Dynamic GPU assignment: assign GPUs as they become available
    # This ensures no two commands run on the same GPU simultaneously
    completed = 0
    failed_commands = []
    available_gpus = [0, 1, 2]
    gpu_futures = {0: None, 1: None, 2: None}  # Track which future is running on each GPU
    command_queue = remaining_commands.copy()
    
    failure_flag = False
    last_error = None
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start initial batch (one per GPU)
        for gpu in available_gpus:
            if command_queue:
                cmd_info = command_queue.pop(0)
                cmd_info["gpu"] = gpu  # Assign GPU dynamically
                print(f"\n{'='*80}")
                print(f"Starting experiment {completed + 1}/{len(remaining_commands)} on GPU {gpu}: {cmd_info['desc']}")
                print(f"{'='*80}\n")
                future = executor.submit(run_command, cmd_info)
                gpu_futures[gpu] = (future, cmd_info)
        
        # Process as they complete and assign new ones
        while any(gpu_futures.values()) or command_queue:
            if failure_flag:
                # If a failure was seen, stop checking/assigning further
                break
            # Wait for any GPU to finish
            done_futures = []
            for gpu, gpu_info in list(gpu_futures.items()):
                if gpu_info is None:
                    continue
                future, cmd_info = gpu_info
                if future.done():
                    done_futures.append((gpu, future, cmd_info))
            
            if not done_futures:
                # Wait a bit if nothing is done yet
                import time
                time.sleep(0.1)
                continue
            
            # Process completed futures
            for gpu, future, cmd_info in done_futures:
                try:
                    success, desc, error_msg = future.result()
                    completed += 1
                    gpu_futures[gpu] = None  # Free up the GPU
                    
                    if not success:
                        failed_commands.append((desc, error_msg))
                        last_error = f"{desc}: {error_msg}"
                        print(f"\n{'!'*80}")
                        print(f"ERROR: Command failed: {desc}")
                        print(f"Error: {error_msg}")
                        print(f"{'!'*80}\n")
                        print("Stopping all remaining commands...")
                        # Cancel remaining futures
                        for gpu_id, gpu_info in gpu_futures.items():
                            if gpu_info is not None:
                                gpu_info[0].cancel()
                        # Prevent new assignments
                        command_queue.clear()
                        failure_flag = True
                        break
                    
                    # Assign next command to this GPU if available
                    if command_queue:
                        next_cmd = command_queue.pop(0)
                        next_cmd["gpu"] = gpu
                        print(f"\n{'='*80}")
                        print(f"Starting experiment {completed + 1}/{len(remaining_commands)} on GPU {gpu}: {next_cmd['desc']}")
                        print(f"{'='*80}\n")
                        next_future = executor.submit(run_command, next_cmd)
                        gpu_futures[gpu] = (next_future, next_cmd)
                except Exception as e:
                    print(f"Error processing result from GPU {gpu}: {e}")
                    gpu_futures[gpu] = None
            
            if failed_commands:
                break
            
            # Remove None entries (no active futures) but keep GPU slots
            gpu_futures = {gpu: info for gpu, info in gpu_futures.items() if info is not None or gpu in available_gpus}
    
    if failed_commands:
        # Print only the last error as the final message before exit
        if last_error is None and failed_commands:
            last_error = f"{failed_commands[-1][0]}: {failed_commands[-1][1]}"
        if last_error:
            print(f"\nFINAL ERROR: {last_error}")
        sys.exit(1)
    else:
        # Final summary on success
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Completed: {completed}/{len(remaining_commands)}")
        if skipped:
            print(f"Skipped (already done): {len(skipped)}")
        print("\n✅ All experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
