#!/usr/bin/env python3
"""
Run all 6 QPS-focused evaluation experiments in parallel across 3 GPUs.
Dynamically assigns GPUs as they become available to avoid conflicts.
Checks existing results and skips completed experiments.
"""
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Change to repo root
repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)

# Results directory
RESULTS_DIR = repo_root / "notebooks" / "outputs" / "qps_results"

# Backend name mapping
BACKEND_MAP = {
    "cagra": "cuvs_cagra",
    "ivf": "ivf"
}

# All 6 QPS experiments
COMMANDS = [
    # IVF experiments
    {
        "cmd": [
            "python", "scripts/qps_shootout_comprehensive.py",
            "--dataset", "laion",
            "--train_backend", "ivf",
            "--size", "10m"
        ],
        "desc": "laion-10m_ivf"
    },
    {
        "cmd": [
            "python", "scripts/qps_shootout_comprehensive.py",
            "--dataset", "t2i",
            "--train_backend", "ivf",
            "--size", "10m"
        ],
        "desc": "t2i-10m_ivf"
    },
    {
        "cmd": [
            "python", "scripts/qps_shootout_comprehensive.py",
            "--dataset", "datacomp",
            "--train_backend", "ivf",
            "--size", "8.2m"
        ],
        "desc": "datacomp-8.2m_ivf"
    },
    # CAGRA experiments
    {
        "cmd": [
            "python", "scripts/qps_shootout_comprehensive.py",
            "--dataset", "laion",
            "--train_backend", "cagra",
            "--size", "10m"
        ],
        "desc": "laion-10m_cuvs_cagra"
    },
    {
        "cmd": [
            "python", "scripts/qps_shootout_comprehensive.py",
            "--dataset", "t2i",
            "--train_backend", "cagra",
            "--size", "10m"
        ],
        "desc": "t2i-10m_cuvs_cagra"
    },
    {
        "cmd": [
            "python", "scripts/qps_shootout_comprehensive.py",
            "--dataset", "datacomp",
            "--train_backend", "cagra",
            "--size", "8.2m"
        ],
        "desc": "datacomp-8.2m_cuvs_cagra"
    },
]


def get_expected_filename(cmd_info):
    """Get the expected output filename for a command"""
    desc = cmd_info["desc"]
    cmd = cmd_info["cmd"]
    
    if any("qps_shootout_comprehensive" in str(arg) for arg in cmd):
        try:
            dataset_idx = cmd.index("--dataset") + 1
            size_idx = cmd.index("--size") + 1
            backend_idx = cmd.index("--train_backend") + 1
            
            dataset = cmd[dataset_idx]
            size = cmd[size_idx]
            backend = cmd[backend_idx]
            backend_name = BACKEND_MAP.get(backend, backend)
            
            return f"{dataset}-{size}_{backend_name}_qps_results.json"
        except (ValueError, IndexError) as e:
            print(f"Error parsing QPS command: {e}")
            return None
    
    return None


def run_command(cmd_info, gpu):
    """Run a single command on specified GPU and return (success, desc, error_msg)"""
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
    print("CHECKING EXISTING QPS RESULT FILES")
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
        remaining_commands.append(cmd_info)
    
    if skipped:
        print(f"\n✓ SKIPPING {len(skipped)} already completed experiments:")
        for desc, filename in skipped:
            print(f"  ✓ {desc}")
            print(f"    → {filename}")
        print()
    
    if not remaining_commands:
        print("\n✅ All 6 QPS experiments are already complete! Nothing to run.")
        sys.exit(0)
    
    print(f"\n📋 RUNNING {len(remaining_commands)} remaining experiments:")
    for cmd_info in remaining_commands:
        filename = get_expected_filename(cmd_info)
        print(f"  → {cmd_info['desc']} (will create: {filename})")
    print()
    print("=" * 80)
    
    # Dynamic GPU assignment: assign GPUs as they become available
    completed = 0
    failed_commands = []
    available_gpus = [0, 1, 2]
    gpu_futures = {0: None, 1: None, 2: None}  # Track which future is running on each GPU
    command_queue = remaining_commands.copy()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start initial batch (one per GPU)
        for gpu in available_gpus:
            if command_queue:
                cmd_info = command_queue.pop(0)
                print(f"\n{'='*80}")
                print(f"Starting experiment {completed + 1}/{len(remaining_commands)} on GPU {gpu}: {cmd_info['desc']}")
                print(f"{'='*80}\n")
                future = executor.submit(run_command, cmd_info, gpu)
                gpu_futures[gpu] = (future, cmd_info)
        
        # Process as they complete and assign new ones
        while any(gpu_futures.values()) or command_queue:
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
                        print(f"\n{'!'*80}")
                        print(f"ERROR: Command failed: {desc}")
                        print(f"Error: {error_msg}")
                        print(f"{'!'*80}\n")
                        print("Stopping all remaining commands...")
                        # Cancel remaining futures
                        for gpu_id, gpu_info in gpu_futures.items():
                            if gpu_info is not None:
                                gpu_info[0].cancel()
                        break
                    
                    # Assign next command to this GPU if available
                    if command_queue:
                        next_cmd = command_queue.pop(0)
                        print(f"\n{'='*80}")
                        print(f"Starting experiment {completed + 1}/{len(remaining_commands)} on GPU {gpu}: {next_cmd['desc']}")
                        print(f"{'='*80}\n")
                        next_future = executor.submit(run_command, next_cmd, gpu)
                        gpu_futures[gpu] = (next_future, next_cmd)
                except Exception as e:
                    print(f"Error processing result from GPU {gpu}: {e}")
                    gpu_futures[gpu] = None
            
            if failed_commands:
                break
            
            # Remove None entries but keep GPU slots
            gpu_futures = {gpu: info for gpu, info in gpu_futures.items() if info is not None or gpu in available_gpus}
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{len(remaining_commands)}")
    if skipped:
        print(f"Skipped (already done): {len(skipped)}")
    
    if failed_commands:
        print(f"\n❌ FAILED COMMANDS ({len(failed_commands)}):")
        for desc, error_msg in failed_commands:
            print(f"  - {desc}: {error_msg}")
        print("\nPlease check the errors above and fix the issues.")
        sys.exit(1)
    else:
        print("\n✅ All QPS experiments completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
