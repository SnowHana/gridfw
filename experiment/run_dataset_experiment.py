import sys
import os
import subprocess
import time
import argparse

# Ensure src is in python path to import grad_fw
sys.path.append(os.path.join(os.getcwd(), "src"))

from grad_fw.data_loader import ALL_DATASETS

def main():
    parser = argparse.ArgumentParser(description="Run critical k benchmarks for datasets.")
    parser.add_argument(
        "dataset", 
        type=str, 
        help="Name of the dataset to run, or 'all' to run all datasets."
    )
    args = parser.parse_args()

    # Define directories
    PROJECT_ROOT = os.getcwd()
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    EXPERIMENT_SCRIPT = os.path.join(PROJECT_ROOT, "experiment", "critical_k_benchmark.py")

    # Determine which datasets to run
    target_datasets = []
    available_datasets = list(ALL_DATASETS.keys())

    if args.dataset.lower() == "all":
        target_datasets = available_datasets
    elif args.dataset.lower() in [d.lower() for d in available_datasets]:
        # Match case-insensitive but use exact key from dictionary
        for d in available_datasets:
            if d.lower() == args.dataset.lower():
                target_datasets = [d]
                break
    else:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {available_datasets}")
        sys.exit(1)

    print(f"--- Starting Benchmark Runner ---")
    print(f"Target Datasets: {target_datasets}")
    print(f"Output directory: {LOG_DIR}")
    print("-" * 30)

    for dataset_name in target_datasets:
        print(f"\n[Running Benchmark] Dataset: {dataset_name}")
        
        output_filename = os.path.join(LOG_DIR, f"critical_k_{dataset_name}.csv")
        
        # Construct command
        # Assumes running inside the python environment where this script is called
        cmd = [sys.executable, EXPERIMENT_SCRIPT, dataset_name, output_filename]
        
        start_time = time.time()
        try:
            # Run the benchmark script as a subprocess
            # capturing output so we don't flood the console, but printing errors
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            elapsed = time.time() - start_time
            
            print(f"  > Success! Time taken: {elapsed:.2f}s")
            print(f"  > Output written to: {output_filename}")
            
            # Print last few non-empty lines of stdout so user sees progress
            output_lines = [l for l in result.stdout.splitlines() if l.strip()]
            if output_lines:
                print(f"  > Last output: {output_lines[-1]}")

        except subprocess.CalledProcessError as e:
            print(f"  > FAILURE: Dataset {dataset_name} failed.")
            print(f"  > Error output:\n{e.stderr}")
            # Optionally print stdout too if needed for debugging
            if e.stdout:
                print(f"  > Standard output:\n{e.stdout}")
        except Exception as e:
            print(f"  > EXCEPTION: {e}")

    print("\n" + "=" * 30)
    print("Benchmark run completed.")

if __name__ == "__main__":
    main()
