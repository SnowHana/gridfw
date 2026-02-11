import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --- IMPORT MODULES ---
from grad_fw.benchmarks import run_experiment, find_critical_k
from grad_fw.data_loader import DatasetLoader

# Configuration to match test_critical_k_time
NUM_POINTS = 30
NUM_REPEATS = 3

def dummy_logger(**kwargs):
    """
    Dummy logger to satisfy find_critical_k's requirement.
    We don't need intermediate binary search logs for this script.
    """
    pass

def main():
    # Usage: python experiment/critical_k_benchmark.py [DATASET_NAME] [OUTPUT_FILENAME]
    if len(sys.argv) < 3:
        print("Usage: python experiment/critical_k_benchmark.py [DATASET_NAME] [OUTPUT_FILENAME] [TARGET_SPEEDUP]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_filename = sys.argv[2]
    target = float(sys.argv[3])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    loader = DatasetLoader()

    # 1. Load Data
    print(f"Loading {dataset_name}...")
    try:
        # data_loader.load returns (A, X_norm) or similar, we just need A
        A_full, _ = loader.load(dataset_name)
        if A_full is None:
            raise ValueError("Dataset load returned None")
        
        p_full = A_full.shape[0]
        # Capitalize for consistency with existing logs if needed, but using arg is fine
        name = dataset_name.capitalize() 
        print(f"Loaded {name} with p_full={p_full}")

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        sys.exit(1)

    # 2. Define Param Space (replicating test_critical_k_time)
    if p_full < 50:
        print(f"Warning: p_full={p_full} is small. Adjusting range.")
        p_values = np.linspace(max(2, int(p_full * 0.01)), p_full, num=min(NUM_POINTS, p_full // 2), dtype=int)
    else:
        p_values = np.linspace(25, p_full, num=NUM_POINTS, dtype=int)
    
    p_values = np.unique(p_values)
    print(f"Targeting p values: {p_values}")

    results_list = []

    # 3. Execution Loop
    # Run test 5 times (replicating @pytest.mark.parametrize("run_id", ... range(5)))
    for run_id in range(NUM_REPEATS):
        print(f"--- Run ID: {run_id} / {NUM_REPEATS - 1} ---")
        
        for p in p_values:
            # Sub-matrix
            indices = np.random.choice(p_full, p, replace=False)
            A_sub = A_full[np.ix_(indices, indices)]

            # Call find_critical_k
            # max_run=10, isTime=True (for time benchmark)
            res = find_critical_k(
                A_sub=A_sub, 
                name_p=name, 
                logger=dummy_logger, 
                max_run=10, 
                isTime=True,
                samples=50,
                target=target
            )

            best_k = res.get("k", -1)
            
            # Replicating critical_k_final_logger fields
            # headers = ["Timestamp", "Dataset", "p", "Final_Critical_k", "Speedup_At_k", "Ratio"]
            entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Dataset": name,
                "p": p,
                "Final_Critical_k": best_k,
                "Speedup_At_k": res.get("speedupx"), # Fixed key
                "Ratio": res.get("ratio"),
                "Steps": res.get("steps"),
                "Samples": res.get("samples"),
                "Run_ID": run_id # Added for tracking
            }
            
            results_list.append(entry)
            print(f"  > p={p}: k={best_k}, Speedup={entry['Speedup_At_k']:.4f}, Ratio={entry['Ratio']:.4f}, Steps={entry['Steps']}, Samples={entry['Samples']}")

    # 4. Save Results
    df = pd.DataFrame(results_list)
    
    # Check if file exists to append or write new
    # The original logger appends. Here we might want to overwrite or append.
    # The prompt asked for "similar to experiment/compare_k.py", which writes a NEW file.
    # But usually benchmarks might want to append. 
    # Let's write to the specified output file. OLD content in that file will be overwritten 
    # if we use 'w' mode (to_csv default).
    # If the user wants to append they can handle merging, or we can use mode='a'.
    # Given compare_k.py does `df.to_csv(output_filename, index=False)`, I will do the same.
    
    df.to_csv(output_filename, index=False)
    print(f"Saved results to: {output_filename}")

if __name__ == "__main__":
    main()
