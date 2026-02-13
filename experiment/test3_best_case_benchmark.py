import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from grad_fw.data_loader import DatasetLoader
from grad_fw.benchmarks.benchmarks import run_experiment

# Configuration to match test_critical_k_time
NUM_POINTS = 5
NUM_REPEATS = 3
MIN_P = 450


def get_adaptive_steps(k):
    return int(min(20 * k, 1000))


def main():
    """main Run ideal case scenario.

    Raises:
        ValueError: _description_
    """
    # Usage: python experiment/critical_k_benchmark.py [DATASET_NAME] [OUTPUT_FILENAME]
    if len(sys.argv) < 3:
        print(
            "Usage: python experiment/best_case_benchmark.py [DATASET_NAME] [OUTPUT_FILENAME] [STEPS] [SAMPLES]"
        )
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_filename = sys.argv[2]

    # Optional arguments
    steps_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    samples_arg = int(sys.argv[4]) if len(sys.argv) > 4 else None

    # Ensure output directory exists if specified
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
    if p_full < MIN_P:
        print(f"Warning: p_full={p_full} is small.")
        return
    else:
        p_values = np.linspace(MIN_P, p_full, num=NUM_POINTS, dtype=int)

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

            k_list = np.linspace(0.1 * p, 0.3 * p, num=5, dtype=int)

            # Call find_critical_k / run_experiment
            for k in k_list:
                # Use provided steps or adaptive steps
                run_steps = (
                    steps_arg if steps_arg is not None else get_adaptive_steps(k)
                )

                res = run_experiment(
                    A_sub,
                    k,
                    experiment_name=f"{dataset_name}_p{p}_k{k}",
                    dataset_name=dataset_name,
                    steps=run_steps,
                    samples=samples_arg,
                )

                # Log data
                entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Dataset": name,
                    "p": int(p),
                    "k": int(k),
                    "Speedupx": res.get("speedupx"),
                    "Ratio": res.get("ratio"),
                    "Steps": res.get("steps"),
                    "Samples": res.get("samples"),
                    "Run_ID": run_id,
                }

                results_list.append(entry)
                print(
                    f"  > p={p}: k={int(k)}, Speedup={entry['Speedupx']:.4f}, Ratio={entry['Ratio']:.4f}, Steps={entry['Steps']}, Samples={entry['Samples']}"
                )

    # 4. Save Results
    df = pd.DataFrame(results_list)
    df.to_csv(output_filename, index=False)
    print(f"Saved results to: {output_filename}")


if __name__ == "__main__":
    main()
