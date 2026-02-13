import sys
import os
import numpy as np
import pandas as pd

# --- IMPORT YOUR MODULES ---
from grad_fw.benchmarks.benchmarks import run_experiment
from grad_fw.data_loader import DatasetLoader

# CONFIGURATION
FIXED_ALPHA = 0.2
FIXED_M = 50
loader = DatasetLoader()


def get_fine_grained_k(p):
    # Scan 1% to 25% (The sparse regime)
    ratios = np.arange(0.01, 0.26, 0.01)
    k_values = sorted(list(set([int(p * r) for r in ratios])))
    return [k for k in k_values if k > 0]


def main():
    # We expect: python run_sniper.py [DATASET_NAME] [OUTPUT_FILENAME]
    if len(sys.argv) < 3:
        print("Usage: python run_sniper.py [DATASET_NAME] [OUTPUT_FILENAME]")
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_filename = sys.argv[2]

    # 1. Load Data
    print(f"Loading {dataset_name}...")
    try:
        # We ignore the second return value (X_norm) using '_'
        A, _ = loader.load(dataset_name)
        p = A.shape[0]

        # We enforce 'name' to be the string from the arguments
        name = dataset_name
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        sys.exit(1)

    # 2. Define K List
    k_list = get_fine_grained_k(p)
    print(f"Targeting k values: {k_list}")

    # 3. Execution Loop
    results = []

    for k in k_list:
        max_steps = max(1000, 20 * k)
        experiment_name = f"sniper_k{k}_a{FIXED_ALPHA}_m{FIXED_M}"

        print(f"Running: k={k}, steps={max_steps}...")

        res = run_experiment(
            A=A,
            k=k,
            experiment_name=experiment_name,
            dataset_name=name,
            samples=FIXED_M,
            alpha=FIXED_ALPHA,
            # steps=max_steps # Uncomment if your function supports this
        )

        # --- BYPASS LOGGER ---
        # Instead of calling logger(**res), we just add the dict to our list
        # We manually add the params to ensure they are in the CSV

        res["dataset_name"] = name
        res["p"] = p
        res["k"] = k
        res["samples"] = FIXED_M
        res["alpha"] = FIXED_ALPHA
        results.append(res)

    # 4. SAVE UNIQUE FILE
    # We save exactly to the path Katana told us to use
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    print(f"Saved unique log to: {output_filename}")


if __name__ == "__main__":
    main()
