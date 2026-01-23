import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Configuration
LOG_FILE = "logs/param_sweep_log.csv"
PLOT_DIR = "logs/plots"

def load_data():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(LOG_FILE)
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def get_x_axis_col(experiment_name):
    """Determines the x-axis column based on the experiment name."""
    exp = experiment_name.lower()
    if 'vary_k' in exp:
        return 'k', 'k (Number of Features)'
    elif 'vary_steps' in exp:
        return 'Steps', 'Steps (Homotopy Steps)'
    elif 'vary_nmc' in exp:
        return 'Samples', 'Samples (Monte Carlo)'
    elif 'vary_p' in exp:
        return 'p', 'p (Problem Dimension)'
    elif 'vary_alpha' in exp:
        # For vary_alpha, the alpha value is usually in the experiment name or we need a column.
        # Assuming we might not have a dedicated alpha column, we might skip or handle differently.
        # But for now, let's assume we plot against Steps or just skip if ambiguous.
        return None, None
    return None, None

def plot_experiment(df, dataset, experiment):
    """Generates the 4 requested graphs for a specific dataset and experiment."""
    subset = df[(df['Dataset'] == dataset) & (df['Experiment'] == experiment)].copy()
    
    if subset.empty:
        return

    x_col, x_label = get_x_axis_col(experiment)
    if not x_col:
        print(f"Skipping experiment '{experiment}' (unknown x-axis)")
        return

    # --- Pre-processing: Deduplicate ---
    # If we have multiple runs for the same x (e.g. k=410), keep the one with MINIMUM Greedy Time.
    # This automatically picks the "stable" run if one exploded and one didn't.
    # We group by x_col and take the row with min Greedy_Time_s
    subset = subset.loc[subset.groupby(x_col)['Greedy_Time_s'].idxmin()]
    
    subset = subset.sort_values(x_col)

    # --- Outlier Filtering (Growth Check) ---
    # The Rolling Median can fail if the instability lasts for many points (a plateau).
    # Instead, we detect the *onset* of instability: a sudden massive jump (e.g. > 5x).
    # We iterate and keep points that are consistent with the recent history.
    
    clean_rows = []
    if not subset.empty:
        # Always keep the first point
        clean_rows.append(subset.iloc[0])
        last_valid_time = subset.iloc[0]['Greedy_Time_s']
        
        for i in range(1, len(subset)):
            row = subset.iloc[i]
            curr_time = row['Greedy_Time_s']
            
            # Allow growth, but check for explosion.
            # If time jumps > 5x from the last valid point AND is > 1.0s (ignore noise), skip it.
            # We use 'last_valid_time' to skip over a chain of bad points.
            if curr_time > 5 * last_valid_time and curr_time > 1.0:
                continue
            
            clean_rows.append(row)
            last_valid_time = curr_time
            
    filtered_subset = pd.DataFrame(clean_rows)
    
    if len(filtered_subset) < len(subset):
        print(f"Filtered {len(subset) - len(filtered_subset)} outliers (Growth Spike) from {experiment}")
    
    subset = filtered_subset
    
    # Ensure plot directory exists
    exp_dir = os.path.join(PLOT_DIR, dataset, experiment)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Common Plot Style
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style if available, else default
    
    # 1. Objective Comparison (Greedy vs FW)
    plt.figure(figsize=(10, 6))
    plt.plot(subset[x_col], subset['Greedy_Obj'], label='Greedy', marker='o', linestyle='-', color='red')
    plt.plot(subset[x_col], subset['FW_Obj'], label='FW-Homotopy', marker='s', linestyle='--', color='blue')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title(f'Objective: Greedy vs FW ({dataset} - {experiment})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(exp_dir, '1_Objective_Comparison.png'))
    plt.close()

    # 2. Time Comparison (Greedy vs FW)
    plt.figure(figsize=(10, 6))
    plt.plot(subset[x_col], subset['Greedy_Time_s'], label='Greedy', marker='o', linestyle='-', color='orange')
    plt.plot(subset[x_col], subset['FW_Time_s'], label='FW-Homotopy', marker='s', linestyle='--', color='green')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title(f'Time: Greedy vs FW ({dataset} - {experiment})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(exp_dir, '2_Time_Comparison.png'))
    plt.close()

    # 3. Ratio
    plt.figure(figsize=(10, 6))
    plt.plot(subset[x_col], subset['Ratio'], label='Approximation Ratio (FW/Greedy)', marker='^', color='purple')
    plt.axhline(1.0, color='gray', linestyle='--', label='1.0 (Parity)')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.title(f'Approximation Ratio ({dataset} - {experiment})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(exp_dir, '3_Ratio.png'))
    plt.close()

    # 4. Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(subset[x_col], subset['Speedup_x'], label='Speedup (Greedy/FW)', marker='D', color='brown')
    plt.axhline(1.0, color='gray', linestyle='--', label='1.0 (No Speedup)')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Speedup Factor (x)', fontsize=12)
    plt.title(f'Speedup ({dataset} - {experiment})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(exp_dir, '4_Speedup.png'))
    plt.close()
    
    print(f"Generated 4 plots for {dataset}/{experiment} in {exp_dir}")

def main():
    print("Loading data...")
    df = load_data()
    
    # Get unique combinations of Dataset and Experiment
    combinations = df[['Dataset', 'Experiment']].drop_duplicates().values
    
    print(f"Found {len(combinations)} experiment configurations.")
    
    for dataset, experiment in combinations:
        # Skip tuning experiments for these specific line graphs (they are better as heatmaps)
        if 'tuning' in experiment:
            continue
            
        plot_experiment(df, dataset, experiment)

    print("\nDone! Check the 'logs/plots' directory.")

if __name__ == "__main__":
    main()