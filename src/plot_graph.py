import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Configuration
LOG_FILE = "logs/param_sweep_log.csv"
PLOT_DIR = "logs/plots"

def load_data():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return None
    return pd.read_csv(LOG_FILE)

def plot_vary_k(df, dataset_name, p_val):
    """Plots for Vary K experiment."""
    subset = df[df['Experiment'] == 'vary_k'].sort_values('k')
    if subset.empty:
        return

    # 1. Time Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(subset['k'], subset['Greedy_Time_s'], label='Greedy', marker='o')
    plt.plot(subset['k'], subset['FW_Time_s'], label='FW-Homotopy', marker='s')
    plt.xlabel('k (Number of Selected Features)')
    plt.ylabel('Time (s)')
    plt.title(f'Time Complexity: Greedy vs FW ({dataset_name}, p={p_val})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_vary_k_time.png")
    plt.close()

    # 2. Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(subset['k'], subset['Speedup_x'], label='Speedup (Greedy/FW)', color='green', marker='^')
    plt.axhline(1.0, color='red', linestyle='--', label='Baseline (1x)')
    plt.xlabel('k')
    plt.ylabel('Speedup Factor (x)')
    plt.title(f'Speedup of FW over Greedy ({dataset_name}, p={p_val})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_vary_k_speedup.png")
    plt.close()

def plot_vary_p(df, dataset_name):
    """Plots for Vary P experiment."""
    subset = df[df['Experiment'] == 'vary_p'].sort_values('p')
    if subset.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(subset['p'], subset['Greedy_Time_s'], label='Greedy', marker='o')
    plt.plot(subset['p'], subset['FW_Time_s'], label='FW-Homotopy', marker='s')
    plt.xlabel('p (Problem Dimension)')
    plt.ylabel('Time (s)')
    plt.title(f'Scalability with Dimension p ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_vary_p_time.png")
    plt.close()

def plot_tuning(df, dataset_name):
    """Plots for Tuning experiment (Heatmap)."""
    # Filter for tuning experiments (tuning_sX_nY)
    tuning_df = df[df['Experiment'].str.startswith('tuning_')].copy()
    if tuning_df.empty:
        return

    # Extract Steps and Samples from columns if available, or parse from Experiment name
    # The log now has 'Steps' and 'Samples' columns, so we use those.
    
    # Pivot for Heatmap: Ratio
    pivot_ratio = tuning_df.pivot_table(index='Steps', columns='Samples', values='Ratio')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_ratio, annot=True, cmap='viridis', fmt=".4f")
    plt.title(f'Approximation Ratio (FW/Greedy) - {dataset_name}')
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_tuning_ratio.png")
    plt.close()

    # Pivot for Heatmap: Speedup
    pivot_speedup = tuning_df.pivot_table(index='Steps', columns='Samples', values='Speedup_x')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_speedup, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Speedup (Greedy/FW) - {dataset_name}')
    plt.savefig(f"{PLOT_DIR}/{dataset_name}_tuning_speedup.png")
    plt.close()

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_data()
    if df is None:
        return

    # Group by Dataset
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        ds_data = df[df['Dataset'] == dataset]
        
        # Get p value (assuming constant p per dataset for most experiments)
        # For vary_p, p changes, so we handle it separately.
        p_val = ds_data['p'].iloc[0] 

        plot_vary_k(ds_data, dataset, p_val)
        plot_vary_p(ds_data, dataset)
        plot_tuning(ds_data, dataset)

    print(f"Plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()