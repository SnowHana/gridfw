import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "logs/param_sweep_log.csv"
PLOT_DIR = "logs/plots"

def plot_sweep():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(LOG_FILE)
    
    # Clean whitespace in headers
    df.columns = df.columns.str.strip()

    # 1. Vary k
    df_k = df[df['Experiment'] == 'vary_k']
    if not df_k.empty:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_k['k'], df_k['Ratio'], marker='o')
        plt.title('Effect of k on Accuracy (Ratio)')
        plt.xlabel('k')
        plt.ylabel('Ratio (FW/Greedy)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(df_k['k'], df_k['Speedup_x'], marker='o', color='orange')
        plt.title('Effect of k on Speedup')
        plt.xlabel('k')
        plt.ylabel('Speedup (Greedy Time / FW Time)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/vary_k.png")
        print(f"Saved {PLOT_DIR}/vary_k.png")
        plt.close()

    # 2. Vary n_mc
    df_nmc = df[df['Experiment'] == 'vary_nmc']
    if not df_nmc.empty:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_nmc['Samples'], df_nmc['Ratio'], marker='o')
        plt.title('Effect of Samples (n_mc) on Accuracy')
        plt.xlabel('Samples')
        plt.ylabel('Ratio')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(df_nmc['Samples'], df_nmc['Speedup_x'], marker='o', color='orange')
        plt.title('Effect of Samples on Speedup')
        plt.xlabel('Samples')
        plt.ylabel('Speedup')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/vary_nmc.png")
        print(f"Saved {PLOT_DIR}/vary_nmc.png")
        plt.close()

    # 3. Vary Steps
    df_steps = df[df['Experiment'] == 'vary_steps']
    if not df_steps.empty:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_steps['Steps'], df_steps['Ratio'], marker='o')
        plt.title('Effect of Steps on Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Ratio')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(df_steps['Steps'], df_steps['Speedup_x'], marker='o', color='orange')
        plt.title('Effect of Steps on Speedup')
        plt.xlabel('Steps')
        plt.ylabel('Speedup')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/vary_steps.png")
        print(f"Saved {PLOT_DIR}/vary_steps.png")
        plt.close()

    # 4. Portfolio Vary k
    df_pk = df[df['Experiment'] == 'portfolio_vary_k']
    if not df_pk.empty:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_pk['k'], df_pk['Ratio'], marker='o', color='green')
        plt.title('Portfolio: Accuracy (Ratio)')
        plt.xlabel('k')
        plt.ylabel('Ratio (FW/Greedy)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(df_pk['k'], df_pk['Speedup_x'], marker='o', color='orange')
        plt.title('Portfolio: Speedup')
        plt.xlabel('k')
        plt.ylabel('Speedup')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/portfolio_vary_k.png")
        print(f"Saved {PLOT_DIR}/portfolio_vary_k.png")
        plt.close()

if __name__ == "__main__":
    plot_sweep()
