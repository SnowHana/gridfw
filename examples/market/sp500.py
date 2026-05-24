import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import kagglehub
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from grad_fw import FWHomotopySolver
from grad_fw.benchmarks.GreedySolver import GreedySolver
from grad_fw.benchmarks.BruteForceSolver import BruteForceSolver

# PATHS
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # examples/market -> repo root
DATA_DIR = os.path.join(REPO_ROOT, "data", "market")
os.makedirs(DATA_DIR, exist_ok=True)

KAGGLE_DATASET = "andrewmvd/sp-500-stocks"
PRICES_CACHE = os.path.join(DATA_DIR, "sp500_prices.csv")
MAX_AGE_DAYS = 30

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"DATA_DIR:  {DATA_DIR}")


# Kaggle Data
def load_kaggle_data(force_update=False):
    companies_path = os.path.join(DATA_DIR, "sp500_companies.csv")
    if not force_update and os.path.exists(companies_path):
        age = (time.time() - os.path.getmtime(companies_path)) / 86400
        if age < MAX_AGE_DAYS:
            print(f"Using cached Kaggle data ({age:.0f} days old).")
        else:
            print(f"Kaggle cache is {age:.0f} days old, re-downloading...")
            kagglehub.dataset_download(KAGGLE_DATASET, output_dir=DATA_DIR)
    else:
        print("Downloading Kaggle dataset...")
        kagglehub.dataset_download(KAGGLE_DATASET, output_dir=DATA_DIR)

    return {
        name: pd.read_csv(os.path.join(DATA_DIR, f"sp500_{name}.csv"))
        for name in ["companies", "index"]
    }


# Price Data (cached)
def load_prices(tickers, start="2015-01-01", end="2024-12-31", force_update=False):
    if not force_update and os.path.exists(PRICES_CACHE):
        age = (time.time() - os.path.getmtime(PRICES_CACHE)) / 86400
        if age < MAX_AGE_DAYS:
            print(f"Loading cached prices ({age:.0f} days old)...")
            return pd.read_csv(PRICES_CACHE, index_col="Date", parse_dates=True)
        print(f"Price cache is {age:.0f} days old, re-downloading...")
    else:
        print("Downloading prices from yfinance (this takes ~2 min)...")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
    raw = raw.dropna(axis=1, thresh=int(0.95 * len(raw)))
    raw = raw.ffill().dropna()
    raw.index.name = "Date"
    raw.to_csv(PRICES_CACHE)
    print(f"Saved {raw.shape[1]} stocks x {raw.shape[0]} days → {PRICES_CACHE}")
    return raw


# Return and Covariance matrix
def build_matrices(prices):
    X = np.log(prices / prices.shift(1)).dropna().values  # (T, p)
    A = np.cov(X.T)
    return X, A


dfs = load_kaggle_data()
tickers = dfs["companies"]["Symbol"].str.replace(".", "-", regex=False).tolist()

# 2. Load prices
prices = load_prices(tickers)
stock_names = prices.columns.tolist()
print(f"Prices shape: {prices.shape}  (days x stocks)")

X, A = build_matrices(prices=prices)
p = A.shape[0]
print(f"X: {X.shape} | A: {A.shape}")


# Run Solver
k = 50
total_variance = np.trace(A)

print(f"\nRunning FW-Homotopy (k={k}, p={p})...")
solver = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
s = solver.solve(verbose=True)
fw_indices = np.where(s > 0.5)[0]
fw_stocks = [stock_names[i] for i in fw_indices]
print(fw_stocks)

print(f"\nRunning Greedy...")
greedy_solver = GreedySolver(A, k)
greedy_s = greedy_solver.solve()[0]

# print(greedy_indices)
greedy_stocks = [stock_names[i] for i in greedy_s]
print(greedy_stocks)


# print(f"\nRunning Brute Force...")
# brute_solver = BruteForceSolver(A, k)
# brute_s = brute_solver.solve()
# brute_stocks = [stock_names[i] for i in brute_s]
# print(brute_stocks)


# ── Visualisation ─────────────────────────────────────────────────────────────


def plot_selection_submatrices(corr, greedy_idx, fw_idx, names, k, save_path):
    """Side-by-side k×k correlation heatmaps for each solver's selection."""
    _, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, idx, title in [
        (axes[0], greedy_idx, f"Greedy (k={k})"),
        (axes[1], fw_idx, f"FW-Homotopy (k={k})"),
    ]:
        labels = [names[i] for i in idx]
        sns.heatmap(
            corr[np.ix_(idx, idx)],
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.3,
        )
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    plt.suptitle("Correlation matrix of selected stocks", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_full_heatmap(corr, greedy_idx, fw_idx, p, k, save_path):
    """Full p×p correlation matrix reordered so selected stocks appear first."""
    greedy_set = set(greedy_idx)
    fw_set = set(fw_idx)
    both = sorted(greedy_set & fw_set)
    greedy_only = sorted(greedy_set - fw_set)
    fw_only = sorted(fw_set - greedy_set)
    rest = sorted(set(range(p)) - greedy_set - fw_set)
    order = both + greedy_only + fw_only + rest

    n_greedy = len(greedy_set)
    n_fw = len(fw_set)
    fw_start = n_greedy - len(both)  # where the FW block starts in reordered matrix

    _, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr[np.ix_(order, order)],
        ax=ax,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6},
    )
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0),
            n_greedy,
            n_greedy,
            fill=False,
            edgecolor="steelblue",
            linewidth=2.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (fw_start, fw_start),
            n_fw,
            n_fw,
            fill=False,
            edgecolor="darkorange",
            linewidth=2.5,
        )
    )
    ax.set_title(
        "Full correlation matrix  (selected stocks ordered first)", fontsize=13
    )
    ax.legend(
        handles=[
            mpatches.Patch(
                edgecolor="steelblue", facecolor="none", lw=2, label=f"Greedy (k={k})"
            ),
            mpatches.Patch(
                edgecolor="darkorange",
                facecolor="none",
                lw=2,
                label=f"FW-Homotopy (k={k})",
            ),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def animate_fw_homotopy(history, names, k, save_path):
    """Bar chart animation of t values evolving over FW-Homotopy iterations.

    Stocks are sorted by their final t value (left = most selected).
    Orange bars = currently in LMO solution s; blue = not selected.
    Bottom panel tracks δ along the homotopy path.
    """
    final_t = history[-1]["t"]
    order = np.argsort(final_t)[::-1]  # stable sort: most-selected stock on the left
    sorted_names = [names[i] for i in order]
    x = np.arange(len(order))

    all_steps = [h["step"] for h in history]
    all_deltas = [h["delta"] for h in history]

    fig, (ax_bar, ax_delta) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.subplots_adjust(hspace=0.35)

    # Bar chart (top)
    init_t = history[0]["t"][order]
    init_s = history[0]["s"][order]
    colors = ["darkorange" if si else "steelblue" for si in init_s]
    bars = ax_bar.bar(x, init_t, color=colors, width=1.0, edgecolor="none")
    ax_bar.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("t  (selection weight)")
    ax_bar.set_xticks(x[:: max(1, len(x) // 20)])
    ax_bar.set_xticklabels(
        sorted_names[:: max(1, len(x) // 20)], rotation=90, fontsize=6
    )
    title = ax_bar.set_title("")

    # Delta curve (bottom)
    ax_delta.plot(all_steps, all_deltas, color="black", linewidth=1)
    ax_delta.set_yscale("log")
    ax_delta.set_xlabel("Step")
    ax_delta.set_ylabel("δ  (log scale)")
    (vline,) = ax_delta.plot(
        [all_steps[0], all_steps[0]],
        ax_delta.get_ylim(),
        color="red",
        linewidth=1,
        linestyle="--",
    )

    def update(frame):
        snap = history[frame]
        t_sorted = snap["t"][order]
        s_sorted = snap["s"][order]
        for bar, h, si in zip(bars, t_sorted, s_sorted):
            bar.set_height(h)
            bar.set_color("darkorange" if si else "steelblue")
        vline.set_xdata([snap["step"], snap["step"]])
        title.set_text(
            f"FW-Homotopy  |  step {snap['step']}/{all_steps[-1]}"
            f"  |  δ={snap['delta']:.2e}  |  k={k}"
        )
        return (*bars, vline, title)

    anim = FuncAnimation(fig, update, frames=len(history), interval=80, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=12))
    plt.show()
    print(f"Animation saved → {save_path}")


# corr = np.corrcoef(X.T)
# plot_selection_submatrices(corr, list(greedy_s), list(fw_indices), stock_names, k,
#                            save_path=os.path.join(DATA_DIR, "selection_submatrix.png"))
# plot_full_heatmap(corr, list(greedy_s), list(fw_indices), p, k,
#                   save_path=os.path.join(DATA_DIR, "full_corr_heatmap.png"))

print("\nRecording FW-Homotopy history for animation...")
solver_anim = FWHomotopySolver(A, k, n_steps=800, n_mc_samples=100)
_, history = solver_anim.solve_with_history(record_every=8)
animate_fw_homotopy(
    history, stock_names, k, save_path=os.path.join(DATA_DIR, "fw_homotopy.gif")
)
