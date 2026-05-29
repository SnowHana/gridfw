import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list


def plot_corr_heatmap(log_return_matrix, threshold=0.2, save_path=None):
    """Plot S&P 500 correlation heatmap with hierarchical clustering and threshold mask.

    Args:
        log_return_matrix: Log-return matrix of shape (days, stocks).
        threshold: Correlations with |r| < threshold are masked white.
        save_path: If provided, saves the figure to this path.
    """
    corr = np.corrcoef(log_return_matrix.T)

    # Hierarchical Clustering
    dist = 1 - np.abs(corr)
    link = linkage(dist, method="ward")
    order = leaves_list(link)
    corr_ordered = corr[np.ix_(order, order)]
    mask = np.abs(corr_ordered) < threshold

    _, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        corr_ordered,
        mask=mask,
        ax=ax,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6, "label": "Pearson correlation"},
    )
    p = corr.shape[0]
    ax.set_title(
        f"S&P 500 return correlation  (p={p}, |r| ≥ {threshold} shown)",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
