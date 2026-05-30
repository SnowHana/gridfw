import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    mask = (np.abs(corr_ordered) < threshold) | np.triu(
        np.ones_like(corr_ordered, dtype=bool), k=1
    )

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
        plt.savefig(save_path, dpi=150, format="pdf")
        print(f"Saved → {save_path}")
    plt.show()


def plot_selection_heatmap(
    log_return_matrix, fw_indices, greedy_indices, threshold=0.2, save_path=None
):
    """Full p×p correlation heatmap with FW and Greedy selections boxed.

    Stocks are reordered as: [fw_only | both | greedy_only | rest].
    This makes both selections contiguous blocks that share an overlap region.

    Args:
        log_return_matrix: Log-return matrix of shape (days, stocks).
        fw_indices: Array of indices selected by FW-Homotopy.
        greedy_indices: Array of indices selected by Greedy.
        threshold: Correlations with |r| < threshold are masked white.
        save_path: If provided, saves the figure to this path.
    """
    corr = np.corrcoef(log_return_matrix.T)
    p = corr.shape[0]

    fw_set = set(int(i) for i in fw_indices)
    greedy_set = set(int(i) for i in greedy_indices)
    both = sorted(fw_set & greedy_set)
    fw_only = sorted(fw_set - greedy_set)
    greedy_only = sorted(greedy_set - fw_set)
    rest = sorted(set(range(p)) - fw_set - greedy_set)

    # Order: fw_only | both | greedy_only | rest
    # FW box covers columns 0 → n_fw (fw_only + both)
    # Greedy box covers columns len(fw_only) → len(fw_only) + n_greedy (both + greedy_only)
    # The overlap (both) sits where the two boxes intersect
    order = fw_only + both + greedy_only + rest
    corr_ordered = corr[np.ix_(order, order)]
    mask = (np.abs(corr_ordered) < threshold) | np.triu(
        np.ones_like(corr_ordered, dtype=bool), k=1
    )

    n_fw = len(fw_set)
    n_greedy = len(greedy_set)
    greedy_start = len(fw_only)  # where greedy's block begins in the reordered matrix

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

    # Bounding boxes — drawn on top of the heatmap
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0),
            n_fw,
            n_fw,
            fill=False,
            edgecolor="darkorange",
            linewidth=2.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (greedy_start, greedy_start),
            n_greedy,
            n_greedy,
            fill=False,
            edgecolor="steelblue",
            linewidth=2.5,
        )
    )

    ax.legend(
        handles=[
            mpatches.Patch(
                edgecolor="darkorange",
                facecolor="none",
                lw=2,
                label=f"FW-Homotopy (k={n_fw}, {len(fw_only)} unique)",
            ),
            mpatches.Patch(
                edgecolor="steelblue",
                facecolor="none",
                lw=2,
                label=f"Greedy (k={n_greedy}, {len(greedy_only)} unique)",
            ),
            mpatches.Patch(
                facecolor="lightgray", lw=0, label=f"Overlap: {len(both)} stocks"
            ),
        ],
        loc="lower right",
        fontsize=9,
    )

    ax.set_title(
        f"S&P 500 correlation — solver selections  (p={p}, |r| ≥ {threshold} shown)",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, format="pdf")
        print(f"Saved → {save_path}")
    plt.show()


def plot_objective_comparison(A, fw_indices, greedy_indices, save_path=None):
    """Bar chart comparing CSSP objective values for FW-Homotopy vs Greedy.

    Args:
        A: Covariance matrix of shape (p, p).
        fw_indices: Array of indices selected by FW-Homotopy.
        greedy_indices: Array of indices selected by Greedy.
        save_path: If provided, saves the figure to this path.
    """
    A2 = A @ A

    def cssp_obj(indices):
        idx = list(indices)
        A_ss = A[np.ix_(idx, idx)]
        A2_ss = A2[np.ix_(idx, idx)]
        return np.trace(np.linalg.inv(A_ss + 1e-9 * np.eye(len(idx))) @ A2_ss)

    k = len(fw_indices)
    fw_obj = cssp_obj(fw_indices)
    greedy_obj = cssp_obj(greedy_indices)

    _, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Greedy", "FW-Homotopy"],
        [greedy_obj, fw_obj],
        color=["steelblue", "darkorange"],
        width=0.5,
        edgecolor="k",
        linewidth=0.7,
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, [greedy_obj, fw_obj]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.005,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Tr( $A_{ss}^{-1}$ $A^2_{ss}$ )", fontsize=11)
    ax.set_title(f"CSSP objective comparison  (k={k})", fontsize=13)
    ax.set_ylim(0, max(fw_obj, greedy_obj) * 1.12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, format="pdf")
        print(f"Saved → {save_path}")
    plt.show()


def plot_submatrix_comparison(
    log_return_matrix, fw_indices, greedy_indices, stock_names=None, save_path=None
):
    """Side-by-side k×k correlation heatmaps for each solver's selection.

    Shows only the selected stocks — directly answers whether each solver
    picked diversified (low off-diagonal) stocks.

    Args:
        log_return_matrix: Log-return matrix of shape (days, stocks).
        fw_indices: Array of indices selected by FW-Homotopy.
        greedy_indices: Array of indices selected by Greedy.
        stock_names: Optional list of stock name strings for tick labels.
        save_path: If provided, saves the figure to this path.
    """
    corr = np.corrcoef(log_return_matrix.T)
    k = len(fw_indices)

    _, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, indices, title, color in [
        (axes[0], greedy_indices, f"Greedy  (k={k})", "steelblue"),
        (axes[1], fw_indices, f"FW-Homotopy  (k={k})", "darkorange"),
    ]:
        idx = list(indices)
        sub = corr[np.ix_(idx, idx)]

        # Cluster within the submatrix so similar stocks appear adjacent
        dist = 1 - np.abs(sub)
        np.fill_diagonal(dist, 0)
        order = leaves_list(linkage(dist, method="ward"))
        sub_ordered = sub[np.ix_(order, order)]
        tri_mask = np.triu(np.ones_like(sub_ordered, dtype=bool), k=1)

        labels = (
            [stock_names[i] for i in np.array(idx)[order]]
            if stock_names is not None
            else False
        )
        fontsize = max(4, 8 - k // 20)

        sns.heatmap(
            sub_ordered,
            mask=tri_mask,
            ax=ax,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.2,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.7, "label": "Pearson correlation"},
        )
        ax.set_title(title, fontsize=13, color=color, fontweight="bold")
        ax.tick_params(axis="x", rotation=90, labelsize=fontsize)
        ax.tick_params(axis="y", rotation=0, labelsize=fontsize)

        # Mean absolute off-diagonal correlation — lower = more diversified
        mask_offdiag = ~np.eye(k, dtype=bool)
        mean_corr = np.abs(sub)[mask_offdiag].mean()
        ax.set_xlabel(f"mean |r| off-diagonal = {mean_corr:.3f}", fontsize=9)

    plt.suptitle("Correlation of selected stocks", fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", format="pdf")
        print(f"Saved → {save_path}")
    plt.show()
