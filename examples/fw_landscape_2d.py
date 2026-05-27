"""
Toy 2D visualisation of FW-Homotopy on a p=2, k=1 CSSP problem.

g_δ(t₁, t₂) — the Boolean relaxation objective — is evaluated *exactly*
(all 2² = 4 Rademacher vectors) and rendered as a 3D surface.  The
algorithm's iterate t is shown as a red dot that travels from the interior
toward the optimal binary vertex as δ grows along the homotopy path.

Run:
    python examples/fw_landscape_2d.py
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

from grad_fw import FWHomotopySolver
from grad_fw.verif.core import BooleanRelaxation

# ── Objective ─────────────────────────────────────────────────────────────────


def fw_objectivev_value(t, delta, A):
    """Calculate Objective Value"""
    p = len(t)
    Pi_inv = BooleanRelaxation.get_pi_inv(p, delta, t, A)
    total = 0.0
    for xi in itertools.product([-1.0, 1.0], repeat=p):
        # Possible Rademacher vectors
        b = A @ np.array(xi)
        total += b @ Pi_inv @ b
    return -total / (2**p)


# ── Animation ─────────────────────────────────────────────────────────────────


def animate_landscape(history, A, save_path, grid_n=32, fps=8):
    """3D animation of -g_δ surface (minimisation objective) + FW iterate.

    Cyan arrow — s − t (FW / LMO step direction, on surface).
    FW gap = -g(t) − (−g(s)) = g(s) − g(t) is shown as a convergence indicator.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    eps = 0.02

    # Grid over feasible simplex C_k = {t₁+t₂ ≤ 1, tᵢ ≥ eps}
    t_vals = np.linspace(eps, 1.0 - eps, grid_n)
    T1, T2 = np.meshgrid(t_vals, t_vals)
    feasible = (T1 + T2) <= (1.0 - eps)
    traj = np.array([h["t"] for h in history])

    # ── Precompute surfaces ───────────────────────────────────────────────────
    print(f"Precomputing {len(history)} surfaces on {grid_n}×{grid_n} grid …")
    Z_frames = []
    for idx, h in enumerate(history):
        Z = np.full_like(T1, np.nan)
        for i in range(grid_n):
            for j in range(grid_n):
                if feasible[i, j]:
                    Z[i, j] = fw_objectivev_value(
                        np.array([T1[i, j], T2[i, j]]), h["delta"], A
                    )
        Z_frames.append(Z)
        if (idx + 1) % max(1, len(history) // 5) == 0:
            print(f"  {idx + 1}/{len(history)}")
    print("Done.\n")

    z_min = np.nanmin(Z_frames)
    z_max = np.nanmax(Z_frames)
    z_range = z_max - z_min
    z_floor = z_min - 0.12 * z_range

    traj_z = np.array(
        [
            fw_objectivev_value(history[i]["t"], history[i]["delta"], A)
            for i in range(len(history))
        ]
    )

    # ── Figure / static elements ──────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Colorbar — created once, persists across ax.cla() calls
    norm = Normalize(vmin=z_min, vmax=z_max)
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.08, aspect=18)
    cbar.set_label("$-g_\\delta(t)$", fontsize=10)
    cbar.ax.tick_params(labelsize=7)

    fig.text(
        0.01,
        0.01,
        "Wujin Kim  |  FW-Homotopy for CSSP",
        fontsize=7,
        color="gray",
        alpha=0.55,
        ha="left",
        va="bottom",
    )

    def update(frame):
        ax.cla()
        snap = history[frame]

        # ── Pane / grid styling ───────────────────────────────────────────────
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#cccccc")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

        # ── Surface with wireframe outline (테두리선) ─────────────────────────
        ax.plot_surface(
            T1,
            T2,
            Z_frames[frame],
            cmap="viridis",
            alpha=0.82,
            linewidth=0.4,
            edgecolors=(0, 0, 0, 0.12),
            antialiased=True,
            vmin=z_min,
            vmax=z_max,
        )

        # ── Floor contour shadow ──────────────────────────────────────────────
        ax.contourf(
            T1,
            T2,
            Z_frames[frame],
            zdir="z",
            offset=z_floor,
            cmap="viridis",
            alpha=0.18,
            levels=10,
            vmin=z_min,
            vmax=z_max,
        )

        # ── Feasible simplex boundary on floor ───────────────────────────────
        corners = np.array([[eps, eps], [1 - eps, eps], [eps, 1 - eps], [eps, eps]])
        ax.plot(
            corners[:, 0], corners[:, 1], np.full(4, z_floor),
            color="white", linewidth=1.5, alpha=0.75, zorder=4,
        )

        # ── Floor trajectory ──────────────────────────────────────────────────
        ax.plot(
            traj[: frame + 1, 0],
            traj[: frame + 1, 1],
            np.full(frame + 1, z_floor),
            color="tomato",
            linewidth=2.2,
            alpha=0.85,
            zorder=5,
        )

        # ── Current iterate: dot on surface + dashed drop-line ───────────────
        t_cur = snap["t"]
        z_cur = traj_z[frame]
        ax.scatter([t_cur[0]], [t_cur[1]], [z_cur], color="crimson", s=85, zorder=8)
        ax.plot(
            [t_cur[0], t_cur[0]],
            [t_cur[1], t_cur[1]],
            [z_floor, z_cur],
            color="crimson",
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
        )

        # ── LMO step arrow: t → s on surface (deepskyblue) ───────────────────
        s_cur = snap["s"]
        z_s = fw_objectivev_value(s_cur, snap["delta"], A)
        ax.quiver(
            t_cur[0],
            t_cur[1],
            z_cur,
            s_cur[0] - t_cur[0],
            s_cur[1] - t_cur[1],
            z_s - z_cur,
            color="deepskyblue",
            linewidth=2.5,
            arrow_length_ratio=0.15,
            alpha=0.92,
        )

        # ── FW gap ────────────────────────────────────────────────────────────
        # gap = g(s) − g(t) = z(t) − z(s);  → 0 as t converges to s
        fw_gap = z_cur - z_s
        ax.text2D(
            0.02, 0.96,
            f"obj  = {z_cur:.4f}",
            transform=ax.transAxes, fontsize=8.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.text2D(
            0.02, 0.90,
            f"gap = {fw_gap:.4f}",
            transform=ax.transAxes, fontsize=8.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

        # ── Binary corner markers ─────────────────────────────────────────────
        # Gold ★ = optimal vertex; grey = suboptimal
        for (c1, c2), label, optimal in [
            ((eps, eps), "", False),
            ((1 - eps, eps), "s=(1,0) ★", True),
            ((eps, 1 - eps), "s=(0,1)", False),
        ]:
            z_c = fw_objectivev_value(np.array([c1, c2]), snap["delta"], A)
            color = "gold" if optimal else "silver"
            ax.scatter([c1], [c2], [z_c], color=color, s=55, zorder=7,
                       edgecolors="k", linewidths=0.5)
            if label:
                ax.text(c1, c2, z_c + 0.025 * z_range, label,
                        fontsize=7.5,
                        color="#b8860b" if optimal else "#888888",
                        fontweight="bold")

        # ── Legend (2D overlay, top-right) ────────────────────────────────────
        ax.text2D(0.68, 0.93, "● crimson : iterate t",
                  transform=ax.transAxes, fontsize=7.5, color="crimson")
        ax.text2D(0.68, 0.88, "▶ cyan : s − t  (LMO step)",
                  transform=ax.transAxes, fontsize=7.5, color="deepskyblue")
        ax.text2D(0.68, 0.83, "★ gold : optimal vertex",
                  transform=ax.transAxes, fontsize=7.5, color="#b8860b")

        # ── Axes ──────────────────────────────────────────────────────────────
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(z_floor, z_max + 0.05 * z_range)
        ax.set_xlabel("$t_1$", fontsize=11, labelpad=8)
        ax.set_ylabel("$t_2$", fontsize=11, labelpad=8)
        ax.set_zlabel("$-g_\\delta(t_1,t_2)$", fontsize=11, labelpad=8)
        ax.set_title(
            f"FW-Homotopy  |  step {snap['step']}  |  $\\delta={snap['delta']:.5f}$",
            fontsize=12,
            pad=12,
            fontweight="semibold",
        )
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=225 + frame * 1.5)

    anim = FuncAnimation(fig, update, frames=len(history), interval=2000 // fps)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.show()
    print(f"Animation saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 2×2 covariance: stock 0 has higher variance and positive correlation
    # A = np.random.rand(2, 2)
    A = np.array([[3.0, 1.2], [1.2, 1.0]])
    k = 1

    # Show which stock is the CSSP optimum
    A2 = A @ A
    print("CSSP objective (k=1) per stock:")
    for i in range(2):
        print(f"  stock {i}: {A2[i, i] / A[i, i]:.4f}")
    print("stock 0 should be selected\n")

    solver = FWHomotopySolver(A, k=k, n_steps=400, n_mc_samples=50, alpha=0.05)
    s, history = solver.solve_with_history(record_every=10, verbose=True)

    print(f"\nFW selected: s = {s}  (stock {int(np.argmax(s))} chosen)")

    animate_landscape(
        history,
        A,
        save_path="fw_landscape_2d.gif",
        grid_n=32,
        fps=8,
    )
