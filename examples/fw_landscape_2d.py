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
    """3D animation of g_δ surface evolving + FW iterate t as a red dot.

    Two arrows drawn at each step:
      • Green  — ∇g_δ(t): analytical gradient (via grad_g_analytical, exact for p=2)
      • Cyan   — s − t:   FW step toward the LMO vertex
    """
    p = A.shape[0]
    eps = 0.02

    # All 2^p Rademacher vectors — exact xi_samples for gradient computation
    xi_exact = [np.array(xi) for xi in itertools.product([-1.0, 1.0], repeat=p)]

    # Grid over feasible triangle C_k = {t₁+t₂ ≤ 1, tᵢ ≥ eps}
    t_vals = np.linspace(eps, 1.0 - eps, grid_n)
    T1, T2 = np.meshgrid(t_vals, t_vals)
    feasible = (T1 + T2) <= (1.0 - eps)

    traj = np.array([h["t"] for h in history])  # (n_frames, p)

    # ── Precompute surfaces ───────────────────────────────────────────────────
    print(f"Precomputing {len(history)} surfaces on {grid_n}×{grid_n} grid …")
    Z_frames = []
    for idx, h in enumerate(history):
        Z = np.full_like(T1, np.nan)
        for i in range(grid_n):
            for j in range(grid_n):
                if feasible[i, j]:
                    t_ij = np.array([T1[i, j], T2[i, j]])
                    Z[i, j] = fw_objectivev_value(t_ij, h["delta"], A)
        Z_frames.append(Z)
        if (idx + 1) % max(1, len(history) // 5) == 0:
            print(f"  {idx + 1}/{len(history)}")
    print("Done.\n")

    # Consistent z-limits across all frames
    z_min = np.nanmin(Z_frames)
    z_max = np.nanmax(Z_frames)
    z_floor = z_min - 0.08 * (z_max - z_min)  # slightly below surface

    # Precompute z-value of the iterate at its own frame's δ
    traj_z = np.array(
        [
            fw_objectivev_value(history[i]["t"], history[i]["delta"], A)
            for i in range(len(history))
        ]
    )

    # ── Build animation ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        snap = history[frame]

        # Surface
        ax.plot_surface(
            T1,
            T2,
            Z_frames[frame],
            cmap="viridis",
            alpha=0.78,
            linewidth=0,
            antialiased=True,
            vmin=z_min,
            vmax=z_max,
        )

        # Trajectory projected onto the floor
        ax.plot(
            traj[: frame + 1, 0],
            traj[: frame + 1, 1],
            np.full(frame + 1, z_floor),
            color="tomato",
            linewidth=1.8,
            alpha=0.7,
        )

        # Current iterate — dot on surface + drop-line to floor
        t_cur = snap["t"]
        z_cur = traj_z[frame]
        ax.scatter([t_cur[0]], [t_cur[1]], [z_cur], color="red", s=70, zorder=6)
        ax.plot(
            [t_cur[0], t_cur[0]],
            [t_cur[1], t_cur[1]],
            [z_floor, z_cur],
            color="red",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
        )

        # ∇g_δ(t) arrow — analytical gradient (green), exact for p=2
        grad_g = BooleanRelaxation.grad_g_analytical(
            p, snap["delta"], t_cur, A, xi_exact
        )
        g_norm = np.linalg.norm(grad_g)
        if g_norm > 1e-10:
            display_len = 0.14
            unit_g = grad_g / g_norm
            ax.quiver(
                t_cur[0],
                t_cur[1],
                z_cur,
                unit_g[0] * display_len,
                unit_g[1] * display_len,
                g_norm * display_len,  # dz = ‖∇g‖·ds along gradient direction
                color="limegreen",
                linewidth=2,
                arrow_length_ratio=0.2,
                alpha=0.9,
            )

        # LMO direction arrow: t → s (FW step direction, cyan)
        s_cur = snap["s"]
        z_s = fw_objectivev_value(s_cur, snap["delta"], A)
        ax.quiver(
            t_cur[0],
            t_cur[1],
            z_cur,
            s_cur[0] - t_cur[0],
            s_cur[1] - t_cur[1],
            z_s - z_cur,
            color="cyan",
            linewidth=2,
            arrow_length_ratio=0.15,
            alpha=0.85,
        )

        # Binary corner markers
        for (c1, c2), label in [
            ((eps, eps), ""),
            ((1 - eps, eps), "t=(1,0)★"),
            ((eps, 1 - eps), "t=(0,1)"),
        ]:
            z_corner = fw_objectivev_value(np.array([c1, c2]), snap["delta"], A)
            ax.scatter([c1], [c2], [z_corner], color="gold", s=40, zorder=5)
            if label:
                ax.text(
                    c1,
                    c2,
                    z_corner + 0.02 * (z_max - z_min),
                    label,
                    fontsize=7,
                    color="goldenrod",
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(z_floor, z_max * 1.04)
        ax.set_xlabel("$t_1$", fontsize=10, labelpad=6)
        ax.set_ylabel("$t_2$", fontsize=10, labelpad=6)
        ax.set_zlabel("$g_\\delta(t_1,t_2)$", fontsize=10, labelpad=6)
        ax.set_title(
            f"FW-Homotopy  step {snap['step']}  |  $\\delta$ = {snap['delta']:.5f}",
            fontsize=11,
            pad=10,
        )
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
