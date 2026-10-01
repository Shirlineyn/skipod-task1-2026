# /// script
# dependencies = ["numpy", "matplotlib"]
# ///
"""Рисунок приближённого решения для отчёта: карта значений в D с контуром
области и поверхность.

  uv run tools/plot_solution.py sol_800x1200.bin [out.png]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solio import in_D, read_solution  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
    x, y, w = read_solution(src)
    X, Y = np.meshgrid(x, y, indexing="ij")
    W = np.where(in_D(X, Y), w, np.nan)  # фиктивная область не показывается

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    im = ax.pcolormesh(X, Y, W, shading="auto", cmap="viridis")
    t = np.linspace(-np.sqrt(2), np.sqrt(2), 400)
    ax.plot(np.sqrt(1 + 4 * t * t), t, "k-", lw=1)
    ax.plot([3, 3], [-np.sqrt(2), np.sqrt(2)], "k-", lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"u(x, y), сетка {len(x) - 1}x{len(y) - 1}")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    step = max(1, (len(x) - 1) // 100)  # для поверхности хватит ~100 линий
    ax3.plot_surface(X[::step, ::step], Y[::step, ::step], W[::step, ::step],
                     cmap="viridis", linewidth=0, antialiased=True)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("u")
    ax3.view_init(elev=30, azim=-120)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
