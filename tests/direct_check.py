# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""Независимая проверка: та же разностная схема, собранная заново в Python
и решённая прямым методом (scipy spsolve), против CG из poisson_seq.

Что здесь сделано иначе, чем в C++-коде, — специально:
  * площади S_ij считаются численным интегрированием (scipy quad),
    а не через первообразную дуги гиперболы;
  * схема собирается для самой w, в исходном виде задания
    (правая часть F + Q v0 плюс граничные слагаемые), без перехода
    к поправке d = w - v0.
Совпадение решений подтверждает и сборку коэффициентов, и CG, и законность
перехода к поправке.

  uv run tests/direct_check.py            # сетки 20x20, 40x40, 80x80
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import integrate, sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from solio import read_solution  # noqa: E402

A1, B1, A2, B2 = 0.91, 3.11, -1.65, 1.65
ROOT = Path(__file__).resolve().parent.parent


def cell_area_quad(xa, xb, ya, yb):
    xr = min(xb, 3.0)
    if xr <= xa:
        return 0.0
    def width(y):
        return max(0.0, xr - max(xa, np.sqrt(1 + 4 * y * y)))
    # Излом подынтегральной функции — там, где дуга входит в ячейку.
    pts = [y for c in (xa, xr) if c > 1 for y in (0.5 * np.sqrt(c * c - 1), -0.5 * np.sqrt(c * c - 1))
           if ya < y < yb]
    val, _ = integrate.quad(width, ya, yb, points=pts or None, epsabs=1e-14, epsrel=1e-12, limit=200)
    return val


def direct_solution(M, N):
    h1, h2 = (B1 - A1) / M, (B2 - A2) / N
    eps = max(h1, h2) ** 2
    x = A1 + h1 * np.arange(M + 1)
    y = A2 + h2 * np.arange(N + 1)
    v0 = np.sin(x[:, None] + y[None, :])
    n = (M - 1) * (N - 1)
    idx = lambda i, j: (i - 1) * (N - 1) + (j - 1)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    for i in range(1, M):
        for j in range(1, N):
            S = cell_area_quad(x[i] - h1 / 2, x[i] + h1 / 2, y[j] - h2 / 2, y[j] + h2 / 2)
            frac = S / (h1 * h2)
            Q = (1 - frac) / eps ** 2
            k = idx(i, j)
            rows.append(k); cols.append(k); vals.append(2 / h1 ** 2 + 2 / h2 ** 2 + Q)
            rhs[k] = frac * 1.0 + Q * v0[i, j]
            for (ii, jj, hh) in ((i - 1, j, h1), (i + 1, j, h1), (i, j - 1, h2), (i, j + 1, h2)):
                if 1 <= ii <= M - 1 and 1 <= jj <= N - 1:
                    rows.append(k); cols.append(idx(ii, jj)); vals.append(-1 / hh ** 2)
                else:
                    rhs[k] += v0[ii, jj] / hh ** 2  # узел на Gamma — в правую часть
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    w = v0.copy()
    w[1:M, 1:N] = spsolve(A.tocsc(), rhs).reshape(M - 1, N - 1)
    return w


def main():
    exe = ROOT / "poisson_seq"
    if not exe.exists():
        sys.exit("сначала make poisson_seq")
    ok = True
    for M, N in ((20, 20), (40, 40), (80, 80)):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sol.bin"
            subprocess.run([str(exe), str(M), str(N), "--delta", "1e-13", "--out", str(out)],
                           check=True, capture_output=True)
            _, _, w_cg = read_solution(out)
        w_dir = direct_solution(M, N)
        diff = np.abs(w_cg - w_dir).max()
        good = diff < 1e-8
        ok &= good
        print(f"{M}x{N}: max|w_cg - w_direct| = {diff:.3e}  {'OK' if good else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
