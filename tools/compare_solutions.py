# /// script
# dependencies = ["numpy"]
# ///
"""Сравнение полных сеточных полей, включая граничные узлы.

uv run tools/compare_solutions.py reference.bin candidate.bin [...]
Допуск абсолютный; сетки должны совпадать. Любое нарушение даёт exit 1.
"""
import argparse
import sys

import numpy as np

from solio import read_solution


def compare(reference, candidate, atol=1e-10):
    x, y, w = read_solution(reference)
    xx, yy, ww = read_solution(candidate)
    if not np.array_equal(x, xx) or not np.array_equal(y, yy) or w.shape != ww.shape:
        raise ValueError("сетки не совпадают")
    if not np.isfinite(w).all() or not np.isfinite(ww).all():
        raise ValueError("поле содержит NaN или бесконечность")
    error = float(np.max(np.abs(w - ww)))
    return error, error <= atol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--atol", type=float, default=1e-10)
    a = ap.parse_args()
    if not np.isfinite(a.atol) or a.atol <= 0:
        ap.error("--atol должен быть конечным и положительным")
    ok = True
    for path in a.candidates:
        try:
            error, good = compare(a.reference, path, a.atol)
            print(f"{path}: max|difference|={error:.3e}, atol={a.atol:.3e} {'OK' if good else 'FAIL'}")
            ok &= good
        except (OSError, ValueError) as exc:
            print(f"{path}: FAIL: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
