# /// script
# dependencies = ["numpy"]
# ///
"""Сверка сеточных решений с эталоном исходной задачи (1)-(2).

Эталон посчитан методом конечных элементов в Mathematica прямо в области D,
без фиктивных областей: tests/reference.wls, вывод — tests/reference.out
(сетка MaxCellMeasure = 5e-5, элементы второго порядка; между двумя самыми
мелкими сетками значения меняются не больше чем на 2e-7).

  uv run tools/compare_reference.py sol_80x80.bin sol_400x600.bin ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solio import bilinear, read_solution  # noqa: E402

POINTS = [(2.0, 0.0), (1.5, 0.3), (2.5, -0.8), (2.8, 1.0), (1.2, -0.2)]
U_REF = [0.791541548692, 0.909701773896, 0.915520114696, -0.549224965113, 0.815692173244]


# Регрессионные пределы для штатного delta=1e-6: округлённые вверх ошибки
# сохранённых проверок с FEM. Это контроль ухудшения, не теоретическая оценка.
MAX_ERROR = {(20, 20): 4.5e-3, (40, 40): 3.8e-3, (80, 80): 3.2e-3,
             (400, 600): 8e-4, (800, 1200): 5e-4}


def main(paths):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-error", type=float, help="явный допуск для всех входных сеток")
    args = ap.parse_args(paths)
    if args.max_error is not None and (not np.isfinite(args.max_error) or args.max_error <= 0):
        ap.error("--max-error должен быть конечным и положительным")
    ok = True
    hdr = "grid".ljust(12) + "".join(f"({px:g},{py:g})".rjust(14) for px, py in POINTS) + "    max err"
    print(hdr)
    print("ref".ljust(12) + "".join(f"{u:14.8f}" for u in U_REF))
    for p in args.paths:
        try:
            x, y, w = read_solution(p)
            grid = (len(x) - 1, len(y) - 1)
            limit = args.max_error if args.max_error is not None else MAX_ERROR.get(grid)
            if limit is None:
                raise ValueError("для этой сетки требуется явный --max-error")
            if not np.isfinite(w).all():
                raise ValueError("поле содержит NaN или бесконечность")
            if not all(x[0] <= px <= x[-1] and y[0] <= py <= y[-1] for px, py in POINTS):
                raise ValueError("контрольные точки находятся вне сетки")
            vals = [bilinear(x, y, w, px, py) for px, py in POINTS]
            error = max(abs(v - u) for v, u in zip(vals, U_REF))
            good = error <= limit
            ok &= good
            name = f"{grid[0]}x{grid[1]}"
            print(name.ljust(12) + "".join(f"{v:14.8f}" for v in vals)
                  + f"   {error:.2e} <= {limit:.2e}  {'OK' if good else 'FAIL'}")
        except (OSError, ValueError) as exc:
            print(f"{p}: FAIL: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
