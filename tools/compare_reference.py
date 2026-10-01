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

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solio import bilinear, read_solution  # noqa: E402

POINTS = [(2.0, 0.0), (1.5, 0.3), (2.5, -0.8), (2.8, 1.0), (1.2, -0.2)]
U_REF = [0.791541548692, 0.909701773896, 0.915520114696, -0.549224965113, 0.815692173244]


def main(paths):
    if not paths:
        sys.exit(__doc__)
    hdr = "grid".ljust(12) + "".join(f"({px:g},{py:g})".rjust(14) for px, py in POINTS) + "    max err"
    print(hdr)
    print("ref".ljust(12) + "".join(f"{u:14.8f}" for u in U_REF))
    for p in paths:
        x, y, w = read_solution(p)
        vals = [bilinear(x, y, w, px, py) for px, py in POINTS]
        errs = [abs(v - u) for v, u in zip(vals, U_REF)]
        name = f"{len(x) - 1}x{len(y) - 1}"
        print(name.ljust(12) + "".join(f"{v:14.8f}" for v in vals) + f"   {max(errs):.2e}")


if __name__ == "__main__":
    main(sys.argv[1:])
