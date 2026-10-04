"""Сравнение вариантов одного опыта по именам заданий.

collect.py собирает таблицы задания и группирует запуски по числу
процессов и нитей — варианты одной конфигурации (например, разные
OMP_WAIT_POLICY) у него сливаются. Здесь группировка по имени задания
(часть имени файла до JOBID), для каждого — число повторов и min / max
полного времени, времени CG и времени обменов и редукций внутри CG.

  python3 tools/compare_runs.py results/polus/s5/*.out
"""

import re
import sys
from collections import defaultdict


def main(paths):
    if not paths:
        sys.exit(__doc__)
    runs = defaultdict(list)
    for p in paths:
        name = re.sub(r"\.\d+\.out$", "", p.split("/")[-1])
        with open(p, errors="replace") as fp:
            text = fp.read()
        m = re.search(r"^RESULT .*$", text, re.M)
        if not m:
            print(f"нет RESULT: {p}")
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", m.group(0)))
        runs[name].append(kv)

    def rng(rs, key):
        vals = [float(r[key]) for r in rs if key in r]
        if not vals:
            return "—".rjust(15)
        return f"{min(vals):.4f}-{max(vals):.4f}".rjust(15)

    print(f"{'задание':36s} {'повт':>4s} {'time, с':>15s} {'cg, с':>15s} {'halo, с':>15s} {'allreduce, с':>15s}")
    for name in sorted(runs):
        rs = runs[name]
        print(f"{name:36s} {len(rs):4d} {rng(rs, 'time')} {rng(rs, 'cg_time')} "
              f"{rng(rs, 'halo')} {rng(rs, 'allreduce')}")


if __name__ == "__main__":
    main(sys.argv[1:])
