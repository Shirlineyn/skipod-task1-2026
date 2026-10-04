# /// script
# dependencies = ["matplotlib"]
# ///
"""Сбор таблиц 1-3 задания из логов прогонов.

Каждая программа печатает строку
  RESULT prog=omp M=400 N=600 procs=1 threads=4 iters=... time=... ...
Скрипт находит такие строки во всех переданных файлах (обычно
results/*.out), группирует их и печатает таблицы в Markdown. Ускорение
считается относительно последовательной программы (prog=seq) на той же
сетке: S = T_seq / T. Если один и тот же запуск повторялся, берётся
минимальное время — оно меньше всего искажено чужой нагрузкой, — а в
столбце «Повт.» указано число повторов и разброс (max - min) / min.

  uv run tools/collect.py results/*.out            # таблицы в stdout
  uv run tools/collect.py --plot speedup.png results/*.out
"""

import argparse
import re
import sys
from collections import defaultdict

ROW_OMP = [(2, "400x600"), (4, "400x600"), (8, "400x600"), (16, "400x600"),
           (4, "800x1200"), (8, "800x1200"), (16, "800x1200"), (32, "800x1200")]
ROW_MPI = ROW_OMP
ROW_HYB = [(2, 1, "400x600"), (2, 2, "400x600"), (2, 4, "400x600"), (2, 8, "400x600"),
           (4, 1, "800x1200"), (4, 2, "800x1200"), (4, 4, "800x1200"), (4, 8, "800x1200")]


def parse(paths):
    runs = defaultdict(list)  # (prog, grid, procs, threads) -> [(time, iters, file)]
    for p in paths:
        with open(p, errors="replace") as fp:
            for line in fp:
                if not line.startswith("RESULT "):
                    continue
                kv = dict(re.findall(r"(\w+)=(\S+)", line))
                grid = f"{kv['M']}x{kv['N']}"
                key = (kv["prog"], grid, int(kv["procs"]), int(kv["threads"]))
                runs[key].append((float(kv["time"]), int(kv["iters"]), p))
    return runs


def best(runs, key):
    if key not in runs:
        return None
    return min(runs[key])


def spread(runs, key):
    ts = [t for t, _, _ in runs.get(key, [])]
    if len(ts) < 2:
        return str(len(ts)) if ts else "—"
    return f"{len(ts)}, {100 * (max(ts) - min(ts)) / min(ts):.0f}%"


def fmt(t):
    return "—" if t is None else f"{t:.4f}"


def table(title, head, rows):
    out = [f"**{title}**", "", "| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--plot", help="сохранить графики ускорения в файл")
    a = ap.parse_args()
    runs = parse(a.files)
    if not runs:
        sys.exit("RESULT-строк не найдено")

    base = {g: best(runs, ("seq", g, 1, 0)) for g in ("400x600", "800x1200")}
    print("Последовательная программа: " + ", ".join(
        f"{g}: {fmt(b[0]) if b else '—'} с, {b[1] if b else '—'} итераций, повт. {spread(runs, ('seq', g, 1, 0))}"
        for g, b in base.items()))
    print()

    def speed(g, r):
        if r is None or base.get(g) is None:
            return "—"
        return f"{base[g][0] / r[0]:.2f}"

    rows = []
    for t, g in ROW_OMP:
        k = ("omp", g, 1, t)
        r = best(runs, k)
        rows.append([str(t), g.replace("x", "×"), str(r[1]) if r else "—", fmt(r and r[0]), speed(g, r), spread(runs, k)])
    print(table("Таблица 1. OpenMP", ["Нитей", "Сетка", "Итераций", "Время, с", "Ускорение", "Повт."], rows))

    rows = []
    for p, g in ROW_MPI:
        k = ("mpi", g, p, 0)
        r = best(runs, k)
        rows.append([str(p), g.replace("x", "×"), str(r[1]) if r else "—", fmt(r and r[0]), speed(g, r), spread(runs, k)])
    print(table("Таблица 2. MPI", ["Процессов", "Сетка", "Итераций", "Время, с", "Ускорение", "Повт."], rows))

    rows = []
    for p, t, g in ROW_HYB:
        k = ("hyb", g, p, t)
        r = best(runs, k)
        rows.append([str(p), str(t), g.replace("x", "×"), str(r[1]) if r else "—", fmt(r and r[0]), speed(g, r), spread(runs, k)])
    print(table("Таблица 3. MPI+OpenMP", ["Процессов", "Нитей", "Сетка", "Итераций", "Время, с", "Ускорение", "Повт."], rows))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, prog, title in zip(axes, ("omp", "mpi", "hyb"), ("OpenMP", "MPI", "MPI+OpenMP")):
            for g in ("400x600", "800x1200"):
                if base.get(g) is None:
                    continue
                pts = sorted((k[2] * max(k[3], 1), base[g][0] / min(v)[0])
                             for k, v in runs.items() if k[0] == prog and k[1] == g)
                if pts:
                    ax.plot(*zip(*pts), "o-", label=g)
            lim = max([1] + [l.get_xdata().max() for l in ax.get_lines()])
            ax.plot([1, lim], [1, lim], "k--", lw=0.8, label="линейное")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xlabel("процессов × нитей")
            ax.set_ylabel("ускорение T_seq / T")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"saved {a.plot}")


if __name__ == "__main__":
    main()
