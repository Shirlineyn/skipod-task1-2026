#!/bin/bash
# Локальная проверка перед кластером: сборка, сходимость по сеткам,
# совпадение всех четырёх программ, прямой решатель, сверка с эталоном.
# Логи и решения — в results/local/<дата-время>/.
#
#   bash tests/run_local.sh
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out="$here/results/local/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
log="$out/run_local.out"
exec > >(tee "$log") 2>&1

echo "== host: $(uname -a)"
make -C "$here" -B
echo

echo "== 1. Последовательная программа на сгущающихся сетках (пункт 1 задания)"
for g in "20 20" "40 40" "80 80" "400 600" "800 1200"; do
    set -- $g
    "$here/poisson_seq" "$1" "$2" --out "$out/sol_${1}x${2}.bin"
    echo
done

echo "== 2. Все программы на 80x80 должны дать то же решение"
check() {
    local tag=$1
    shift
    "$@" --out "$out/$tag.bin"
    uv run --quiet "$here/tools/compare_solutions.py" "$out/sol_80x80.bin" "$out/$tag.bin"
}
for t in 1 4 16; do OMP_NUM_THREADS=$t check "omp_t$t" "$here/poisson_omp" 80 80; done
for p in 1 2 4; do check "mpi_p$p" mpirun --oversubscribe -n "$p" "$here/poisson_mpi" 80 80; done
for p in 1 2; do OMP_NUM_THREADS=4 check "hyb_p$p" mpirun --oversubscribe -n "$p" "$here/poisson_hyb" 80 80; done
echo

echo "== 3. Прямой решатель (scipy) против CG"
uv run --quiet "$here/tests/direct_check.py"
echo

echo "== 4. Сверка с эталоном исходной задачи (Mathematica FEM)"
uv run --quiet "$here/tools/compare_reference.py" "$out"/sol_*.bin
echo

echo "== 5. Рисунок решения на самой мелкой сетке"
uv run --quiet "$here/tools/plot_solution.py" "$out/sol_800x1200.bin" "$out/solution_800x1200.png"
echo "== 6. Регрессионные проверки ошибок и неравномерного разбиения"
uv run --quiet "$here/tests/regression.py" --output-dir "$out/regression"
echo "logs: $out"
