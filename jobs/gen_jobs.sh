#!/bin/bash
# Генерация командных файлов LSF для всех прогонов задания.
# Один файл — один запуск (одна строка таблицы): ресурсы под каждую
# конфигурацию свои, и в логе JOBID однозначно соответствует строке.
#
#   bash jobs/gen_jobs.sh          # пишет jobs/lsf/<группа>/<имя>.lsf
#
# Группы:
#   check — пункты 1, 2, 5, 7 задания: сетки 20, 40, 80; 80x80 на 1/4/16 нитях,
#           1/2/4 процессах, 1 и 2 процессах по 4 нити;
#   base  — последовательная программа на сетках таблиц (база для ускорения);
#   t1, t2, t3 — строки таблиц 1 (OpenMP), 2 (MPI), 3 (MPI+OpenMP).
#
# Как выбраны ресурсы (узел Polus: 20 ядер POWER8 по 8 аппаратных нитей,
# слот LSF = ядро):
#   OpenMP, T <= 20 нитей: одно ядро на нить, affinity[core(T)], нити
#     раскладывает по ядрам /polusfs/lsf/openmp/launchOpenMP.py.
#   OpenMP, T = 32: на узле только 20 ядер, поэтому 16 ядер по 2 нити
#     (SMT2) — нагрузка на все ядра одинаковая, в отличие от 20 ядер,
#     где 12 ядер получили бы по 2 нити, а 8 — по одной.
#   MPI: один процесс на ядро; до 16 процессов — на одном узле, 32 —
#     поровну на два узла по 16.
#   MPI+OpenMP: на процесс T ядер; если P*T > 20, процессы поровну на два узла.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf lsf
WTIME=00:10

# emit группа имя число_слотов "строка -R" команда
emit() {
    local group=$1 name=$2 slots=$3 res=$4 cmd=$5
    mkdir -p "lsf/$group"
    cat > "lsf/$group/$name.lsf" <<EOF
#BSUB -J $name
#BSUB -q short
#BSUB -W $WTIME
#BSUB -n $slots
#BSUB -R "$res"
#BSUB -o results/polus/$name.%J.out
#BSUB -e results/polus/$name.%J.err
source /polusfs/setenv/setup.SMPI
echo "job \$LSB_JOBID ($name) hosts: \$LSB_HOSTS"
echo "bind cpu list: \${LSB_BIND_CPU_LIST:-none}"
date
$cmd
date
EOF
}

seq_job() {  # группа M N [доп. аргументы]
    local group=$1 M=$2 N=$3; shift 3
    emit "$group" "seq_${M}x${N}" 1 "affinity[core(1)]" "./poisson_seq $M $N $*"
}

omp_job() {  # группа M N T
    local group=$1 M=$2 N=$3 T=$4 cores=$4
    [ "$T" -gt 20 ] && cores=$((T / 2))
    emit "$group" "omp_${M}x${N}_t${T}" 1 "affinity[core($cores)]" \
"export OMP_NUM_THREADS=$T OMP_PROC_BIND=close
/polusfs/lsf/openmp/launchOpenMP.py ./poisson_omp $M $N"
}

mpi_job() {  # группа M N P
    local group=$1 M=$2 N=$3 P=$4 span="span[hosts=1]"
    [ "$P" -gt 16 ] && span="span[ptile=$((P / 2))]"
    emit "$group" "mpi_${M}x${N}_p${P}" "$P" "affinity[core(1)] $span" \
        "mpiexec ./poisson_mpi $M $N"
}

hyb_job() {  # группа M N P T
    local group=$1 M=$2 N=$3 P=$4 T=$5 span="span[hosts=1]"
    [ $((P * T)) -gt 20 ] && span="span[ptile=$((P / 2))]"
    # OMP_NUM_THREADS передаётся процессам явно (-x): на втором узле
    # окружение задания само по себе может не появиться. Программа
    # предупредит, если число нитей у процессов разное.
    emit "$group" "hyb_${M}x${N}_p${P}_t${T}" "$P" "affinity[core($T)] $span" \
"export OMP_NUM_THREADS=$T
mpiexec -x OMP_NUM_THREADS ./poisson_hyb $M $N"
}

# check: пункты 1, 2, 5, 7
for g in 20 40 80; do seq_job check $g $g --out results/polus/sol_${g}x${g}.bin; done
for t in 1 4 16; do omp_job check 80 80 $t; done
for p in 1 2 4; do mpi_job check 80 80 $p; done
for p in 1 2; do hyb_job check 80 80 $p 4; done

# base: последовательная программа на сетках таблиц; на крупной — рисунок
seq_job base 400 600 --out results/polus/sol_400x600.bin
seq_job base 800 1200 --out results/polus/sol_800x1200.bin

# Таблицы 1-3
for t in 2 4 8 16; do omp_job t1 400 600 $t; done
for t in 4 8 16 32; do omp_job t1 800 1200 $t; done
for p in 2 4 8 16; do mpi_job t2 400 600 $p; done
for p in 4 8 16 32; do mpi_job t2 800 1200 $p; done
for t in 1 2 4 8; do hyb_job t3 400 600 2 $t; done
for t in 1 2 4 8; do hyb_job t3 800 1200 4 $t; done

find lsf -name '*.lsf' | sort | sed 's/^/  /'
echo "$(find lsf -name '*.lsf' | wc -l | tr -d ' ') files"
