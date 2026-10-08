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
#   t1, t2, t3 — строки таблиц 1 (OpenMP), 2 (MPI), 3 (MPI+OpenMP);
#   bind  — гибрид из таблицы 3 в двух вариантах: free — как в t3, нити
#           процесса не привязаны; places — OMP_PLACES=cores,
#           OMP_PROC_BIND=close. Оба с --binding: программа печатает, на
#           каком CPU сидит каждая нить. В серии 3 гибрид 4x2 на 800x1200
#           дал эффективность 0.79 против 0.91 у 4x4 — гипотеза, что нити
#           без привязки садятся на одно физическое ядро. Итог серии 4:
#           не садятся, а places замедляет в 90-400 раз (XL ставит все
#           процессы узла на одни и те же ядра). Группа оставлена как
#           воспроизводимый опыт;
#   wait  — режим ожидания нитей OpenMP для гибрида 4xT на 800x1200
#           (T = 1, 2, 4): default — как в t3, active — OMP_WAIT_POLICY=ACTIVE
#           (нити между параллельными областями крутятся, а не засыпают),
#           passive — OMP_WAIT_POLICY=PASSIVE. Рядом для сравнения MPI на 4 и
#           8 процессах. Везде OMP_DISPLAY_ENV=true — среда XL печатает в .err
#           действующие настройки (так видно, что параметр подхватился, и
#           видны значения XLSMPOPTS spins/yields по умолчанию). Проверка
#           гипотезы серии 4: гибрид 4x2 теряет на ожидании в allreduce;
#   nodes — одна и та же последовательная задача на каждом узле (-m), по
#           два раза: в серии 1 все прогоны на polus-c2 были в ~1.6 раза
#           медленнее, чем на c3 и c4;
#   cuda_check — MPI+CUDA-программа: 80x80 на 1, 2 и 4 картах (решение
#           должно совпасть с CPU-версиями), сетки таблиц на одной карте;
#   t4    — таблица 4 (своя, в задании её нет): MPI+CUDA на 1, 2 и 4 картах
#           на сетках таблиц и на двух более крупных, 1600x2400 и 3200x4800,
#           где карте есть чем загрузиться; для них же — последовательная
#           программа и MPI на 32 процессах как база для сравнения.
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
#   MPI+CUDA: процесс на карту, у процесса одно ядро; на узле две карты
#     P100, поэтому 1-2 карты — один узел, 4 — два узла по две.
#     mode=exclusive_process: на карте только наш процесс, иначе её может
#     делить чужая задача. mpiexec — обязательно с ключом -gpu: без него
#     Spectrum MPI не завершается после выхода программы, работающей с CUDA
#     (результат напечатан, а задача висит до лимита времени; проверено
#     08.10.2026 на фронтенде: без -gpu снято по таймауту через 20 с,
#     с -gpu — 0.3 с).
#
# Узлы. В серии 1 (04.10.2026) все прогоны, попавшие на polus-c2, шли в
# ~1.65 раза медленнее (группа nodes: seq 800x1200 — 8.27 с на c2 против
# 5.05 с на c3/c4), хотя заявленная частота и загрузка узлов одинаковые.
# Поэтому по умолчанию все задания идут только на c3 и c4 (#BSUB -m);
# двух узлов хватает и для 32 процессов.
#
# Монопольный узел. В серии 2 разброс между повторами доходил до 58%:
# медленные повторы шли одновременно с другой моей задачей на том же узле
# (ядра разные, но память общая, а CG упирается в пропускную способность
# памяти). С EXCLUSIVE=1 каждое задание забирает узел целиком (#BSUB -x).
#
# Серия. Логи пишутся в results/polus/$SERIES, чтобы серии не смешивались:
#   SERIES=s4 EXCLUSIVE=0 bash jobs/gen_jobs.sh
set -euo pipefail
cd "$(dirname "$0")"
rm -rf lsf
WTIME=00:10
SERIES=${SERIES:-s5}
EXCLUSIVE=${EXCLUSIVE:-1}
OUT=results/polus/$SERIES
HOSTS="polus-c3-ib polus-c4-ib"
mkdir -p "../$OUT"

# emit группа имя число_слотов "строка -R" команда [доп. строки #BSUB]
emit() {
    local group=$1 name=$2 slots=$3 res=$4 cmd=$5
    local excl=""
    [ "$EXCLUSIVE" = 1 ] && excl="#BSUB -x
"
    local extra=${6:-"#BSUB -m \"$HOSTS\"
$excl"}
    mkdir -p "lsf/$group"
    cat > "lsf/$group/$name.lsf" <<EOF
#BSUB -J $name
#BSUB -q short
#BSUB -W $WTIME
#BSUB -n $slots
#BSUB -R "$res"
#BSUB -o $OUT/$name.%J.out
#BSUB -e $OUT/$name.%J.err
${extra}source /polusfs/setenv/setup.SMPI
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

cuda_job() {  # группа M N G
    local group=$1 M=$2 N=$3 G=$4 span="span[hosts=1]" per=$4
    [ "$G" -gt 2 ] && { span="span[ptile=2]"; per=2; }
    local excl=""
    [ "$EXCLUSIVE" = 1 ] && excl="#BSUB -x
"
    emit "$group" "cuda_${M}x${N}_g${G}" "$G" "affinity[core(1)] $span" \
        "mpiexec -gpu ./poisson_cuda $M $N" \
        "#BSUB -gpu \"num=$per:mode=exclusive_process\"
#BSUB -m \"$HOSTS\"
$excl"
}

# check: пункты 1, 2, 5, 7
for g in 20 40 80; do seq_job check $g $g --out $OUT/sol_${g}x${g}.bin; done
for t in 1 4 16; do omp_job check 80 80 $t; done
for p in 1 2 4; do mpi_job check 80 80 $p; done
for p in 1 2; do hyb_job check 80 80 $p 4; done

# base: последовательная программа на сетках таблиц; на крупной — рисунок
seq_job base 400 600 --out $OUT/sol_400x600.bin
seq_job base 800 1200 --out $OUT/sol_800x1200.bin

# bind: гибрид без привязки нитей и с OMP_PLACES=cores
hyb_bind() {  # M N P T вариант
    local M=$1 N=$2 P=$3 T=$4 v=$5 span="span[hosts=1]" env
    [ $((P * T)) -gt 20 ] && span="span[ptile=$((P / 2))]"
    if [ "$v" = places ]; then
        env="export OMP_NUM_THREADS=$T OMP_PLACES=cores OMP_PROC_BIND=close
mpiexec -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND"
    else
        env="export OMP_NUM_THREADS=$T
mpiexec -x OMP_NUM_THREADS"
    fi
    emit bind "hyb_${M}x${N}_p${P}_t${T}_$v" "$P" "affinity[core($T)] $span" \
        "$env ./poisson_hyb $M $N --binding"
}
for v in free places; do
    for t in 1 2 4 8; do hyb_bind 400 600 2 $t $v; done
    for t in 1 2 4 8; do hyb_bind 800 1200 4 $t $v; done
done

# wait: режим ожидания нитей в гибриде
hyb_wait() {  # M N P T вариант
    local M=$1 N=$2 P=$3 T=$4 v=$5 pol=""
    case $v in
        active)  pol=" OMP_WAIT_POLICY=ACTIVE" ;;
        passive) pol=" OMP_WAIT_POLICY=PASSIVE" ;;
    esac
    local fwd="-x OMP_NUM_THREADS -x OMP_DISPLAY_ENV"
    [ -n "$pol" ] && fwd="$fwd -x OMP_WAIT_POLICY"
    emit wait "hyb_${M}x${N}_p${P}_t${T}_$v" "$P" "affinity[core($T)] span[hosts=1]" \
"export OMP_NUM_THREADS=$T OMP_DISPLAY_ENV=true$pol
mpiexec $fwd ./poisson_hyb $M $N --binding"
}
for v in default active passive; do
    for t in 1 2 4; do hyb_wait 800 1200 4 $t $v; done
done
mpi_job wait 800 1200 4
mpi_job wait 800 1200 8

# Таблицы 1-3
for t in 2 4 8 16; do omp_job t1 400 600 $t; done
for t in 4 8 16 32; do omp_job t1 800 1200 $t; done
for p in 2 4 8 16; do mpi_job t2 400 600 $p; done
for p in 4 8 16 32; do mpi_job t2 800 1200 $p; done
for t in 1 2 4 8; do hyb_job t3 400 600 2 $t; done
for t in 1 2 4 8; do hyb_job t3 800 1200 4 $t; done

# nodes: скорость узлов по отдельности
for node in c2 c3 c4; do
    for rep in 1 2; do
        emit nodes "node_${node}_seq_800x1200_r$rep" 1 "affinity[core(1)]" \
            "./poisson_seq 800 1200" "#BSUB -m polus-$node-ib
"
    done
done

# cuda_check: правильность MPI+CUDA
for g in 1 2 4; do cuda_job cuda_check 80 80 $g; done
cuda_job cuda_check 400 600 1
cuda_job cuda_check 800 1200 1

# t4: MPI+CUDA; WTIME у последовательной на 3200x4800 — с запасом
for grid in "400 600" "800 1200" "1600 2400" "3200 4800"; do
    for g in 1 2 4; do cuda_job t4 $grid $g; done
done
seq_job t4 1600 2400
WTIME=00:25 seq_job t4 3200 4800
for grid in "1600 2400" "3200 4800"; do mpi_job t4 $grid 32; done

find lsf -name '*.lsf' | sort | sed 's/^/  /'
echo "$(find lsf -name '*.lsf' | wc -l | tr -d ' ') files"
