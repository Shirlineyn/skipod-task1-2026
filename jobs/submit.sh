#!/bin/bash
# Постановка заданий в очередь на Polus. Запускать из корня task1 на polus:
#
#   bash jobs/submit.sh check            # одна группа
#   bash jobs/submit.sh base t1 t2 t3    # несколько
#   nohup bash jobs/submit.sh all > results/polus/submit.log 2>&1 &
#
# Правила Polus: в очередях short и normal не больше 4 задач одновременно
# на пользователя. Скрипт ставит следующую задачу, только когда своих
# задач в очереди (PEND + RUN) меньше MAX_JOBS, и ждёт иначе. Ожидание —
# это sleep на фронтенде, счёта там нет.
#
# REPEAT=3 bash jobs/submit.sh base t1 t2 t3 — каждую группу трижды;
# повторы идут кругами (весь список, потом снова), а не подряд, чтобы
# разброс времени отражал разные моменты, а не один.
#
# Перед постановкой: make (исполняемые файлы нельзя пересобирать, пока
# задачи с ними стоят в очереди или считаются).
set -euo pipefail
MAX_JOBS=${MAX_JOBS:-4}
REPEAT=${REPEAT:-1}
cd "$(dirname "$0")/.."
mkdir -p results/polus

[ $# -eq 0 ] && { sed -n '2,18p' "$0"; exit 1; }
groups=("$@")
[ "${groups[0]}" = all ] && groups=(check base t1 t2 t3)

progs="poisson_seq poisson_omp poisson_mpi poisson_hyb"
case " ${groups[*]} " in *" cuda_check "*|*" t4 "*) progs="$progs poisson_cuda" ;; esac
for b in $progs; do
    [ -x "$b" ] || { echo "нет ./$b — сначала: module load SpectrumMPI && make"; exit 1; }
done

my_jobs() { bjobs -noheader 2>/dev/null | grep -c -E ' (PEND|RUN) ' || true; }

for rep in $(seq "$REPEAT"); do
    for g in "${groups[@]}"; do
        for f in jobs/lsf/"$g"/*.lsf; do
            while [ "$(my_jobs)" -ge "$MAX_JOBS" ]; do sleep 20; done
            id=$(bsub < "$f" | sed -n 's/.*<\([0-9]*\)>.*/\1/p')
            echo "$(date '+%F %T') $id $f rep=$rep" | tee -a results/polus/submitted.log
            sleep 2
        done
    done
done
echo "all submitted; watch: bjobs; results: results/polus/"
