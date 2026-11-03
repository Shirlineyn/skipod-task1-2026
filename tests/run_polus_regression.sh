#!/bin/bash
# Запускать из корня репозитория внутри задания LSF, после сборки.
# Поля затем сравнить tools/compare_solutions.py с seq той же сетки.
set -euo pipefail
out=${1:?usage: bash tests/run_polus_regression.sh NEW_OUTPUT_DIR}
mkdir "$out"
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
failure() {
    local name=$1 pattern=$2
    shift 2
    if "$@" > "$out/$name.log" 2>&1; then
        echo "FAIL: $name returned success"
        exit 1
    fi
    if grep -q '^RESULT ' "$out/$name.log"; then
        echo "FAIL: $name printed RESULT"
        exit 1
    fi
    grep -q "$pattern" "$out/$name.log"
    echo "PASS: $name"
}
for grid in '80 80' '19 23'; do
    read -r m n <<< "$grid"
    ./poisson_seq "$m" "$n" --out "$out/seq_${m}x${n}.bin"
    ./poisson_omp "$m" "$n" --out "$out/omp_${m}x${n}.bin"
    for prog in mpi hyb; do
        mpiexec -n 2 "./poisson_$prog" "$m" "$n" --out "$out/${prog}_${m}x${n}.bin"
    done
    if [ -x ./poisson_cuda ]; then
        for p in 1 2; do
            mpiexec -gpu -n "$p" ./poisson_cuda "$m" "$n" --out "$out/cuda_p${p}_${m}x${n}.bin"
        done
    fi
done
for prog in seq omp mpi hyb cuda; do
    [ -x "./poisson_$prog" ] || continue
    cmd=("./poisson_$prog")
    case "$prog" in
        mpi|hyb) cmd=(mpiexec -n 2 "./poisson_$prog");;
        cuda) cmd=(mpiexec -gpu -n 2 ./poisson_cuda);;
    esac
    failure "${prog}_maxit" 'status=max_iterations' "${cmd[@]}" 20 20 --maxit 1
    failure "${prog}_write" 'cannot write' "${cmd[@]}" 20 20 --out "$out/missing/sol.bin"
    failure "${prog}_nan" 'usage:' "${cmd[@]}" 20 20 --delta nan
done
echo ALL_RUNTIME_CHECKS_PASSED
