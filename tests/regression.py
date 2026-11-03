# /// script
# dependencies = ["numpy"]
# ///
"""Регрессии интерфейсов запуска и сравнение полных полей.

make && uv run tests/regression.py --output-dir /path/to/new/test-results
Файлы и логи сохраняются для проверки; существующий каталог не используется.
"""
import argparse
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from compare_solutions import compare
from solio import read_solution

OUTPUT = None


def run(command, *, env=None, preexec_fn=None):
    merged = dict(os.environ, OMP_NUM_THREADS="4")
    merged.update(env or {})
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True,
                            env=merged, timeout=60, preexec_fn=preexec_fn)
    with (OUTPUT / "commands.log").open("a") as fp:
        fp.write(shlex.join([str(x) for x in command]) + f"\nexit={result.returncode}\n")
        fp.write(result.stdout + result.stderr + "\n")
    return result


def variants():
    return [(name, [ROOT / f"poisson_{name}"]) for name in ("seq", "omp")] + [
        (name, ["mpirun", "--oversubscribe", "-n", "2", ROOT / f"poisson_{name}"])
        for name in ("mpi", "hyb")]


def alter(source, target, changes):
    data = bytearray(source.read_bytes())
    values = np.frombuffer(data, dtype=np.float64, offset=40)
    changes(values)
    target.write_bytes(data)


class Regression(unittest.TestCase):
    def test_solver_status(self):
        exe = OUTPUT / "solver_status"
        command = shlex.split(os.environ.get("CXX", "c++"))
        result = run(command + ["-std=c++11", "-O0", "-I", ROOT / "src",
                                ROOT / "tests/solver_status.cpp", "-o", exe])
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run([exe])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_solves_and_writes(self):
        for name, command in variants():
            with self.subTest(name=name, case="maxit"):
                path = OUTPUT / f"{name}_incomplete.bin"
                r = run(command + ["20", "20", "--maxit", "1", "--out", path])
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("status=max_iterations", r.stderr)
                self.assertNotIn("RESULT ", r.stdout)
                self.assertFalse(path.exists())
            with self.subTest(name=name, case="write"):
                r = run(command + ["20", "20", "--out", OUTPUT / "missing" / "sol.bin"])
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("cannot write", r.stderr)
                self.assertNotIn("RESULT ", r.stdout)

    def test_buffered_write_failure(self):
        # В отличие от отсутствующего каталога, fopen здесь проходит;
        # ограничение размера обнаруживается в fwrite или fclose.
        def limit():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))
        r = run([ROOT / "poisson_seq", "20", "20", "--out", OUTPUT / "short.bin"],
                preexec_fn=limit)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("cannot write", r.stderr)
        self.assertNotIn("RESULT ", r.stdout)

    def test_bad_arguments(self):
        cases = [["20foo", "20"], ["2147483647", "20"], ["50000", "50000"],
                 ["20", "20", "--delta", "nan"], ["20", "20", "--delta", "inf"],
                 ["20", "20", "--delta", "0"], ["20", "20", "--delta", "-1"],
                 ["20", "20", "--delta", "1e-6x"], ["20", "20", "--delta", "1e999"],
                 ["20", "20", "--maxit", "0"], ["20", "20", "--maxit", "362"],
                 ["20", "20", "--maxit", "2foo"], ["20", "20", "--verbose", "-1"],
                 ["20", "20", "--verbose", "2147483648"], ["20", "20", "--delta"]]
        for name, command in variants():
            for args in cases:
                with self.subTest(name=name, args=args):
                    r = run(command + args)
                    self.assertNotEqual(r.returncode, 0)
                    self.assertNotIn("RESULT ", r.stdout)
                    self.assertIn("usage:", r.stderr)

    def test_full_fields(self):
        cases = [("omp", 1, t) for t in (1, 4, 16)]
        cases += [("mpi", p, 1) for p in (1, 2, 4)]
        cases += [("hyb", p, 4) for p in (1, 2)]
        for m, n in ((80, 80), (20, 30), (19, 23)):
            reference = OUTPUT / f"seq_{m}x{n}.bin"
            r = run([ROOT / "poisson_seq", m, n, "--out", reference])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertRegex(r.stdout, r"RESULT .* status=(converged|exact_residual)")
            for name, procs, threads in cases:
                with self.subTest(grid=(m, n), name=name, procs=procs, threads=threads):
                    dest = OUTPUT / f"{name}_{m}x{n}_p{procs}_t{threads}.bin"
                    command = [ROOT / f"poisson_{name}", m, n, "--out", dest]
                    if name in ("mpi", "hyb"):
                        command = ["mpirun", "--oversubscribe", "-n", procs] + command
                    r = run(command, env={"OMP_NUM_THREADS": str(threads)})
                    self.assertEqual(r.returncode, 0, r.stderr)
                    error, good = compare(reference, dest)
                    self.assertTrue(good, f"max|difference|={error}")
                    print(f"field {dest.name}: max|difference|={error:.3e}", flush=True)

    def test_comparators_detect_corruption(self):
        reference = OUTPUT / "check20.bin"
        r = run([ROOT / "poisson_seq", "20", "20", "--out", reference])
        self.assertEqual(r.returncode, 0)
        r = run([sys.executable, ROOT / "tools/compare_reference.py", reference])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        swapped = OUTPUT / "same_sum.bin"
        # Сумма не меняется, но локальные значения заведомо испорчены.
        def perturb(values):
            values[10 * 21 + 10] += 0.1
            values[11 * 21 + 10] -= 0.1
        alter(reference, swapped, perturb)
        self.assertAlmostEqual(read_solution(reference)[2].sum(), read_solution(swapped)[2].sum())
        r = run([sys.executable, ROOT / "tools/compare_solutions.py", reference, swapped])
        self.assertEqual(r.returncode, 1)
        broken = OUTPUT / "wrong_reference.bin"
        alter(reference, broken, lambda values: values.fill(0))
        r = run([sys.executable, ROOT / "tools/compare_reference.py", broken])
        self.assertEqual(r.returncode, 1)
        nan = OUTPUT / "nan.bin"
        alter(reference, nan, lambda values: values.__setitem__(0, np.nan))
        for script, args in (("compare_solutions.py", [reference, nan]), ("compare_reference.py", [nan])):
            r = run([sys.executable, ROOT / "tools" / script] + args)
            self.assertEqual(r.returncode, 1)
        r = run([sys.executable, ROOT / "tools/compare_solutions.py", reference, reference, "--atol", "nan"])
        self.assertNotEqual(r.returncode, 0)

    def test_collect_rejects_failed_status(self):
        from collect import parse
        source = OUTPUT / "status.out"
        base = "RESULT prog=seq M=400 N=600 procs=1 threads=0 iters=5 time="
        source.write_text(base + "1 status=converged\n" + base + "0.1 status=max_iterations\n"
                          + base + "0.2 status=breakdown\n" + base + "2\n")
        rows = parse([str(source)])[("seq", "400x600", 1, 0)]
        self.assertEqual(sorted(r[0] for r in rows), [1.0, 2.0])
        failed = OUTPUT / "failed.out"
        failed.write_text(base + "0.1 status=max_iterations\n")
        r = run([sys.executable, ROOT / "tools/collect.py", failed])
        self.assertNotEqual(r.returncode, 0)
        r = run([sys.executable, ROOT / "tools/compare_runs.py", failed])
        self.assertNotIn("0.1000", r.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Regression)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"Evidence: {OUTPUT}")
    sys.exit(not result.wasSuccessful())
