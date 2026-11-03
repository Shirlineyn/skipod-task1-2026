// Последовательная и OpenMP-версия.
//
// Один исходный файл даёт две программы:
//   без -fopenmp / -qsmp=omp директивы игнорируются — последовательный код;
//   с ними — OpenMP-код, число нитей задаёт OMP_NUM_THREADS.
// Так последовательная версия гарантированно считает то же самое, что и
// параллельная, и сравнение времени честное.
//
// Запуск: ./poisson_seq M N [--delta 1e-6] [--out sol.bin]

#include <cstdio>
#include <vector>
#include "common.h"
#include "problem.h"
#include "solver.h"

// Блок один на всю сетку; его теневые узлы — граница прямоугольника,
// где поправка d = 0. Обмены не нужны.
struct NoComm : Comm {
    void halo(double*) {}
    void allreduce(double*, int) {}
};

int main(int argc, char** argv)
{
    Options opt;
    if (!parse_options(argc, argv, opt)) {
        usage(argv[0]);
        return 1;
    }
    problem::Grid gr(opt.M, opt.N);
    problem::Block bl;
    bl.init(opt.M - 1, opt.N - 1, 0, 0);

    std::printf("poisson_omp: M=%d N=%d h1=%.6g h2=%.6g eps=%.6g delta=%g threads=%d\n",
                gr.M, gr.N, gr.h1, gr.h2, gr.eps, opt.delta, omp_threads());

    const double t0 = wtime();
    problem::Coeffs c;
    problem::build_coeffs(gr, bl, c);
    std::vector<double> d(bl.size(), 0.0);  // начальное приближение w(0) = v0, т. е. d = 0
    const double t1 = wtime();

    NoComm comm;
    CGParams prm;
    prm.delta = opt.delta;
    prm.maxit = opt.maxit;
    prm.verbose = opt.verbose;
    prm.rank = 0;
    CGResult res = cg_solve(gr, bl, c, d, comm, prm);
    const double t2 = wtime();

    // Самопроверка геометрии: площади ячеек должны сложиться в площадь D.
    double area = 0.0;
    for (int li = 1; li <= bl.nx; ++li)
        for (int lj = 1; lj <= bl.ny; ++lj)
            area += (1.0 - c.Q[bl.idx(li, lj)] * gr.eps * gr.eps) * gr.h1 * gr.h2;

    double st[3];
    solution_stats(gr, bl, d, st);

    std::printf("iters=%ld restarts=%d last_step=%.3e resid=%.3e J=%.12e\n",
                res.iters, res.restarts, res.last_step, res.resid, res.J);
    if (!res.converged()) {
        std::fprintf(stderr, "SOLVER_ERROR status=%s iters=%ld last_step=%.6e resid=%.6e\n",
                     cg_status_name(res.status), res.iters, res.last_step, res.resid);
        return 3;
    }
    std::printf("area_cells=%.10f area_D=%.10f  int_D(w)=%.10f  max_D|w|=%.10f  nodes_in_D=%.0f\n",
                area, problem::area_D(), st[0], st[1], st[2]);
    std::printf("time: setup=%.4f cg=%.4f total=%.4f s\n", t1 - t0, t2 - t1, t2 - t0);
    if (opt.binding) std::printf("binding: %s\n", thread_cpus().c_str());

    if (!opt.out.empty()) {
        std::vector<double> w;
        fill_v0(gr, w);
        for (int li = 1; li <= bl.nx; ++li)
            for (int lj = 1; lj <= bl.ny; ++lj)
                w[(size_t)li * (gr.N + 1) + lj] += d[bl.idx(li, lj)];
        if (!write_solution(opt.out, gr, w)) {
            std::fprintf(stderr, "cannot write %s\n", opt.out.c_str());
            return 2;
        }
        std::printf("solution written to %s\n", opt.out.c_str());
    }
    // Строка для сбора таблиц скриптом tools/collect.py.
    std::printf("RESULT prog=%s M=%d N=%d procs=1 threads=%d iters=%ld time=%.6f cg_time=%.6f "
                "intw=%.12e maxw=%.12e resid=%.3e restarts=%d status=%s\n",
                omp_threads() ? "omp" : "seq", gr.M, gr.N, omp_threads(), res.iters, t2 - t0,
                t2 - t1, st[0], st[1], res.resid, res.restarts, cg_status_name(res.status));

    return 0;
}
