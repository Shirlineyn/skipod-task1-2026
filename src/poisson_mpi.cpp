// MPI-версия и гибридная MPI+OpenMP-версия.
//
// Один исходный файл, две программы: без OpenMP-флага — чистый MPI
// (poisson_mpi), с флагом — гибрид (poisson_hyb), где каждый процесс
// дополнительно распараллеливает циклы по своему блоку нитями OpenMP.
//
// Декомпозиция. Внутренние узлы сетки, (M-1) x (N-1), делятся на Px x Py
// прямоугольных блоков, Px*Py = P. По каждой оси узлы раздаются поровну,
// остаток — по одному первым блокам, поэтому размеры любых двух блоков по
// x (и по y) отличаются не больше чем на единицу. Из всех разложений
// P = Px*Py выбирается то, при котором отношение nx/ny в каждом блоке
// ближе всего к 1; требование задания — nx/ny в [1/2, 2].
//
// Связь между блоками — два вида операций на итерацию CG:
//   обмен теневыми слоями вектора p перед применением оператора A
//   (пятиточечный шаблон, нужны только соседи по сторонам, без углов);
//   MPI_Allreduce для скалярных произведений (2 раза по 2 числа).
//
// Запуск: mpiexec -n P ./poisson_mpi M N [--delta 1e-6] [--out sol.bin]

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "common.h"
#include "problem.h"
#include "solver.h"

struct Decomp {
    int Px, Py;
    bool ok;       // выполнено ли требование 1/2 <= nx/ny <= 2 во всех блоках
    double worst;  // наихудшее отношение max(nx/ny, ny/nx) по блокам
};

// Размер и смещение части c из parts при делении n узлов поровну.
static void split(int n, int parts, int c, int& len, int& off)
{
    const int base = n / parts, rem = n % parts;
    len = base + (c < rem ? 1 : 0);
    off = c * base + (c < rem ? c : rem);
}

static Decomp choose_decomp(int P, int nxg, int nyg)
{
    Decomp best;
    best.Px = 0;
    best.Py = 0;
    best.ok = false;
    best.worst = 1e300;
    for (int Px = 1; Px <= P; ++Px) {
        if (P % Px) continue;
        const int Py = P / Px;
        if (Px > nxg || Py > nyg) continue;
        // Крайние размеры блоков по каждой оси.
        const double xmin = nxg / Px, xmax = (nxg + Px - 1) / Px;
        const double ymin = nyg / Py, ymax = (nyg + Py - 1) / Py;
        const double worst = std::max(std::max(xmax / ymin, ymin / xmax),
                                      std::max(xmin / ymax, ymax / xmin));
        if (worst < best.worst) {
            best.Px = Px;
            best.Py = Py;
            best.worst = worst;
            best.ok = worst <= 2.0;
        }
    }
    return best;
}

// Обмены для блока внутри декартовой решётки процессов.
struct MpiComm : Comm {
    MPI_Comm cart;
    const problem::Block* bl;
    int left, right, down, up;  // соседи по x (i-1, i+1) и по y (j-1, j+1)
    std::vector<double> sbuf_d, sbuf_u, rbuf_d, rbuf_u;
    double t_halo, t_reduce;    // накопленное время обменов

    MpiComm(MPI_Comm c, const problem::Block* b) : cart(c), bl(b), t_halo(0), t_reduce(0)
    {
        MPI_Cart_shift(cart, 0, 1, &left, &right);
        MPI_Cart_shift(cart, 1, 1, &down, &up);
        sbuf_d.resize(bl->nx);
        sbuf_u.resize(bl->nx);
        rbuf_d.resize(bl->nx);
        rbuf_u.resize(bl->nx);
    }

    void halo(double* v)
    {
        const double t = MPI_Wtime();
        const int nx = bl->nx, ny = bl->ny;
        MPI_Request rq[8];
        int n = 0;
        // По x: строки li = 1 и li = nx лежат в памяти подряд (ny чисел).
        MPI_Irecv(v + bl->idx(0, 1), ny, MPI_DOUBLE, left, 0, cart, &rq[n++]);
        MPI_Irecv(v + bl->idx(nx + 1, 1), ny, MPI_DOUBLE, right, 1, cart, &rq[n++]);
        MPI_Isend(v + bl->idx(1, 1), ny, MPI_DOUBLE, left, 1, cart, &rq[n++]);
        MPI_Isend(v + bl->idx(nx, 1), ny, MPI_DOUBLE, right, 0, cart, &rq[n++]);
        // По y: столбцы lj = 1 и lj = ny идут с шагом ny+2 — упаковываем.
        for (int li = 1; li <= nx; ++li) {
            sbuf_d[li - 1] = v[bl->idx(li, 1)];
            sbuf_u[li - 1] = v[bl->idx(li, ny)];
        }
        MPI_Irecv(&rbuf_d[0], nx, MPI_DOUBLE, down, 2, cart, &rq[n++]);
        MPI_Irecv(&rbuf_u[0], nx, MPI_DOUBLE, up, 3, cart, &rq[n++]);
        MPI_Isend(&sbuf_d[0], nx, MPI_DOUBLE, down, 3, cart, &rq[n++]);
        MPI_Isend(&sbuf_u[0], nx, MPI_DOUBLE, up, 2, cart, &rq[n++]);
        MPI_Waitall(n, rq, MPI_STATUSES_IGNORE);
        // У границы Pi соседа нет (MPI_PROC_NULL): теневой слой остаётся
        // нулевым, как и положено для поправки d на Gamma.
        if (down != MPI_PROC_NULL)
            for (int li = 1; li <= nx; ++li) v[bl->idx(li, 0)] = rbuf_d[li - 1];
        if (up != MPI_PROC_NULL)
            for (int li = 1; li <= nx; ++li) v[bl->idx(li, ny + 1)] = rbuf_u[li - 1];
        t_halo += MPI_Wtime() - t;
    }

    void allreduce(double* a, int n)
    {
        const double t = MPI_Wtime();
        double tmp[4];
        MPI_Allreduce(a, tmp, n, MPI_DOUBLE, MPI_SUM, cart);
        for (int k = 0; k < n; ++k) a[k] = tmp[k];
        t_reduce += MPI_Wtime() - t;
    }
};

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Options opt;
    if (!parse_options(argc, argv, opt)) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    problem::Grid gr(opt.M, opt.N);
    const int nxg = opt.M - 1, nyg = opt.N - 1;  // внутренние узлы

    Decomp dc = choose_decomp(size, nxg, nyg);
    if (dc.Px == 0) {
        if (rank == 0) std::fprintf(stderr, "cannot split %dx%d nodes into %d blocks\n", nxg, nyg, size);
        MPI_Finalize();
        return 1;
    }
    int dims[2] = {dc.Px, dc.Py}, periods[2] = {0, 0}, coords[2];
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, coords);

    problem::Block bl;
    int nx, ny, i0, j0;
    split(nxg, dc.Px, coords[0], nx, i0);
    split(nyg, dc.Py, coords[1], ny, j0);
    bl.init(nx, ny, i0, j0);

    // Число нитей у всех процессов должно совпадать: на чужом узле
    // OMP_NUM_THREADS может не дойти через mpiexec — тогда это видно здесь.
    int thr_l = omp_threads(), thr_min, thr_max;
    MPI_Allreduce(&thr_l, &thr_min, 1, MPI_INT, MPI_MIN, cart);
    MPI_Allreduce(&thr_l, &thr_max, 1, MPI_INT, MPI_MAX, cart);
    if (rank == 0 && thr_min != thr_max)
        std::printf("WARNING: OpenMP threads differ across ranks: %d..%d\n", thr_min, thr_max);

    if (rank == 0) {
        std::printf("poisson_mpi: M=%d N=%d h1=%.6g h2=%.6g eps=%.6g delta=%g procs=%d threads=%d "
                    "(MPI thread level %d)\n",
                    gr.M, gr.N, gr.h1, gr.h2, gr.eps, opt.delta, size, omp_threads(), provided);
        std::printf("decomposition %d x %d, blocks %d..%d x %d..%d nodes, worst nx/ny ratio %.3f%s\n",
                    dc.Px, dc.Py, nxg / dc.Px, (nxg + dc.Px - 1) / dc.Px, nyg / dc.Py,
                    (nyg + dc.Py - 1) / dc.Py, dc.worst,
                    dc.ok ? "" : "  WARNING: outside [1/2, 2]");
    }

    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();
    problem::Coeffs c;
    problem::build_coeffs(gr, bl, c);
    std::vector<double> d(bl.size(), 0.0);
    const double t1 = MPI_Wtime();

    MpiComm comm(cart, &bl);
    CGParams prm;
    prm.delta = opt.delta;
    prm.maxit = opt.maxit;
    prm.verbose = opt.verbose;
    prm.rank = rank;
    CGResult res = cg_solve(gr, bl, c, d, comm, prm);
    const double t2 = MPI_Wtime();

    // Время — по самому медленному процессу.
    double tl[4] = {t1 - t0, t2 - t1, comm.t_halo, comm.t_reduce}, tm[4];
    MPI_Reduce(tl, tm, 4, MPI_DOUBLE, MPI_MAX, 0, cart);

    double area_l = 0.0;
    for (int li = 1; li <= bl.nx; ++li)
        for (int lj = 1; lj <= bl.ny; ++lj)
            area_l += (1.0 - c.Q[bl.idx(li, lj)] * gr.eps * gr.eps) * gr.h1 * gr.h2;
    double st_l[3], st[3], area;
    solution_stats(gr, bl, d, st_l);
    MPI_Reduce(&area_l, &area, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    MPI_Reduce(st_l, st, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    MPI_Reduce(st_l + 1, st + 1, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(st_l + 2, st + 2, 1, MPI_DOUBLE, MPI_SUM, 0, cart);

    if (rank == 0) {
        std::printf("iters=%ld restarts=%d last_step=%.3e resid=%.3e J=%.12e\n",
                    res.iters, res.restarts, res.last_step, res.resid, res.J);
        std::printf("area_cells=%.10f area_D=%.10f  int_D(w)=%.10f  max_D|w|=%.10f  nodes_in_D=%.0f\n",
                    area, problem::area_D(), st[0], st[1], st[2]);
        std::printf("time: setup=%.4f cg=%.4f total=%.4f s; inside cg: halo=%.4f allreduce=%.4f s\n",
                    tm[0], tm[1], tm[0] + tm[1], tm[2], tm[3]);
        std::printf("RESULT prog=%s M=%d N=%d procs=%d threads=%d iters=%ld time=%.6f cg_time=%.6f "
                    "intw=%.12e maxw=%.12e resid=%.3e restarts=%d decomp=%dx%d halo=%.6f allreduce=%.6f\n",
                    omp_threads() ? "hyb" : "mpi", gr.M, gr.N, size, omp_threads(), res.iters,
                    tm[0] + tm[1], tm[1], st[0], st[1], res.resid, res.restarts, dc.Px, dc.Py,
                    tm[2], tm[3]);
    }

    if (!opt.out.empty()) {
        // Сбор поля d на процессе 0: сначала геометрия блоков, потом значения.
        int meta[4] = {bl.nx, bl.ny, bl.i0, bl.j0};
        std::vector<int> metas(rank == 0 ? 4 * size : 0);
        MPI_Gather(meta, 4, MPI_INT, rank == 0 ? &metas[0] : 0, 4, MPI_INT, 0, cart);
        std::vector<double> mine((size_t)bl.nx * bl.ny);
        for (int li = 1; li <= bl.nx; ++li)
            for (int lj = 1; lj <= bl.ny; ++lj)
                mine[(size_t)(li - 1) * bl.ny + (lj - 1)] = d[bl.idx(li, lj)];
        std::vector<int> counts(size), displs(size);
        std::vector<double> all;
        if (rank == 0) {
            int tot = 0;
            for (int r = 0; r < size; ++r) {
                counts[r] = metas[4 * r] * metas[4 * r + 1];
                displs[r] = tot;
                tot += counts[r];
            }
            all.resize(tot);
        }
        MPI_Gatherv(&mine[0], (int)mine.size(), MPI_DOUBLE, rank == 0 ? &all[0] : 0,
                    &counts[0], &displs[0], MPI_DOUBLE, 0, cart);
        if (rank == 0) {
            std::vector<double> w;
            fill_v0(gr, w);
            for (int r = 0; r < size; ++r) {
                const int bnx = metas[4 * r], bny = metas[4 * r + 1];
                const int bi0 = metas[4 * r + 2], bj0 = metas[4 * r + 3];
                for (int a = 0; a < bnx; ++a)
                    for (int b = 0; b < bny; ++b)
                        w[(size_t)(bi0 + 1 + a) * (gr.N + 1) + (bj0 + 1 + b)] +=
                            all[displs[r] + (size_t)a * bny + b];
            }
            if (write_solution(opt.out, gr, w))
                std::printf("solution written to %s\n", opt.out.c_str());
            else
                std::fprintf(stderr, "cannot write %s\n", opt.out.c_str());
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
