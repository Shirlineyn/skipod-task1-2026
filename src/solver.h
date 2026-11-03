// Метод сопряжённых градиентов с диагональным предобуславливанием.
//
// Решатель работает с одним блоком сетки (problem::Block) и ничего не знает
// о том, есть ли другие блоки. Всё, что требует связи между блоками, вынесено
// в интерфейс Comm:
//   halo(v)          — заполнить теневые узлы v значениями соседних блоков;
//   allreduce(a, n)  — сложить n чисел по всем блокам.
// В последовательной и OpenMP-программе это пустые операции (блок один,
// его теневые узлы — граница Gamma, где d = 0). В MPI-программе — обмены.
//
// Каждый цикл по узлам распараллелен директивой OpenMP по строкам li.
// Вызовы Comm стоят между циклами, вне параллельных областей, поэтому для
// гибридной версии достаточно уровня MPI_THREAD_FUNNELED.
//
// Работа над строкой li вынесена в функции row_*: все скаляры передаются
// по значению, массивы — указателями на начало строки с __restrict__.
// Иначе общие (shared) переменные параллельного цикла видны компилятору
// через указатели, он не может доказать, что запись в r[k] их не меняет,
// и отказывается векторизовать внутренний цикл. На этом OpenMP-версия
// с одной нитью теряла вдвое против последовательной (проверено на clang).

#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include "problem.h"

struct Comm {
    virtual void halo(double* v) = 0;
    virtual void allreduce(double* a, int n) = 0;
    virtual ~Comm() {}
};

struct CGParams {
    double delta;   // критерий остановки ||w(k+1) - w(k)||_E < delta
    long maxit;     // предел числа итераций (не больше числа неизвестных)
    int verbose;    // печатать каждую verbose-ю итерацию; 0 — молчать
    int rank;       // печатает только процесс 0
};

enum CGStatus { CG_CONVERGED, CG_EXACT_RESIDUAL, CG_MAXIT, CG_BREAKDOWN, CG_RESTART_LIMIT };

inline const char* cg_status_name(CGStatus status)
{
    switch (status) {
    case CG_CONVERGED: return "converged";
    case CG_EXACT_RESIDUAL: return "exact_residual";
    case CG_MAXIT: return "max_iterations";
    case CG_RESTART_LIMIT: return "restart_limit";
    default: return "breakdown";
    }
}

inline bool cg_finite(double x)
{
    return x == x && std::fabs(x) <= std::numeric_limits<double>::max();
}

struct CGResult {
    long iters;        // число выполненных итераций
    int restarts;      // перезапусков из-за нарушения монотонности J
    double last_step;  // ||w(k+1) - w(k)||_E на последней итерации
    double J;          // значение функционала на последней итерации
    double resid;      // ||B - A d||_E по окончании
    CGStatus status;
    CGResult() : iters(0), restarts(0), last_step(0), J(0), resid(0), status(CG_MAXIT) {}
    bool converged() const { return status == CG_CONVERGED || status == CG_EXACT_RESIDUAL; }
};

inline void finish_cg_result(CGResult& res)
{
    if (!cg_finite(res.resid) || !cg_finite(res.J) || !cg_finite(res.last_step))
        res.status = CG_BREAKDOWN;
    // Нулевое направление само по себе не доказывает точного решения:
    // проверяем заново вычисленную невязку, а не только рекуррентную.
    else if (res.resid == 0.0)
        res.status = CG_EXACT_RESIDUAL;
}

// ---- Операции над одной строкой li; индекс j = 1..ny, j = 0 и ny+1 — тень.

// y = A v; накапливает (y, v) и (v, v) по строке (без веса h1*h2).
// vm, vc, vp — строки li-1, li, li+1 вектора v.
static inline void row_A(int ny, double ih1, double ih2, const double* __restrict__ q,
                         const double* __restrict__ vm, const double* __restrict__ vc,
                         const double* __restrict__ vp, double* __restrict__ y,
                         double& yv, double& vv)
{
    double a = 0.0, b = 0.0;
    for (int j = 1; j <= ny; ++j) {
        const double t = (2.0 * vc[j] - vp[j] - vm[j]) * ih1 +
                         (2.0 * vc[j] - vc[j + 1] - vc[j - 1]) * ih2 + q[j] * vc[j];
        y[j] = t;
        a += t * vc[j];
        b += vc[j] * vc[j];
    }
    yv += a;
    vv += b;
}

// Шаг CG: w += alpha p, r -= alpha Ap, z = D^{-1} r;
// накапливает (z, r) и (B + r, w) — последнее нужно для J(w).
static inline void row_step(int ny, double alpha, const double* __restrict__ p,
                            const double* __restrict__ Ap, const double* __restrict__ dinv,
                            const double* __restrict__ B, double* __restrict__ w,
                            double* __restrict__ r, double* __restrict__ z,
                            double& zr, double& Jw)
{
    double a = 0.0, b = 0.0;
    for (int j = 1; j <= ny; ++j) {
        w[j] += alpha * p[j];
        r[j] -= alpha * Ap[j];
        z[j] = dinv[j] * r[j];
        a += z[j] * r[j];
        b += (B[j] + r[j]) * w[j];
    }
    zr += a;
    Jw += b;
}

// (Пере)запуск по уже посчитанному Aw: r = B - Aw, z = D^{-1} r, p = z.
static inline void row_restart(int ny, const double* __restrict__ Aw,
                               const double* __restrict__ dinv, const double* __restrict__ B,
                               const double* __restrict__ w, double* __restrict__ r,
                               double* __restrict__ z, double* __restrict__ p,
                               double& zr, double& Jw)
{
    double a = 0.0, b = 0.0;
    for (int j = 1; j <= ny; ++j) {
        r[j] = B[j] - Aw[j];
        z[j] = dinv[j] * r[j];
        p[j] = z[j];
        a += z[j] * r[j];
        b += (B[j] + r[j]) * w[j];
    }
    zr += a;
    Jw += b;
}

// p = z + beta p.
static inline void row_dir(int ny, double beta, const double* __restrict__ z,
                           double* __restrict__ p)
{
    for (int j = 1; j <= ny; ++j) p[j] = z[j] + beta * p[j];
}

// y += a x.
static inline void row_axpy(int ny, double a, const double* __restrict__ x,
                            double* __restrict__ y)
{
    for (int j = 1; j <= ny; ++j) y[j] += a * x[j];
}

// (B - Ax, B - Ax) по строке.
static inline double row_resid2(int ny, const double* __restrict__ B,
                                const double* __restrict__ Ax)
{
    double s = 0.0;
    for (int j = 1; j <= ny; ++j) {
        const double t = B[j] - Ax[j];
        s += t * t;
    }
    return s;
}

// ---- Операции над блоком: параллельный цикл по строкам.

// y = A v. Теневые узлы v должны быть заполнены. Возвращает (Av, v), (v, v)
// по блоку с весом h1*h2 (ещё не просуммированные по блокам).
static inline void block_A(const problem::Grid& gr, const problem::Block& bl,
                           const problem::Coeffs& c, const double* v, double* y,
                           double* out2)
{
    const double ih1 = 1.0 / (gr.h1 * gr.h1), ih2 = 1.0 / (gr.h2 * gr.h2);
    const int s = bl.stride, nx = bl.nx, ny = bl.ny;
    const double* q = &c.Q[0];
    double yv = 0.0, vv = 0.0;
#pragma omp parallel for schedule(static) reduction(+:yv,vv)
    for (int li = 1; li <= nx; ++li) {
        const int o = li * s;
        row_A(ny, ih1, ih2, q + o, v + o - s, v + o, v + o + s, y + o, yv, vv);
    }
    out2[0] = yv * gr.h1 * gr.h2;
    out2[1] = vv * gr.h1 * gr.h2;
}

// Решение A d = B. На входе d — начальное приближение (обычно 0),
// на выходе — приближённое решение.
inline CGResult cg_solve(const problem::Grid& gr, const problem::Block& bl,
                         const problem::Coeffs& c, std::vector<double>& d,
                         Comm& comm, const CGParams& prm)
{
    const int sz = bl.size(), s = bl.stride, nx = bl.nx, ny = bl.ny;
    const double cell = gr.h1 * gr.h2;  // вес скалярного произведения (8)
    const double* Dinv = &c.Dinv[0];
    const double* B = &c.B[0];
    std::vector<double> r_(sz, 0.0), z_(sz, 0.0), p_(sz, 0.0), Ap_(sz, 0.0);
    double* r = &r_[0];
    double* z = &z_[0];
    double* p = &p_[0];
    double* Ap = &Ap_[0];
    double* w = &d[0];

    CGResult res;
    res.iters = 0;
    res.restarts = 0;
    res.last_step = 0.0;
    res.J = 0.0;
    res.resid = 0.0;

    double zr = 0.0, J_old = 0.0;
    bool need_restart = true;  // первая итерация — скорейший спуск

    while (true) {
        if (need_restart) {
            // r = B - A w, z = D^{-1} r, p = z; (z, r) и J(w) = -0.5 (B + r, w).
            double tmp[2];
            comm.halo(w);
            block_A(gr, bl, c, w, Ap, tmp);
            double zr_l = 0.0, Jw_l = 0.0;
#pragma omp parallel for schedule(static) reduction(+:zr_l,Jw_l)
            for (int li = 1; li <= nx; ++li) {
                const int o = li * s;
                row_restart(ny, Ap + o, Dinv + o, B + o, w + o, r + o, z + o, p + o, zr_l, Jw_l);
            }
            double acc[2] = {zr_l * cell, Jw_l * cell};
            comm.allreduce(acc, 2);
            zr = acc[0];
            J_old = -0.5 * acc[1];
            res.J = J_old;
            need_restart = false;
        }
        if (res.iters >= prm.maxit) break;
        if (!cg_finite(zr) || zr < 0.0 || !cg_finite(J_old)) {
            res.status = CG_BREAKDOWN;
            break;
        }

        // 1) Ap = A p, скалярные произведения (Ap, p) и (p, p).
        comm.halo(p);
        double acc1[2];
        block_A(gr, bl, c, p, Ap, acc1);
        comm.allreduce(acc1, 2);
        const double pAp = acc1[0], pp = acc1[1];
        if (!(pAp > 0.0) || !cg_finite(pAp) || !(pp > 0.0) || !cg_finite(pp)) {
            res.status = CG_BREAKDOWN;
            break;
        }
        const double alpha = zr / pAp;
        if (!(alpha > 0.0) || !cg_finite(alpha)) {
            res.status = CG_BREAKDOWN;
            break;
        }

        // 2) w += alpha p, r -= alpha Ap, z = D^{-1} r; (z, r) и J(w).
        double zr_l = 0.0, Jw_l = 0.0;
#pragma omp parallel for schedule(static) reduction(+:zr_l,Jw_l)
        for (int li = 1; li <= nx; ++li) {
            const int o = li * s;
            row_step(ny, alpha, p + o, Ap + o, Dinv + o, B + o, w + o, r + o, z + o, zr_l, Jw_l);
        }
        double acc2[2] = {zr_l * cell, Jw_l * cell};
        comm.allreduce(acc2, 2);
        const double zr_new = acc2[0];
        const double J_new = -0.5 * acc2[1];
        ++res.iters;
        const double previous_step = res.last_step;
        res.last_step = std::fabs(alpha) * std::sqrt(pp);  // ||w(k+1) - w(k)||_E
        res.J = J_new;
        if (!cg_finite(zr_new) || zr_new < 0.0 || !cg_finite(J_new) || !cg_finite(res.last_step)) {
            res.status = CG_BREAKDOWN;
            break;
        }

        if (prm.verbose > 0 && prm.rank == 0 && res.iters % prm.verbose == 0)
            std::printf("  it %6ld  step %.3e  J %.12e\n", res.iters, res.last_step, J_new);

        // Контроль монотонности J. Порог относительный: J вычисляется суммой
        // по всем узлам с ошибкой округления порядка 1e-13 |J|; рост меньше
        // порога — шум, а не нарушение сходимости.
        if (J_new > J_old + 1e-10 * std::fabs(J_old)) {
            ++res.restarts;
            if (prm.rank == 0)
                std::printf("  it %ld: J grew (%.15e -> %.15e), restart from w(k)\n",
                            res.iters, J_old, J_new);
            // Откат к w(k), на которой монотонность ещё соблюдалась.
#pragma omp parallel for schedule(static)
            for (int li = 1; li <= nx; ++li) row_axpy(ny, -alpha, p + li * s, w + li * s);
            res.J = J_old;
            res.last_step = previous_step;
            need_restart = true;
            if (res.restarts > 20) {
                res.status = CG_RESTART_LIMIT;
                break;
            }
            continue;
        }
        J_old = J_new;

        if (res.last_step < prm.delta) {
            res.status = CG_CONVERGED;
            break;
        }

        // 3) p = z + beta p.
        const double beta = zr_new / zr;
        if (!cg_finite(beta)) {
            res.status = CG_BREAKDOWN;
            break;
        }
        zr = zr_new;
#pragma omp parallel for schedule(static)
        for (int li = 1; li <= nx; ++li) row_dir(ny, beta, z + li * s, p + li * s);
    }

    // Контрольная невязка, посчитанная заново, а не накопленная рекуррентно.
    double tmp[2];
    comm.halo(w);
    block_A(gr, bl, c, w, Ap, tmp);
    double rr_l = 0.0;
#pragma omp parallel for schedule(static) reduction(+:rr_l)
    for (int li = 1; li <= nx; ++li) rr_l += row_resid2(ny, B + li * s, Ap + li * s);
    double acc[1] = {rr_l * cell};
    comm.allreduce(acc, 1);
    res.resid = std::sqrt(acc[0]);
    finish_cg_result(res);
    return res;
}

// Сводные характеристики решения w = v0 + d по узлам блока, лежащим в D:
// [0] сумма w*h1*h2 (грубое приближение интеграла u по D), [1] max |w|,
// [2] число узлов в D. По блокам их складывает (а [1] — берёт max) вызывающий.
inline void solution_stats(const problem::Grid& gr, const problem::Block& bl,
                           const std::vector<double>& d, double* out)
{
    double sum = 0.0, mx = 0.0, cnt = 0.0;
    for (int li = 1; li <= bl.nx; ++li)
        for (int lj = 1; lj <= bl.ny; ++lj) {
            const double x = gr.x(bl.i0 + li), y = gr.y(bl.j0 + lj);
            if (!problem::in_D(x, y)) continue;
            const double w = problem::v0(x, y) + d[bl.idx(li, lj)];
            sum += w;
            mx = std::max(mx, std::fabs(w));
            cnt += 1.0;
        }
    out[0] = sum * gr.h1 * gr.h2;
    out[1] = mx;
    out[2] = cnt;
}

#endif
