// Метод сопряжённых градиентов на GPU: ядра CUDA и управляющий код.
//
// Раскладка векторов в памяти GPU та же, что на CPU (problem::Block):
// (nx+2) x (ny+2) с теневым слоем, idx = li*(ny+2) + lj. Нить ядра —
// один внутренний узел; соседние нити варпа берут соседние lj, поэтому
// чтения и записи идут сплошными отрезками памяти (coalesced).
//
// Итерация CG — те же три шага, что в cg_solve (solver.h):
//   k_A     Ap = A p, суммы (Ap, p), (p, p);
//   k_step  w += alpha p, r -= alpha Ap, z = D^{-1} r, суммы (z, r), (B + r, w);
//   k_dir   p = z + beta p.
// Скалярные произведения считаются в два этапа: каждый блок нитей
// складывает свои узлы в разделяемой памяти и пишет частичную сумму,
// затем одно ядро k_finish складывает частичные суммы в фиксированном
// порядке. Атомарные сложения не используются, чтобы результат не зависел
// от порядка исполнения блоков: повторный запуск даёт те же цифры.
// Итог (два числа) копируется на хост — alpha, beta, контроль J и
// остановка решаются на CPU, как и в MPI-версии, и там же делается
// MPI_Allreduce.
//
// Обмен теневыми слоями — через хост: k_pack собирает четыре края блока в
// один буфер, он копируется на хост, процессы обмениваются им по MPI,
// ответ копируется обратно и k_unpack раскладывает его в теневые слои.
// CUDA-aware MPI не используется: в Spectrum MPI он включается отдельным
// ключом запуска, а краёв мало (2(nx + ny) чисел против nx*ny узлов).

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "cg_cuda.h"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(e_));                       \
            std::exit(3);                                                         \
        }                                                                         \
    } while (0)

namespace {

// Блок нитей 32 x 8: 32 узла по lj (один варп — одна строка) и 8 строк li.
const int BX = 32, BY = 8, NT = BX * BY;
const int FIN = 1024;  // нитей в k_finish

struct Dims {
    int nx, ny, s;  // s = ny + 2
};

// Сумма двух величин по блоку нитей; частичные суммы блока — в
// pa[номер блока], pb[номер блока].
__device__ inline void block_sum2(double a, double b, double* pa, double* pb)
{
    __shared__ double sa[NT], sb[NT];
    const int t = threadIdx.y * BX + threadIdx.x;
    sa[t] = a;
    sb[t] = b;
    __syncthreads();
    for (int k = NT / 2; k > 0; k >>= 1) {
        if (t < k) {
            sa[t] += sa[t + k];
            sb[t] += sb[t + k];
        }
        __syncthreads();
    }
    if (t == 0) {
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;
        pa[bid] = sa[0];
        pb[bid] = sb[0];
    }
}

// Узел нити: li = 1..nx, lj = 1..ny; false — нить вне блока (хвост сетки).
__device__ inline bool node(const Dims& d, int& k)
{
    const int lj = blockIdx.x * BX + threadIdx.x + 1;
    const int li = blockIdx.y * BY + threadIdx.y + 1;
    k = li * d.s + lj;
    return li <= d.nx && lj <= d.ny;
}

__global__ void k_coeffs(problem::Grid gr, Dims d, int i0, int j0, double* Q, double* Dinv,
                         double* B)
{
    const int lj = blockIdx.x * BX + threadIdx.x + 1;
    const int li = blockIdx.y * BY + threadIdx.y + 1;
    if (li > d.nx || lj > d.ny) return;
    const int k = li * d.s + lj;
    problem::node_coeffs(gr, i0 + li, j0 + lj, Q[k], Dinv[k], B[k]);
}

// y = A v; частичные суммы (y, v) и (v, v).
__global__ void k_A(Dims d, double ih1, double ih2, const double* __restrict__ q,
                    const double* __restrict__ v, double* __restrict__ y, double* part,
                    int nb)
{
    int k;
    double a = 0.0, b = 0.0;
    if (node(d, k)) {
        const double vc = v[k];
        const double t = (2.0 * vc - v[k + d.s] - v[k - d.s]) * ih1 +
                         (2.0 * vc - v[k + 1] - v[k - 1]) * ih2 + q[k] * vc;
        y[k] = t;
        a = t * vc;
        b = vc * vc;
    }
    block_sum2(a, b, part, part + nb);
}

// w += alpha p, r -= alpha Ap, z = D^{-1} r; частичные суммы (z, r), (B + r, w).
__global__ void k_step(Dims d, double alpha, const double* __restrict__ p,
                       const double* __restrict__ Ap, const double* __restrict__ dinv,
                       const double* __restrict__ B, double* __restrict__ w,
                       double* __restrict__ r, double* __restrict__ z, double* part, int nb)
{
    int k;
    double a = 0.0, b = 0.0;
    if (node(d, k)) {
        const double wk = w[k] + alpha * p[k];
        const double rk = r[k] - alpha * Ap[k];
        const double zk = dinv[k] * rk;
        w[k] = wk;
        r[k] = rk;
        z[k] = zk;
        a = zk * rk;
        b = (B[k] + rk) * wk;
    }
    block_sum2(a, b, part, part + nb);
}

// (Пере)запуск по посчитанному Aw: r = B - Aw, z = D^{-1} r, p = z;
// частичные суммы (z, r), (B + r, w).
__global__ void k_restart(Dims d, const double* __restrict__ Aw,
                          const double* __restrict__ dinv, const double* __restrict__ B,
                          const double* __restrict__ w, double* __restrict__ r,
                          double* __restrict__ z, double* __restrict__ p, double* part, int nb)
{
    int k;
    double a = 0.0, b = 0.0;
    if (node(d, k)) {
        const double rk = B[k] - Aw[k];
        const double zk = dinv[k] * rk;
        r[k] = rk;
        z[k] = zk;
        p[k] = zk;
        a = zk * rk;
        b = (B[k] + rk) * w[k];
    }
    block_sum2(a, b, part, part + nb);
}

// p = z + beta p.
__global__ void k_dir(Dims d, double beta, const double* __restrict__ z, double* __restrict__ p)
{
    int k;
    if (node(d, k)) p[k] = z[k] + beta * p[k];
}

// y += a x.
__global__ void k_axpy(Dims d, double a, const double* __restrict__ x, double* __restrict__ y)
{
    int k;
    if (node(d, k)) y[k] += a * x[k];
}

// Частичные суммы (B - Ax, B - Ax); вторая величина не нужна.
__global__ void k_resid2(Dims d, const double* __restrict__ B, const double* __restrict__ Ax,
                         double* part, int nb)
{
    int k;
    double a = 0.0;
    if (node(d, k)) {
        const double t = B[k] - Ax[k];
        a = t * t;
    }
    block_sum2(a, 0.0, part, part + nb);
}

// Сложить nb частичных сумм каждой из двух величин; итог — out[0], out[1].
// Один блок: нить t складывает элементы t, t + FIN, ... по порядку, затем
// сложение деревом. Порядок фиксирован — результат воспроизводим.
__global__ void k_finish(const double* part, int nb, double* out)
{
    __shared__ double sa[FIN], sb[FIN];
    const int t = threadIdx.x;
    double a = 0.0, b = 0.0;
    for (int i = t; i < nb; i += FIN) {
        a += part[i];
        b += part[nb + i];
    }
    sa[t] = a;
    sb[t] = b;
    __syncthreads();
    for (int k = FIN / 2; k > 0; k >>= 1) {
        if (t < k) {
            sa[t] += sa[t + k];
            sb[t] += sb[t + k];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[0] = sa[0];
        out[1] = sb[0];
    }
}

// Края блока -> буфер: [строка 1 | строка nx | столбец 1 | столбец ny].
__global__ void k_pack(Dims d, const double* __restrict__ v, double* __restrict__ buf)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = d.nx, ny = d.ny, s = d.s;
    if (t < ny) buf[t] = v[1 * s + t + 1];
    else if (t < 2 * ny) buf[t] = v[nx * s + (t - ny) + 1];
    else if (t < 2 * ny + nx) buf[t] = v[(t - 2 * ny + 1) * s + 1];
    else if (t < 2 * ny + 2 * nx) buf[t] = v[(t - 2 * ny - nx + 1) * s + ny];
}

// Буфер -> теневые слои: [строка 0 | строка nx+1 | столбец 0 | столбец ny+1].
__global__ void k_unpack(Dims d, const double* __restrict__ buf, double* __restrict__ v)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = d.nx, ny = d.ny, s = d.s;
    if (t < ny) v[0 * s + t + 1] = buf[t];
    else if (t < 2 * ny) v[(nx + 1) * s + (t - ny) + 1] = buf[t];
    else if (t < 2 * ny + nx) v[(t - 2 * ny + 1) * s + 0] = buf[t];
    else if (t < 2 * ny + 2 * nx) v[(t - 2 * ny - nx + 1) * s + ny + 1] = buf[t];
}

double now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

}  // namespace

void gpu_select(int local_rank, char* descr, int len)
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    if (n == 0) {
        std::fprintf(stderr, "no CUDA devices\n");
        std::exit(3);
    }
    const int dev = local_rank % n;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(0));  // создать контекст сейчас, а не в замере
    cudaDeviceProp pr;
    CUDA_CHECK(cudaGetDeviceProperties(&pr, dev));
    char bus[32];
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
    std::snprintf(descr, len, "device %d of %d: %s, sm_%d%d, %d SMs, bus %s", dev, n, pr.name,
                  pr.major, pr.minor, pr.multiProcessorCount, bus);
}

GpuCG::GpuCG(const problem::Grid& g, const problem::Block& b)
    : gr(g), bl(b), t_halo(0.0), t_reduce(0.0)
{
    const size_t sz = (size_t)bl.size() * sizeof(double);
    double** vecs[8] = {&Q, &Dinv, &B, &w, &r, &z, &p, &Ap};
    for (int i = 0; i < 8; ++i) {
        CUDA_CHECK(cudaMalloc(vecs[i], sz));
        // Теневые слои и узлы на Gamma должны быть нулями: d = 0 на Gamma.
        CUDA_CHECK(cudaMemset(*vecs[i], 0, sz));
    }
    const int gx = (bl.ny + BX - 1) / BX, gy = (bl.nx + BY - 1) / BY;
    nblocks = gx * gy;
    CUDA_CHECK(cudaMalloc(&part, 2 * (size_t)nblocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sums, 2 * sizeof(double)));
    const size_t ne = 2 * (size_t)(bl.nx + bl.ny);
    CUDA_CHECK(cudaMalloc(&ebuf_d, 2 * ne * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ebuf_h, 2 * ne * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&sums_h, 2 * sizeof(double)));
    std::memset(ebuf_h, 0, 2 * ne * sizeof(double));

    Dims d = {bl.nx, bl.ny, bl.stride};
    k_coeffs<<<dim3(gx, gy), dim3(BX, BY)>>>(gr, d, bl.i0, bl.j0, Q, Dinv, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

GpuCG::~GpuCG()
{
    double* vecs[11] = {Q, Dinv, B, w, r, z, p, Ap, part, sums, ebuf_d};
    for (int i = 0; i < 11; ++i) cudaFree(vecs[i]);
    cudaFreeHost(ebuf_h);
    cudaFreeHost(sums_h);
}

void GpuCG::download(double* d, double* Qh) const
{
    const size_t sz = (size_t)bl.size() * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d, w, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Qh, Q, sz, cudaMemcpyDeviceToHost));
}

CGResult GpuCG::solve(EdgeComm& comm, const CGParams& prm)
{
    const Dims d = {bl.nx, bl.ny, bl.stride};
    const dim3 grid((bl.ny + BX - 1) / BX, (bl.nx + BY - 1) / BY), thr(BX, BY);
    const int nb = nblocks;
    const double cell = gr.h1 * gr.h2;  // вес скалярного произведения (8)
    const double ih1 = 1.0 / (gr.h1 * gr.h1), ih2 = 1.0 / (gr.h2 * gr.h2);
    const int ne = 2 * (bl.nx + bl.ny);
    const bool exchange = comm.has_neighbors();

    // Теневые слои v от соседей. Без соседей теневые слои — Gamma, там нули.
    struct Halo {
        GpuCG& g;
        EdgeComm& c;
        Dims d;
        int ne;
        bool on;
        void operator()(double* v)
        {
            if (!on) return;
            const double t = now();
            const int nt = 256, nbk = (ne + nt - 1) / nt;
            k_pack<<<nbk, nt>>>(d, v, g.ebuf_d);
            CUDA_CHECK(cudaMemcpy(g.ebuf_h, g.ebuf_d, ne * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            c.edges(g.ebuf_h, g.ebuf_h + ne);
            CUDA_CHECK(cudaMemcpy(g.ebuf_d + ne, g.ebuf_h + ne, ne * sizeof(double),
                                  cudaMemcpyHostToDevice));
            k_unpack<<<nbk, nt>>>(d, g.ebuf_d + ne, v);
            g.t_halo += now() - t;
        }
    } halo = {*this, comm, d, ne, exchange};

    // Досчитать частичные суммы и сложить по всем процессам. Копирование
    // на хост ждёт окончания всех ядер, запущенных до него.
    struct Reduce {
        GpuCG& g;
        EdgeComm& c;
        int nb;
        double cell;
        void operator()(double* acc)
        {
            const double t = now();
            k_finish<<<1, FIN>>>(g.part, nb, g.sums);
            CUDA_CHECK(cudaMemcpy(g.sums_h, g.sums, 2 * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            acc[0] = g.sums_h[0] * cell;
            acc[1] = g.sums_h[1] * cell;
            c.allreduce(acc, 2);
            g.t_reduce += now() - t;
        }
    } reduce = {*this, comm, nb, cell};

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
            halo(w);
            k_A<<<grid, thr>>>(d, ih1, ih2, Q, w, Ap, part, nb);
            k_restart<<<grid, thr>>>(d, Ap, Dinv, B, w, r, z, p, part, nb);
            double acc[2];
            reduce(acc);
            zr = acc[0];
            J_old = -0.5 * acc[1];
            need_restart = false;
        }
        if (res.iters >= prm.maxit) break;

        // 1) Ap = A p, (Ap, p) и (p, p).
        halo(p);
        k_A<<<grid, thr>>>(d, ih1, ih2, Q, p, Ap, part, nb);
        double acc1[2];
        reduce(acc1);
        const double pAp = acc1[0], pp = acc1[1];
        if (!(pAp > 0.0)) break;  // невязка обнулилась точно — шагать некуда
        const double alpha = zr / pAp;

        // 2) w += alpha p, r -= alpha Ap, z = D^{-1} r; (z, r) и J(w).
        k_step<<<grid, thr>>>(d, alpha, p, Ap, Dinv, B, w, r, z, part, nb);
        double acc2[2];
        reduce(acc2);
        const double zr_new = acc2[0];
        const double J_new = -0.5 * acc2[1];
        ++res.iters;
        res.last_step = std::fabs(alpha) * std::sqrt(pp);  // ||w(k+1) - w(k)||_E
        res.J = J_new;

        if (prm.verbose > 0 && prm.rank == 0 && res.iters % prm.verbose == 0)
            std::printf("  it %6ld  step %.3e  J %.12e\n", res.iters, res.last_step, J_new);

        // Контроль монотонности J — как в cg_solve.
        if (J_new > J_old + 1e-10 * std::fabs(J_old)) {
            ++res.restarts;
            if (prm.rank == 0)
                std::printf("  it %ld: J grew (%.15e -> %.15e), restart from w(k)\n",
                            res.iters, J_old, J_new);
            k_axpy<<<grid, thr>>>(d, -alpha, p, w);
            need_restart = true;
            if (res.restarts > 20) break;
            continue;
        }
        J_old = J_new;

        if (res.last_step < prm.delta) break;

        // 3) p = z + beta p.
        const double beta = zr_new / zr;
        zr = zr_new;
        k_dir<<<grid, thr>>>(d, beta, z, p);
    }

    // Контрольная невязка, посчитанная заново.
    halo(w);
    k_A<<<grid, thr>>>(d, ih1, ih2, Q, w, Ap, part, nb);
    k_resid2<<<grid, thr>>>(d, B, Ap, part, nb);
    double acc[2];
    reduce(acc);
    res.resid = std::sqrt(acc[0]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return res;
}
