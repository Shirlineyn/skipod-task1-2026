// Метод сопряжённых градиентов на GPU (CUDA): интерфейс для MPI-программы.
//
// Реализация — в cg_cuda.cu, собирается nvcc. Этот заголовок обычный C++
// без типов CUDA: его включает poisson_mpi.cpp, который собирает
// MPI-компилятор (на Polus — IBM XL), а с кодом для GPU связывается на этапе
// линковки.
//
// Один MPI-процесс — один блок сетки — одна видеокарта. Все векторы CG и
// коэффициенты схемы лежат в памяти GPU и считаются там же; на хост за
// итерацию уходят только скалярные произведения (2 раза по 2 числа) и края
// блока для обмена с соседями.

#ifndef CG_CUDA_H
#define CG_CUDA_H

#include "problem.h"
#include "solver.h"

// Связь между блоками для GPU-решателя. Края блока передаются одним
// буфером на хосте в порядке
//   [строка li=1 | строка li=nx | столбец lj=1 | столбец lj=ny],
// длины ny, ny, nx, nx. Ответ соседей приходит в recv в том же порядке:
//   [строка li=0 | строка li=nx+1 | столбец lj=0 | столбец lj=ny+1].
// Где соседа нет, recv не трогается (там нули — граница Gamma).
struct EdgeComm {
    virtual bool has_neighbors() = 0;
    virtual void edges(const double* send, double* recv) = 0;
    virtual void allreduce(double* a, int n) = 0;
    virtual ~EdgeComm() {}
};

// Выбрать видеокарту для процесса: local_rank — номер процесса на узле.
// Возвращает описание карты для печати; при ошибке завершает программу.
// Заодно создаёт контекст CUDA (это занимает до секунды), чтобы его
// создание не попало в замер времени.
void gpu_select(int local_rank, char* descr, int len);

struct GpuCG {
    problem::Grid gr;
    problem::Block bl;
    // Указатели на память GPU.
    double *Q, *Dinv, *B;           // коэффициенты схемы
    double *w, *r, *z, *p, *Ap;     // векторы CG; w — поправка d
    double *part, *sums;            // частичные суммы блоков и итог редукции
    double *ebuf_d;                 // края блока: 2(nx+ny) отправить + 2(nx+ny) принять
    // Закреплённая (pinned) память хоста.
    double *ebuf_h, *sums_h;
    int nblocks;                    // число блоков нитей в ядрах по узлам
    // Накопленное время: обмен краями целиком (упаковка, копирование
    // GPU <-> хост, MPI) и ожидание скалярных произведений (досчёт ядер,
    // копирование на хост, MPI_Allreduce).
    double t_halo, t_reduce;

    // Выделить память и посчитать на GPU коэффициенты блока.
    GpuCG(const problem::Grid& g, const problem::Block& b);
    ~GpuCG();
    // Решить A d = B с d(0) = 0. Логика остановки, контроль J и
    // перезапуски — как в cg_solve (solver.h).
    CGResult solve(EdgeComm& comm, const CGParams& prm);
    // Скопировать на хост поправку d и коэффициент Q (массивы bl.size()).
    void download(double* d, double* Qh) const;

private:
    GpuCG(const GpuCG&);
    GpuCG& operator=(const GpuCG&);
};

#endif
