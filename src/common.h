// Общее для обеих программ: разбор аргументов, таймер, запись решения.

#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <limits>
#include <string>
#include <vector>
#include <sys/time.h>
#include <set>
#ifdef __linux__
#include <sched.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "problem.h"

struct Options {
    int M, N;
    double delta;
    long maxit;        // 1..(M-1)(N-1), по умолчанию верхняя граница
    int verbose;
    int binding;       // печатать, на каких CPU сидят нити
    std::string out;   // файл для решения; пусто — не писать
};

inline void usage(const char* prog)
{
    std::fprintf(stderr,
                 "usage: %s M N [--delta D] [--maxit K] [--verbose K] [--out FILE] [--binding]\n"
                 "  M, N      число отрезков сетки по x и y\n"
                 "  --delta   критерий остановки ||w(k+1)-w(k)||_E < D (по умолчанию 1e-6)\n"
                 "  --maxit   предел итераций (по умолчанию (M-1)(N-1))\n"
                 "  --verbose печатать каждую K-ю итерацию\n"
                 "  --out     записать решение на всей сетке в двоичный файл\n"
                 "  --binding напечатать логические CPU нитей OpenMP после расчёта\n",
                 prog);
}

inline bool parse_integer(const char* text, long low, long high, long& value)
{
    if (!text[0] || text[0] < '0' || text[0] > '9') return false;
    char* end = 0;
    errno = 0;
    value = std::strtol(text, &end, 10);
    return errno == 0 && *end == '\0' && value >= low && value <= high;
}

inline bool parse_options(int argc, char** argv, Options& o)
{
    if (argc < 3) return false;
    long m, n;
    if (!parse_integer(argv[1], 2, INT_MAX - 1, m) ||
        !parse_integer(argv[2], 2, INT_MAX - 1, n)) return false;
    // Индексы блока, размеры сообщений MPI и смещения Gatherv имеют тип int.
    // Проверяем произведение делением, до сложения/умножения в int.
    if (m + 1 > INT_MAX / (n + 1)) return false;
    o.M = (int)m;
    o.N = (int)n;
    const long unknowns = (m - 1) * (n - 1);
    o.delta = 1e-6;
    o.maxit = unknowns;
    o.verbose = 0;
    o.binding = 0;
    o.out.clear();
    for (int a = 3; a < argc; ++a) {
        if (!std::strcmp(argv[a], "--delta") && a + 1 < argc) {
            const char* text = argv[++a];
            char* end = 0;
            errno = 0;
            o.delta = std::strtod(text, &end);
            if (errno || end == text || *end || !(o.delta > 0.0) ||
                o.delta > std::numeric_limits<double>::max()) return false;
        }
        else if (!std::strcmp(argv[a], "--maxit") && a + 1 < argc) {
            if (!parse_integer(argv[++a], 1, unknowns, o.maxit)) return false;
        }
        else if (!std::strcmp(argv[a], "--verbose") && a + 1 < argc) {
            long value;
            if (!parse_integer(argv[++a], 0, INT_MAX, value)) return false;
            o.verbose = (int)value;
        }
        else if (!std::strcmp(argv[a], "--out") && a + 1 < argc) {
            o.out = argv[++a];
            if (o.out.empty()) return false;
        }
        else if (!std::strcmp(argv[a], "--binding")) o.binding = 1;
        else return false;
    }
    return true;
}

inline double wtime()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
#endif
}

inline int omp_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;  // собрано без OpenMP
#endif
}

// Где сейчас работают нити: "cpu/ядро" для каждой нити и число разных
// физических ядер. На POWER8 у ядра 8 аппаратных нитей, ядро = cpu / 8
// (CPUS_PER_CORE). Две нити процесса на одном ядре делят его конвейер —
// так проверяется, разошлись ли нити по ядрам без явной привязки.
const int CPUS_PER_CORE = 8;

inline std::string thread_cpus()
{
    const int n = omp_threads() > 0 ? omp_threads() : 1;
    std::vector<int> cpu(n, -1);
#pragma omp parallel
    {
#ifdef _OPENMP
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
#ifdef __linux__
        cpu[t] = sched_getcpu();
#else
        cpu[t] = -1;  // sched_getcpu есть только в Linux
#endif
    }
    std::string s;
    std::set<int> cores;
    char buf[32];
    for (int t = 0; t < n; ++t) {
        std::snprintf(buf, sizeof buf, "%s%d/%d", t ? " " : "", cpu[t], cpu[t] / CPUS_PER_CORE);
        s += buf;
        cores.insert(cpu[t] / CPUS_PER_CORE);
    }
    std::snprintf(buf, sizeof buf, "  [%d cores for %d threads]", (int)cores.size(), n);
    return s + buf;
}

// Двоичный файл решения: int32 M, N; double A1, B1, A2, B2;
// затем (M+1)*(N+1) значений w(x_i, y_j) в порядке i*(N+1)+j, включая границу.
inline bool write_solution(const std::string& path, const problem::Grid& gr,
                           const std::vector<double>& w)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    int hdr[2] = {gr.M, gr.N};
    double box[4] = {problem::A1, problem::B1, problem::A2, problem::B2};
    bool ok = std::fwrite(hdr, sizeof(int), 2, fp) == 2;
    ok = (std::fwrite(box, sizeof(double), 4, fp) == 4) && ok;
    ok = (std::fwrite(&w[0], sizeof(double), w.size(), fp) == w.size()) && ok;
    // Ошибка буферизованной записи может появиться только при fclose.
    if (std::fclose(fp) != 0) ok = false;
    return ok;
}

// w = v0 во всех узлах сетки (M+1)x(N+1); поправку d во внутренних узлах
// добавляет вызывающий код.
inline void fill_v0(const problem::Grid& gr, std::vector<double>& w)
{
    w.assign((size_t)(gr.M + 1) * (gr.N + 1), 0.0);
    for (int i = 0; i <= gr.M; ++i)
        for (int j = 0; j <= gr.N; ++j)
            w[(size_t)i * (gr.N + 1) + j] = problem::v0(gr.x(i), gr.y(j));
}

#endif
