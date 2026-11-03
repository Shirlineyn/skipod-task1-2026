// Геометрические домены и единственное владение внутренними неизвестными.
// Общие границы принадлежат двум замкнутым доменам; неизвестную считает
// ровно один процесс. Теневые слои не используются для проверки формы домена.
#ifndef DECOMP_H
#define DECOMP_H
#include <algorithm>

struct Decomp {
    int Px, Py;
    bool ok;       // отношение чисел узлов геометрического домена в [1/2,2]
    double worst;  // max(nx/ny, ny/nx) для узлов доменов, включая общую границу
};

// Размер и смещение части c из parts при делении n узлов поровну.
inline void split(int n, int parts, int c, int& len, int& off)
{
    const int base = n / parts, rem = n % parts;
    len = base + (c < rem ? 1 : 0);
    off = c * base + (c < rem ? c : rem);
}

inline Decomp choose_decomp(int P, int nxg, int nyg)
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
        // Геометрические домены покрывают Pi и пересекаются только границами.
        // На оси: [off, off+len] для не последнего, [off, M] для последнего.
        // Число узлов домена = len+1+(last?1:0); крайние размеры ниже.
        const double xmin = (nxg + 1) / Px + 1, xmax = (nxg + Px) / Px + 1;
        const double ymin = (nyg + 1) / Py + 1, ymax = (nyg + Py) / Py + 1;
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

#endif
