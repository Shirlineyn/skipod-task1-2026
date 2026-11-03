// Проверки реальных геометрических интервалов, а не только формулы экстремумов.
#include <cstdio>
#include <cmath>
#include "decomp.h"

double inspect(int M, int N, int px, int py)
{
    double worst = 1;
    for (int axis = 0; axis < 2; ++axis) {
        const int m = axis ? N : M, p = axis ? py : px;
        int last = 0, lo = m + 1, hi = 0, owned = 0;
        for (int c = 0; c < p; ++c) {
            int len, off;
            split(m - 1, p, c, len, off);
            if (len < 1 || off != owned || off != last) return -1;
            const int end = c == p - 1 ? m : off + len;
            lo = std::min(lo, end - off + 1); hi = std::max(hi, end - off + 1);
            owned += len; last = end;
        }
        if (last != m || owned != m - 1 || hi - lo > 1) return -1;
    }
    for (int x = 0; x < px; ++x)
        for (int y = 0; y < py; ++y) {
            int nx, ox, ny, oy;
            split(M - 1, px, x, nx, ox); split(N - 1, py, y, ny, oy);
            nx += 1 + (x == px - 1); ny += 1 + (y == py - 1);
            worst = std::max(worst, std::max(double(nx)/ny, double(ny)/nx));
        }
    return worst;
}

int main()
{
    int cases = 0;
    for (int M = 2; M <= 50; ++M)
        for (int N = 2; N <= 50; ++N)
            for (int p = 1; p <= 16; ++p) {
                Decomp dc = choose_decomp(p, M-1, N-1);
                double best = 1e300;
                for (int px = 1; px <= p; ++px)
                    if (p % px == 0 && px < M && p/px < N) {
                        const double w = inspect(M, N, px, p/px);
                        if (w < 0) return 1;
                        best = std::min(best, w);
                    }
                if (dc.Px == 0) { if (best != 1e300) return 2; }
                else if (std::fabs(inspect(M,N,dc.Px,dc.Py)-best)>1e-14 || dc.ok != (best<=2)) return 3;
                ++cases;
            }
    Decomp a = choose_decomp(2,79,79), b = choose_decomp(4,3,5);
    if (!a.ok || std::fabs(a.worst-81.0/41)>1e-14 || !b.ok || b.Px!=2 || b.Py!=2) return 4;
    std::printf("%d grid/process combinations OK; 80x80/P2 domain ratio=%.12f; 4x6/P4=%dx%d\n",
                 cases,a.worst,b.Px,b.Py);
    return 0;
}
