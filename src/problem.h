// Постановка задачи 1, вариант 12, и сборка разностной схемы.
//
// Область D = { (x,y) : x^2 - 4y^2 > 1, 1 < x < 3 } — часть плоскости между
// правой ветвью гиперболы x = sqrt(1 + 4y^2) и прямой x = 3. При x > 1 условие
// x^2 - 4y^2 > 1 равносильно x > g(y), g(y) = sqrt(1 + 4y^2), а g выпукла,
// поэтому D выпукла: D = { g(y) < x < 3 }, |y| < sqrt(2).
//
// Уравнение: -Laplace(u) = f = 1 в D, u = phi = sin(x+y) на границе gamma.
//
// Метод фиктивных областей: в прямоугольнике Pi решается
//   -Laplace(v) + Q (v - v0) = F,   v = v0 на границе Pi,
// Q = 0 в D и 1/eps^2 вне D, F = f в D и 0 вне D, eps = h^2, h = max(h1, h2).
// Продолжение граничной функции v0 = sin(x+y) во всём Pi.
//
// Прямоугольник Pi = (0.91, 3.11) x (-1.65, 1.65). Замыкание D = [1,3] x
// [-sqrt(2), sqrt(2)] лежит строго внутри. Стороны относятся как 2:3 — как
// сетки из таблиц задания (400x600, 800x1200), поэтому там h1 = h2.
//
// Сдвиг 0.91 выбран так, чтобы прямая x = 3 проходила через узел сетки при
// любом M, кратном 20: 3 - 0.91 = 2.09 = 19 * 2.2/20. Зачем: в ячейке,
// которую пересекает граница, Q = (1 - S/(h1 h2))/eps^2 ~ 1/h^4 много больше
// 1/h^2 из разностного лапласиана, и узел фактически закрепляется на v0.
// Если x = 3 падает между узлами, граница сдвигается до ближайшего узла
// с разрезанной ячейкой. При исходном выборе (0.9, 3.1) таким узлом на всех
// вложенных сетках 20, 40, 80 оказывался x = 2.99, и ошибка в точках
// сверки застревала на 4e-3, не убывая при измельчении.

#ifndef PROBLEM_H
#define PROBLEM_H

#include <cmath>
#include <math.h>  // ::asinh — в <cmath> до C++11 его нет
#include <vector>
#include <algorithm>

namespace problem {

const double A1 = 0.91, B1 = 3.11;     // Pi по x
const double A2 = -1.65, B2 = 1.65;  // Pi по y
const double X_RIGHT = 3.0;          // правая сторона D

inline double f(double /*x*/, double /*y*/) { return 1.0; }
inline double phi(double x, double y) { return std::sin(x + y); }
// Продолжение phi с границы gamma в фиктивную область: та же формула.
inline double v0(double x, double y) { return phi(x, y); }

// Левая граница D: x = g(y).
inline double g(double y) { return std::sqrt(1.0 + 4.0 * y * y); }
// Первообразная g: G'(y) = g(y).
inline double G(double y) { return 0.5 * y * g(y) + 0.25 * ::asinh(2.0 * y); }

// Принадлежность точки области D (для подсчёта норм только по D).
inline bool in_D(double x, double y) { return x > g(y) && x < X_RIGHT; }

// Точная площадь пересечения прямоугольника [xa,xb] x [ya,yb] с D.
//
// При фиксированном y сечение D — отрезок (g(y), 3), поэтому
//   S = int_{ya}^{yb} max(0, min(xb,3) - max(xa, g(y))) dy.
// Подынтегральная функция на каждом участке монотонности g принимает один
// из трёх видов: 0, xr - xa или xr - g(y). Точки смены вида — корни
// g(y) = xa, g(y) = xr и y = 0 (минимум g). Между ними вид определяется по
// середине участка, а интеграл от g берётся через первообразную G.
// Задание разрешает заменять дугу хордой; здесь дуга учитывается точно,
// так что S не зависит от того, как кривая пересекает ячейку.
inline double cell_area(double xa, double xb, double ya, double yb)
{
    const double xr = std::min(xb, X_RIGHT);
    if (xr <= xa || yb <= ya) return 0.0;

    double pts[8];
    int n = 0;
    pts[n++] = ya;
    pts[n++] = yb;
    const double cs[2] = {xa, xr};
    for (int k = 0; k < 2; ++k) {
        if (cs[k] > 1.0) {
            const double yc = 0.5 * std::sqrt(cs[k] * cs[k] - 1.0);
            if (yc > ya && yc < yb) pts[n++] = yc;
            if (-yc > ya && -yc < yb) pts[n++] = -yc;
        }
    }
    if (0.0 > ya && 0.0 < yb) pts[n++] = 0.0;
    std::sort(pts, pts + n);

    double s = 0.0;
    for (int k = 0; k + 1 < n; ++k) {
        const double a = pts[k], b = pts[k + 1];
        if (b <= a) continue;
        const double gm = g(0.5 * (a + b));
        if (gm >= xr) continue;                     // сечение пусто
        if (gm <= xa) s += (xr - xa) * (b - a);     // ячейка перекрыта по x целиком
        else s += xr * (b - a) - (G(b) - G(a));     // левый край — дуга гиперболы
    }
    return s;
}

// Точная площадь D: 3*sqrt(2) - asinh(2*sqrt(2))/2. Для самопроверки.
inline double area_D() { return 3.0 * std::sqrt(2.0) - 0.5 * ::asinh(2.0 * std::sqrt(2.0)); }

struct Grid {
    int M, N;          // число отрезков разбиения по x и по y
    double h1, h2;     // шаги
    double eps;        // параметр метода фиктивных областей, eps = h^2
    Grid(int M_, int N_) : M(M_), N(N_)
    {
        h1 = (B1 - A1) / M;
        h2 = (B2 - A2) / N;
        const double h = std::max(h1, h2);
        eps = h * h;
    }
    double x(int i) const { return A1 + i * h1; }
    double y(int j) const { return A2 + j * h2; }
};

// Прямоугольный блок внутренних узлов сетки с одним слоем теневых узлов.
// Глобальные индексы узлов блока: i = i0+1 .. i0+nx, j = j0+1 .. j0+ny.
// Локальный индекс (li, lj), li = 0..nx+1, lj = 0..ny+1; li = 0 и li = nx+1 —
// теневые слои (узлы соседнего блока или граница Gamma прямоугольника).
// Хранение по строкам x: idx = li*(ny+2) + lj, т. е. соседи по y лежат рядом.
// В последовательной/OpenMP программе блок один: i0 = j0 = 0, nx = M-1, ny = N-1.
struct Block {
    int nx, ny, i0, j0;
    int stride;  // ny + 2
    int size() const { return (nx + 2) * (ny + 2); }
    int idx(int li, int lj) const { return li * stride + lj; }
    void init(int nx_, int ny_, int i0_, int j0_)
    {
        nx = nx_; ny = ny_; i0 = i0_; j0 = j0_;
        stride = ny + 2;
    }
};

// Коэффициенты разностной схемы на блоке.
//
// Решение ищется в виде w = v0 + d. Подстановка в схему
//   -w_xx - w_yy + Q w = F + Q v0,  w = v0 на Gamma,
// даёт для поправки d задачу с нулевыми граничными условиями
//   A d = -d_xx - d_yy + Q d = F + (v0)_xx + (v0)_yy =: B.
// Это ровно метод сопряжённых градиентов для исходной схемы с начальным
// приближением w(0) = v0 (задание разрешает любое w(0)); выгода в том,
// что слагаемое Q v0 ~ 1/h^4 сокращается аналитически и не съедает
// значащие цифры F в правой части.
struct Coeffs {
    std::vector<double> Q;     // Q_ij — среднее по ячейке Pi_ij
    std::vector<double> Dinv;  // 1 / (2/h1^2 + 2/h2^2 + Q_ij) — диагональный предобуславливатель
    std::vector<double> B;     // правая часть для поправки d
};

inline void build_coeffs(const Grid& gr, const Block& bl, Coeffs& c)
{
    const int sz = bl.size();
    c.Q.assign(sz, 0.0);
    c.Dinv.assign(sz, 0.0);
    c.B.assign(sz, 0.0);
    const double h1 = gr.h1, h2 = gr.h2, cell = h1 * h2;
    const double inv_eps2 = 1.0 / (gr.eps * gr.eps);
    const double ih1 = 1.0 / (h1 * h1), ih2 = 1.0 / (h2 * h2);

#pragma omp parallel for schedule(static)
    for (int li = 1; li <= bl.nx; ++li) {
        const int i = bl.i0 + li;
        const double x = gr.x(i);
        for (int lj = 1; lj <= bl.ny; ++lj) {
            const int j = bl.j0 + lj;
            const double y = gr.y(j);
            const double S = cell_area(x - 0.5 * h1, x + 0.5 * h1, y - 0.5 * h2, y + 0.5 * h2);
            const double frac = S / cell;           // доля ячейки внутри D
            const double Qij = (1.0 - frac) * inv_eps2;
            // F_ij = S_ij f(x*, y*) / (h1 h2); f = 1, точка (x*, y*) ни на что не влияет.
            const double Fij = frac * f(x, y);
            const double lap_v0 =
                (v0(x + h1, y) - 2.0 * v0(x, y) + v0(x - h1, y)) * ih1 +
                (v0(x, y + h2) - 2.0 * v0(x, y) + v0(x, y - h2)) * ih2;
            const int k = bl.idx(li, lj);
            c.Q[k] = Qij;
            c.Dinv[k] = 1.0 / (2.0 * ih1 + 2.0 * ih2 + Qij);
            c.B[k] = Fij + lap_v0;
        }
    }
}

}  // namespace problem

#endif
