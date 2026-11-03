// Диагностический последовательный запуск того же cg_solve, что в программе.
// trace_cg M N reference.bin points.csv trace.csv delta
// Наблюдатель только читает d; его работа исключена из замеров ускорения.
#include <cstdio>
#include <vector>
#include <algorithm>
#include "common.h"
#include "solver.h"

struct LocalComm : Comm {
    void halo(double*) {}
    void allreduce(double*, int) {}
};

struct Trace : CGObserver {
    const problem::Grid& gr;
    const problem::Block& bl;
    const problem::Coeffs& c;
    const std::vector<double>& reference;
    const std::vector<double>& px;
    const std::vector<double>& py;
    FILE* file;
    bool ok;
    std::vector<double> Ad;
    Trace(const problem::Grid& g, const problem::Block& b, const problem::Coeffs& coeff,
          const std::vector<double>& ref, const std::vector<double>& x,
          const std::vector<double>& y, FILE* f)
        : gr(g), bl(b), c(coeff), reference(ref), px(x), py(y), file(f), ok(true), Ad(b.size(), 0.0)
    {
        std::fprintf(file, "iteration,accepted,step,residual,J,error_l2,error_inf");
        for (size_t i = 0; i < px.size(); ++i) std::fprintf(file, ",u%lu", (unsigned long)i);
        std::fprintf(file, "\n");
    }
    double value(const std::vector<double>& d, int i, int j) const
    {
        return problem::v0(gr.x(i), gr.y(j)) + d[bl.idx(i, j)];
    }
    void sample(long k, double step, bool accepted, const std::vector<double>& d)
    {
        double ignored[2];
        block_A(gr, bl, c, &d[0], &Ad[0], ignored);
        double rr = 0, ee = 0, em = 0, J = 0;
        for (int i = 1; i < gr.M; ++i)
            for (int j = 1; j < gr.N; ++j) {
                const int at = bl.idx(i, j);
                const double r = c.B[at] - Ad[at];
                const double e = value(d, i, j) - reference[at];
                rr += r * r;
                ee += e * e;
                em = std::max(em, std::fabs(e));
                J += (0.5 * Ad[at] - c.B[at]) * d[at];
            }
        const double cell = gr.h1 * gr.h2;
        std::fprintf(file, "%ld,%d,%.17e,%.17e,%.17e,%.17e,%.17e", k, accepted ? 1 : 0,
                     step, std::sqrt(rr * cell), J * cell, std::sqrt(ee * cell), em);
        for (size_t p = 0; p < px.size(); ++p) {
            const int i = std::min(gr.M - 1, std::max(0, (int)((px[p] - problem::A1) / gr.h1)));
            const int j = std::min(gr.N - 1, std::max(0, (int)((py[p] - problem::A2) / gr.h2)));
            const double tx = (px[p] - gr.x(i)) / gr.h1, ty = (py[p] - gr.y(j)) / gr.h2;
            const double u = (1-tx)*(1-ty)*value(d,i,j) + tx*(1-ty)*value(d,i+1,j)
                           + (1-tx)*ty*value(d,i,j+1) + tx*ty*value(d,i+1,j+1);
            std::fprintf(file, ",%.17e", u);
        }
        std::fprintf(file, "\n");
        ok = ok && !std::ferror(file);
    }
};

int main(int argc, char** argv)
{
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s M N reference.bin points.csv trace.csv delta\n", argv[0]);
        return 1;
    }
    char flag[] = "--delta";
    char* args[] = {argv[0], argv[1], argv[2], flag, argv[6]};
    Options opt;
    if (!parse_options(5, args, opt)) return 1;
    problem::Grid gr(opt.M, opt.N);
    problem::Block bl;
    bl.init(gr.M - 1, gr.N - 1, 0, 0);
    std::vector<double> reference(bl.size());
    FILE* in = std::fopen(argv[3], "rb");
    if (!in) { std::perror("reference"); return 2; }
    int dims[2]; double box[4];
    bool good = std::fread(dims, sizeof(int), 2, in) == 2;
    good = (std::fread(box, sizeof(double), 4, in) == 4) && good;
    good = (std::fread(&reference[0], sizeof(double), reference.size(), in) == reference.size()) && good;
    good = (std::fgetc(in) == EOF) && good;
    std::fclose(in);
    if (!good || dims[0] != gr.M || dims[1] != gr.N || box[0] != problem::A1 ||
        box[1] != problem::B1 || box[2] != problem::A2 || box[3] != problem::B2) {
        std::fprintf(stderr, "reference has wrong grid or size\n"); return 1;
    }
    for (size_t k = 0; k < reference.size(); ++k)
        if (!cg_finite(reference[k])) { std::fprintf(stderr, "nonfinite reference\n"); return 1; }
    FILE* points = std::fopen(argv[4], "r");
    if (!points) { std::perror("points"); return 2; }
    std::vector<double> px, py;
    double x, y; int count;
    while ((count = std::fscanf(points, " %lf,%lf", &x, &y)) == 2) {
        if (!cg_finite(x) || !cg_finite(y) || x < problem::A1 || x > problem::B1 ||
            y < problem::A2 || y > problem::B2) break;
        px.push_back(x); py.push_back(y);
    }
    std::fclose(points);
    if (count != EOF || px.empty()) { std::fprintf(stderr, "invalid points\n"); return 1; }
    FILE* out = std::fopen(argv[5], "w");
    if (!out) { std::perror("trace"); return 2; }
    problem::Coeffs c;
    problem::build_coeffs(gr, bl, c);
    std::vector<double> d(bl.size(), 0.0);
    LocalComm comm;
    CGParams prm = {opt.delta, opt.maxit, 0, 0};
    Trace trace(gr, bl, c, reference, px, py, out);
    CGResult res = cg_solve(gr, bl, c, d, comm, prm, &trace);
    good = trace.ok;
    if (std::fclose(out) != 0) good = false;
    std::printf("TRACE status=%s iters=%ld restarts=%d step=%.17e resid=%.17e\n",
                 cg_status_name(res.status), res.iters, res.restarts, res.last_step, res.resid);
    if (!good) return 2;
    return res.converged() ? 0 : 3;
}
