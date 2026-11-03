// Изолированные случаи, которые нельзя получить штатной правой частью варианта 12.
#include <cstdio>
#include <limits>
#include "solver.h"

struct LocalComm : Comm {
    void halo(double*) {}
    void allreduce(double*, int) {}
};

int main()
{
    problem::Grid gr(2, 2);
    problem::Block bl;
    bl.init(1, 1, 0, 0);
    problem::Coeffs c;
    problem::build_coeffs(gr, bl, c);
    const int k = bl.idx(1, 1);
    const double lap_diag = 2 / (gr.h1 * gr.h1) + 2 / (gr.h2 * gr.h2);
    CGParams prm = {1e-12, 1, 0, 0};
    LocalComm comm;
    for (int test = 0; test < 5; ++test) {
        std::vector<double> d(bl.size(), 0.0);
        c.Q[k] = (test == 2 ? -1.0 : 1.0) - lap_diag;
        c.Dinv[k] = 1.0;
        c.B[k] = test == 0 ? 0.0 : 1.0;
        if (test == 3) c.B[k] = std::numeric_limits<double>::quiet_NaN();
        if (test == 4) d[k] = 1.0;
        CGResult r = cg_solve(gr, bl, c, d, comm, prm);
        const bool good = test < 2 || test == 4
            ? r.converged() && r.resid == 0.0 && d[k] == c.B[k] &&
              (test != 4 || (r.iters == 0 && std::fabs(r.J + 0.5 * gr.h1 * gr.h2) < 1e-14))
            : !r.converged() && r.status == CG_BREAKDOWN;
        std::printf("case=%d status=%s resid=%g %s\n", test, cg_status_name(r.status),
                    r.resid, good ? "OK" : "FAIL");
        if (!good) return 1;
    }
    return 0;
}
