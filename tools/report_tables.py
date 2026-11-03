# /// script
# dependencies = ["matplotlib"]
# ///
"""Таблицы LaTeX из сохранённых логов Polus и диагностических JSON."""
import json
from pathlib import Path
import subprocess
import sys

from collect import parse, best, spread, ROW_OMP, ROW_MPI, ROW_HYB

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'report/tables'


def sci(value):
    if value==0: return '$0$'
    mantissa, exponent=f'{value:.2e}'.split('e')
    return f'${mantissa}\\cdot10^{{{int(exponent)}}}$'


def table(name,head,rows):
    text='\\small\n\\begin{tabular}{'+('r'*len(head))+'}\n\\toprule\n'
    text+=' & '.join(head)+r'\\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r'\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (OUT/f'{name}.tex').write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'report/data/summary.json').read_text())
    grid=lambda s:'$'+s.replace('x',r'\times')+'$'
    table('grid',['Сетка', '$h$', 'Итераций', 'Ошибка в 5 точках'],[
        [grid(r['grid']),f"{r['h']:.5f}",str(r['iterations']),sci(r['fem_point_error'])] for r in rows])
    table('iteration',['Сетка', '$s_k$', '$\\rho_k$', '$e_k$', '$\\|w^{(k)}-\\widetilde w_h\\|_\\infty$'],[
        [grid(r['grid']),sci(r['step']),sci(r['residual']),sci(r['iteration_error_l2']),sci(r['iteration_error_inf'])] for r in rows])
    tol=json.loads((ROOT/'report/data/tolerance.json').read_text())
    table('tolerance',['$\\delta$', 'Итераций', 'Ошибка к сеточному эталону', 'Ошибка в 5 точках'],[
        [sci(r['delta']),str(r['iterations']),sci(r['error_inf']),sci(r['fem_point_error'])] for r in tol])
    files=sorted(str(p) for p in (ROOT/'results/polus/s3').glob('*.out'))
    runs=parse(files)
    base={g:best(runs,('seq',g,1,0))[0] for g in ['400x600','800x1200']}
    for name,spec in [('omp',ROW_OMP),('mpi',ROW_MPI),('hyb',ROW_HYB)]:
        out=[]
        for row in spec:
            if name=='hyb':
                p,t,g=row; first=[str(p),str(t)]
            elif name=='mpi':
                p,g=row;t=0;first=[str(p)]
            else:
                t,g=row;p=1;first=[str(t)]
            key=(name,g,p,t);r=best(runs,key)
            assert r is not None and len(runs[key])==3
            width=spread(runs,key).split(', ')[1].replace('%',r'\%')
            out.append(first+[grid(g),str(r[1]),f'{r[0]:.4f}',f'{base[g]/r[0]:.2f}',width])
        head=(['Проц.','Нитей'] if name=='hyb' else ['Нитей' if name=='omp' else 'Проц.'])
        table(name,head+['Сетка','Итер.','$T$, с','$S$','Разброс'],out)
    subprocess.run([sys.executable,str(ROOT/'tools/collect.py'),'--plot',str(ROOT/'report/figures/speedup.png')]+files,
                   check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__': main()
