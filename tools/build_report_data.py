# /// script
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Воспроизводимые CSV, рисунки и сводка для отчёта.

uv run tools/build_report_data.py --work-dir /path/to/report-work
Сырые бинарные поля и журналы сохраняются в work-dir; в report/ — CSV и рисунки.
Повтор использует только результаты с тем же хешем исходников и аргументов.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_reference import POINTS, U_REF
from solio import read_solution, bilinear
from plot_solution import main as plot_solution

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "report/data", ROOT / "report/figures"
GRIDS = [(20,20),(40,40),(80,80),(400,600),(800,1200)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True)
    args = ap.parse_args()
    work = args.work_dir.resolve(); work.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    points = work / "points.csv"
    points.write_text("".join(f"{x:.17g},{y:.17g}\n" for x,y in POINTS))
    source = hashlib.sha256()
    for p in sorted((ROOT/'src').glob('*')) + [ROOT/'tools/trace_cg.cpp']:
        source.update(p.read_bytes())
    fingerprint = source.hexdigest()
    def run(tag, cmd, output):
        stamp = work / f"{tag}.json"
        key = {"source":fingerprint,"command":[str(x) for x in cmd]}
        if stamp.exists() and json.loads(stamp.read_text()) == key and output.exists():
            print("cached",tag,flush=True)
            return (work/f'{tag}.log').read_text()
        p = subprocess.run([str(x) for x in cmd],capture_output=True,text=True,
                           env=dict(os.environ,OMP_NUM_THREADS='1'),timeout=240)
        text=p.stdout+p.stderr
        (work/f'{tag}.log').write_text(text)
        if p.returncode: raise RuntimeError(f'{tag} failed: {text}')
        stamp.write_text(json.dumps(key,ensure_ascii=False,indent=2))
        print(tag, text.splitlines()[-1],flush=True)
        return text
    summaries=[]
    traces={}
    for m,n in GRIDS:
        key=f'{m}x{n}'
        refs=[]
        for exponent in (12,13):
            dest=work/f'reference_{key}_{exponent}.bin'
            run(f'reference_{key}_{exponent}',[ROOT/'poisson_seq',m,n,'--delta',f'1e-{exponent}','--out',dest],dest)
            refs.append(dest)
        x,y,wref=read_solution(refs[-1])
        gap=float(np.max(np.abs(read_solution(refs[0])[2]-wref)))
        csv=DATA/f'trace_{key}.csv'
        trace_log=run(f'trace_{key}',[ROOT/'trace_cg',m,n,refs[-1],points,csv,'1e-6'],csv)
        t=np.genfromtxt(csv,delimiter=',',names=True)
        assert t[0]['iteration']==0 and np.isfinite(np.array(t.tolist())).all()
        assert np.array_equal(t['iteration'],np.arange(len(t)))
        assert (t['accepted']==1).all(), 'restarts need separate plotting'
        normal=work/f'solution_{key}.bin'
        normal_log=run(f'solution_{key}',[ROOT/'poisson_seq',m,n,'--out',normal],normal)
        _,_,wnormal=read_solution(normal)
        # Наблюдение не меняет расчёт: конечные значения в точках и норма ошибки
        # совпадают с обычным запуском без наблюдателя.
        endpoint=np.array([t[-1][f'u{i}'] for i in range(len(POINTS))])
        wanted=np.array([bilinear(x,y,wnormal,*p) for p in POINTS])
        assert np.max(np.abs(endpoint-wanted))<1e-12
        assert abs(t[-1]['error_inf']-np.max(np.abs(wnormal-wref)))<1e-14
        point_error=np.max(np.abs(np.column_stack([t[f'u{i}'] for i in range(len(POINTS))])-U_REF),axis=1)
        traces[key]=(t,point_error)
        summaries.append(dict(grid=key,M=m,N=n,h=max(2.2/m,3.3/n),iterations=int(t[-1]['iteration']),
                              step=float(t[-1]['step']),residual=float(t[-1]['residual']),
                              iteration_error_l2=float(t[-1]['error_l2']),iteration_error_inf=float(t[-1]['error_inf']),
                              fem_point_error=float(point_error[-1]),reference_delta=1e-13,
                              reference_stability_inf=gap))
    # Влияние допуска на самую мелкую сетку: изменение интеграла не
    # заменяет оценку ошибки поля. Сохраняем оба типа ошибки раздельно.
    tolerance=[]
    x,y,wref=read_solution(work/'reference_800x1200_13.bin')
    for exponent in (6,8,10,13):
        dest = work/'solution_800x1200.bin' if exponent==6 else (
            work/'reference_800x1200_13.bin' if exponent==13 else work/f'tolerance_800x1200_{exponent}.bin')
        if exponent in (8,10):
            log=run(f'tolerance_800x1200_{exponent}',[ROOT/'poisson_seq',800,1200,'--delta',f'1e-{exponent}','--out',dest],dest)
        else:
            log=(work/('solution_800x1200.log' if exponent==6 else 'reference_800x1200_13.log')).read_text()
        import re
        count=int(re.search(r'iters=(\d+)',log).group(1))
        w=read_solution(dest)[2]
        tolerance.append(dict(delta=10.0**(-exponent),iterations=count,
            error_inf=float(np.max(np.abs(w-wref))),
            fem_point_error=max(abs(bilinear(x,y,w,*p)-u) for p,u in zip(POINTS,U_REF))))
    tight_csv=DATA/'trace_800x1200_tight.csv'
    run('trace_800x1200_tight',[ROOT/'trace_cg',800,1200,work/'reference_800x1200_13.bin',points,tight_csv,'1e-10'],tight_csv)
    tight=np.genfromtxt(tight_csv,delimiter=',',names=True)
    tight_pe=np.max(np.abs(np.column_stack([tight[f'u{i}'] for i in range(len(POINTS))])-U_REF),axis=1)
    assert int(tight[-1]['iteration'])==tolerance[2]['iterations']
    assert abs(tight[-1]['error_inf']-tolerance[2]['error_inf'])<1e-14
    (DATA/'tolerance.json').write_text(json.dumps(tolerance,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    axes[0].semilogy(tight['iteration'],tight['error_inf'],label='Ошибка к сеточному эталону')
    axes[0].semilogy(tight['iteration'],tight_pe,label='Отклонение от FEM в пяти точках')
    axes[1].semilogy(tight['iteration'][1:],tight['step'][1:],label='Норма шага')
    for ax in axes:
        for item,color in zip(tolerance[:3],['#ad3838','#638c36','#6a56a6']):
            ax.axvline(item['iterations'],lw=.8,ls='--',color=color,label=f"δ={item['delta']:.0e}")
        ax.set_xlabel('Номер итерации k');ax.grid(True,alpha=.25);ax.legend(fontsize=7)
    axes[0].set_title('800×1200: различие двух ошибок',fontsize=10)
    axes[1].set_title('Остановки по норме шага',fontsize=10)
    fig.tight_layout();fig.savefig(FIG/'tolerance.png',dpi=180);plt.close(fig)
    # Независимый прямой решатель для малых высокоточных опорных полей.
    sys.path.insert(0,str(ROOT/'tests'))
    from direct_check import direct_solution
    for item in summaries[:3]:
        m,n=item['M'],item['N']
        exact=direct_solution(m,n)
        wref=read_solution(work/f'reference_{m}x{n}_13.bin')[2]
        item['direct_reference_inf']=float(np.max(np.abs(exact-wref)))
        assert item['direct_reference_inf']<1e-8
    (DATA/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.25})
    colors=plt.cm.viridis(np.linspace(.1,.9,len(GRIDS)))
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.1))
    for ((key,(t,pe)),color) in zip(traces.items(),colors):
        for ax,column in zip(axes.flat,('step','residual','error_l2','error_inf')):
            ix=t['iteration']>0
            ax.semilogy(t['iteration'][ix],t[column][ix],label=key,color=color,lw=1.2)
            ax.set_xlabel('Номер итерации k')
    for ax,title in zip(axes.flat,('Норма шага ‖wᵏ−wᵏ⁻¹‖E','Истинная невязка ‖B−Adᵏ‖E',
                                    'Отклонение от сеточного эталона, E-норма','Отклонение от сеточного эталона, max-норма')):
        ax.set_title(title,fontsize=10)
    axes[0,0].axhline(1e-6,color='black',ls='--',lw=.8,label='δ = 10⁻⁶')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(FIG/'iterations.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for ((key,(t,pe)),color) in zip(traces.items(),colors):
        axes[0].semilogy(t['iteration'],pe,label=key,color=color)
        axes[1].plot(t['iteration'],t['J'],label=key,color=color)
    axes[0].set_title('Максимум отклонений в пяти точках от FEM',fontsize=10)
    axes[0].set_ylabel('max по пяти контрольным точкам')
    axes[1].set_title('Функционал для поправки d = w − v₀',fontsize=10)
    axes[1].set_ylabel('J(d)')
    for ax in axes: ax.set_xlabel('Номер итерации k')
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'fem_and_functional.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    hs=[s['h'] for s in summaries];es=[s['fem_point_error'] for s in summaries]
    ax.loglog(hs,es,'o-',label='Максимум ошибки в пяти точках')
    for s in summaries: ax.annotate(s['grid'],(s['h'],s['fem_point_error']),xytext=(6,-12),textcoords='offset points',fontsize=8)
    ax.set_xlabel('h = max(h₁, h₂)');ax.set_ylabel('Отклонение от FEM');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'grid_error.png',dpi=180);plt.close(fig)
    # Геометрическая область и два домена, разделённые y=0.
    fig,ax=plt.subplots(figsize=(6.8,4.5))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((.91,-1.65),2.2,1.65,facecolor='#d9eaf3',edgecolor='#29617b'))
    ax.add_patch(Rectangle((.91,0),2.2,1.65,facecolor='#f6e3c5',edgecolor='#8f622d'))
    yy=np.linspace(-np.sqrt(2),np.sqrt(2),300)
    ax.plot(np.sqrt(1+4*yy*yy),yy,'k',lw=1.2);ax.plot([3,3],[-np.sqrt(2),np.sqrt(2)],'k',lw=1.2)
    ax.text(2.01,-.9,'Домен 0: 81 × 41 узел\nВладение: 79 × 40',ha='center',fontsize=10)
    ax.text(2.01,.9,'Домен 1: 81 × 41 узел\nВладение: 79 × 39',ha='center',fontsize=10)
    ax.annotate('Общая граница y = 0',xy=(1.4,0),xytext=(.5,1.9),arrowprops={'arrowstyle':'->'},fontsize=9)
    ax.set(xlim=(.4,3.5),ylim=(-1.9,2.2),xlabel='x',ylabel='y',aspect='equal')
    ax.grid(False);fig.tight_layout();fig.savefig(FIG/'decomposition.png',dpi=180);plt.close(fig)
    sys.argv=['plot_solution',str(work/'solution_800x1200.bin'),str(FIG/'solution.png')]
    plot_solution()
    (DATA/'provenance.json').write_text(json.dumps({'source_sha256':fingerprint,'diagnostic_delta':1e-6,
        'reference_deltas':[1e-12,1e-13],'norm_domain':'all interior nodes of Pi',
        'fem_points':POINTS,'fem_values':U_REF,'diagnostic_timings_excluded':True},indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()
