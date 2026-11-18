"""Generate manuscript figures and table only from recorded executed results."""
from pathlib import Path
import json,math,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def midpoint(s):return float(s.split('+/-')[0].strip(' []'))

def main():
    plt.rcParams.update({'font.size':14,'axes.labelsize':14,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14})
    rows=json.loads((ROOT/'results/predictive_pilot.json').read_text())['rows']
    rob=json.loads((ROOT/'results/robustness.json').read_text())['rows']
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    k=[r['degree'] for r in rows];delta=[100*r['deletion_fraction_float'] for r in rows]
    ax[0].loglog(k,delta,'o-',color='#235c91');ax[0].set(xlabel='Moment degree k',ylabel='Deleted fraction (%)')
    margin=[min(midpoint(r['certificates']['plus']['query_score_interval']),-midpoint(r['certificates']['minus']['query_score_interval'])) for r in rows]
    ax[1].loglog(k,margin,'o-',color='#235c91');ax[1].set(xlabel='Moment degree k',ylabel='Retained-query logit magnitude')
    for kk, marker, style in [(2,'o','-'),(4,'s','--'),(8,'^',':')]:
        rr=[r for r in rob if r['degree']==kk]
        ax[2].semilogx([r['common_rows'] for r in rr],[midpoint(r['probability_gap_interval']) for r in rr],marker=marker,linestyle=style,label=f'k={kk}')
    ax[2].set(xlabel='Common balanced probe rows',ylabel='Reference probability gap',ylim=(0,1.05));ax[2].legend(frameon=False)
    for a in ax:a.grid(True,which='major',alpha=.2)
    f=ROOT/'figures';f.mkdir(exist_ok=True)
    fig.savefig(f/'Fig1.pdf');fig.savefig(f/'Fig1.png',dpi=200);plt.close(fig)
    lines=[]
    for r,m in zip(rows,margin):
        lines.append(f"{r['degree']} & {math.log10(int(r['N'])):.2f} & {100*r['deletion_fraction_float']:.4f} & {m:.2f} \\\\")
    (f/'numeric_table.tex').write_text('\n'.join(lines)+'\n')
    print('Figure and seven-row table generated from executed outputs')

if __name__=='__main__':main()
