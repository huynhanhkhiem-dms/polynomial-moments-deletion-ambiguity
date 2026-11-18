"""Exact replication identities and high-precision (not interval-certified) fits."""
from pathlib import Path
from fractions import Fraction as Q
import sys,json
import mpmath as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,weighted_fit,U
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    for k in [2,4,8,16,32]:
        s=specification(k); N0=s['N']; n0=s['n']; mp.mp.dps=150
        def scalar(world):
            ridge=mp.mpf(s['kappa'].numerator)/s['kappa'].denominator
            atoms=[(mp.mpf(a['x'].numerator)/a['x'].denominator,a['count']) for a in s['atoms'] if a['world']==world]
            def grad(t):return ridge*t-sum(2*c*x/(1+mp.exp(x*t)) for x,c in atoms)
            l=mp.mpf(0);h=mp.mpf(1)
            while grad(h)<0:h*=2
            for _ in range(240):
                m=(l+h)/2
                if grad(m)<0:l=m
                else:h=m
            return (l+h)/2
        a,b=scalar('A'),scalar('B')
        fits={w:weighted_fit(s,w,init,150) for w,init in [('plus',[a,b]),('minus',[b,a])]}
        scores={w:sum(mp.mpf(x)*mp.mpf(u.numerator)/u.denominator for x,u in zip(f['weights'],U)) for w,f in fits.items()}
        assert scores['plus']>0>scores['minus'] and abs(scores['plus']+scores['minus'])<mp.mpf('1e-40')
        for d in [1,2,3]:
            r=N0**(d-1); N=r*N0; n=r*n0; lam=Q(1,4*N0)
            assert N==N0**d and Q(r*s['deleted'],N)==Q(s['deleted'],N0)
            assert 4*n*lam/r==s['kappa']
            rows.append({'degree':k,'alpha':f'1/{d}','initial_count':str(N),'replication':str(r),'ridge':str(lam),'effective_ridge':str(s['kappa']),'plus_logit':mp.nstr(scores['plus'],40),'minus_logit':mp.nstr(scores['minus'],40),'gradient_l1_mp':{w:f['gradient_l1_mp'] for w,f in fits.items()},'scope':'exact replication identity; shared base numerical solve, not independent replicated fits'})
    (ROOT/'results/power_ridge.json').write_text(json.dumps({'rows':rows,'status':'completed'},indent=2)+'\n')
    print('PASS: 15 replication identities, 10 high-precision base fits')
if __name__=='__main__':main()
