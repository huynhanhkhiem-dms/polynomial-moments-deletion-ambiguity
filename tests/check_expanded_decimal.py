"""Refit saved expanded arrays and bound exact-target error using Decimal intervals."""
from pathlib import Path
from decimal import Decimal as D
import sys,json,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from secant import fit
from predictive_family import specification,retained_rows,embed_intercept,U
from check_decimal_certificates import verify

def main():
    results=[]
    for k in [2,4]:
        s=specification(k)
        for world in ['plus','minus']:
            f=ROOT/'data/generated'/f'expanded_k{k}_{world}.npz'
            with np.load(f) as a: X=a['X']; y=a['y']; lam=float(a['solver_lambda'])
            assert len(y)==s['n'] and X.shape==(s['n'],3)
            expected=[(tuple(float(v) for v in embed_intercept(x)),label,c) for x,label,c in retained_rows(s,world)]
            # Check the actual expanded rows, including multiplicity and labels, against the exact construction.
            offset=0
            for x,label,c in expected:
                assert np.all(X[offset:offset+c]==x) and np.all(y[offset:offset+c]==label)
                offset+=c
            assert offset==len(y)
            w,info=fit(X,y,lam)
            assert info['converged'],info
            atoms=[(tuple((2*y-1)*v for v in embed_intercept(x)),c) for x,y,c in retained_rows(s,world)]
            score,prob,r=verify(atoms,[D.from_float(float(v)) for v in w],s['n']*s['ridge_with_intercept'],embed_intercept(U))
            assert score.a>0 if world=='plus' else score.b<0
            assert r<D('0.001'),r
            results.append({'degree':k,'world':world,'rows':len(y),'weights':w.tolist(),'radius_upper':str(r),'query_score_interval':score.bounds(),'converged':True})
            print(k,world,len(y),'radius',r,flush=True)
    (ROOT/'results/expanded_decimal_replay.json').write_text(json.dumps({'status':'PASS','python':platform.python_version(),'numpy':np.__version__,'rows':results,'scope':'fresh binary64 fits on captured expanded arrays, exact-target Decimal residual bounds'},indent=2)+'\n')
if __name__=='__main__': main()
