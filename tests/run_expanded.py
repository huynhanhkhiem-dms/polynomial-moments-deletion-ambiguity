"""Actually materialize two dyadic witnesses and fit the 3D intercept models."""
from pathlib import Path
import sys,json,time
from fractions import Fraction as Q
import numpy as np
from flint import arb,ctx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,retained_rows,embed_intercept,U
from secant import fit
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'results/expanded_protocol.json'
    if p.exists():raise FileExistsError('preserve expanded experiments')
    p.write_text(json.dumps({'degrees':[2,4],'families':['dyadic ridge logistic with explicit penalized intercept'],
      'actual_expansion':True,'gate':'solver convergence; verified exact-target residual bound and opposite query decisions',
      'written_before_results_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    rows=[]
    for k in [2,4]:
        s=specification(k)
        for world in ['plus','minus']:
            t=time.monotonic();X=[];y=[]
            atoms=[(embed_intercept(x),label,c) for x,label,c in retained_rows(s,world)]
            for x,label,c in atoms:
                z=np.array([float(v) for v in x]);assert all(Q(float(a))==b for a,b in zip(z,x))
                X.append(np.repeat(z[None,:],c,axis=0));y.append(np.full(c,label,dtype=float))
            X=np.concatenate(X);y=np.concatenate(y);assert len(y)==s['n']
            w,info=fit(X,y,float(s['ridge_with_intercept']));assert info['converged'],info
            ctx.prec=768;ridge=arb(s['n'])*arb(s['ridge_with_intercept'].numerator)/s['ridge_with_intercept'].denominator
            g=[ridge*arb(float(z)) for z in w]
            for x,label,c in atoms:
                z=[arb(v.numerator)/v.denominator for v in x];score=sum((z[j]*arb(float(w[j])) for j in range(3)),arb(0))
                residual=1/(1+(-score).exp())-label
                for j in range(3):g[j]+=c*z[j]*residual
            radius=(sum((abs(z).upper() for z in g),arb(0))/ridge).upper()
            query=embed_intercept(U);score=sum((arb(x.numerator)/x.denominator*arb(float(w[j])) for j,x in enumerate(query)),arb(0))+arb(0,radius)
            probability=1/(1+(-score).exp())
            assert score>0 if world=='plus' else score<0
            row={'degree':k,'world':world,'actual_expanded_rows':len(y),'dimension':3,'initial_count':str(s['N']),
                'exact_lambda':str(s['ridge_with_intercept']),'fit_lambda_float':float(s['ridge_with_intercept']),
                'feature_dyadic_exactly_binary64':True,'weights':w.tolist(),'fit':info,
                'exact_target_optimizer_error_upper':str(radius),'query_score_interval':str(score),'query_probability_interval':str(probability),
                'seconds':time.monotonic()-t,'scope':'actual expanded unweighted rows fitted; exact-target gradient grouped by known atom multiplicities'}
            rows.append(row);(ROOT/'results/expanded.json').write_text(json.dumps({'status':'running','rows':rows},indent=2))
            print(k,world,len(y),'score',str(score),'radius',str(radius),flush=True)
            del X,y
    (ROOT/'results/expanded.json').write_text(json.dumps({'status':'PASS','rows':rows},indent=2))

if __name__=='__main__':main()
