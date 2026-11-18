from pathlib import Path
import sys,json,hashlib
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,retained_rows,embed_intercept,source_root,analytic_bounds
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]

def alphas(d,total):
    if d==0:yield ()
    else:
        for a in range(total+1):
            for rest in alphas(d-1,total-a):yield (a,)+rest

def main():
    outputs=[];total=0
    for k in [2,4,8,16,32]:
        s=specification(k);grid=2**(s['b']+1);states={}
        for world in ['plus','minus']:
            rows=[]
            for x,y,c in retained_rows(s,world):
                coords=[v*grid for v in embed_intercept(x)];assert all(v.denominator==1 for v in coords)
                coords=[int(v) for v in coords];powers=[[v**a for a in range(k+1)] for v in coords]
                rows.append((y,c,powers))
            state=[]
            for alpha in alphas(3,k):
                for label in [0,1]:
                    state.append(sum(c*p[0][alpha[0]]*p[1][alpha[1]]*p[2][alpha[2]] for y,c,p in rows if y==label))
            states[world]=state
        assert states['plus']==states['minus'];total+=len(states['plus'])
        serialized=json.dumps(states['plus'],separators=(',',':')).encode()
        outputs.append({'degree':k,'integer_moments_per_world':len(states['plus']),'grid_denominator':grid,
          'equal_integer_numerators':True,'canonical_state_sha256':hashlib.sha256(serialized).hexdigest()})
    bounds=[]
    for k in [2,4,8,16,32,64,128,256]:
        s=specification(k);a=source_root(s,'A',768);b=source_root(s,'B',768);A,B=analytic_bounds(s,768)
        assert a>A and b<B
        aq=lambda x:arb(x.numerator)/x.denominator
        c=(A-B)/2-s['p']/aq(s['kappa']);numeric=(a-b)/2-s['p']/aq(s['kappa'])
        bounds.append({'degree':k,'A_lower':str(A),'B_upper':str(B),'analytic_margin_lower':str(c),
          'interval_root_margin_lower':str(numeric),'analytic_margin_positive':bool(c>0),'root_margin_positive':bool(numeric>0)})
    (ROOT/'results/integer_state_and_bounds.json').write_text(json.dumps({'status':'PASS','integer_state_checks':total,
       'states':outputs,'finite_bound_checks':bounds,'scope':'specified exact integer accumulator, not arbitrary IEEE reduction'},indent=2))
    print(json.dumps({'status':'PASS','integer_state_checks':total,'source_bound_pairs':len(bounds)}))

if __name__=='__main__':main()
