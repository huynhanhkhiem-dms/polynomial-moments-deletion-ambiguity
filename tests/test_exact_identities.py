"""Enumerate all 3D class moments for small k, plus initial training identities."""
from pathlib import Path
from fractions import Fraction as Q
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,retained_rows,deletion_rows,embed_intercept,U
ROOT=Path(__file__).resolve().parents[1]

def exponents(d,degree):
    if d==0:
        yield ()
    else:
        for a in range(degree+1):
            for rest in exponents(d-1,degree-a):yield (a,)+rest

def main():
    results=[];checks=0
    for k in [2,4,8,16,32]:
        spec=specification(k)
        datasets={w:[(embed_intercept(x),y,c) for x,y,c in retained_rows(spec,w)] for w in ['plus','minus']}
        delete=[(embed_intercept(x),y,c) for x,y,c in deletion_rows(spec)]
        assert sum(c for x,y,c in delete)==spec['deleted']
        for world,rows in datasets.items():
            assert sum(c for x,y,c in rows)==spec['n']
            assert all(Q(1,4)<=sum(z*z for z in x)<=Q(1,2) for x,y,c in rows+delete)
            initial=rows+delete
            assert sum(c*(2*y-1) for x,y,c in initial)==0
            assert all(sum(c*(2*y-1)*x[j] for x,y,c in initial)==0 for j in range(3))
        for alpha in exponents(3,k):
            for label in [0,1]:
                vals={}
                for world,rows in datasets.items():
                    val=Q(0)
                    for x,y,c in rows:
                        if y!=label:continue
                        term=Q(c)
                        for z,a in zip(x,alpha):
                            if a:term*=z**a
                        val+=term
                    vals[world]=val
                assert vals['plus']==vals['minus'],(k,alpha,label)
                checks+=1
        grams={w:[[sum(c*x[i]*x[j] for x,y,c in rows+delete) for j in range(3)] for i in range(3)] for w,rows in datasets.items()}
        assert grams['plus']==grams['minus']
        results.append({'degree':k,'feature_norms_between_one_half_and_one_over_sqrt_two':True,'balanced_labels':True,'initial_zero_gradient':True,
          'identical_initial_hessian':True,'retained_count':str(spec['n']),'initial_count':str(spec['N'])})
    out={'status':'PASS','all_3d_class_moment_checks':checks,'degrees':results,
       'scope':'full enumeration of all monomials through the listed degrees on exact rational atoms; large counts implicit'}
    (ROOT/'results/exact_identities.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

if __name__=='__main__':main()
