"""Outward-rounded independent Decimal checks of stored and recomputed candidates."""
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as Q
from pathlib import Path
import json, sys, platform
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from predictive_family import specification, signed_atoms, retained_rows, embed_intercept, U, probe_block, weighted_fit
from public_anchor import load_public
P=180
L=Context(prec=P,rounding=ROUND_FLOOR,Emax=999999999,Emin=-999999999)
H=Context(prec=P,rounding=ROUND_CEILING,Emax=999999999,Emin=-999999999)
M=Context(prec=P,Emax=999999999,Emin=-999999999)
class I:
    def __init__(self,x=0,b=None):
        if isinstance(x,I): self.a,self.b=x.a,x.b
        elif isinstance(x,Q): self.a=L.divide(D(x.numerator),D(x.denominator)); self.b=H.divide(D(x.numerator),D(x.denominator))
        else: self.a=D(x); self.b=self.a if b is None else D(b)
    def __add__(self,y):
        y=I(y); return I(L.add(self.a,y.a),H.add(self.b,y.b))
    __radd__=__add__
    def __neg__(self): return I(self.b.copy_negate(),self.a.copy_negate())
    def __sub__(self,y): return self+-I(y)
    def __mul__(self,y):
        y=I(y); pairs=[(a,b) for a in (self.a,self.b) for b in (y.a,y.b)]
        return I(min(L.multiply(a,b) for a,b in pairs),max(H.multiply(a,b) for a,b in pairs))
    __rmul__=__mul__
    def __truediv__(self,y):
        y=I(y); assert y.a>0 or y.b<0
        return self*I(L.divide(D(1),y.b),H.divide(D(1),y.a))
    def __rtruediv__(self,y): return I(y)/self
    # Decimal exp/ln are correctly rounded to nearest; expand by one representable step.
    def exp(self):return I(M.next_minus(M.exp(self.a)),M.next_plus(M.exp(self.b)))
    def ln(self):return I(M.next_minus(M.ln(self.a)),M.next_plus(M.ln(self.b)))
    def abs_upper(self):return max(self.a.copy_abs(),self.b.copy_abs())
    def bounds(self):return [str(self.a),str(self.b)]
def verify(rows,w,ridge,query):
    w=[I(v) for v in w]; ridge=I(ridge); g=[ridge*v for v in w]
    for x,c in rows:
        v=list(map(I,x)); score=sum(a*b for a,b in zip(v,w)); r=1/(1+score.exp())
        for j in range(len(w)):g[j]-=c*v[j]*r
    radius=(sum(I(v.abs_upper()) for v in g)/ridge).b
    score=sum(I(x)*v for x,v in zip(query,w))+I(radius.copy_negate(),radius)
    return score,1/(1+(-score).exp()),radius
def atomfit(s,world):
    mp.mp.dps=180
    mq=lambda x: mp.mpf(x.numerator)/x.denominator
    def scalar(which):
        def g(t):return mq(s['kappa'])*t-sum(2*a['count']*mq(a['x'])/(1+mp.exp(mq(a['x'])*t)) for a in s['atoms'] if a['world']==which)
        l=mp.mpf(0);h=mp.mpf(1)
        while g(h)<0:h*=2
        for _ in range(240):
            m=(l+h)/2
            if g(m)<0:l=m
            else:h=m
        return (l+h)/2
    a,b=scalar('A'),scalar('B')
    return weighted_fit(s,world,[a,b] if world=='plus' else [b,a],180)['weights']
def main():
    result=[]
    def load(name):return json.loads((ROOT/'results'/f'{name}.json').read_text())['rows']
    blocks={name:load_public(name)[0] for name in ['spambase','magic','optdigits','covertype']}
    for name in ['predictive_pilot','public_anchor','robustness','regularization_regimes']:
        for row in load(name):
            k=row['degree']; common=blocks[row['dataset']] if name=='public_anchor' else probe_block()*row.get('probe_repetitions',1)
            s=specification(k,common)
            if name=='regularization_regimes':s['kappa']=4*s['n']*Q(row['lambda'])
            cert={}
            for world in ['plus','minus']:
                w=row['fits'][world]['weights'] if 'fits' in row else atomfit(s,world)
                cert[world]=verify(signed_atoms(s,world),w,s['kappa'],U)
            gap=cert['plus'][1]-cert['minus'][1]
            absolute_lo=max(D(0),gap.a,gap.b.copy_negate());absolute_hi=gap.abs_upper()
            expected=row.get('probability_gap_above_0_99',row.get('gap_above_0_99',row.get('gap_certified_above_0_99')))
            assert expected==(absolute_lo>D('.99')),(name,k,gap.bounds())
            if name=='regularization_regimes':assert row['gap_certified_below_0_01']==(absolute_hi<D('.01'))
            if name in ['predictive_pilot','public_anchor']:assert cert['plus'][0].a>0>cert['minus'][0].b
            result.append({'experiment':name,'degree':k,'dataset':row.get('dataset'),'probe_repetitions':row.get('probe_repetitions'),'policy':row.get('policy'),'ridge':row.get('lambda'),'gap':gap.bounds(),'score_intervals':{w:v[0].bounds() for w,v in cert.items()},'radius_upper':{w:str(v[2]) for w,v in cert.items()},'gap_above_0_99':absolute_lo>D('.99'),'gap_below_0_01':absolute_hi<D('.01')})
        print(name,'verified',flush=True)
    expanded=[]
    for row in load('expanded'):
        s=specification(row['degree']); world=row['world'];w=[D.from_float(v) for v in row['weights']]
        atoms=[(tuple((2*y-1)*v for v in embed_intercept(x)),c) for x,y,c in retained_rows(s,world)]
        score,prob,r=verify(atoms,w,s['n']*s['ridge_with_intercept'],embed_intercept(U))
        assert score.a>0 if world=='plus' else score.b<0
        assert r < D('5.4e-11' if row['degree']==2 else '2.3e-6')
        expanded.append({'degree':row['degree'],'world':world,'rows':row['actual_expanded_rows'],'score':score.bounds(),'radius_upper':str(r)})
    source=[]
    for k in [2,4,8,16,32,64,128,256]:
        s=specification(k); q=s['q']; h=I(s['h']); Lg=I(2*q*2**s['b']).ln();A=(Lg-(2*Lg).ln())/h;B=2*I(8*q).ln()/(3*h)
        def g(t,w):return I(s['kappa'])*t-sum(2*a['count']*I(a['x'])/(1+(I(a['x'])*t).exp()) for a in s['atoms'] if a['world']==w)
        assert g(A,'A').b<0 and g(B,'B').a>0
        source.append({'degree':k,'A_gradient':g(A,'A').bounds(),'B_gradient':g(B,'B').bounds()})
    out={'status':'PASS','precision_decimal_digits':P,'python':platform.python_version(),'paired_cases':result,'expanded_archived_candidates':expanded,'scalar_bound_checks':source,'scope':'interval checks of archived base/public/expanded candidates and newly computed ablation/regime candidates; not an Arb replay or a new expanded training run'}
    (ROOT/'results/decimal_certificates.json').write_text(json.dumps(out,indent=2)+'\n')
    print('PASS',len(result),'paired cases;',len(expanded),'expanded candidates;',len(source),'scalar bounds',flush=True)
if __name__=='__main__':main()
