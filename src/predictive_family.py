"""Exact dyadic binomial construction for retained-point reference ambiguity.

Multiplicities are literal unweighted counts, represented implicitly except in
separate expanded experiments. Global polynomial moments and a common initial
trained state are supported. Arbitrary nonlinear radial groups are excluded.
"""
from fractions import Fraction as Q
from math import comb
U=(Q(1,2),Q(-1,2))
RHO=Q(1,2)
INTERCEPT=Q(1,2)

def probe_block():
    return [(U,1),(U,0),(tuple(-x for x in U),1),(tuple(-x for x in U),0)]

def specification(k,common=None):
    if not isinstance(k,int) or k<2:raise ValueError('degree at least two')
    q=k+1;b=(q*q-1).bit_length();h=Q(1,2**b);T=2**(3*b)
    atoms=[{'j':j,'x':(2*j-1)*h,'count':T*comb(q,j),'world':'A' if j%2 else 'B'} for j in range(q+1)]
    m=T*2**q;mu=(q-1)*h;S=m*mu
    assert S.denominator==1;S=int(S);assert 0<S<m and S%2==0
    common=probe_block() if common is None else common
    assert all(len(x)==2 and sum(z*z for z in x)<=1 and y in [0,1] for x,y in common)
    # Central reflection makes the penalized intercept exactly zero for every active parameter.
    from collections import Counter
    counts=Counter((tuple(x),y) for x,y in common)
    assert all(counts[(tuple(-v for v in x),1-y)]==n for (x,y),n in counts.items())
    p=len(common);N=2*m+2*S+2*p;n=2*m+p
    return {'degree':k,'q':q,'b':b,'h':h,'T':T,'atoms':atoms,'m':m,'mu':mu,'S':S,'p':p,
        'N':N,'n':n,'deleted':2*S+p,'kappa':Q(n,N),'source_lambda':Q(n,m*N),
        'ridge_with_intercept':RHO*RHO/N,'common':common}

def core_rows(spec,world):
    out=[]
    for axis,wanted in [(0,'A' if world=='plus' else 'B'),(1,'B' if world=='plus' else 'A')]:
        for a in spec['atoms']:
            if a['world']!=wanted:continue
            for label in [1,0]:
                x=[Q(0),Q(0)];x[axis]=(2*label-1)*a['x']
                out.append((tuple(x),label,a['count']))
    return out

def retained_rows(spec,world):return core_rows(spec,world)+[(x,y,1) for x,y in spec['common']]

def deletion_rows(spec):
    out=[]
    for axis in [0,1]:
        for sign,label in [(1,0),(-1,1)]:
            x=[Q(0),Q(0)];x[axis]=Q(sign);out.append((tuple(x),label,spec['S']//2))
    return out+[(x,1-y,1) for x,y in spec['common']]

def embed_intercept(x):return tuple(RHO*z for z in x)+(INTERCEPT,)

def signed_atoms(spec,world):
    # Aggregate literal duplicates; the resulting counts still denote every row.
    out={}
    for x,y,c in retained_rows(spec,world):
        z=tuple((2*y-1)*v for v in x);out[z]=out.get(z,0)+c
    return list(out.items())

def source_root(spec,world,bits=512):
    from flint import arb,ctx
    ctx.prec=bits
    aq=lambda x:arb(x.numerator)/x.denominator
    ridge=aq(spec['kappa'])
    def grad(t):
        return ridge*t-sum((2*a['count']*aq(a['x'])/(1+(aq(a['x'])*t).exp()) for a in spec['atoms'] if a['world']==world),arb(0))
    lo=arb(0);hi=arb(1);assert grad(lo)<0
    for _ in range(100):
        if grad(hi)>0:break
        hi*=2
    else:raise ArithmeticError('source bracket failed')
    for _ in range(240):
        mid=(lo+hi)/2;v=grad(mid)
        if v<0:lo=mid
        elif v>0:hi=mid
        else:break
    return arb(((lo+hi)/2).mid(),((hi-lo)/2).upper())

def analytic_bounds(spec,bits=512):
    from flint import arb,ctx
    ctx.prec=bits;q=spec['q'];h=arb(1)/2**spec['b']
    L=arb(2*q*2**spec['b']).log()
    A=(L-(2*L).log())/h
    B=arb(2)/3*arb(8*q).log()/h
    return A,B

def weighted_fit(spec,world,initial,dps=160):
    """Newton solve on the finite atom representation, with arbitrary precision."""
    import mpmath as mp
    mp.mp.dps=dps
    def mq(x):return mp.mpf(x.numerator)/x.denominator
    atoms=[(tuple(mq(z) for z in x),mp.mpf(c)) for x,c in signed_atoms(spec,world)]
    ridge=mq(spec['kappa']);w=mp.matrix(initial);history=[];dim=len(U)
    def values(w,hessian=True):
        val=ridge*sum(z*z for z in w)/2;g=ridge*w
        H=ridge*mp.eye(dim) if hessian else None
        for x,c in atoms:
            t=sum(x[j]*w[j] for j in range(dim))
            if t>=0:
                e=mp.exp(-t);r=e/(1+e);loss=mp.log1p(e)
            else:
                e=mp.exp(t);r=1/(1+e);loss=-t+mp.log1p(e)
            val+=c*loss
            for j in range(dim):g[j]-=c*x[j]*r
            if hessian:
                h=c*e/(1+e)**2
                for j in range(dim):
                    for l in range(dim):H[j,l]+=h*x[j]*x[l]
        return val,g,H
    for it in range(80):
        val,g,H=values(w);res=sum(abs(z) for z in g)
        history.append(mp.nstr(res,10))
        if res<mp.mpf('1e-50'):
            return {'weights':[mp.nstr(z,dps-15) for z in w],'iterations':it,
                    'gradient_l1_mp':mp.nstr(res,12),'history':history,'dps':dps}
        step=mp.lu_solve(H,g);decrease=sum(g[j]*step[j] for j in range(dim));alpha=mp.mpf(1)
        for ls in range(60):
            trial=w-alpha*step;nextval,_,_=values(trial,False)
            if nextval<=val-mp.mpf('0.01')*alpha*decrease:break
            alpha/=2
        else:raise ArithmeticError('weighted Newton line search failed')
        w=trial
    raise ArithmeticError('weighted Newton iteration limit')

def verify_fit(spec,world,fit,query=U,bits=1024):
    """Arb residual verifies the exact-rational optimizer of the implicit multiset."""
    from flint import arb,ctx
    ctx.prec=bits;dim=len(U)
    aq=lambda x:arb(x.numerator)/x.denominator
    ridge=aq(spec['kappa']);w=[arb(z) for z in fit['weights']];g=[ridge*z for z in w]
    for x,c in signed_atoms(spec,world):
        v=[aq(z) for z in x];t=sum((v[j]*w[j] for j in range(dim)),arb(0))
        r=1/(1+t.exp())
        for j in range(dim):g[j]-=c*v[j]*r
    residual=sum((abs(z).upper() for z in g),arb(0)).upper()
    radius=(residual/ridge).upper()
    score=sum((aq(query[j])*w[j] for j in range(dim)),arb(0))
    exact_score=score+arb(0,radius)
    probability=1/(1+(-exact_score).exp())
    return {'bits':bits,'gradient_l1_upper':str(residual),'effective_parameter_error_upper':str(radius),
            'query_score_interval':str(exact_score),'query_probability_interval':str(probability),
            'scope':'exact rational weighted rows; no materialization of large replication counts'}
