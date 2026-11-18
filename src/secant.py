"""Rowwise secant enclosures for L2-regularized binary logistic ERM.

Real-arithmetic enclosure theorem; FP64 is an empirical implementation, NOT an
outward-rounded proof certificate. All coefficients including the intercept are
regularized. Losses are averaged over the RETAIN set.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve, eigh
from scipy.special import expit

def curvature(z):
    q=np.exp(-np.abs(z)); return q/(1+q)**2

def phi(a):
    a=np.asarray(a,dtype=float)
    return np.divide(-np.expm1(-a),a,out=np.ones_like(a),where=a!=0)

def secant_floor(z,a):
    """Exact inf_h [sigmoid(z+h)-sigmoid(z)]/h for |h|<=a."""
    z=np.asarray(z);a=np.asarray(a)
    if np.any(a<0): raise ValueError('radius must be nonnegative')
    q=np.exp(-np.abs(z))
    return q*phi(a)/((1+q)*(1+q*np.exp(-a)))

def objective(X,y,w,lam):
    z=X@w
    return float(np.mean(np.logaddexp(0,z)-y*z)+lam/2*(w@w))

def derivatives(X,y,w,lam):
    z=X@w; q=curvature(z)
    g=X.T@(expit(z)-y)/len(y)+lam*w
    H=(X.T*q)@X/len(y)+lam*np.eye(X.shape[1])
    return g,H

def fit(X,y,lam,start=None,tol=2e-11,maxiter=100):
    """Damped Newton reference solver with explicit first-order residual."""
    w=np.zeros(X.shape[1]) if start is None else start.copy()
    history=[];begin=time.perf_counter()
    for k in range(maxiter):
        g,H=derivatives(X,y,w,lam)
        gnorm=float(np.linalg.norm(g));history.append(gnorm)
        if gnorm<=tol: break
        step=cho_solve((cholesky(H,lower=True),True),g)
        f=objective(X,y,w,lam);dec=float(g@step);scale=1.
        while scale>2**-35:
            trial=w-scale*step
            if objective(X,y,trial,lam)<=f-1e-4*scale*dec+2e-15: break
            scale*=.5
        w=trial
    g,_=derivatives(X,y,w,lam)
    return w,{'iterations':k,'gradient_norm':float(np.linalg.norm(g)),
              'seconds':time.perf_counter()-begin,'converged':bool(np.linalg.norm(g)<=tol),
              'residual_history':history}

@dataclass
class Enclosure:
    radius: float
    method: str
    seconds: float
    evaluations: int
    lhs: float
    nu: float
    valid_real_arithmetic: bool
    lower_radius: float = 0.

class Geometry:
    def __init__(self,X,y,c,lam):
        begin=time.perf_counter()
        self.X=X;self.y=y;self.c=c;self.lam=lam;self.z=X@c
        self.g,self.H=derivatives(X,y,c,lam)
        self.L=cholesky(self.H,lower=True)
        self.V=solve_triangular(self.L,X.T,lower=True).T
        self.ell=np.linalg.norm(self.V,axis=1)
        invL=solve_triangular(self.L,np.eye(X.shape[1]),lower=True)
        self.reg=lam*(invL@invL.T)
        self.q=curvature(self.z)
        self.nu=float(np.linalg.norm(solve_triangular(self.L,self.g,lower=True)))
        self.mu_reg=float(eigh(self.reg,subset_by_index=[0,0],eigvals_only=True)[0])
        self.preparation_seconds=time.perf_counter()-begin

    def lower_matrix(self,R,method):
        a=R*self.ell
        if method in ['compressed','compressed_upper']:
            qexp=np.exp(-np.abs(self.z))
            relative=phi(a)*(1+qexp)/(1+qexp*np.exp(-a))
            if method=='compressed':
                extrema=np.full(self.group_count,np.inf)
                np.minimum.at(extrema,self.group,relative)
                extrema[~np.isfinite(extrema)]=0.
            else:
                extrema=np.zeros(self.group_count)
                np.maximum.at(extrema,self.group,relative)
            return self.reg+np.einsum('b,bij->ij',extrema,self.group_grams,optimize=False)
        if method=='secant': coeff=secant_floor(self.z,a)
        elif method=='row_exp': coeff=self.q*phi(a)
        elif method=='hessian_floor': coeff=curvature(np.abs(self.z)+a)
        elif method=='uniform_exp':
            f=float(phi(np.array(R*float(np.max(self.ell)))))
            return self.reg+f*(np.eye(self.X.shape[1])-self.reg)
        else: raise ValueError(method)
        return self.reg+(self.V.T*coeff)@self.V/len(self.y)

    def prepare_compression(self,groups=64,reference_R=None):
        """Fixed nested partition; min relative curvature keeps a PSD floor.

        Group formation never uses the retrained optimum. Reference radius is
        twice the directly observed Newton decrement by default.
        """
        begin=time.perf_counter()
        groups=min(int(groups),len(self.y))
        R=2*self.nu if reference_R is None else reference_R
        a=R*self.ell;qexp=np.exp(-np.abs(self.z))
        relative=phi(a)*(1+qexp)/(1+qexp*np.exp(-a))
        order=np.argsort(relative,kind='stable')
        self.group=np.empty(len(self.y),dtype=int)
        self.group[order]=(np.arange(len(self.y))*groups)//len(self.y)
        self.group_count=groups
        self.group_grams=np.empty((groups,self.X.shape[1],self.X.shape[1]))
        offsets=np.r_[0,np.cumsum(np.bincount(self.group,minlength=groups))]
        for b in range(groups):
            ids=order[offsets[b]:offsets[b+1]];V=self.V[ids]
            self.group_grams[b]=(V.T*self.q[ids])@V/len(self.y)
        return time.perf_counter()-begin

    def compression_spread(self,R):
        a=R*self.ell;qexp=np.exp(-np.abs(self.z))
        relative=phi(a)*(1+qexp)/(1+qexp*np.exp(-a))
        minima=np.full(self.group_count,np.inf);maxima=np.zeros(self.group_count)
        np.minimum.at(minima,self.group,relative);np.maximum.at(maxima,self.group,relative)
        return float(np.max(np.log(np.maximum(maxima,1e-300)/np.maximum(minima,1e-300))))

    def adaptive_enclose(self,relative_tolerance=.05,max_groups=256):
        """Return a true lower-matrix enclosure and an approximation bracket.

        The upper matrix is used ONLY to bound the uncompressed radius from
        below. It is not an enclosure of the optimizer. Fallback is explicit.
        All claims here are under exact real arithmetic, as elsewhere.
        """
        begin=time.perf_counter();attempts=[]
        target=self.nu*(1+1e-9)+1e-15
        if self.nu==0:
            cert=self.enclose('secant')
            return cert,{'groups':0,'fallback':False,'lower_radius':0.,'bracket_ratio':1.,
                         'seconds':time.perf_counter()-begin,'attempts':[]}
        for count in [4,16,64,256]:
            if count>max_groups:break
            prep=self.prepare_compression(count)
            lower=self.enclose('compressed')
            # One upper-matrix evaluation establishes the required lower root
            # bracket; solving a second root at each level is unnecessary.
            candidate_lo=lower.radius/(1+relative_tolerance)
            upper_lhs=self.lhs(candidate_lo,'compressed_upper')
            attempts.append({'groups':count,'preparation_seconds':prep,'upper_test_gap':upper_lhs-target})
            if upper_lhs<target:
                return lower,{'groups':count,'fallback':False,'lower_radius':candidate_lo,
                    'bracket_ratio':1+relative_tolerance,'seconds':time.perf_counter()-begin,'attempts':attempts}
        exact=self.enclose('secant')
        return exact,{'groups':0,'fallback':True,'lower_radius':exact.lower_radius,
            'bracket_ratio':exact.radius/max(exact.lower_radius,1e-300),
            'seconds':time.perf_counter()-begin,'attempts':attempts}

    def lhs(self,R,method):
        if method=='uniform_exp':
            f=float(phi(np.array(R*float(np.max(self.ell)))))
            return R*(self.mu_reg+f*(1-self.mu_reg))
        M=self.lower_matrix(R,method)
        return R*float(eigh(M,subset_by_index=[0,0],eigvals_only=True)[0])

    def enclose(self,method='secant',rtol=1e-6):
        begin=time.perf_counter();count=0
        if self.nu==0:
            return Enclosure(0.,method,time.perf_counter()-begin,0,0.,0.,True)
        # λ>0 implies an enclosure exists. A strict slack is retained to avoid
        # accepting equality due to scalar root-finder rounding. This slack is
        # NOT a substitute for interval arithmetic on the matrix computations.
        target=self.nu if method=='compressed_upper' else self.nu*(1+1e-9)+1e-15
        lo=0.;hi=max(self.nu*1.05,1e-10)
        for _ in range(90):
            val=self.lhs(hi,method);count+=1
            if val>target: break
            hi*=2
        else:
            return Enclosure(float('inf'),method,time.perf_counter()-begin,count,val,self.nu,False)
        # Monotonicity applies to secant and both integrated exponential rules.
        # Hessian-floor need not be monotone; do not use it in a bisection.
        if method=='hessian_floor':
            return Enclosure(hi,method,time.perf_counter()-begin,count,val,self.nu,True)
        for _ in range(55):
            mid=(lo+hi)/2;val=self.lhs(mid,method);count+=1
            if val>target: hi=mid
            else: lo=mid
            if hi-lo <= rtol*max(hi,1e-12): break
        val=self.lhs(hi,method);count+=1
        return Enclosure(hi,method,time.perf_counter()-begin,count,val,self.nu,
                         method!='compressed_upper' and val>self.nu,lower_radius=lo)

    def norm(self,w):
        return float(np.linalg.norm(self.L.T@(w-self.c)))

    def prediction_halfwidth(self,Z,enclosure):
        return enclosure.radius*np.linalg.norm(solve_triangular(self.L,Z.T,lower=True).T,axis=1)
