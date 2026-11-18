from pathlib import Path
from fractions import Fraction as Q
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,source_root,weighted_fit,verify_fit
from flint import arb,ctx
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]

def main():
    protocol=ROOT/'results/regularization_protocol.json'
    if protocol.exists():raise FileExistsError('preserve regime outputs')
    protocol.write_text(json.dumps({'fixed_ridges':['1/10','1/100','1/1000'],'fixed_degrees':[2,4,8,16,32],
       'refreshed_ridge':'1/(4 retained_count)','refreshed_degrees':[2,8,32],
       'gate':'report every case; fixed-ridge high-degree ambiguity expected small, not forced by assertions',
       'written_before_results_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    cases=[(k,Q(lam),'fixed') for lam in ['1/10','1/100','1/1000'] for k in [2,4,8,16,32]]
    cases += [(k,None,'refreshed_inverse') for k in [2,8,32]]
    rows=[]
    for k,lam,policy in cases:
        s=specification(k)
        if lam is None:lam=Q(1,4*s['n'])
        s['ridge_with_intercept']=lam;s['kappa']=4*s['n']*lam;s['source_lambda']=s['kappa']/s['m']
        a=source_root(s,'A',768);b=source_root(s,'B',768);mp.mp.dps=180
        mid=lambda z:mp.mpf(str(z.mid()).strip(' []').split('+/-')[0]);aa,bb=mid(a),mid(b)
        certificates={}
        for world,initial in [('plus',[aa,bb]),('minus',[bb,aa])]:
            f=weighted_fit(s,world,initial,180);certificates[world]=verify_fit(s,world,f,bits=1024)
        ctx.prec=1024;pp=arb(certificates['plus']['query_probability_interval']);pm=arb(certificates['minus']['query_probability_interval'])
        gap=abs(pp-pm)
        rows.append({'policy':policy,'degree':k,'lambda':str(lam),'certificates':certificates,
           'absolute_probability_gap_interval':str(gap),'gap_certified_below_0_01':bool(gap<arb(1)/100),
           'gap_certified_above_0_99':bool(gap>arb(99)/100)})
        print(policy,k,str(lam),'small',rows[-1]['gap_certified_below_0_01'],'large',rows[-1]['gap_certified_above_0_99'],flush=True)
    (ROOT/'results/regularization_regimes.json').write_text(json.dumps({'status':'completed','rows':rows},indent=2))

if __name__=='__main__':main()
