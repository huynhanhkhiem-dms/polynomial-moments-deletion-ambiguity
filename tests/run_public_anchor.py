from pathlib import Path
import sys,json,time
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from public_anchor import load_public
from predictive_family import specification,weighted_fit,verify_fit,U,signed_atoms,source_root
from flint import arb,ctx
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'results/public_anchor_protocol.json'
    if p.exists():raise FileExistsError('preserve completed public anchor outputs')
    protocol={'datasets':['spambase','magic','optdigits','covertype'],'seed':743,'original_rows_per_dataset':16,
      'degrees':[2,4,8,16],'query':'first selected public row after declared fixed representation',
      'gate':'retain all degrees and failures; degree 16 must pass signed margins for every dataset',
      'written_before_results_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    p.write_text(json.dumps(protocol,indent=2));rows=[];data=[]
    for name in protocol['datasets']:
        common,metadata=load_public(name);data.append(metadata)
        for k in protocol['degrees']:
            s=specification(k,common);a=source_root(s,'A',512);b=source_root(s,'B',512)
            mp.mp.dps=160
            midpoint=lambda z:mp.mpf(str(z.mid()).strip(' []').split('+/-')[0])
            aa,bb=midpoint(a),midpoint(b);certs={};fits={}
            for world,initial in [('plus',[aa,bb]),('minus',[bb,aa])]:
                fits[world]=weighted_fit(s,world,initial,160);certs[world]=verify_fit(s,world,fits[world],bits=1024)
            ctx.prec=1024;lp=arb(certs['plus']['query_score_interval']);lm=arb(certs['minus']['query_score_interval'])
            probp=arb(certs['plus']['query_probability_interval']);probm=arb(certs['minus']['query_probability_interval'])
            # Same quadratic Taylor model in both compatible worlds; it has extra rounding, so is an illustrative baseline.
            mq=lambda x:mp.mpf(x.numerator)/x.denominator
            G=mp.matrix(2);signed=mp.matrix(2,1)
            for x,c in signed_atoms(s,'plus'):
                z=[mq(v) for v in x]
                for i in range(2):
                    signed[i]+=c*z[i]
                    for j in range(2):G[i,j]+=c*z[i]*z[j]
            wt=mp.lu_solve(G+4*mq(s['kappa'])*mp.eye(2),2*signed)
            taylor_score=sum(mq(U[j])*wt[j] for j in range(2));taylor_prob=1/(1+mp.exp(-taylor_score))
            row={'dataset':name,'degree':k,'N':str(s['N']),'common_rows':s['p'],'deletion_fraction_float':s['deleted']/s['N'],
              'opposite_classifications':bool(lp>0 and lm<0),'probability_gap_interval':str(probp-probm),
              'gap_above_0_99':bool(probp-probm>arb(99)/100),'certificates':certs,'fits':fits,
              'quadratic_taylor_query_probability':mp.nstr(taylor_prob,40),
              'constant_baseline_probability':'0.5','best_pair_midpoint_probability':str((probp+probm)/2),
              'pair_minimax_absolute_error_lower':str(((probp-probm)/2).lower())}
            rows.append(row);(ROOT/'results/public_anchor.json').write_text(json.dumps({'status':'running','data':data,'rows':rows},indent=2))
            print(name,k,'opposite',row['opposite_classifications'],'gap99',row['gap_above_0_99'],flush=True)
    assert all(r['opposite_classifications'] and r['gap_above_0_99'] for r in rows if r['degree']==16)
    (ROOT/'results/public_anchor.json').write_text(json.dumps({'status':'PASS','data':data,'rows':rows,
      'scope':'16 paired public-data-anchored adversarial cases; 32 weighted atom fits, not natural-data benchmark efficacy'},indent=2))

if __name__=='__main__':main()
