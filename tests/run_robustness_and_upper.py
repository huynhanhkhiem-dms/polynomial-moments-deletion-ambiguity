from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,probe_block,weighted_fit,verify_fit,source_root,U
from flint import arb,ctx
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'results/robustness_protocol.json'
    if p.exists():raise FileExistsError('preserve completed robustness')
    p.write_text(json.dumps({'degrees':[2,4,8],'probe_repetitions':[1,8,64,512],
      'gate':'report every probe multiplicity; no promised success at fixed degree as common query weight grows',
      'written_before_results_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    rows=[]
    for k in [2,4,8]:
        for repetitions in [1,8,64,512]:
            s=specification(k,probe_block()*repetitions);a=source_root(s,'A',512);b=source_root(s,'B',512);mp.mp.dps=160
            mid=lambda v:mp.mpf(str(v.mid()).strip(' []').split('+/-')[0]);aa,bb=mid(a),mid(b)
            certificates={}
            for world,initial in [('plus',[aa,bb]),('minus',[bb,aa])]:
                fit=weighted_fit(s,world,initial,160);certificates[world]=verify_fit(s,world,fit,bits=1024)
            ctx.prec=1024;pa=arb(certificates['plus']['query_probability_interval']);pb=arb(certificates['minus']['query_probability_interval'])
            gap=pa-pb
            rows.append({'degree':k,'probe_repetitions':repetitions,'common_rows':s['p'],'N':str(s['N']),
              'deletion_fraction':s['deleted']/s['N'],'certificates':certificates,'probability_gap_interval':str(gap),
              'gap_above_0_99':bool(gap>arb(99)/100)})
            print(k,repetitions,'gap99',rows[-1]['gap_above_0_99'],flush=True)
    (ROOT/'results/robustness.json').write_text(json.dumps({'status':'completed','rows':rows},indent=2))
    ctx.prec=768;upper=[]
    for lam_text in ['0.1','0.01','0.001']:
        lam=arb(lam_text);R=(2*arb(2).log()/lam).sqrt();a=arb.pi()/(2*R);rho=a+(1+a*a).sqrt()
        for degree in [2,4,8,16,32,64,128,256,512,1024]:
            log_bound=-(degree-1)*rho.log()-lam.log()-(rho-1).log()
            upper.append({'lambda':lam_text,'degree':degree,'log_unclipped_prediction_diameter_upper':str(log_bound.upper()),
                'informative_below_one':bool(log_bound<0),'prediction_diameter_below_0_01':bool(log_bound<arb('0.01').log())})
    (ROOT/'results/fixed_ridge_upper.json').write_text(json.dumps({'status':'PASS','rows':upper,
      'scope':'evaluation of an analytic universal bound; not empirical error or a new approximation method'},indent=2))

if __name__=='__main__':main()
