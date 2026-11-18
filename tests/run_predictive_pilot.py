"""Prospective kill-test before revising the manuscript."""
from pathlib import Path
import sys,json,time
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,weighted_fit,verify_fit,U,source_root,analytic_bounds
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'results/predictive_pilot_protocol.json'
    if p.exists():raise FileExistsError('preserve pilot protocol and output')
    p.write_text(json.dumps({'degrees':[2,4,8,16,32,64,128],'common_block':'four balanced dyadic bounded probe rows; their label reversals are also deleted',
      'gate':'exact initial identities separate; signed query margins must be certified; finite failures retained',
      'public_data_gate':'separate anchored stress test before manuscript revision',
      'written_before_results_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    rows=[]
    for k in [2,4,8,16,32,64,128]:
        s=specification(k);a=source_root(s,'A',max(384,8*k))
        b=source_root(s,'B',max(384,8*k))
        # The same midpoint source coordinates seed separate full 2D Newton fits.
        amid=str(a.mid());bmid=str(b.mid())
        def mpval(text):
            import mpmath as mp
            return mp.mpf(text.strip(' []').split('+/-')[0])
        fits={};certs={}
        for world,initial in [('plus',[amid,bmid]),('minus',[bmid,amid])]:
            import mpmath as mp
            mp.mp.dps=max(160,2*k+80)
            fits[world]=weighted_fit(s,world,[mpval(z) for z in initial],max(160,2*k+80))
            certs[world]=verify_fit(s,world,fits[world],bits=max(768,16*k))
        ctx.prec=max(768,16*k)
        plus=arb(certs['plus']['query_score_interval']);minus=arb(certs['minus']['query_score_interval'])
        pa=arb(certs['plus']['query_probability_interval']);pb=arb(certs['minus']['query_probability_interval'])
        gap=pa-pb
        row={'degree':k,'q':s['q'],'grid_bits':s['b'],'N':str(s['N']),'retained_count':str(s['n']),'deleted_count':str(s['deleted']),
             'deletion_fraction':str(Q(s['deleted'],s['N'])),'deletion_fraction_float':s['deleted']/s['N'],
             'ridge_with_intercept':str(s['ridge_with_intercept']),'source_lambda':str(s['source_lambda']),
             'base_optimizer_A':str(a),'base_optimizer_B':str(b),
             'fits':fits,'certificates':certs,'opposite_retained_query_classifications':bool(plus>0 and minus<0),
             'probability_gap_interval':str(gap),'probability_gap_above_0_99':bool(gap>arb(99)/100)}
        rows.append(row);(ROOT/'results/predictive_pilot.json').write_text(json.dumps({'status':'running','rows':rows},indent=2))
        print(k,'deleted',row['deletion_fraction_float'],'opposite',row['opposite_retained_query_classifications'],'gap',str(gap),flush=True)
    assert all(r['opposite_retained_query_classifications'] for r in rows)
    (ROOT/'results/predictive_pilot.json').write_text(json.dumps({'status':'PASS','rows':rows,
       'scope':'weighted exact-rational atom fits, not expanded N-row experiments; hypothetical novelty still under audit'},indent=2))

if __name__=='__main__':main()
