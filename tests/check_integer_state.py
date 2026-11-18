from pathlib import Path
import sys,json,hashlib
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,retained_rows,embed_intercept
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
    (ROOT/'results/integer_state_replay.json').write_text(json.dumps({'status':'PASS','integer_state_checks':total,'states':outputs},indent=2)+'\n')
    print('PASS:',total,'integer numerators')
if __name__=='__main__':main()
