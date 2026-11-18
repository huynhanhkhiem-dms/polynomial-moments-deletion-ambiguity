"""Exact first distinguishing moment of the constructed pair."""
from pathlib import Path
from math import factorial
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from predictive_family import specification,retained_rows,embed_intercept
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=[]
    for k in [2,4,8,16,32,64,128,256]:
        s=specification(k);q=s['q'];grid=2**(s['b']+1);values={}
        for world in ['plus','minus']:
            values[world]=sum(c*int(embed_intercept(x)[0]*grid)**q for x,y,c in retained_rows(s,world) if y==1)
        actual=values['plus']-values['minus'];expected=(-1)**(q+1)*s['T']*factorial(q)*2**q
        assert actual==expected and actual!=0
        rows.append({'hidden_degree':k,'first_distinguishing_degree':q,'integer_moment_difference':str(actual),
             'exact_formula_match':True,'scope':'pair-specific distinction at k+1, not a universal optimizer recovery bound'})
    (ROOT/'results/degree_threshold.json').write_text(json.dumps({'status':'PASS','rows':rows},indent=2))
    print('PASS: eight exact pair-specific degree thresholds')

if __name__=='__main__':main()
