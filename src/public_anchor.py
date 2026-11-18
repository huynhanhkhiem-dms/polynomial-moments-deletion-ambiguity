"""Fixed public rows mapped to bounded rational features, then query-aligned.

This is a public-data-anchored adversarial stress test, not a natural training
benchmark or a claim about generalization. Central reflection balances labels.
"""
from pathlib import Path
from fractions import Fraction as Q
import gzip,zipfile,io,hashlib
import numpy as np
from predictive_family import U
ROOT=Path(__file__).resolve().parents[1]

def load_public(name,seed=743,count=16):
    raw=ROOT/'data/raw'
    if name=='covertype':
        p=raw/'covertype.zip'
        with zipfile.ZipFile(p) as z:
            candidate=next(n for n in z.namelist() if n.endswith('covtype.data.gz'))
            with gzip.GzipFile(fileobj=io.BytesIO(z.read(candidate))) as g:
                lines=[g.readline().decode() for _ in range(2048)]
        arr=np.loadtxt(io.StringIO(''.join(lines)),delimiter=',')
    else:
        filename={'spambase':'spambase.data','magic':'magic.data','optdigits':'optdigits.data'}[name]
        p=raw/filename
        if name=='magic':
            lines=p.read_text().splitlines();arr=np.array([[float(v) for v in line.split(',')[:-1]]+[float(line.split(',')[-1]=='g')] for line in lines])
        else:arr=np.loadtxt(p,delimiter=',')
    rng=np.random.default_rng(seed);ids=rng.choice(len(arr),size=count,replace=False)
    values=arr[ids,:-1];scale=np.maximum(1.,np.max(np.abs(values),axis=0))
    # Every binary64 value is converted to its exact rational before the map.
    signs=rng.choice([-1,1],size=values.shape[1]);public=[]
    for index,vals in zip(ids,values):
        projected=sum((int(sign)*Q(float(v))/Q(float(sc)) for sign,v,sc in zip(signs,vals,scale)),Q(0))/len(vals)
        den=1+projected*projected
        a=2*projected/den;bb=(1-projected*projected)/den
        x=((a+bb)/2,(-a+bb)/2)
        assert sum(v*v for v in x)==Q(1,2)
        y=int(arr[index,-1])
        if name=='optdigits':y=int(y>=5)
        if name=='covertype':y=int(y==2)
        public.append((x,y))
    q=public[0][0];v=tuple(U[j]-q[j] for j in range(2));den=sum(z*z for z in v)
    H=[[Q(int(i==j))-(2*v[i]*v[j]/den if den else 0) for j in range(2)] for i in range(2)]
    def transform(x):return tuple(sum(H[i][j]*x[j] for j in range(2)) for i in range(2))
    assert transform(q)==U
    assert all(sum(H[i][l]*H[j][l] for l in range(2))==int(i==j) for i in range(2) for j in range(2))
    common=[]
    for x,y in public:
        z=transform(x);common.extend([(z,y),(tuple(-v for v in z),1-y)])
    metadata={'dataset':name,'seed':seed,'selected_original_rows':count,'common_rows_after_central_reflection':len(common),
        'row_indices':ids.tolist(),'query_original_row_index':int(ids[0]),'all_feature_max_abs_scale':scale.tolist(),'projection_signs':signs.tolist(),
        'input_file':p.name,'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'covertype_prefix_rows':2048 if name=='covertype' else None,
        'transform':'exact rational signed-mean projection of all scaled features and stereographic circle map, followed by a rational norm-squared-one-half map, Householder alignment of first selected query; central reflection with opposite label',
        'scope':'public-data-anchored adversarial stress test; not natural-data utility or accuracy'}
    return common,metadata
