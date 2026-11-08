"""Exact inference given position-wise hypothetical label subsets, at fixed scores."""
import math
from .core import check, infer, score
from .model import potentials,features,K,F,SIZE


def allowed_rows(allowed,n,k):
    if not isinstance(allowed,list) or len(allowed)!=n:raise ValueError('allowed must have one label-subset list per position')
    result=[]
    for row in allowed:
        if not isinstance(row,list) or any(type(j)is not int or not 0<=j<k for j in row) or len(set(row))!=len(row):raise ValueError('allowed labels must be unique integer indices in range')
        result.append(sorted(row))
    return result


def condition(emissions,transition,start,end,allowed):
    k=check(emissions,transition,start,end);n=len(emissions)
    if n>256:raise ValueError('conditioning supports at most 256 positions')
    rows=allowed_rows(allowed,n,k);base=infer(emissions,transition,start,end)
    count=math.prod(len(row) for row in rows)
    result=dict(schema='taglab-label-event-v1',allowed=rows,support_count=str(count),unrestricted_log_partition=base['log_partition'],unrestricted_node=base['node'])
    if not count:return dict(result,possible=False,log_event_probability=None,event_probability=0.,probability_underflow=False,conditional=None,impossible_reason='At least one position permits no label. A conditional distribution is undefined.')
    neg=-math.inf
    def lse(xs):
        m=max(xs);return neg if m==neg else m+math.log(math.fsum(math.exp(x-m) for x in xs))
    a=[[start[j]+emissions[0][j] if j in rows[0] else neg for j in range(k)]]
    for t in range(1,n):a.append([emissions[t][j]+lse([a[t-1][i]+transition[i][j] for i in rows[t-1]]) if j in rows[t] else neg for j in range(k)])
    z=lse([a[-1][j]+end[j] for j in rows[-1]])
    b=[[neg]*k for _ in range(n)];b[-1]=[end[j] if j in rows[-1] else neg for j in range(k)]
    for t in range(n-2,-1,-1):
        b[t]=[lse([transition[i][j]+emissions[t+1][j]+b[t+1][j] for j in rows[t+1]]) if i in rows[t] else neg for i in range(k)]
    node=[[math.exp(a[t][j]+b[t][j]-z) if j in rows[t] else 0. for j in range(k)] for t in range(n)]
    edge=[[[math.exp(a[t][i]+transition[i][j]+emissions[t+1][j]+b[t+1][j]-z) if i in rows[t] and j in rows[t+1] else 0. for j in range(k)] for i in range(k)] for t in range(n-1)]
    v=[[start[j]+emissions[0][j] if j in rows[0] else neg for j in range(k)]];back=[[None]*k]
    for t in range(1,n):
        choices=[max(rows[t-1],key=lambda i:v[-1][i]+transition[i][j]) if j in rows[t] else None for j in range(k)]
        v.append([emissions[t][j]+v[-1][choices[j]]+transition[choices[j]][j] if j in rows[t] else neg for j in range(k)]);back.append(choices)
    path=[max(rows[-1],key=lambda j:v[-1][j]+end[j])]
    for t in range(n-1,0,-1):path.append(back[t][path[-1]])
    path.reverse();best=score(emissions,transition,start,end,path)
    expected=math.fsum(start[j]*node[0][j]+end[j]*node[-1][j] for j in range(k))+math.fsum(row[j]*p[j] for row,p in zip(emissions,node) for j in range(k))+math.fsum(transition[i][j]*p[i][j] for p in edge for i in range(k) for j in range(k))
    logp=min(0.,z-base['log_partition']);prob=math.exp(logp)
    # Null encodes excluded (-infinity) messages for strict JSON; never a finite sentinel.
    clean=lambda table:[[None if x==neg else x for x in row] for row in table]
    conditional=dict(log_partition=z,node=node,edge=edge,alpha=clean(a),beta=clean(b),viterbi=path,best_score=best,path_log_probability=best-z,entropy=z-expected,marginal_modes=[max(range(k),key=lambda j:row[j]) for row in node],viterbi_scores=clean(v),backpointers=back)
    return dict(result,possible=True,log_event_probability=logp,event_probability=prob,probability_underflow=prob==0.,conditional=conditional)


def expected_features(symbols,r):
    rows=features(symbols);g=[0.]*SIZE
    for t,row in enumerate(rows):
        for j in range(K):
            for f,value in enumerate(row):g[j*F+f]+=r['node'][t][j]*value
    for pair in r['edge']:
        for i in range(K):
            for j in range(K):g[K*F+i*K+j]+=pair[i][j]
    for j in range(K):g[33+j]=r['node'][0][j];g[36+j]=r['node'][-1][j]
    return g


def label_event(theta,symbols,allowed):
    p=potentials(theta,symbols);out=condition(*p,allowed)
    if out['possible']:
        before=expected_features(symbols,infer(*p));after=expected_features(symbols,out['conditional'])
        out['log_event_gradient']=[a-b for a,b in zip(after,before)]
    else:out['log_event_gradient']=None
    return out
