"""Fixed-score exact occurrence counts via an augmented (count,label) chain."""
import math
from .core import check,infer,score
from .conditioning import expected_features
from .model import potentials

NEG=-math.inf

def ls(xs):
    xs=list(xs)
    if not xs:return NEG
    m=max(xs)
    return m if m==NEG else m+math.log(math.fsum(math.exp(x-m) for x in xs))

def count_event(emissions,transition,start,end,label,count):
    k=check(emissions,transition,start,end);n=len(emissions)
    if n>64:raise ValueError('count inference supports at most 64 positions')
    if type(label)is not int or not 0<=label<k:raise ValueError('label must be an integer index in range')
    if type(count)is not int or not 0<=count<=n:raise ValueError('count must be an integer from zero to sequence length')
    ids=range(k);cs=range(n+1);hit=[int(j==label) for j in ids]
    table=lambda:[[[NEG]*k for _ in cs] for _ in emissions]
    a=table();v=table();back=[[[None]*k for _ in cs] for _ in emissions]
    for j in ids:a[0][hit[j]][j]=v[0][hit[j]][j]=start[j]+emissions[0][j]
    for t in range(1,n):
        for c in range(t+2):
            for j in ids:
                prev=c-hit[j]
                if prev<0:continue
                a[t][c][j]=emissions[t][j]+ls(a[t-1][prev][i]+transition[i][j] for i in ids)
                i=max(ids,key=lambda i:v[t-1][prev][i]+transition[i][j])
                if v[t-1][prev][i]!=NEG:v[t][c][j]=emissions[t][j]+v[t-1][prev][i]+transition[i][j];back[t][c][j]=i
    zs=[ls(a[-1][c][j]+end[j] for j in ids) for c in cs];z=zs[count];base=infer(emissions,transition,start,end);logz=base['log_partition']
    # b[t][q][i] counts occurrences strictly AFTER t, given current label i.
    b=table()
    for i in ids:b[-1][0][i]=end[i]
    for t in range(n-2,-1,-1):
        for q in range(n-t):
            for i in ids:b[t][q][i]=ls(transition[i][j]+emissions[t+1][j]+b[t+1][q-hit[j]][j] for j in ids if q>=hit[j])
    node=[[math.exp(ls(a[t][c][j]+b[t][count-c][j] for c in range(count+1))-z) for j in ids] for t in range(n)]
    edge=[[[math.exp(ls(a[t][c][i]+transition[i][j]+emissions[t+1][j]+b[t+1][count-c-hit[j]][j] for c in range(count-hit[j]+1))-z) for j in ids] for i in ids] for t in range(n-1)]
    j=max(ids,key=lambda j:v[-1][count][j]+end[j]);path=[j];c=count
    for t in range(n-1,0,-1):i=back[t][c][j];c-=hit[j];j=i;path.append(j)
    path.reverse();best=score(emissions,transition,start,end,path)
    expected=math.fsum(start[j]*node[0][j]+end[j]*node[-1][j] for j in ids)+math.fsum(row[j]*p[j] for row,p in zip(emissions,node) for j in ids)+math.fsum(transition[i][j]*p[i][j] for p in edge for i in ids for j in ids)
    # Independent draws from these conditioned node marginals do not enforce count.
    independent=[1.]
    for row in node:
        p=min(1.,max(0.,row[label]));nextrow=[0.]*(len(independent)+1)
        for c,w in enumerate(independent):nextrow[c]+=w*(1-p);nextrow[c+1]+=w*p
        independent=nextrow
    clean=lambda x:[clean(v) if isinstance(v,list) else None if v==NEG else v for v in x]
    logps=[min(0.,q-logz) for q in zs];probs=[math.exp(q) for q in logps];logp=logps[count]
    return dict(schema='taglab-count-event-v1',label=label,count=count,length=n,support_count=str(math.comb(n,count)*(k-1)**(n-count)),unrestricted_log_partition=logz,unrestricted_node=base['node'],count_log_partitions=zs,count_log_probabilities=logps,count_probabilities=probs,log_event_probability=logp,event_probability=probs[count],probability_underflow=probs[count]==0.,alpha=clean(a),suffix_beta=clean(b),conditional=dict(log_partition=z,node=node,edge=edge,viterbi=path,best_score=best,path_log_probability=best-z,entropy=z-expected,marginal_modes=[max(ids,key=lambda j:row[j]) for row in node]),independent_node_count_probabilities=independent,independent_node_event_probability=independent[count],conditioned_expected_count=math.fsum(row[label] for row in node),unrestricted_expected_count=math.fsum(row[label] for row in base['node']),refitted=False)

def label_count(theta,symbols,label,count):
    p=potentials(theta,symbols);out=count_event(*p,label,count);before=expected_features(symbols,infer(*p));after=expected_features(symbols,out['conditional']);out['log_event_gradient']=[a-b for a,b in zip(after,before)];return out
