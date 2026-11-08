"""Linear-chain conditional scores. Transition rows are FROM, columns TO."""
import math


def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('expected finite real scores, not booleans')
    return float(value)


def lse(values):
    m=max(values)
    return m+math.log(math.fsum(math.exp(x-m) for x in values))


def check(emissions, transition, start, end):
    if not isinstance(emissions,list) or not emissions:
        raise ValueError('a nonempty sequence is required')
    if not isinstance(start,list) or not 2<=len(start)<=8:
        raise ValueError('two to eight labels are required')
    k=len(start)
    if not isinstance(end,list) or len(end)!=k or not isinstance(transition,list) or len(transition)!=k:
        raise ValueError('inconsistent score dimensions')
    for row in emissions+transition:
        if not isinstance(row,list) or len(row)!=k:raise ValueError('inconsistent score row')
        for x in row:finite(x)
    for x in start+end:finite(x)
    if max(abs(x) for row in emissions+transition for x in row)>1e6 or max(abs(x) for x in start+end)>1e6:
        raise ValueError('score magnitude exceeds supported bound')
    return k


def score(emissions, transition, start, end, path):
    k=check(emissions,transition,start,end)
    if not isinstance(path,list) or len(path)!=len(emissions) or any(type(i) is not int or not 0<=i<k for i in path):
        raise ValueError('invalid label path')
    return math.fsum([start[path[0]],end[path[-1]]]+[row[i] for row,i in zip(emissions,path)]+[transition[a][b] for a,b in zip(path,path[1:])])


def infer(emissions, transition, start, end):
    k=check(emissions,transition,start,end);n=len(emissions)
    alpha=[[start[j]+emissions[0][j] for j in range(k)]]
    for row in emissions[1:]:
        alpha.append([row[j]+lse([alpha[-1][i]+transition[i][j] for i in range(k)]) for j in range(k)])
    logz=lse([alpha[-1][j]+end[j] for j in range(k)])
    beta=[[0.]*k for _ in emissions];beta[-1]=list(end)
    for t in range(n-2,-1,-1):
        beta[t]=[lse([transition[i][j]+emissions[t+1][j]+beta[t+1][j] for j in range(k)]) for i in range(k)]
    node=[[math.exp(alpha[t][j]+beta[t][j]-logz) for j in range(k)] for t in range(n)]
    edge=[[[math.exp(alpha[t][i]+transition[i][j]+emissions[t+1][j]+beta[t+1][j]-logz) for j in range(k)] for i in range(k)] for t in range(n-1)]
    v=[[start[j]+emissions[0][j] for j in range(k)]];back=[[0]*k]
    for row in emissions[1:]:
        choices=[max(range(k),key=lambda i:v[-1][i]+transition[i][j]) for j in range(k)]
        v.append([row[j]+v[-1][choices[j]]+transition[choices[j]][j] for j in range(k)]);back.append(choices)
    last=max(range(k),key=lambda j:v[-1][j]+end[j]);path=[last]
    for t in range(n-1,0,-1):path.append(back[t][path[-1]])
    path.reverse();best=score(emissions,transition,start,end,path)
    expected=math.fsum(start[j]*node[0][j]+end[j]*node[-1][j] for j in range(k))+math.fsum(row[j]*p[j] for row,p in zip(emissions,node) for j in range(k))+math.fsum(transition[i][j]*p[i][j] for p in edge for i in range(k) for j in range(k))
    return dict(log_partition=logz,alpha=alpha,beta=beta,node=node,edge=edge,viterbi=path,viterbi_scores=v,backpointers=back,best_score=best,path_log_probability=best-logz,entropy=logz-expected,marginal_modes=[max(range(k),key=lambda j:row[j]) for row in node])
