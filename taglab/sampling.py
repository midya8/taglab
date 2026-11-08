"""Backward-conditional path sampling with supplied uniforms and visible CDF bins."""
import math
from .core import score,finite
from .conditioning import condition


def uniforms(seed,count):
    """Portable xorshift32 mid-bin lattice; reproducible, not cryptographic randomness."""
    if type(seed)is not int or not 1<=seed<2**32:raise ValueError('seed must be an integer from 1 to 4294967295')
    if type(count)is not int or not 1<=count<=131072:raise ValueError('uniform count must be 1 through 131072')
    out=[];s=seed
    for _ in range(count):
        s^=(s<<13)&0xffffffff;s^=s>>17;s^=(s<<5)&0xffffffff;s&=0xffffffff
        out.append((s+.5)/2**32)
    return out


def prepare(emissions,transition,start,end,allowed):
    r=condition(emissions,transition,start,end,allowed)
    if not r['possible']:raise ValueError('cannot sample an impossible label event')
    return dict(emissions=[row[:] for row in emissions],transition=[row[:] for row in transition],start=start[:],end=end[:],event=r)


def draw(prepared,values,mode='joint'):
    if mode not in ['joint','independent']:raise ValueError('mode must be joint or independent')
    e=prepared['emissions'];tr=prepared['transition'];start=prepared['start'];end=prepared['end'];event=prepared['event'];r=event['conditional'];n=len(e);k=len(start)
    if not isinstance(values,list) or len(values)!=n:raise ValueError('one uniform per position is required')
    for u in values:
        if not 0<=finite(u)<1:raise ValueError('uniforms must be in [0,1)')
    path=[None]*n;trace=[];numeric_logp=0.;model_kernel_logp=0.
    for t in range(n-1,-1,-1):
        if mode=='independent':logs=[None if j not in event['allowed'][t] else r['alpha'][t][j]+r['beta'][t][j] for j in range(k)]
        else:logs=[None if j not in event['allowed'][t] else r['alpha'][t][j]+(end[j] if t==n-1 else tr[j][path[t+1]]) for j in range(k)]
        m=max(x for x in logs if x is not None);weights=[0. if x is None else math.exp(x-m) for x in logs];total=math.fsum(weights);probs=[w/total for w in weights];logtotal=m+math.log(total);logprobs=[None if x is None else x-logtotal for x in logs]
        cdf=[0.]
        for p in probs:cdf.append(min(1.,cdf[-1]+p))
        last=max(j for j,p in enumerate(probs) if p>0);cdf[last+1:]=[1.]*(k-last)
        j=next((j for j,p in enumerate(probs) if p>0 and values[t]<cdf[j+1]),last);path[t]=j
        numeric_logp+=math.log(probs[j]);model_kernel_logp+=logprobs[j]
        trace.append(dict(position=t,next_label=path[t+1] if mode=='joint' and t+1<n else None,uniform=values[t],probabilities=probs,log_probabilities=logprobs,cdf=cdf,chosen=j,bin=[cdf[j],cdf[j+1]],underflow_labels=[j for j in range(k) if logs[j] is not None and weights[j]==0],collapsed_cdf_labels=[j for j in range(k) if probs[j]>0 and cdf[j+1]<=cdf[j]]))
    model_logp=score(e,tr,start,end,path)-r['log_partition']
    return dict(schema='taglab-sample-v1',mode=mode,path=path,uniforms=list(values),trace=trace,model_path_log_probability=model_logp,proposal_log_probability=model_kernel_logp,numeric_kernel_log_probability=numeric_logp,conditional_event_log_probability=event['log_event_probability'],finite_precision=True)


def sample(emissions,transition,start,end,allowed,values,mode='joint'):
    return draw(prepare(emissions,transition,start,end,allowed),values,mode)
