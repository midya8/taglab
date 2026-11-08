"""Fixed observed features, analytic expected-minus-observed gradient and Armijo fit."""
import math
from .core import finite,infer,score
K=3
F=8
FEATURES=['bias','current / 2','left / 2 (zero at edge)','right / 2 (zero at edge)','absolute current / 2','left edge','right edge','left times current / 4']
SIZE=K*F+K*K+2*K


def features(symbols):
    if not isinstance(symbols,list) or not symbols or len(symbols)>256 or any(type(x) is not int or not -2<=x<=2 for x in symbols):raise ValueError('symbols must be a nonempty list of at most 256 integers from -2 to 2')
    return [[1,x/2,(symbols[t-1]/2 if t else 0),(symbols[t+1]/2 if t+1<len(symbols) else 0),abs(x)/2,int(t==0),int(t+1==len(symbols)),(symbols[t-1]*x/4 if t else 0)] for t,x in enumerate(symbols)]


def parameters(theta):
    if not isinstance(theta,list) or len(theta)!=SIZE:raise ValueError('wrong parameter length')
    for x in theta:finite(x)
    w=[theta[i*F:(i+1)*F] for i in range(K)];a=[theta[K*F+i*K:K*F+(i+1)*K] for i in range(K)]
    return w,a,theta[K*F+K*K:K*F+K*K+K],theta[-K:]


def potentials(theta,symbols):
    w,a,s,e=parameters(theta);x=features(symbols)
    return [[math.fsum(v*z for v,z in zip(row,weight)) for weight in w] for row in x],a,s,e


def predict(theta,symbols):return infer(*potentials(theta,symbols))


def validate_data(data):
    if not isinstance(data,list) or not data:raise ValueError('nonempty labeled examples required')
    for item in data:
        if not isinstance(item,dict) or set(item)!={'x','y'}:raise ValueError('expected x/y examples')
        x=features(item['x']);y=item['y']
        if not isinstance(y,list) or len(y)!=len(x) or any(type(j)is not int or not 0<=j<K for j in y):raise ValueError('invalid labels')


def objective(theta,data,regularization,independent=False):
    parameters(theta);validate_data(data);lam=finite(regularization)
    if lam<=0:raise ValueError('strictly positive L2 regularization required')
    if type(independent)is not bool:raise ValueError('independent must be boolean')
    if independent and any(v!=0 for v in theta[K*F:]):raise ValueError('independent control has fixed zero sequence scores')
    grad=[0.]*SIZE;loss=0.
    for item in data:
        x=features(item['x']);y=item['y'];p=potentials(theta,item['x']);r=infer(*p);loss+=r['log_partition']-score(*p,y)
        for t,row in enumerate(x):
            for j in range(K):
                d=r['node'][t][j]-(y[t]==j)
                for f,value in enumerate(row):grad[j*F+f]+=d*value
        if not independent:
            for t,pair in enumerate(r['edge']):
                for i in range(K):
                    for j in range(K):grad[K*F+i*K+j]+=pair[i][j]-(y[t]==i and y[t+1]==j)
            for j in range(K):grad[K*F+K*K+j]+=r['node'][0][j]-(y[0]==j);grad[-K+j]+=r['node'][-1][j]-(y[-1]==j)
    active=K*F if independent else SIZE
    for j in range(active):grad[j]=grad[j]/len(data)+lam*theta[j]
    loss=loss/len(data)+lam/2*math.fsum(v*v for v in theta)
    return loss,grad


def fit(data,regularization,independent=False,max_steps=100,tolerance=1e-5):
    validate_data(data)
    if type(max_steps)is not int or not 1<=max_steps<=1000:raise ValueError('invalid update budget')
    tol=finite(tolerance)
    if tol<=0:raise ValueError('invalid gradient tolerance')
    theta=[0.]*SIZE;history=[];status='budget_exhausted'
    for iteration in range(max_steps+1):
        loss,g=objective(theta,data,regularization,independent);norm=math.sqrt(math.fsum(v*v for v in g))
        entry=dict(iteration=iteration,theta=list(theta),objective=loss,gradient=list(g),gradient_norm=norm,step=0.,backtracks=0);history.append(entry)
        if norm<=tol:status='gradient_tolerance';break
        if iteration==max_steps:break
        step=1.;accepted=False
        for backtracks in range(40):
            trial=[v-step*d for v,d in zip(theta,g)];new,_=objective(trial,data,regularization,independent)
            if new<=loss-1e-4*step*norm*norm:accepted=True;break
            step/=2
        if not accepted:status='line_search_failed';break
        entry['step']=step;entry['backtracks']=backtracks;theta=trial
    return dict(theta=theta,regularization=regularization,independent=independent,max_steps=max_steps,tolerance=tolerance,status=status,history=history)


def metrics(theta,data):
    validate_data(data);nll=0.;correct=0;exact=0;modecorrect=0;tokens=0
    for item in data:
        p=potentials(theta,item['x']);r=infer(*p);nll+=r['log_partition']-score(*p,item['y']);tokens+=len(item['y']);correct+=sum(a==b for a,b in zip(r['viterbi'],item['y']));modecorrect+=sum(a==b for a,b in zip(r['marginal_modes'],item['y']));exact+=r['viterbi']==item['y']
    return dict(sequences=len(data),tokens=tokens,nll_per_token=nll/tokens,viterbi_token_errors=tokens-correct,marginal_token_errors=tokens-modecorrect,viterbi_sequence_errors=len(data)-exact)
