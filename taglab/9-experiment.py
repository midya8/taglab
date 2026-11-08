import copy,gzip,hashlib,json,base64,math
from importlib.resources import files
from .fixture import dataset
from .model import fit,metrics,predict,objective,SIZE


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def experiment(data=None,max_steps=80):
    data=dataset() if data is None else copy.deepcopy(data)
    models=[]
    for independent in [False,True]:
        for lam in [.03,.3,3.]:
            m=fit(data['train'],lam,independent,max_steps=max_steps);m['id']=('local' if independent else 'chain')+'-'+str(lam)
            m['train_digest']=digest(data['train']);m['validation']=metrics(m['theta'],data['validation']);models.append(m)
    selected={family:min((m for m in models if m['independent']==independent),key=lambda m:(m['validation']['nll_per_token'],m['id']))['id'] for family,independent in [('chain',False),('local',True)]}
    # Test results are computed only after selection is frozen.
    for m in models:m['test']=metrics(m['theta'],data['test'])
    witnesses={m['id']:[predict(m['theta'],item['x']) for item in data['test']] for m in models if m['id'] in selected.values()}
    return dict(schema='taglab-1',data=data,models=models,selected=selected,witnesses=witnesses,selection='minimum validation conditional NLL/token within family; test never selects')


def close(a,b):
    if type(a)is not type(b) and not (isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool)):raise ValueError('type changed')
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('keys changed')
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('length changed')
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):
        if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>2e-10*(1+abs(b)):raise ValueError('numeric mismatch')
    elif a!=b:raise ValueError('value changed')


def audit(report):
    if report.get('schema')!='taglab-1':raise ValueError('schema')
    from .model import metrics,predict,validate_data
    if set(report['data'])!={'train','validation','test'}:raise ValueError('partition keys')
    for rows in report['data'].values():validate_data(rows)
    expected={(('local' if local else 'chain')+'-'+str(lam),lam,local) for local in [False,True] for lam in [.03,.3,3.]}
    if len(report['models'])!=6 or {(m['id'],m['regularization'],m['independent']) for m in report['models']}!=expected:raise ValueError('model roster')
    for m in report['models']:
        if m['train_digest']!=digest(report['data']['train']):raise ValueError('training binding')
        if len(m['theta'])!=SIZE:raise ValueError('parameters')
        h=m['history']
        if h[0]['theta']!=[0.]*SIZE:raise ValueError('initialization')
        for i,entry in enumerate(h):
            loss,g=objective(entry['theta'],report['data']['train'],m['regularization'],m['independent']);close(loss,entry['objective']);close(g,entry['gradient']);close(math.sqrt(sum(v*v for v in g)),entry['gradient_norm'])
            if entry['iteration']!=i:raise ValueError('iteration')
            if i+1<len(h):
                step=entry['step']
                if not 0<step<=1 or step!=2.**(-entry['backtracks']):raise ValueError('step')
                for b in range(entry['backtracks']):
                    earlier=2.**(-b);candidate=[v-earlier*d for v,d in zip(entry['theta'],g)];candidate_loss,_=objective(candidate,report['data']['train'],m['regularization'],m['independent'])
                    if candidate_loss<=loss-1e-4*earlier*sum(v*v for v in g):raise ValueError('skipped acceptable step')
                close([v-step*d for v,d in zip(entry['theta'],g)],h[i+1]['theta'])
                if h[i+1]['objective']>loss-1e-4*step*sum(v*v for v in g)+1e-11:raise ValueError('Armijo')
            elif entry['step']!=0 or entry['backtracks']!=0:raise ValueError('terminal step')
        close(m['theta'],h[-1]['theta'])
        expected='gradient_tolerance' if h[-1]['gradient_norm']<=m['tolerance'] else 'budget_exhausted' if len(h)==m['max_steps']+1 else 'line_search_failed'
        if m['status']!=expected:raise ValueError('status')
        close(metrics(m['theta'],report['data']['validation']),m['validation']);close(metrics(m['theta'],report['data']['test']),m['test'])
    chosen={family:min((m for m in report['models'] if m['independent']==independent),key=lambda m:(m['validation']['nll_per_token'],m['id']))['id'] for family,independent in [('chain',False),('local',True)]}
    if report['selected']!=chosen:raise ValueError('selection')
    for model_id,w in report['witnesses'].items():
        m=next(m for m in report['models'] if m['id']==model_id);close([predict(m['theta'],item['x']) for item in report['data']['test']],w)
    if set(report['witnesses'])!=set(chosen.values()):raise ValueError('witness keys')
    return dict(models=len(report['models']),states=sum(len(m['history']) for m in report['models']),selected=chosen,valid=True)


def saved():
    root=files('taglab').joinpath('artifacts');meta=json.loads(root.joinpath('report.json').read_text());raw=gzip.decompress(base64.b64decode(''.join(root.joinpath(p).read_text().strip() for p in meta['parts']),validate=True))
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('archive hash')
    return json.loads(raw)
