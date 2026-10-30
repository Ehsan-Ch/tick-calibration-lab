import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix

def validate_probabilities(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-6):raise ValueError('Invalid class probabilities')
    return p

def losses(y,p):
    p=validate_probabilities(p);y=np.asarray(y)
    if len(y)!=len(p) or not np.isin(y,[0,1,2]).all():raise ValueError('Invalid target')
    return -np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))

def temperature_scale(p,t):
    if not np.isfinite(t) or t<=0:raise ValueError('Temperature must be positive')
    return softmax(np.log(np.clip(validate_probabilities(p),1e-12,1))/t,axis=1)

def fit_temperature(y,p):
    fit=minimize_scalar(lambda lt:float(losses(y,temperature_scale(p,np.exp(lt))).mean()),bounds=(np.log(.25),np.log(4)),method='bounded')
    candidates=[1.,float(np.exp(fit.x))]
    return min(candidates,key=lambda t:float(losses(y,temperature_scale(p,t)).mean()))

def reliability(y,p,bins=10):
    pred=p.argmax(1);confidence=p.max(1);correct=(pred==y);idx=np.minimum((confidence*bins).astype(int),bins-1)
    rows=[]
    for b in range(bins):
        mask=idx==b;n=int(mask.sum())
        rows.append({'lo':b/bins,'hi':(b+1)/bins,'n':n,'confidence':float(confidence[mask].mean()) if n else None,'accuracy':float(correct[mask].mean()) if n else None})
    return rows

def metrics(y,p):
    p=validate_probabilities(p);pred=p.argmax(1);rel=reliability(y,p)
    return {'n':len(y),'log_loss':float(losses(y,p).mean()),'brier':float(((p-np.eye(3)[y])**2).sum(1).mean()),
        'accuracy':float((pred==y).mean()),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
        'ece':float(sum(r['n']*abs(r['confidence']-r['accuracy']) for r in rel if r['n'])/len(y)),
        'reliability':rel,'confusion':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
        'class_counts':np.bincount(y,minlength=3).tolist()}

def selective_metrics(y,p):
    output=[]
    for threshold in [.34,.4,.5,.6,.7,.8,.9]:
        mask=p.max(1)>=threshold;n=int(mask.sum())
        output.append({'threshold':threshold,'n':n,'coverage':float(mask.mean()),'accuracy':float((p[mask].argmax(1)==y[mask]).mean()) if n else None})
    return output

def block_bootstrap_difference(y,p,reference,days,replicates=300,block=60,seed=42):
    diff=losses(y,p)-losses(y,reference);rng=np.random.default_rng(seed);estimates=[]
    groups=[np.flatnonzero(days==d) for d in np.unique(days)]
    if block<=0 or replicates<2:raise ValueError('Invalid bootstrap configuration')
    for _ in range(replicates):
        selected=[]
        for g in groups:
            starts=rng.integers(0,len(g),size=int(np.ceil(len(g)/block)))
            ix=((starts[:,None]+np.arange(block))%len(g)).ravel()[:len(g)]
            selected.append(g[ix])
        estimates.append(float(diff[np.concatenate(selected)].mean()))
    return {'difference':float(diff.mean()),'low':float(np.quantile(estimates,.025)),'high':float(np.quantile(estimates,.975)),
        'method':'paired circular moving-block bootstrap within each test day','block_rows':block,'replicates':replicates}

def psi(reference,observed):
    edges=np.unique(np.quantile(reference,np.linspace(0,1,11)))
    if len(edges)<3:return 0.
    edges[0]=-np.inf;edges[-1]=np.inf
    a=np.histogram(reference,edges)[0].astype(float)+.5;b=np.histogram(observed,edges)[0].astype(float)+.5
    a/=a.sum();b/=b.sum();return float(((b-a)*np.log(b/a)).sum())
