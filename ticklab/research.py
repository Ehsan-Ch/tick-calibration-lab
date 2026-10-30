"""Run the locked experiment. Raw data is streamed; compact model matrices fit RAM."""
from __future__ import annotations
import csv,gzip,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import sklearn,scipy
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import ROOT,build_dataset,sha256
from .features import FEATURES,split_masks
from .metrics import metrics,losses,fit_temperature,temperature_scale,selective_metrics,block_bootstrap_difference,psi

def iso(ms):return datetime.fromtimestamp(int(ms)/1000,timezone.utc).isoformat().replace('+00:00','Z')

def fit_linear(x,y,alpha,epochs,seed):
    scaler=StandardScaler()
    for start in range(0,len(x),4096):scaler.partial_fit(x[start:start+4096])
    model=SGDClassifier(loss='log_loss',alpha=alpha,random_state=seed,average=True,shuffle=False)
    for _ in range(epochs):
        for start in range(0,len(x),4096):model.partial_fit(scaler.transform(x[start:start+4096]),y[start:start+4096],classes=np.arange(3))
    return model,scaler

def run(root=ROOT):
    start=time.perf_counter();config=json.loads((root/'config.json').read_text());data,quality=build_dataset(config,root)
    out=root/'docs/data';out.mkdir(parents=True,exist_ok=True)
    results=[];folds=[];all_predictions=[];runtime=[];chosen=[];monitor=[]
    for fold in config['folds']:
        masks=split_masks(data,fold);x=data['x'];y=data['y'];tr=masks['train'];va=masks['select'];ca=masks['calibrate'];te=masks['test']
        ranges={name:{'n':int(m.sum()),'first_prediction':iso(data['times'][m].min()),'last_prediction':iso(data['times'][m].max()),'last_label_available':iso(data['ends'][m].max())} for name,m in masks.items()}
        folds.append({'id':fold['id'],'ranges':ranges})
        print(f"{fold['id']}: {int(tr.sum()):,} train / {int(te.sum()):,} test",flush=True)
        prior=(np.bincount(y[tr],minlength=3)+1)/(tr.sum()+3)
        models={'prior':(np.tile(prior,(ca.sum(),1)),np.tile(prior,(te.sum(),1)))}
        selection=[]
        for family in ['linear','boosted']:
            candidates=[]
            parameters=config['linear_alphas'] if family=='linear' else config['boosted_leaves']
            for parameter in parameters:
                t=time.perf_counter()
                if family=='linear':
                    model,scaler=fit_linear(x[tr],y[tr],parameter,config['linear_epochs'],config['seed'])
                    def predict(z,model=model,scaler=scaler):return model.predict_proba(scaler.transform(z))
                else:
                    model=HistGradientBoostingClassifier(max_iter=config['boosted_iterations'],max_leaf_nodes=parameter,learning_rate=.07,l2_regularization=1.,min_samples_leaf=100,early_stopping=False,random_state=config['seed'])
                    model.fit(x[tr],y[tr]);predict=model.predict_proba
                score=float(losses(y[va],predict(x[va])).mean())
                selection.append({'model':family,'parameter':parameter,'selection_log_loss':score})
                candidates.append((score,parameter,predict));runtime.append({'fold':fold['id'],'model':family,'parameter':parameter,'fit_and_select_seconds':time.perf_counter()-t})
            score,parameter,predict=min(candidates,key=lambda item:(item[0],item[1]))
            models[family]=(predict(x[ca]),predict(x[te]))
            t=time.perf_counter()
            for _ in range(10):predict(x[te][:1024])
            runtime.append({'fold':fold['id'],'model':family,'batch_1024_mean_ms':(time.perf_counter()-t)*100})
            chosen.append({'fold':fold['id'],'model':family,'parameter':parameter,'selection_log_loss':score})
        folds[-1]['selection']=selection
        # Champion is named using selection data only; no test or calibration ranking.
        folds[-1]['selection_champion']=min([c for c in chosen if c['fold']==fold['id']],key=lambda c:c['selection_log_loss'])['model']
        test_p={}
        for family,(cal,test) in models.items():
            temp=fit_temperature(y[ca],cal) if family!='prior' else 1.
            variants=[('raw',test)] if family=='prior' else [('raw',test),('calibrated',temperature_scale(test,temp))]
            for variant,p in variants:
                key=family+'_'+variant;test_p[key]=p
                result={'fold':fold['id'],'model':family,'variant':variant,'temperature':temp if variant=='calibrated' else 1.,
                        **metrics(y[te],p),'selective':selective_metrics(y[te],p),'daily':[]}
                for day in fold['test']:
                    m=data['days'][te]==day;result['daily'].append({'day':int(day),'date':iso(data['times'][te][m][0])[:10],**metrics(y[te][m],p[m])})
                results.append(result)
        for day in fold['test']:
            m=te&(data['days']==day)
            monitor.append({'fold':fold['id'],'day':day,'date':iso(data['times'][m][0])[:10],
                'features':[{'feature':FEATURES[j],'psi':psi(x[tr,j],x[m,j])} for j in [0,3,7,11,14]]})
        all_predictions.append({'fold':fold['id'],'times':data['times'][te],'ends':data['ends'][te],'days':data['days'][te],'y':y[te],'p':test_p})
    keys=list(all_predictions[0]['p']);labels=np.concatenate([p['y'] for p in all_predictions]);days=np.concatenate([p['days'] for p in all_predictions])
    pooled={key:np.concatenate([p['p'][key] for p in all_predictions]) for key in keys};summary=[]
    for key,p in pooled.items():
        model,variant=key.split('_');summary.append({'model':model,'variant':variant,**metrics(labels,p),'selective':selective_metrics(labels,p),
            'vs_prior':block_bootstrap_difference(labels,p,pooled['prior_raw'],days,config['bootstrap_replicates'],config['bootstrap_block_rows'],config['seed'])})
    replay=[]
    for pred in all_predictions:
        for day in np.unique(pred['days']):
            for i in np.flatnonzero(pred['days']==day)[:120]:
                replay.append({'fold':pred['fold'],'time':iso(pred['times'][i]),'label_available':iso(pred['ends'][i]),'actual':int(pred['y'][i]),'probabilities':{k:[round(float(v),6) for v in p[i]] for k,p in pred['p'].items()}})
    predictions_path=out/'predictions.csv.gz'
    with open(predictions_path,'wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        import io
        with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['fold','prediction_time','label_available','actual']+[f'{k}_{c}' for k in keys for c in ['down','flat','up']])
            for pred in all_predictions:
                for i in range(len(pred['y'])):w.writerow([pred['fold'],iso(pred['times'][i]),iso(pred['ends'][i]),int(pred['y'][i])]+[format(float(v),'.12g') for k in keys for v in pred['p'][k][i]])
    report={'schema_version':1,'title':'TickLab — Market Probability Research','config':config,'config_sha256':sha256(root/'config.json'),
        'source':'Binance Vision ETHUSDT aggregate trades','data_license':'CC BY-NC-SA 4.0','scope':'Historical crypto research proxy; not equities validation or executable trading signals.',
        'features':FEATURES,'quality':quality,'folds':folds,'chosen':chosen,'summary':summary,'results':results,'monitor':monitor,'replay':replay,
        'prediction_file':{'path':'data/predictions.csv.gz','sha256':sha256(predictions_path),'rows':len(labels)},
        'source_manifest_sha256':sha256(root/'data/source_manifest.json'),'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/runtime.json').write_text(json.dumps({'total_seconds':time.perf_counter()-start,'measurements':runtime,'note':'Local single-thread wall-clock timings; not exchange latency, not controlled hardware comparisons.'},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps([{'model':s['model'],'variant':s['variant'],'loss':s['log_loss'],'ece':s['ece'],'ci':s['vs_prior']} for s in summary],indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
