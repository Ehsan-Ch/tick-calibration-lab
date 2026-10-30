"""Independently reconcile stored predictions with the public report, without retraining."""
import csv,gzip,json
from pathlib import Path
import numpy as np
from ticklab.metrics import metrics
from ticklab.data import sha256
root=Path(__file__).parent
report=json.loads((root/'docs/data/report.json').read_text(encoding='utf-8'))
path=root/'docs'/report['prediction_file']['path']
assert sha256(path)==report['prediction_file']['sha256'],'Prediction digest differs'
assert sha256(root/'config.json')==report['config_sha256'],'Protocol changed'
assert sha256(root/'data/source_manifest.json')==report['source_manifest_sha256'],'Source manifest changed'
with gzip.open(path,'rt',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==report['prediction_file']['rows']
for result in report['summary']+report['results']:
    subset=[r for r in rows if 'fold' not in result or r['fold']==result['fold']]
    y=np.array([int(r['actual']) for r in subset]);key=result['model']+'_'+result['variant']
    p=np.array([[float(r[key+'_'+c]) for c in ['down','flat','up']] for r in subset]);m=metrics(y,p)
    for field in ['log_loss','brier','accuracy','balanced_accuracy','macro_f1','ece']:assert abs(m[field]-result[field])<1e-8,(key,field)
    assert m['confusion']==result['confusion']
assert len({(r['fold'],r['prediction_time']) for r in rows})==len(rows),'Duplicate forecasts'
assert all(r['prediction_time']<r['label_available'] for r in rows),'Invalid target timing'
print(f'Reconciled {len(rows):,} forecasts across {len(report["results"])} fold/model variants; hashes and metrics match.')
