"""Checksum-pinned acquisition and bounded-memory trade aggregation."""
from __future__ import annotations
import csv, hashlib, io, json, math, urllib.request, zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://data.binance.vision/data/spot/daily/aggTrades'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def acquire(config, root=ROOT):
    raw=root/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'data/source_manifest.json'
    prior=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    pins={r['date']:r['sha256'] for r in prior}
    records=[]
    for d in range(config['days']):
        day=str(date.fromisoformat(config['start'])+timedelta(days=d))
        name=f"{config['symbol']}-aggTrades-{day}.zip";url=f"{BASE}/{config['symbol']}/{name}"
        with urllib.request.urlopen(url+'.CHECKSUM',timeout=60) as r:expected=r.read().decode().split()[0]
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('Invalid source checksum')
        if day in pins and expected!=pins[day]:raise ValueError(f'Source revision detected for {day}; review before updating the pin')
        path=raw/name
        if not path.exists():
            tmp=path.with_suffix('.partial')
            with urllib.request.urlopen(url,timeout=90) as r,open(tmp,'wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
            if sha256(tmp)!=expected:raise ValueError('Downloaded archive checksum mismatch')
            tmp.replace(path)
        if sha256(path)!=expected:raise ValueError(f'Cached archive checksum mismatch: {name}')
        records.append({'date':day,'url':url,'checksum_url':url+'.CHECKSUM','sha256':expected,'bytes':path.stat().st_size})
        print(f'Verified {day}: {path.stat().st_size:,} bytes',flush=True)
    manifest_path.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8',newline='\n')
    return records

def aggregate_rows(rows, day, seconds=10):
    if seconds<=0 or 86400%seconds:raise ValueError('Bar width must divide a UTC day')
    start=int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()*1000)
    n=86400//seconds;step=seconds*1000
    close=np.full(n,np.nan);high=np.full(n,-np.inf);low=np.full(n,np.inf)
    qty=np.zeros(n);signed=np.zeros(n);count=np.zeros(n);last=np.full(n,np.nan)
    previous_id=-1;previous_time=-1;total=0;id_gaps=0
    for row in rows:
        if len(row)!=8:raise ValueError('Expected 8 aggregate-trade columns')
        ident,first,final=int(row[0]),int(row[3]),int(row[4]);price=float(row[1]);size=float(row[2]);stamp=int(row[5])
        maker=row[6].lower()
        if maker not in ('true','false') or row[7].lower() not in ('true','false'):raise ValueError('Invalid boolean')
        if ident<=previous_id or stamp<previous_time:raise ValueError('Unordered or duplicate aggregate trade')
        if previous_id>=0 and ident!=previous_id+1:id_gaps+=ident-previous_id-1
        if first>final or not math.isfinite(price) or not math.isfinite(size) or price<=0 or size<=0:raise ValueError('Invalid trade value')
        if not start<=stamp<start+86400000:raise ValueError('Timestamp outside UTC day (2024 archives require milliseconds)')
        j=(stamp-start)//step
        close[j]=price;high[j]=max(high[j],price);low[j]=min(low[j],price);qty[j]+=size
        signed[j]+=size*(1 if maker=='false' else -1);count[j]+=1;last[j]=stamp
        previous_id=ident;previous_time=stamp;total+=1
    if not total:raise ValueError('Empty archive')
    # Only forward-fill after the first real trade, never from future observations.
    for j in range(1,n):
        if np.isnan(close[j]):close[j]=close[j-1];last[j]=last[j-1]
    empty=count==0;high[empty]=close[empty];low[empty]=close[empty]
    end=start+np.arange(1,n+1)*step
    return {'end':end,'close':close,'high':high,'low':low,'qty':qty,'signed':signed,'count':count,'age':(end-last)/1000},{'date':day,'aggregate_trades':total,'empty_bins':int(empty.sum()),'missing_aggregate_ids':id_gaps}

def aggregate_archive(path, day, seconds=10):
    with zipfile.ZipFile(path) as z:
        expected=Path(path).stem+'.csv'
        if z.namelist()!=[expected]:raise ValueError('Unexpected archive members')
        with z.open(expected) as raw:
            return aggregate_rows(csv.reader(io.TextIOWrapper(raw,encoding='utf-8')),day,seconds)

def build_dataset(config, root=ROOT):
    from .features import make_features
    records=json.loads((root/'data/source_manifest.json').read_text())
    parts=[];quality=[]
    for index,r in enumerate(records):
        path=root/'data/raw'/r['url'].split('/')[-1]
        if sha256(path)!=r['sha256']:raise ValueError('Input integrity check failed')
        bars,q=aggregate_archive(path,r['date'],config['bar_seconds'])
        x,y,returns,times,ends=make_features(bars,config['bar_seconds'],config['horizon_seconds'],config['threshold_bps'])
        q['examples']=len(y);quality.append(q);parts.append((x,y,returns,times,ends,np.full(len(y),index+1)))
        print(f"Prepared {r['date']}: {q['aggregate_trades']:,} trades -> {len(y):,} examples",flush=True)
    arrays=[np.concatenate([p[i] for p in parts]) for i in range(6)]
    cache=root/'data/cache';cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache/'dataset.npz',**dict(zip(['x','y','returns','times','ends','days'],arrays)))
    return dict(zip(['x','y','returns','times','ends','days'],arrays)),quality

if __name__=='__main__':
    config=json.loads((ROOT/'config.json').read_text());acquire(config)
