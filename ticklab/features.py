"""Causal time-series features and non-crossing forward labels."""
import numpy as np

FEATURES=['return_10s_bps','return_60s_bps','return_300s_bps','volatility_60s','volatility_300s',
          'range_10s_bps','range_60s_bps','log_volume_10s','log_volume_60s','log_count_10s','log_count_60s',
          'taker_imbalance_10s','taker_imbalance_60s','taker_imbalance_300s','last_trade_age_s',
          'return_lag1','return_lag2','return_lag3','return_lag4','return_lag5']

def rolling_sum(x,k):
    s=np.r_[0.,np.cumsum(x)];out=np.full(len(x),np.nan);out[k-1:]=s[k:]-s[:-k];return out

def causal_features(bars,seconds=10):
    if seconds!=10:raise ValueError('This feature specification requires 10-second bars')
    p=bars['close'];n=len(p);ret=np.zeros(n)
    with np.errstate(divide='ignore',invalid='ignore'):
        ret[1:]=1e4*np.log(p[1:]/p[:-1])
    # Missing early prices remain invalid and are discarded below, not backfilled.
    safe=np.nan_to_num(ret,nan=0);features=[]
    for k in [1,6,30]:
        r=np.full(n,np.nan);r[k:]=1e4*np.log(p[k:]/p[:-k]);features.append(r)
    for k in [6,30]:features.append(np.sqrt(np.maximum(0,rolling_sum(safe**2,k)/k-(rolling_sum(safe,k)/k)**2)))
    features.append(1e4*(bars['high']-bars['low'])/p)
    rng=np.full(n,np.nan)
    for i in range(5,n):rng[i]=1e4*(np.max(bars['high'][i-5:i+1])-np.min(bars['low'][i-5:i+1]))/p[i]
    features.append(rng)
    for field in ['qty','count']:
        features.extend([np.log1p(bars[field]),np.log1p(rolling_sum(bars[field],6))])
    for k in [1,6,30]:
        v=rolling_sum(bars['qty'],k);s=rolling_sum(bars['signed'],k);features.append(np.divide(s,v,out=np.zeros(n),where=v>0))
    features.append(bars['age'])
    for lag in range(1,6):features.append(np.r_[np.full(lag,np.nan),ret[:-lag]])
    return np.column_stack(features)

def make_features(bars,seconds=10,horizon=60,threshold=3.):
    if horizon<=0 or horizon%seconds or threshold<0:raise ValueError('Invalid target specification')
    x=causal_features(bars,seconds);h=horizon//seconds;n=len(x)
    idx=np.arange(30,n-h-1) # strict endpoint before the next UTC day / partition
    future=1e4*np.log(bars['close'][idx+h]/bars['close'][idx])
    valid=np.isfinite(x[idx]).all(axis=1)&np.isfinite(future)&(bars['age'][idx]<=30)&(bars['age'][idx+h]<=30)
    idx=idx[valid];future=future[valid]
    y=np.where(future>threshold,2,np.where(future < -threshold,0,1)).astype(int)
    return x[idx],y,future,bars['end'][idx],bars['end'][idx+h]

def split_masks(data,fold):
    days=data['days'];result={
        'train':days<=fold['train_end'],'select':days==fold['select'],
        'calibrate':days==fold['calibrate'],'test':np.isin(days,fold['test'])}
    previous_end=None
    for name,m in result.items():
        if not m.any():raise ValueError(f'Empty partition: {name}')
        if previous_end is not None and data['times'][m].min()<=previous_end:raise ValueError('Label overlap across partitions')
        previous_end=data['ends'][m].max()
    return result
