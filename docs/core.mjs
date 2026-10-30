export const modelNames={prior:'Historical prior',linear:'Incremental linear',boosted:'Boosted trees'};
export const classes=['Down','Flat','Up'];
export function selectResult(report,fold,model,calibrated){
  const variant=model==='prior'?'raw':calibrated?'calibrated':'raw';
  const rows=fold==='all'?report.summary:report.results.filter(r=>r.fold===fold);
  const result=rows.find(r=>r.model===model && r.variant===variant);
  if(!result)throw Error('Result not found');return result;
}
export function availableReplay(rows,asOf,key,limit=8){
  const cutoff=Date.parse(asOf);if(!Number.isFinite(cutoff))throw Error('Invalid replay clock');
  return rows.filter(r=>Date.parse(r.time)<=cutoff).slice(-limit).reverse().map(r=>({time:r.time,label_available:r.label_available,probabilities:r.probabilities[key],actual:Date.parse(r.label_available)<=cutoff?r.actual:null}));
}
export function selectedCSV(result,fold){
  const fields=['fold','model','variant','n','log_loss','brier','accuracy','balanced_accuracy','macro_f1','ece'];
  const row={...result,fold};return fields.join(',')+'\n'+fields.map(k=>JSON.stringify(row[k])).join(',')+'\n';
}
export function bestClass(p){return p.indexOf(Math.max(...p));}
