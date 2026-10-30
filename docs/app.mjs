import {modelNames,classes,selectResult,availableReplay,selectedCSV,bestClass} from './core.mjs';
const $=s=>document.querySelector(s),fmt=(n,d=3)=>Number(n).toFixed(d),pct=(n,d=1)=>n===null?'—':fmt(100*n,d)+'%';
let report,replayRows=[];
function cell(text,tag='td'){const n=document.createElement(tag);n.textContent=text;return n;}
function tr(values){const row=document.createElement('tr');for(const v of values)row.append(cell(v));return row;}
function state(){return {fold:$('#fold').value,model:$('#model').value,cal:$('#calibrated').checked};}
function reliabilityChart(result){
  const left=48,top=18,width=420,height=245;let svg='<title>Top-label confidence versus empirical accuracy</title>';
  for(let t=0;t<=1.01;t+=.2){const x=left+t*width,y=top+(1-t)*height;svg+=`<path d="M${left} ${y}h${width}" stroke="#e5e9ef"/><text x="${left-10}" y="${y+4}" text-anchor="end" fill="#738095" font-size="10">${t.toFixed(1)}</text><text x="${x}" y="${top+height+21}" text-anchor="middle" fill="#738095" font-size="10">${t.toFixed(1)}</text>`;}
  svg+=`<path d="M${left} ${top+height}L${left+width} ${top}" stroke="#b3bdcb" stroke-width="1.5" stroke-dasharray="5 5"/><text x="255" y="310" text-anchor="middle" fill="#637084" font-size="11">Mean predicted confidence</text><text transform="translate(12 140) rotate(-90)" text-anchor="middle" fill="#637084" font-size="11">Observed accuracy</text>`;
  for(const b of result.reliability){if(!b.n)continue;const x=left+b.confidence*width,y=top+(1-b.accuracy)*height;svg+=`<circle cx="${x}" cy="${y}" r="${4+10*Math.sqrt(b.n/result.n)}" fill="#365ae3" fill-opacity=".78" stroke="white" stroke-width="2"><title>${b.n.toLocaleString()} forecasts; confidence ${pct(b.confidence)}; accuracy ${pct(b.accuracy)}</title></circle>`;}
  $('#reliability').innerHTML=svg;
}
function render(){
  const {fold,model,cal}=state(),result=selectResult(report,fold,model,cal);$('#calibrated').disabled=model==='prior';
  $('#loss').textContent=fmt(result.log_loss);$('#ece').textContent=pct(result.ece,2);$('#f1').textContent=fmt(result.macro_f1);$('#brier').textContent=fmt(result.brier);
  $('#selection-note').textContent=`${result.n.toLocaleString()} test forecasts · ${modelNames[model]} · ${result.variant==='calibrated'?(fold==='all'?'Temperature fitted independently in each fold':`Calibration temperature ${fmt(result.temperature)}`):'Raw probabilities'} · Lower log loss and ECE are better.`;
  reliabilityChart(result);$('#selective').replaceChildren(...result.selective.map(r=>tr([pct(r.threshold),`${pct(r.coverage)} (${r.n.toLocaleString()})`,pct(r.accuracy)])));
  $('#uncertainty').textContent=result.vs_prior?`Paired log-loss difference vs prior: ${fmt(result.vs_prior.difference,4)}. 95% block-bootstrap interval: [${fmt(result.vs_prior.low,4)}, ${fmt(result.vs_prior.high,4)}]. Negative favors this model. This interval conditions on four observed test days.`:'Use “All out-of-sample days” to inspect the paired block-bootstrap interval.';
  const comparisons=fold==='all'?report.summary:report.results.filter(r=>r.fold===fold);
  $('#comparison').replaceChildren(...comparisons.map(r=>{const row=tr([modelNames[r.model],r.variant,fmt(r.log_loss),pct(r.ece,2),fmt(r.macro_f1)]);if(r===result)row.className='selected';return row;}));
  const monitors=fold==='all'?report.monitor:report.monitor.filter(r=>r.fold===fold);
  $('#drift').replaceChildren(...monitors.map(r=>{const d=document.createElement('div');d.className='drift-day';const peak=r.features.reduce((a,b)=>a.psi>b.psi?a:b);d.append(cell(r.date,'span'),cell(`${peak.feature}: ${fmt(peak.psi)}${peak.psi>.2?' · review':''}`,'span'));if(peak.psi>.2)d.classList.add('flag');return d;}));
  renderReplay();
}
function renderReplay(){
  const {model,cal}=state(),key=model+'_'+(model==='prior'?'raw':cal?'calibrated':'raw');
  replayRows=report.replay.filter(r=>r.time.slice(0,10)===$('#replay-day').value);
  const index=Math.min(Number($('#clock-slider').value),replayRows.length-1),time=replayRows[index].time;
  $('#clock').textContent=time.replace('T',' ').replace('Z',' UTC');$('#clock-slider').max=replayRows.length-1;
  $('#replay-rows').replaceChildren(...availableReplay(replayRows,time,key).map(r=>{const pred=bestClass(r.probabilities),row=tr([r.time.slice(11,19),classes[pred],pct(Math.max(...r.probabilities)),r.actual===null?'Awaiting label':classes[r.actual]]);row.lastChild.className=r.actual===null?'pending':'outcome';return row;}));
}
async function init(){
  const res=await fetch('data/report.json');if(!res.ok)throw Error('Report download failed');report=await res.json();
  if(report.schema_version!==1)throw Error('Unsupported report format');
  $('#trade-count').textContent=report.quality.reduce((s,r)=>s+r.aggregate_trades,0).toLocaleString();$('#test-count').textContent=report.prediction_file.rows.toLocaleString();
  const prior=report.summary.find(r=>r.model==='prior'),boosted=report.summary.find(r=>r.model==='boosted'&&r.variant==='calibrated');
  $('#finding').textContent=`Calibrated boosted trees reduced pooled test log loss by ${fmt(100*(1-boosted.log_loss/prior.log_loss),1)}% versus the historical prior (${fmt(prior.log_loss)} → ${fmt(boosted.log_loss)}). The linear model remained worse than the prior. Calibration slightly worsened boosted-tree ECE despite improving log loss.`;
  $('#fold-cards').replaceChildren(...report.folds.map(f=>{const box=document.createElement('article');box.className='fold-card';const title=document.createElement('div');title.className='fold-name';title.textContent=f.id+' / EXPANDING WINDOW';box.append(title);for(const [name,r]of Object.entries(f.ranges)){const p=document.createElement('p');p.append(cell(name[0].toUpperCase()+name.slice(1),'span'),cell(`${r.first_prediction.slice(8,10)}–${r.last_prediction.slice(8,10)} Jan · ${r.n.toLocaleString()}`,'span'));box.append(p);}const note=document.createElement('small');note.textContent=`Selected model family: ${modelNames[f.selection_champion]}. Labels end before the next partition begins.`;box.append(note);return box;}));
  $('#replay-day').replaceChildren(...[...new Set(report.replay.map(r=>r.time.slice(0,10)))].map(d=>new Option(d,d)));
  $('#fold').addEventListener('change',render);$('#model').addEventListener('change',render);$('#calibrated').addEventListener('change',render);
  $('#replay-day').addEventListener('change',()=>{$('#clock-slider').value=0;renderReplay();});$('#clock-slider').addEventListener('input',renderReplay);
  $('#advance').addEventListener('click',()=>{$('#clock-slider').value=Math.min(Number($('#clock-slider').max),Number($('#clock-slider').value)+6);renderReplay();});
  $('#export').addEventListener('click',()=>{const {fold,model,cal}=state();const blob=new Blob([selectedCSV(selectResult(report,fold,model,cal),fold)],{type:'text/csv'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`ticklab-${fold}-${model}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  $('#loading').hidden=true;$('#app').hidden=false;render();
}
init().catch(e=>{$('#loading').textContent=`The report could not load: ${e.message}. Reload or open the repository for the saved results.`;});
