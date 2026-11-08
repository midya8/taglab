'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),C=require('../docs/counts.js');
const close=(a,b,tol=1e-8)=>{if(a===null||b===null)return assert.equal(a,b);if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));}else if(typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort());for(const k of Object.keys(a))close(a[k],b[k],tol);}else if(typeof a==='number')assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);else assert.equal(a,b);};
test('864 full Python/JS count-state, pair, path and gradient witnesses',()=>{
const text=execFileSync('python',['-c',`import json
from taglab.experiment import saved
from taglab.counts import label_count
r=saved();out=[]
for m in r['models']:
 for d in r['data']['test']:
  x=d['x']
  for label in range(3):
   for c in [0,len(x)//2,len(x)]:out.append([m['theta'],x,label,c,label_count(m['theta'],x,label,c)])
print(json.dumps(out,allow_nan=False))`],{cwd:require('node:path').resolve(__dirname,'..'),maxBuffer:128e6,encoding:'utf8'});const cases=JSON.parse(text);assert.equal(cases.length,864);for(const [theta,x,label,c,w]of cases){const a=C.labelCount(theta,x,label,c),p=TagInference.potentials(theta,x);assert.equal(a.conditional.viterbi.filter(j=>j===label).length,c);close(TagInference.score(p,a.conditional.viterbi),TagInference.score(p,w.conditional.viterbi));for(let t=0;t<x.length;t++)close(a.conditional.node[t][a.conditional.marginal_modes[t]],a.conditional.node[t][w.conditional.marginal_modes[t]]);delete a.conditional.viterbi;delete w.conditional.viterbi;delete a.conditional.marginal_modes;delete w.conditional.marginal_modes;close(a,w);}
});
test('sixty independent BigInt complete-path controls, every label and count',()=>{
 let state=31;const next=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return 1+state%7;};let checked=0;
 for(let q=0;q<60;q++){const k=2+q%2,n=1+q%5,u=Array.from({length:n},()=>Array.from({length:k},next)),tr=Array.from({length:k},()=>Array.from({length:k},next)),s=Array.from({length:k},next),e=Array.from({length:k},next),paths=[];function walk(y){if(y.length===n){let w=BigInt(s[y[0]]*e[y.at(-1)]);for(let t=0;t<n;t++)w*=BigInt(u[t][y[t]]);for(let t=1;t<n;t++)w*=BigInt(tr[y[t-1]][y[t]]);paths.push([y,w]);return;}for(let j=0;j<k;j++)walk([...y,j]);}walk([]);const z=paths.reduce((a,p)=>a+p[1],0n),p={emissions:u.map(row=>row.map(Math.log)),transition:tr.map(row=>row.map(Math.log)),start:s.map(Math.log),end:e.map(Math.log)};
 for(let label=0;label<k;label++)for(let c=0;c<=n;c++){const r=C.countEvent(p,label,c),set=paths.filter(([y])=>y.filter(j=>j===label).length===c),weight=set.reduce((a,p)=>a+p[1],0n);assert.equal(r.support_count,String(set.length));close(r.event_probability,Number(weight)/Number(z));close(r.conditioned_expected_count,c);assert.equal(r.conditional.viterbi.filter(j=>j===label).length,c);for(let t=0;t<n;t++)for(let j=0;j<k;j++)close(r.conditional.node[t][j],Number(set.reduce((a,[y,w])=>a+(y[t]===j?w:0n),0n))/Number(weight));for(let t=0;t<n-1;t++)for(let i=0;i<k;i++)for(let j=0;j<k;j++)close(r.conditional.edge[t][i][j],Number(set.reduce((a,[y,w])=>a+(y[t]===i&&y[t+1]===j?w:0n),0n))/Number(weight));checked++;}}
 assert.equal(checked,600);
});
test('nonlocal dependence, singleton, underflow, strict validation and 64-position limit',()=>{
 const p={emissions:[[0,0],[0,0]],transition:[[0,0],[0,0]],start:[0,0],end:[0,0]},r=C.countEvent(p,0,1);close(r.conditional.node,[[.5,.5],[.5,.5]]);close(r.conditional.edge,[[[0,.5],[.5,0]]]);assert.equal(r.independent_node_event_probability,.5);for(const [l,c]of [[true,1],[0,true],[0,3],[-1,0],[2,0],[0,NaN]])assert.throws(()=>C.countEvent(p,l,c));const under=C.countEvent({...p,emissions:[[0,-2000]]},1,1);assert.equal(under.event_probability,0);assert.equal(under.log_event_probability,-2000);close(under.conditional.node,[[0,1]]);const large={emissions:Array.from({length:64},()=>[1e6,-1e6]),transition:[[1e6,-1e6],[-1e6,1e6]],start:[1e6,-1e6],end:[1e6,-1e6]},w=C.countEvent(large,0,32);close(w.conditioned_expected_count,32,1e-6);assert.ok(!JSON.stringify(w).includes('Infinity'));assert.throws(()=>C.countEvent({...large,emissions:[...large.emissions,[0,0]]},0,1));
});
