'use strict';
// Count occurrence, not segment runs. A fixed-score model; no refitting.
(()=>{
if(typeof module!=='undefined'){globalThis.TagInference=require('./inference.js');globalThis.TagConditioning=require('./conditioning.js');}
const sum=xs=>xs.reduce((a,b)=>a+b,0),ls=xs=>{if(!xs.length)return -Infinity;const m=Math.max(...xs);return m===-Infinity?m:m+Math.log(sum(xs.map(x=>Math.exp(x-m))));},arg=(ids,f)=>ids.reduce((a,b)=>f(b)>f(a)?b:a),clean=x=>x.map(v=>Array.isArray(v)?clean(v):v===-Infinity?null:v);
function countEvent(p,label,count){
 const base=TagInference.infer(p),{emissions:u,transition:tr,start:s,end:e}=p,n=u.length,k=s.length,ids=Array.from({length:k},(_,j)=>j),cs=Array.from({length:n+1},(_,c)=>c),hit=ids.map(j=>Number(j===label));
 if(n>64||!Number.isInteger(label)||label<0||label>=k||!Number.isInteger(count)||count<0||count>n)throw Error('Use at most64 positions, an integer label index and an integer count from0 through length.');
 const table=()=>u.map(()=>cs.map(()=>Array(k).fill(-Infinity))),a=table(),v=table(),back=u.map(()=>cs.map(()=>Array(k).fill(null)));
 for(const j of ids)a[0][hit[j]][j]=v[0][hit[j]][j]=s[j]+u[0][j];
 for(let t=1;t<n;t++)for(let c=0;c<=t+1;c++)for(const j of ids){const prev=c-hit[j];if(prev<0)continue;a[t][c][j]=u[t][j]+ls(ids.map(i=>a[t-1][prev][i]+tr[i][j]));const i=arg(ids,i=>v[t-1][prev][i]+tr[i][j]);if(v[t-1][prev][i]!==-Infinity){v[t][c][j]=u[t][j]+v[t-1][prev][i]+tr[i][j];back[t][c][j]=i;}}
 const zs=cs.map(c=>ls(ids.map(j=>a[n-1][c][j]+e[j]))),z=zs[count],logz=base.log_partition,b=table();
 for(const i of ids)b[n-1][0][i]=e[i];
 for(let t=n-2;t>=0;t--)for(let q=0;q<n-t;q++)for(const i of ids)b[t][q][i]=ls(ids.filter(j=>q>=hit[j]).map(j=>tr[i][j]+u[t+1][j]+b[t+1][q-hit[j]][j]));
 const node=u.map((_,t)=>ids.map(j=>Math.exp(ls(cs.slice(0,count+1).map(c=>a[t][c][j]+b[t][count-c][j]))-z))),edge=[];
 for(let t=0;t<n-1;t++)edge.push(ids.map(i=>ids.map(j=>Math.exp(ls(cs.slice(0,count-hit[j]+1).map(c=>a[t][c][i]+tr[i][j]+u[t+1][j]+b[t+1][count-c-hit[j]][j]))-z))));
 let j=arg(ids,j=>v[n-1][count][j]+e[j]),c=count;const path=[j];for(let t=n-1;t>0;t--){const i=back[t][c][j];c-=hit[j];j=i;path.push(j);}path.reverse();const best=TagInference.score(p,path);
 let expected=sum(ids.map(j=>s[j]*node[0][j]+e[j]*node[n-1][j]));for(let t=0;t<n;t++)for(const j of ids)expected+=u[t][j]*node[t][j];for(const pair of edge)for(const i of ids)for(const j of ids)expected+=tr[i][j]*pair[i][j];
 let independent=[1];for(const row of node){const p=Math.min(1,Math.max(0,row[label])),next=Array(independent.length+1).fill(0);independent.forEach((w,c)=>{next[c]+=w*(1-p);next[c+1]+=w*p;});independent=next;}
 let choose=1n;for(let i=1;i<=count;i++)choose=choose*BigInt(n-i+1)/BigInt(i);const logps=zs.map(q=>Math.min(0,q-logz)),probs=logps.map(Math.exp);
 return {schema:'taglab-count-event-v1',label,count,length:n,support_count:String(choose*BigInt(k-1)**BigInt(n-count)),unrestricted_log_partition:logz,unrestricted_node:base.node,count_log_partitions:zs,count_log_probabilities:logps,count_probabilities:probs,log_event_probability:logps[count],event_probability:probs[count],probability_underflow:probs[count]===0,alpha:clean(a),suffix_beta:clean(b),conditional:{log_partition:z,node,edge,viterbi:path,best_score:best,path_log_probability:best-z,entropy:z-expected,marginal_modes:node.map(row=>arg(ids,j=>row[j]))},independent_node_count_probabilities:independent,independent_node_event_probability:independent[count],conditioned_expected_count:sum(node.map(row=>row[label])),unrestricted_expected_count:sum(base.node.map(row=>row[label])),refitted:false};
}
function labelCount(theta,symbols,label,count){const p=TagInference.potentials(theta,symbols),out=countEvent(p,label,count),before=TagConditioning.expectedFeatures(symbols,TagInference.infer(p)),after=TagConditioning.expectedFeatures(symbols,out.conditional);out.log_event_gradient=after.map((v,j)=>v-before[j]);return out;}
globalThis.TagCounts={countEvent,labelCount};if(typeof module!=='undefined')module.exports=TagCounts;
})();
