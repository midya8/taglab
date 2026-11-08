'use strict';
// Position-wise subset conditioning only. Original finite potentials stay unchanged.
(()=>{
const sum=xs=>xs.reduce((a,b)=>a+b,0),lse=xs=>{const m=Math.max(...xs);return m===-Infinity?m:m+Math.log(sum(xs.map(x=>Math.exp(x-m))));},argmax=(indices,fn)=>indices.reduce((a,b)=>fn(b)>fn(a)?b:a),clean=table=>table.map(row=>row.map(v=>v===-Infinity?null:v));
function condition(p,allowed){
 const base=TagInference.infer(p),{emissions:u,transition:a,start:s,end:e}=p,n=u.length,k=s.length,ids=Array.from({length:k},(_,j)=>j);
 if(n>256||!Array.isArray(allowed)||allowed.length!==n||allowed.some(row=>!Array.isArray(row)||new Set(row).size!==row.length||row.some(j=>!Number.isInteger(j)||j<0||j>=k)))throw Error('Use one unique integer label-subset list per position (maximum256 positions).');
 const rows=allowed.map(row=>[...row].sort((a,b)=>a-b)),count=rows.reduce((a,row)=>a*BigInt(row.length),1n),out={schema:'taglab-label-event-v1',allowed:rows,support_count:String(count),unrestricted_log_partition:base.log_partition,unrestricted_node:base.node};
 if(count===0n)return {...out,possible:false,log_event_probability:null,event_probability:0,probability_underflow:false,conditional:null,impossible_reason:'At least one position permits no label. A conditional distribution is undefined.'};
 const alpha=[ids.map(j=>rows[0].includes(j)?s[j]+u[0][j]:-Infinity)];
 for(let t=1;t<n;t++)alpha.push(ids.map(j=>rows[t].includes(j)?u[t][j]+lse(rows[t-1].map(i=>alpha[t-1][i]+a[i][j])):-Infinity));
 const z=lse(rows.at(-1).map(j=>alpha.at(-1)[j]+e[j])),beta=Array.from({length:n},()=>Array(k).fill(-Infinity));beta[n-1]=ids.map(j=>rows.at(-1).includes(j)?e[j]:-Infinity);
 for(let t=n-2;t>=0;t--)beta[t]=ids.map(i=>rows[t].includes(i)?lse(rows[t+1].map(j=>a[i][j]+u[t+1][j]+beta[t+1][j])):-Infinity);
 const node=alpha.map((row,t)=>row.map((v,j)=>rows[t].includes(j)?Math.exp(v+beta[t][j]-z):0)),edge=[];
 for(let t=0;t<n-1;t++)edge.push(ids.map(i=>ids.map(j=>rows[t].includes(i)&&rows[t+1].includes(j)?Math.exp(alpha[t][i]+a[i][j]+u[t+1][j]+beta[t+1][j]-z):0)));
 const v=[ids.map(j=>rows[0].includes(j)?s[j]+u[0][j]:-Infinity)],back=[Array(k).fill(null)];
 for(let t=1;t<n;t++){const choices=ids.map(j=>rows[t].includes(j)?argmax(rows[t-1],i=>v[t-1][i]+a[i][j]):null);v.push(choices.map((i,j)=>i===null?-Infinity:u[t][j]+v[t-1][i]+a[i][j]));back.push(choices);}
 const path=[argmax(rows.at(-1),j=>v.at(-1)[j]+e[j])];for(let t=n-1;t>0;t--)path.push(back[t][path.at(-1)]);path.reverse();const best=TagInference.score(p,path);
 let expected=sum(node[0].map((v,j)=>v*s[j]))+sum(node.at(-1).map((v,j)=>v*e[j]));for(let t=0;t<n;t++)expected+=sum(u[t].map((v,j)=>v*node[t][j]));for(const pair of edge)for(let i=0;i<k;i++)for(let j=0;j<k;j++)expected+=a[i][j]*pair[i][j];
 const logp=Math.min(0,z-base.log_partition),prob=Math.exp(logp);
 return {...out,possible:true,log_event_probability:logp,event_probability:prob,probability_underflow:prob===0,conditional:{log_partition:z,node,edge,alpha:clean(alpha),beta:clean(beta),viterbi:path,best_score:best,path_log_probability:best-z,entropy:z-expected,marginal_modes:node.map(row=>argmax(ids,j=>row[j])),viterbi_scores:clean(v),backpointers:back}};
}
function expectedFeatures(symbols,r){const rows=TagInference.features(symbols),g=Array(39).fill(0);for(let t=0;t<rows.length;t++)for(let j=0;j<3;j++)for(let f=0;f<8;f++)g[j*8+f]+=r.node[t][j]*rows[t][f];for(const pair of r.edge)for(let i=0;i<3;i++)for(let j=0;j<3;j++)g[24+i*3+j]+=pair[i][j];for(let j=0;j<3;j++){g[33+j]=r.node[0][j];g[36+j]=r.node.at(-1)[j];}return g;}
function labelEvent(theta,symbols,allowed){const p=TagInference.potentials(theta,symbols),out=condition(p,allowed);if(out.possible){const before=expectedFeatures(symbols,TagInference.infer(p)),after=expectedFeatures(symbols,out.conditional);out.log_event_gradient=after.map((v,j)=>v-before[j]);}else out.log_event_gradient=null;return out;}
globalThis.TagConditioning={condition,labelEvent,expectedFeatures};if(typeof module!=='undefined')module.exports=TagConditioning;
})();
