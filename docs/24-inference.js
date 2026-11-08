'use strict';
// Independent scalar JavaScript. A[from][to], no tensor or model API.
(() => {
 const K=3,F=8,SIZE=39;
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 const sum=xs=>xs.reduce((a,b)=>a+b,0);
 const lse=xs=>{const m=Math.max(...xs);return m+Math.log(sum(xs.map(v=>Math.exp(v-m))));};
 const argmax=xs=>xs.reduce((best,v,j)=>v>xs[best]?j:best,0);
 function features(x){
  if(!Array.isArray(x)||!x.length||x.length>256||x.some(v=>!Number.isInteger(v)||v< -2||v>2))throw Error('Use 1-256 integer observations from -2 to 2.');
  return x.map((v,t)=>[1,v/2,t?x[t-1]/2:0,t+1<x.length?x[t+1]/2:0,Math.abs(v)/2,Number(t===0),Number(t+1===x.length),t?x[t-1]*v/4:0]);
 }
 function potentials(theta,x,scale=1){
  if(!Array.isArray(theta)||theta.length!==SIZE||theta.some(v=>!finite(v)))throw Error('Invalid parameters');
  if(!finite(scale)||Math.abs(scale)>2)throw Error('Transition scale must be between -2 and 2.');
  const rows=features(x),u=rows.map(row=>Array.from({length:K},(_,j)=>sum(row.map((v,f)=>v*theta[j*F+f]))));
  return {emissions:u,transition:Array.from({length:K},(_,i)=>theta.slice(24+i*3,27+i*3).map(v=>v*scale)),start:theta.slice(33,36),end:theta.slice(36)};
 }
 function score(p,path){return p.start[path[0]]+p.end[path.at(-1)]+sum(path.map((j,t)=>p.emissions[t][j]))+sum(path.slice(1).map((j,t)=>p.transition[path[t]][j]));}
 function infer(p){
  const {emissions:u,transition:a,start:s,end:e}=p,n=u.length,k=s.length;
  if(!n||k<2||k>8||e.length!==k||a.length!==k||[...u,...a].some(row=>row.length!==k||row.some(v=>!finite(v)||Math.abs(v)>1e6))||[...s,...e].some(v=>!finite(v)||Math.abs(v)>1e6))throw Error('Invalid score dimensions or values');
  const alpha=[s.map((v,j)=>v+u[0][j])];
  for(let t=1;t<n;t++)alpha.push(u[t].map((v,j)=>v+lse(alpha[t-1].map((v,i)=>v+a[i][j]))));
  const z=lse(alpha.at(-1).map((v,j)=>v+e[j])),beta=Array.from({length:n},()=>Array(k).fill(0));beta[n-1]=[...e];
  for(let t=n-2;t>=0;t--)beta[t]=a.map(row=>lse(row.map((v,j)=>v+u[t+1][j]+beta[t+1][j])));
  const node=alpha.map((row,t)=>row.map((v,j)=>Math.exp(v+beta[t][j]-z))),edge=[];
  for(let t=0;t<n-1;t++)edge.push(a.map((row,i)=>row.map((v,j)=>Math.exp(alpha[t][i]+v+u[t+1][j]+beta[t+1][j]-z))));
  const viterbiScores=[s.map((v,j)=>v+u[0][j])],back=[Array(k).fill(0)];
  for(let t=1;t<n;t++){const choices=u[t].map((_,j)=>argmax(viterbiScores[t-1].map((v,i)=>v+a[i][j])));back.push(choices);viterbiScores.push(choices.map((i,j)=>u[t][j]+viterbiScores[t-1][i]+a[i][j]));}
  const path=[argmax(viterbiScores.at(-1).map((v,j)=>v+e[j]))];for(let t=n-1;t>0;t--)path.push(back[t][path.at(-1)]);path.reverse();const best=score(p,path);
  let expectation=sum(node[0].map((v,j)=>v*s[j]))+sum(node.at(-1).map((v,j)=>v*e[j]));
  for(let t=0;t<n;t++)expectation+=sum(u[t].map((v,j)=>v*node[t][j]));
  for(let t=0;t<n-1;t++)for(let i=0;i<k;i++)expectation+=sum(a[i].map((v,j)=>v*edge[t][i][j]));
  return {log_partition:z,alpha,beta,node,edge,viterbi:path,viterbi_scores:viterbiScores,backpointers:back,best_score:best,path_log_probability:best-z,entropy:z-expectation,marginal_modes:node.map(argmax)};
 }
 function objective(theta,data,lam,independent=false){
  const g=Array(SIZE).fill(0);let loss=0;
  for(const {x,y} of data){const p=potentials(theta,x),r=infer(p),rows=features(x);loss+=r.log_partition-score(p,y);
   for(let t=0;t<x.length;t++)for(let j=0;j<K;j++)for(let f=0;f<F;f++)g[j*F+f]+=(r.node[t][j]-Number(y[t]===j))*rows[t][f];
   if(!independent){for(let t=0;t<x.length-1;t++)for(let i=0;i<K;i++)for(let j=0;j<K;j++)g[24+i*3+j]+=r.edge[t][i][j]-Number(y[t]===i&&y[t+1]===j);for(let j=0;j<K;j++){g[33+j]+=r.node[0][j]-Number(y[0]===j);g[36+j]+=r.node.at(-1)[j]-Number(y.at(-1)===j);}}
  }
  for(let j=0;j<(independent?24:SIZE);j++)g[j]=g[j]/data.length+lam*theta[j];
  return {objective:loss/data.length+lam/2*sum(theta.map(v=>v*v)),gradient:g};
 }
 function normalization(denominator){
  if(![1,3,9,27,81].includes(denominator))throw Error('Invalid evidence weight');
  const gcd=(a,b)=>b?gcd(b,a%b):a,ratio=(a,b)=>{const d=gcd(a,b);return a/d+(b/d===1?'':'/'+b/d);};
  return {paths:[['A','C'],['B','A'],['B','C']],raw_weights:[ratio(1,denominator),'1',ratio(1,denominator)],global_partition:ratio(denominator+2,denominator),globally_normalized:[ratio(1,denominator+2),ratio(denominator,denominator+2),ratio(1,denominator+2)],locally_normalized:['1/2',ratio(denominator,2*(denominator+1)),ratio(1,2*(denominator+1))],global_branch_A:ratio(1,denominator+2),local_branch_A:'1/2'};
 }
 globalThis.TagInference={features,potentials,infer,score,objective,normalization};
 if(typeof module!=='undefined')module.exports=TagInference;
})();
