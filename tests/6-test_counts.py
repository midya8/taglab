import unittest,itertools,math,random,json,subprocess,sys
from fractions import Fraction
from copy import deepcopy
from taglab.counts import count_event,label_count
from taglab.core import score,infer
from taglab.model import SIZE,potentials
from taglab.experiment import saved

class CountTests(unittest.TestCase):
 def test_sixty_exact_path_oracles_all_labels_all_counts(self):
  rng=random.Random(877);checked=0
  for case in range(60):
   k=2+case%2;n=1+case%5;u=[[rng.randint(1,7) for _ in range(k)] for _ in range(n)];tr=[[rng.randint(1,7) for _ in range(k)] for _ in range(k)];s=[rng.randint(1,7) for _ in range(k)];e=[rng.randint(1,7) for _ in range(k)];p=([[math.log(v) for v in row] for row in u],[[math.log(v) for v in row] for row in tr],list(map(math.log,s)),list(map(math.log,e)))
   paths=list(itertools.product(range(k),repeat=n));weights={y:s[y[0]]*e[y[-1]]*math.prod(u[t][j] for t,j in enumerate(y))*math.prod(tr[a][b] for a,b in zip(y,y[1:])) for y in paths};total=sum(weights.values())
   for label in range(k):
    for c in range(n+1):
     r=count_event(*p,label,c);chosen={y:w for y,w in weights.items() if y.count(label)==c};z=sum(chosen.values());checked+=1
     self.assertEqual(int(r['support_count']),len(chosen));self.assertAlmostEqual(r['event_probability'],float(Fraction(z,total)),places=11);self.assertAlmostEqual(sum(r['count_probabilities']),1,places=11)
     for q in range(n+1):self.assertAlmostEqual(r['count_probabilities'][q],sum(w for y,w in weights.items() if y.count(label)==q)/total,places=11)
     cond=r['conditional'];self.assertEqual(cond['viterbi'].count(label),c);self.assertEqual(weights[tuple(cond['viterbi'])],max(chosen.values()))
     self.assertAlmostEqual(r['conditioned_expected_count'],c,places=10)
     self.assertAlmostEqual(cond['entropy'],-sum(float(Fraction(w,z))*math.log(float(Fraction(w,z))) for w in chosen.values()),places=10)
     for t in range(n):
      for j in range(k):self.assertAlmostEqual(cond['node'][t][j],sum(w for y,w in chosen.items() if y[t]==j)/z,places=11)
     for t in range(n-1):
      for i in range(k):
       for j in range(k):self.assertAlmostEqual(cond['edge'][t][i][j],sum(w for y,w in chosen.items() if y[t:t+2]==(i,j))/z,places=11)
  self.assertEqual(checked,600)
 def test_nonlocal_constraint_even_for_independent_model(self):
  r=count_event([[0,0],[0,0]],[[0,0],[0,0]],[0,0],[0,0],0,1)
  self.assertEqual(r['count_probabilities'],[.25,.5,.25]);self.assertEqual(r['conditional']['node'],[[.5,.5],[.5,.5]])
  self.assertEqual(r['conditional']['edge'],[[[0,.5],[.5,0]]]);self.assertEqual(r['independent_node_count_probabilities'],[.25,.5,.25]);self.assertEqual(r['independent_node_event_probability'],.5)
 def test_all_39_count_event_gradients(self):
  theta=[math.sin(i+.3)*.2 for i in range(SIZE)];x=[-2,0,1,2];eps=1e-5
  for label,c in [(0,0),(1,2),(2,4)]:
   r=label_count(theta,x,label,c)
   for i,g in enumerate(r['log_event_gradient']):
    a=theta[:];b=theta[:];a[i]+=eps;b[i]-=eps
    fd=(label_count(a,x,label,c)['log_event_probability']-label_count(b,x,label,c)['log_event_probability'])/(2*eps)
    self.assertAlmostEqual(g,fd,places=8)
 def test_partition_recomposition_prefix_suffix_and_ties(self):
  p=([[0]*3 for _ in range(5)],[[0]*3 for _ in range(3)],[0]*3,[0]*3)
  r=count_event(*p,1,2);self.assertEqual(r['conditional']['viterbi'],[1,1,0,0,0]);self.assertAlmostEqual(r['conditional']['entropy'],math.log(80))
  for t in range(5):
   terms=[]
   for c in range(3):
    for j in range(3):
     a=r['alpha'][t][c][j];b=r['suffix_beta'][t][2-c][j]
     if a is not None and b is not None:terms.append(math.exp(a+b))
   self.assertAlmostEqual(sum(terms),80)
 def test_underflow_long_scores_and_strict_validation(self):
  p=([[0,-2000]],[[0,0],[0,0]],[0,0],[0,0]);r=count_event(*p,1,1);self.assertEqual(r['event_probability'],0);self.assertEqual(r['log_event_probability'],-2000);self.assertTrue(r['probability_underflow']);self.assertEqual(r['conditional']['node'],[[0,1]])
  p=([[1e6,-1e6] for _ in range(64)],[[1e6,-1e6],[-1e6,1e6]],[1e6,-1e6],[1e6,-1e6]);r=count_event(*p,0,32);self.assertEqual(r['conditional']['viterbi'].count(0),32);self.assertAlmostEqual(r['conditioned_expected_count'],32,places=5);json.dumps(r,allow_nan=False)
  before=deepcopy(p)
  for l,c in [(True,1),(0,True),(-1,1),(2,1),(0,-1),(0,65),(0,1.0),(0,float('nan'))]:
   with self.assertRaises(ValueError):count_event(*p,l,c)
  self.assertEqual(before,p)
  with self.assertRaises(ValueError):count_event([[0,0]]*65,[[0,0],[0,0]],[0,0],[0,0],0,2)

 def test_cli(self):
  out=subprocess.run([sys.executable,'-m','taglab','count','--symbols','[0,1]','--label','0','--count','1'],capture_output=True,text=True)
  self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(json.loads(out.stdout)['inference']['conditional']['viterbi'].count(0),1)
  out=subprocess.run([sys.executable,'-m','taglab','count','--symbols','[0,1]','--label','0','--count','3'],capture_output=True,text=True)
  self.assertNotEqual(out.returncode,0)
