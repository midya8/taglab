import unittest,math,itertools,random,json,subprocess,sys,copy
from fractions import Fraction
from taglab.sampling import sample,prepare,draw,uniforms

class SamplingTests(unittest.TestCase):
 def test_exhaustive_conditional_product(self):
  rng=random.Random(91);checked=0
  for trial in range(60):
   k=2+trial%2;n=1+trial%4;u=[[rng.randrange(1,8) for _ in range(k)] for _ in range(n)];a=[[rng.randrange(1,6) for _ in range(k)] for _ in range(k)];s=[rng.randrange(1,5) for _ in range(k)];e=[rng.randrange(1,5) for _ in range(k)];allowed=[[j for j in range(k) if rng.randrange(3)] for _ in range(n)];allowed=[r or [0] for r in allowed]
   paths=list(itertools.product(*allowed));weights={p:Fraction(s[p[0]]*e[p[-1]]*math.prod(u[t][p[t]] for t in range(n))*math.prod(a[p[t]][p[t+1]] for t in range(n-1))) for p in paths};z=sum(weights.values());args=([[math.log(v) for v in r] for r in u],[[math.log(v) for v in r] for r in a],list(map(math.log,s)),list(map(math.log,e)),allowed);prepared=prepare(*args)
   for path in paths:
    vals=[0.]*n;prob=Fraction(1)
    for t in range(n-1,-1,-1):
     compatible=[p for p in paths if p[t+1:]==path[t+1:]];den=sum(weights[p] for p in compatible);row=[sum(weights[p] for p in compatible if p[t]==j)/den for j in range(k)];prob*=row[path[t]];vals[t]=float(sum(row[:path[t]])+row[path[t]]/2)
    out=draw(prepared,vals);self.assertEqual(out['path'],list(path));self.assertEqual(prob,weights[path]/z);self.assertAlmostEqual(out['proposal_log_probability'],math.log(float(prob)),places=10);self.assertAlmostEqual(out['model_path_log_probability'],math.log(float(prob)),places=10)
    for tr in out['trace']:
     self.assertLessEqual(tr['bin'][0],tr['uniform']);self.assertLess(tr['uniform'],tr['bin'][1]);self.assertAlmostEqual(sum(tr['probabilities']),1,places=12)
    checked+=1
  self.assertEqual(checked,428)
 def test_prng_and_boundaries(self):
  self.assertEqual(uniforms(1,3),[270369.5/2**32,67634689.5/2**32,2647435461.5/2**32]);p=prepare([[0,0],[0,0]],[[0,0],[0,0]],[0,0],[0,0],[[0,1],[0,1]])
  self.assertEqual(draw(p,[0,0])['path'],[0,0]);self.assertEqual(draw(p,[.5,.5])['path'],[1,1]);self.assertEqual(draw(p,[math.nextafter(1,0)]*2)['path'],[1,1]);self.assertEqual(draw(p,[.2,.8],'joint')['path'],draw(p,[.2,.8],'independent')['path'])
 def test_independent_node_is_different_joint(self):
  args=([[0,0],[0,0]],[[math.log(9),0],[0,math.log(9)]],[0,0],[0,0],[[0,1],[0,1]])
  r=prepare(*args)['event']['conditional'];self.assertAlmostEqual(sum(abs(r['edge'][0][i][j]-r['node'][0][i]*r['node'][1][j]) for i in range(2) for j in range(2))/2,.4)
  joint=sample(*args,[.6,.4]);wrong=sample(*args,[.6,.4],'independent');self.assertEqual(joint['path'],[0,0]);self.assertEqual(wrong['path'],[1,0]);self.assertAlmostEqual(math.exp(wrong['proposal_log_probability']),.25);self.assertAlmostEqual(math.exp(wrong['model_path_log_probability']),.05)
 def test_strict_validation_and_no_mutation(self):
  args=([[0,0],[0,0]],[[0,0],[0,0]],[0,0],[0,0],[[0],[1]])
  orig=copy.deepcopy(args);self.assertEqual(sample(*args,[0,.999])['path'],[0,1]);self.assertEqual(args,orig)
  for vals in [[],[0],[False,.5],[-.1,.5],[1,.5],[math.nan,.5],[math.inf,.5],['0',.5]]:
   with self.assertRaises((ValueError,TypeError)):sample(*args,vals)
  with self.assertRaises(ValueError):sample(*args,[0,0],'wrong')
  with self.assertRaises(ValueError):prepare(*args[:-1],[[],[0]])
  for seed in [0,True,1.1,2**32,-1]:
   with self.assertRaises(ValueError):uniforms(seed,1)
  for count in [0,True,1.1,131073]:
   with self.assertRaises(ValueError):uniforms(1,count)
 def test_underflow_and_long_scores(self):
  tiny=sample([[0,-40]],[[0,0],[0,0]],[0,0],[0,0],[[0,1]],[0]);self.assertEqual(tiny['trace'][0]['collapsed_cdf_labels'],[1])
  r=sample([[0,-2000]],[[0,0],[0,0]],[0,0],[0,0],[[0,1]],[0]);self.assertEqual(r['path'],[0]);self.assertEqual(r['trace'][0]['underflow_labels'],[1]);self.assertEqual(r['trace'][0]['log_probabilities'],[0,-2000]);json.dumps(r,allow_nan=False)
  r=sample([[1e6,-1e6]]*256,[[1e6,-1e6],[-1e6,1e6]],[1e6,-1e6],[1e6,-1e6],[[0,1]]*256,uniforms(17,256));self.assertEqual(r['path'],[0]*256);self.assertTrue(math.isfinite(r['model_path_log_probability']))

 def test_cli(self):
  out=subprocess.run([sys.executable,'-m','taglab','sample','--symbols','[0,1]','--allowed','[[0,1,2],[0,1,2]]','--uniforms','[0.2,0.7]'],text=True,capture_output=True);self.assertEqual(out.returncode,0,out.stderr);r=json.loads(out.stdout)['inference'];self.assertEqual(r['uniforms'],[.2,.7]);self.assertEqual(len(r['trace']),2)
  out=subprocess.run([sys.executable,'-m','taglab','sample','--symbols','[0]','--allowed','[[]]'],text=True,capture_output=True);self.assertNotEqual(out.returncode,0);self.assertIn('impossible',out.stderr)
