import unittest,math,itertools,random,json,subprocess,sys,copy
from fractions import Fraction
from taglab.conditioning import condition,label_event
from taglab.core import infer
from taglab.model import SIZE

class ConditioningTests(unittest.TestCase):
    def test_fraction_path_oracles(self):
        rng=random.Random(180)
        for trial in range(60):
            k=2+trial%2;n=1+trial%5
            u=[[rng.randrange(1,8) for _ in range(k)] for _ in range(n)];a=[[rng.randrange(1,8) for _ in range(k)] for _ in range(k)];s=[rng.randrange(1,5) for _ in range(k)];e=[rng.randrange(1,5) for _ in range(k)]
            allowed=[[j for j in range(k) if rng.randrange(3)] for _ in range(n)]
            weights={path:Fraction(s[path[0]]*e[path[-1]]*math.prod(u[t][j] for t,j in enumerate(path))*math.prod(a[i][j] for i,j in zip(path,path[1:]))) for path in itertools.product(range(k),repeat=n)}
            event={path:w for path,w in weights.items() if all(j in allowed[t] for t,j in enumerate(path))};r=condition([[math.log(v) for v in row] for row in u],[[math.log(v) for v in row] for row in a],list(map(math.log,s)),list(map(math.log,e)),allowed)
            self.assertEqual(r['possible'],bool(event));self.assertEqual(int(r['support_count']),len(event))
            if not event:self.assertIsNone(r['conditional']);json.dumps(r,allow_nan=False);continue
            z=sum(event.values());self.assertAlmostEqual(r['event_probability'],float(z/sum(weights.values())),places=11);c=r['conditional'];self.assertAlmostEqual(c['log_partition'],math.log(z),places=11)
            for t in range(n):
                for j in range(k):self.assertAlmostEqual(c['node'][t][j],float(sum(w for p,w in event.items() if p[t]==j)/z),places=11)
            for t in range(n-1):
                for i in range(k):
                    for j in range(k):self.assertAlmostEqual(c['edge'][t][i][j],float(sum(w for p,w in event.items() if p[t:t+2]==(i,j))/z),places=11)
            self.assertAlmostEqual(c['entropy'],-sum(float(w/z)*math.log(float(w/z)) for w in event.values()),places=10)
            self.assertEqual(weights[tuple(c['viterbi'])],max(event.values()));json.dumps(r,allow_nan=False)
    def test_allfree_and_pointmass(self):
        u=[[.2,-.3],[.4,-.2],[0.,.5]];a=[[.1,.3],[-.5,.6]];s=[.1,-.1];e=[-.2,.3];base=infer(u,a,s,e);r=condition(u,a,s,e,[[1,0]]*3)
        self.assertEqual(r['log_event_probability'],0.);self.assertEqual(r['event_probability'],1.)
        for key in ['node','edge','viterbi','alpha','beta','entropy']:self.assertEqual(r['conditional'][key],base[key])
        r=condition(u,a,s,e,[[1],[0],[1]]);self.assertEqual(r['conditional']['viterbi'],[1,0,1]);self.assertAlmostEqual(r['conditional']['entropy'],0,places=12);self.assertEqual(r['conditional']['node'],[[0.,1.],[1.,0.],[0.,1.]])
    def test_every_gradient_coordinate(self):
        rng=random.Random(31);theta=[rng.uniform(-.3,.3) for _ in range(SIZE)];x=[-2,1,0,2];allowed=[[0,2],[0,1,2],[1],[1,2]];r=label_event(theta,x,allowed);eps=1e-5
        for j in range(SIZE):
            plus=theta[:];minus=theta[:];plus[j]+=eps;minus[j]-=eps
            fd=(label_event(plus,x,allowed)['log_event_probability']-label_event(minus,x,allowed)['log_event_probability'])/(2*eps);self.assertAlmostEqual(fd,r['log_event_gradient'][j],places=8)
        self.assertTrue(all(abs(v)<1e-12 for v in label_event(theta,x,[[0,1,2]]*4)['log_event_gradient']))
    def test_independent_unconstrained_positions_unchanged(self):
        theta=[(j%5-2)/10 for j in range(24)]+[0.]*15;r=label_event(theta,[-2,1,0,2],[[0,1,2],[1],[0,1,2],[0,2]])
        for t in [0,2]:
            for a,b in zip(r['unrestricted_node'][t],r['conditional']['node'][t]):self.assertAlmostEqual(a,b,places=13)
    def test_rare_not_impossible(self):
        r=condition([[0.,-1000.]]*2,[[0.,0.],[0.,0.]],[0.,0.],[0.,0.],[[1],[1]])
        self.assertTrue(r['possible']);self.assertTrue(r['probability_underflow']);self.assertEqual(r['event_probability'],0);self.assertEqual(r['log_event_probability'],-2000.);self.assertEqual(r['conditional']['node'],[[0.,1.],[0.,1.]])
    def test_validation_and_no_mutation(self):
        u=[[0.,0.]];a=[[0.,0.],[0.,0.]];s=[0.,0.];e=[0.,0.];before=copy.deepcopy([u,a,s,e])
        for allowed in [None,[],[[True]],[[0,0]],[[2]],[[.5]],['0'],[[0],[1]]]:
            with self.assertRaises(ValueError):condition(u,a,s,e,allowed)
        condition(u,a,s,e,[[1,0]]);self.assertEqual([u,a,s,e],before)
        with self.assertRaises(ValueError):condition(u*257,a,s,e,[[0]]*257)
    def test_long_extremes_ties_and_zero_event(self):
        r=condition([[1e6,-1e6]]*256,[[1e6,-1e6],[-1e6,1e6]],[0.,0.],[0.,0.],[[0,1]]*255+[[1]])
        self.assertTrue(math.isfinite(r['log_event_probability']));self.assertEqual(r['conditional']['viterbi'][-1],1);json.dumps(r,allow_nan=False)
        r=condition([[0.]*3]*4,[[0.]*3]*3,[0.]*3,[0.]*3,[[2,1],[2,1],[2,1],[2,1]]);self.assertEqual(r['conditional']['viterbi'],[1]*4)
        r=condition([[0.,0.]]*3,[[0.,0.]]*2,[0.,0.],[0.,0.],[[0],[],[1]]);self.assertFalse(r['possible']);self.assertIsNone(r['log_event_probability']);json.dumps(r,allow_nan=False)
    def test_cli_strict_json(self):
        r=subprocess.run([sys.executable,'-m','taglab','condition','--symbols','[0,1]','--allowed','[[0],[0,1,2]]'],capture_output=True,text=True,check=True);self.assertTrue(json.loads(r.stdout)['inference']['possible'])
        r=subprocess.run([sys.executable,'-m','taglab','condition','--symbols','[0]','--allowed','[[true]]'],capture_output=True,text=True);self.assertNotEqual(r.returncode,0)
    def test_event_algebra_and_entropy_can_increase(self):
        u=[[math.log(1000),0.],[0.,0.]];a=[[math.log(1000),0.],[0.,0.]];s=e=[0.,0.]
        free=condition(u,a,s,e,[[0,1],[0,1]]);left=condition(u,a,s,e,[[0],[0,1]]);right=condition(u,a,s,e,[[1],[0,1]]);nested=condition(u,a,s,e,[[1],[1]])
        self.assertAlmostEqual(left['event_probability']+right['event_probability'],1.,places=12)
        self.assertLessEqual(nested['event_probability'],right['event_probability'])
        self.assertAlmostEqual(right['event_probability'],free['conditional']['node'][0][1],places=12)
        self.assertGreater(right['conditional']['entropy'],free['conditional']['entropy'])
