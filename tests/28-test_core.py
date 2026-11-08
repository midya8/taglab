import unittest,math,itertools,random
from fractions import Fraction as Q
from taglab.core import infer,score
from taglab.model import SIZE,F,K,objective,fit,features,potentials,metrics
from taglab.normalization import comparison


def exact_reference(unary,transition,start,end):
    n=len(unary);k=len(start);paths=list(itertools.product(range(k),repeat=n));weights=[]
    for path in paths:
        w=start[path[0]]*end[path[-1]]
        for t,j in enumerate(path):w*=unary[t][j]
        for a,b in zip(path,path[1:]):w*=transition[a][b]
        weights.append(w)
    z=sum(weights);node=[[Q(0) for _ in range(k)] for _ in range(n)];edge=[[[Q(0) for _ in range(k)] for _ in range(k)] for _ in range(n-1)]
    for path,w in zip(paths,weights):
        for t,j in enumerate(path):node[t][j]+=Q(w,z)
        for t,(a,b) in enumerate(zip(path,path[1:])):edge[t][a][b]+=Q(w,z)
    return z,node,edge,max(weights),paths,weights


class Core(unittest.TestCase):
    def test_sixty_exact_integer_potential_oracles(self):
        rng=random.Random(731)
        for c in range(60):
            k=2+c%3;n=1+c%5;u=[[rng.randint(1,4) for _ in range(k)] for _ in range(n)];a=[[rng.randint(1,4) for _ in range(k)] for _ in range(k)];s=[rng.randint(1,4) for _ in range(k)];e=[rng.randint(1,4) for _ in range(k)]
            z,node,edge,best,paths,weights=exact_reference(u,a,s,e);r=infer([[math.log(v) for v in row] for row in u],[[math.log(v) for v in row] for row in a],list(map(math.log,s)),list(map(math.log,e)))
            self.assertAlmostEqual(r['log_partition'],math.log(z),places=11);self.assertAlmostEqual(r['best_score'],math.log(best),places=11)
            for t in range(n):
                for j in range(k):self.assertAlmostEqual(r['node'][t][j],float(node[t][j]),places=11)
            for t in range(n-1):
                for i in range(k):
                    for j in range(k):self.assertAlmostEqual(r['edge'][t][i][j],float(edge[t][i][j]),places=11)
            entropy=-sum(float(Q(w,z))*math.log(float(Q(w,z))) for w in weights);self.assertAlmostEqual(r['entropy'],entropy,places=10)
    def test_edge_marginal_consistency(self):
        r=infer([[.3,-.8,.7],[.1,.4,-.9],[.8,-.2,.1]],[[.2,.6,-.2],[-.9,.2,.5],[.4,-.7,.9]],[.1,.4,-.2],[.9,.1,-.4])
        for t,edge in enumerate(r['edge']):
            for j in range(3):self.assertAlmostEqual(sum(edge[j]),r['node'][t][j]);self.assertAlmostEqual(sum(row[j] for row in edge),r['node'][t+1][j])
    def test_single_token_has_no_edge(self):
        r=infer([[1.,2.]],[[9.,8.],[7.,6.]],[.2,.4],[.3,.7]);self.assertEqual(r['edge'],[]);self.assertEqual(r['viterbi'],[1])
    def test_row_shift_gauge(self):
        u=[[.2,.4],[-.3,.7],[.2,.8]];a=[[.1,.4],[-.2,.8]];s=[0.,1.];e=[.7,-.3];r=infer(u,a,s,e);q=infer([[x+z for x in row] for row,z in zip(u,[900.,-200.,100.])],a,s,e)
        self.assertAlmostEqual(q['log_partition']-r['log_partition'],800.);self.assertEqual(q['viterbi'],r['viterbi'])
        for p,v in zip(q['node'],r['node']):
            for x,y in zip(p,v):self.assertAlmostEqual(x,y,places=10)
    def test_long_extreme_scores_stay_logarithmic(self):
        r=infer([[-1000.,-1001.]]*1000,[[-20.,-21.],[-19.,-20.]],[0.,0.],[0.,0.]);self.assertTrue(math.isfinite(r['log_partition']));self.assertLess(r['log_partition'],-1e6)
        for row in r['node']:self.assertAlmostEqual(sum(row),1.,places=6)
    def test_ties_follow_declared_recursion(self):
        r=infer([[0.,0.]]*4,[[0.,0.],[0.,0.]],[0.,0.],[0.,0.]);self.assertEqual(r['viterbi'],[0]*4);self.assertEqual(r['marginal_modes'],[0]*4);self.assertEqual(r['backpointers'],[[0,0]]*4)
    def test_invalid_dimensions_values_and_paths(self):
        good=([[0.,0.]],[[0.,0.],[0.,0.]],[0.,0.],[0.,0.])
        for bad in [([],good[1],good[2],good[3]),([[0.]],good[1],good[2],good[3]),([[True,0.]],good[1],good[2],good[3]),([[float('nan'),0.]],good[1],good[2],good[3]),([[float('inf'),0.]],good[1],good[2],good[3]),([[1e7,0.]],good[1],good[2],good[3])]:
            with self.assertRaises(ValueError):infer(*bad)
        for path in [[],[True],[2],[-1],[0,1]]:
            with self.assertRaises(ValueError):score(*good,path)
    def test_exact_label_bias_example(self):
        for denominator in [1,3,9,27,81]:
            r=comparison(Q(1,denominator));p=list(map(Q,r['globally_normalized']));local=list(map(Q,r['locally_normalized']));self.assertEqual(sum(p),1);self.assertEqual(sum(local),1);self.assertEqual(p[0],Q(1,denominator+2));self.assertEqual(local[0],Q(1,2))
        self.assertEqual(comparison()['globally_normalized'],['1/11','9/11','1/11']);self.assertEqual(comparison()['locally_normalized'],['1/2','9/20','1/20'])


class Gradient(unittest.TestCase):
    data=[dict(x=[-2,0,1],y=[0,1,2]),dict(x=[1,-1],y=[1,0])]
    def test_all39_parameters_finite_differences(self):
        theta=[math.sin(i*.7)*.3 for i in range(SIZE)];loss,g=objective(theta,self.data,.17)
        for j in range(SIZE):
            a=theta.copy();b=theta.copy();a[j]+=1e-5;b[j]-=1e-5;fd=(objective(a,self.data,.17)[0]-objective(b,self.data,.17)[0])/2e-5;self.assertAlmostEqual(fd,g[j],places=8)
    def test_exhaustive_expected_feature_gradient(self):
        theta=[(i%7-3)/7 for i in range(SIZE)];lam=.17;loss,g=objective(theta,self.data,lam);want=[lam*v for v in theta];nll=lam/2*sum(v*v for v in theta)
        def counts(x,path):
            c=[0.]*SIZE
            # Independent feature construction, no model.features call.
            for t,j in enumerate(path):
                left=x[t-1] if t else 0;right=x[t+1] if t+1<len(x) else 0
                for f,v in enumerate([1,x[t]/2,left/2,right/2,abs(x[t])/2,int(t==0),int(t+1==len(x)),left*x[t]/4]):c[j*8+f]+=v
            for a,b in zip(path,path[1:]):c[24+a*3+b]+=1
            c[33+path[0]]+=1;c[36+path[-1]]+=1
            return c
        for item in self.data:
            paths=list(itertools.product(range(3),repeat=len(item['x'])));cs=[counts(item['x'],path) for path in paths];scores=[sum(v*w for v,w in zip(theta,c)) for c in cs];weights=[math.exp(s) for s in scores];z=sum(weights);observed=counts(item['x'],item['y']);nll+=(math.log(z)-sum(v*w for v,w in zip(theta,observed)))/len(self.data)
            for j in range(SIZE):want[j]+=(sum(w*c[j] for w,c in zip(weights,cs))/z-observed[j])/len(self.data)
        self.assertAlmostEqual(loss,nll,places=12)
        for a,b in zip(g,want):self.assertAlmostEqual(a,b,places=12)
    def test_independent_factorization_and_gradient(self):
        theta=[math.cos(i)*.2 for i in range(24)]+[0.]*15;loss,g=objective(theta,self.data,.3,True);manual=.15*sum(v*v for v in theta)
        for item in self.data:
            u,*_=potentials(theta,item['x'])
            for row,j in zip(u,item['y']):manual+=(-row[j]+math.log(sum(math.exp(v) for v in row)))/len(self.data)
        self.assertAlmostEqual(loss,manual,places=12);self.assertEqual(g[24:],[0.]*15)
    def test_descent_trace_and_zero_control_scores(self):
        m=fit(self.data,.3,True,max_steps=8);self.assertEqual(m['theta'][24:],[0.]*15)
        for a,b in zip(m['history'],m['history'][1:]):self.assertLess(b['objective'],a['objective']);self.assertEqual([v-a['step']*g for v,g in zip(a['theta'],a['gradient'])],b['theta'])
    def test_strong_convexity_direction_check(self):
        theta=[math.sin(i)*.2 for i in range(SIZE)];d=[math.cos(i*.3) for i in range(SIZE)];eps=1e-4;lam=.3
        gp=objective([v+eps*w for v,w in zip(theta,d)],self.data,lam)[1];gm=objective([v-eps*w for v,w in zip(theta,d)],self.data,lam)[1];curvature=sum(w*(a-b)/(2*eps) for w,a,b in zip(d,gp,gm));self.assertGreaterEqual(curvature,lam*sum(v*v for v in d)-1e-7)
    def test_strict_features_and_training(self):
        for x in [[],[True],[3],[1.0],[-3],list(range(257))]:
            with self.assertRaises(ValueError):features(x)
        for data,lam in [([],.3),([dict(x=[1],y=[True])],.3),(self.data,0),(self.data,float('nan'))]:
            with self.assertRaises(ValueError):objective([0.]*SIZE,data,lam)
        with self.assertRaises(ValueError):objective([.1]*SIZE,self.data,.3,True)
