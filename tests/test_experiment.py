import copy,json,unittest,tempfile,subprocess,sys
from pathlib import Path
from taglab.experiment import saved,audit,experiment,close
from taglab.export import export
from taglab.fixture import dataset
from taglab.model import metrics


class Experiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=saved()
    def test_all327_states_replay(self):self.assertEqual(audit(self.report),dict(models=6,states=327,selected={'chain':'chain-0.3','local':'local-0.3'},valid=True))
    def test_all_six_fits_reproduce(self):close(experiment(),self.report)
    def test_test_labels_do_not_change_training_or_selection(self):
        data=dataset()
        for row in data['test']:row['y']=[(j+1)%3 for j in row['y']]
        q=experiment(data);self.assertEqual(q['selected'],self.report['selected'])
        for a,b in zip(q['models'],self.report['models']):close(a['theta'],b['theta']);close(a['history'],b['history']);close(a['validation'],b['validation'])
        self.assertNotEqual(q['models'][0]['test'],self.report['models'][0]['test'])
    def test_winning_nll_is_not_winning_token_error(self):
        models={m['id']:m for m in self.report['models']};c=models[self.report['selected']['chain']]['test'];l=models[self.report['selected']['local']]['test'];self.assertLess(c['nll_per_token'],l['nll_per_token']);self.assertGreater(c['viterbi_token_errors'],l['viterbi_token_errors']);self.assertEqual((c['tokens'],c['viterbi_token_errors'],l['viterbi_token_errors']),(164,66,56))
    def test_mutations_are_rejected(self):
        mutations=[lambda r:r['models'][0]['history'][1]['theta'].__setitem__(0,3.),lambda r:r['models'][0]['history'][0].__setitem__('backtracks',29),lambda r:r['models'][0].__setitem__('status','converged'),lambda r:r['selected'].__setitem__('chain','chain-0.03'),lambda r:r['models'].pop(),lambda r:r['data']['train'][0]['y'].__setitem__(0,2),lambda r:r['witnesses']['chain-0.3'][0]['node'][0].__setitem__(0,.1)]
        for mutate in mutations:
            r=copy.deepcopy(self.report);mutate(r)
            with self.assertRaises(ValueError):audit(r)
    def test_archive_exports_byte_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=export(folder)
            for p in paths:self.assertEqual(Path(folder,p).read_bytes(),Path(__file__).resolve().parent.parent.joinpath('docs',p).read_bytes())
    def test_partition_identity_and_original_generator(self):
        self.assertEqual(self.report['data'],dataset());self.assertEqual([len(self.report['data'][p]) for p in ['train','validation','test']],[24,12,16]);self.assertEqual(sum(len(r['x']) for r in self.report['data']['test']),164)
    def test_cli_and_bad_values(self):
        root=Path(__file__).resolve().parent.parent
        for args in [['demo'],['normalization'],['infer','--symbols','[-2,0,2]'],['infer','--family','local','--symbols','[0]']]:
            x=subprocess.run([sys.executable,'-m','taglab']+args,cwd=root,capture_output=True,text=True);self.assertEqual(x.returncode,0,x.stderr);self.assertIsInstance(json.loads(x.stdout),dict)
        for args in [['infer','--symbols','[true]'],['infer','--symbols','[]'],['export'],['audit','--input','missing.json']]:
            x=subprocess.run([sys.executable,'-m','taglab']+args,cwd=root,capture_output=True,text=True);self.assertNotEqual(x.returncode,0);self.assertNotIn('Traceback',x.stderr)
