import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('cache_ab',Path(__file__).resolve().parents[1]/'scripts/topic02_cache_ab.py')
ab=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

class CacheABTests(unittest.TestCase):
    def test_same_teacher_forced_positions(self):
        self.assertEqual(len(ab.PROMPT),42)
        self.assertEqual(len(ab.CONTINUATION),10)
        self.assertEqual(ab.CONTINUATION[-1],151645)
        self.assertEqual(ab.plan(False),[(0,n) for n in range(42,52)])
        self.assertEqual(ab.plan(True),[(0,42)]+[(n-1,n) for n in range(43,52)])
        self.assertEqual(sum(e-s for s,e in ab.plan(False)),465)
        self.assertEqual(sum(e-s for s,e in ab.plan(True)),51)

    def test_paired_contract_and_rejection(self):
        rows=[]
        for i in range(5):
            for mode in (['no_cache','cache'] if i%2==0 else ['cache','no_cache']):
                rows.append(dict(pair=i,mode=mode,wall_ms=20 if mode=='no_cache' else 10,
                    cuda_interval_ms=9,logits_allclose=True,argmax_equal=True))
        self.assertEqual(ab.summarize(rows)['ratio_of_medians'],2)
        for key,value in [('wall_ms',float('nan')),('wall_ms',0),('logits_allclose',False),('argmax_equal',False),('pair',9)]:
            broken=copy.deepcopy(rows);broken[0][key]=value
            with self.assertRaises(ValueError):ab.summarize(broken)
        with self.assertRaises(ValueError):ab.summarize(rows[:-1])

if __name__=='__main__':unittest.main()
