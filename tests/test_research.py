import unittest
from datetime import datetime,timezone
import numpy as np
from ticklab.data import aggregate_rows
from ticklab.features import causal_features,make_features,split_masks,FEATURES
from ticklab.metrics import metrics,fit_temperature,temperature_scale,block_bootstrap_difference,psi,losses

START=int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()*1000)
def row(i,ms,price=100,size=1,maker='false'):
    return [str(i),str(price),str(size),str(i),str(i),str(START+ms),maker,'true']
def fixture(n=200):
    p=100*np.exp(np.arange(n)*.00005)
    return {'end':START+(np.arange(n)+1)*10000,'close':p,'high':p+.01,'low':p-.01,'qty':np.arange(n)%5+1.,'signed':np.ones(n),'count':np.ones(n)*4,'age':np.ones(n)}

class IngestionTests(unittest.TestCase):
    def test_bin_boundary_and_side(self):
        b,q=aggregate_rows([row(1,9999,100,2),row(2,10000,102,3,'true')],'2024-01-01')
        self.assertEqual(b['close'][0],100);self.assertEqual(b['close'][1],102)
        self.assertEqual(b['signed'][0],2);self.assertEqual(b['signed'][1],-3);self.assertEqual(q['aggregate_trades'],2)
    def test_no_backfill_before_first_trade(self):
        b,_=aggregate_rows([row(1,15000)],'2024-01-01');self.assertTrue(np.isnan(b['close'][0]));self.assertEqual(b['close'][2],100);self.assertEqual(b['qty'][2],0)
    def test_duplicates_and_time_reversal_rejected(self):
        for rows in [[row(1,10),row(1,11)],[row(1,20),row(2,10)]]:
            with self.assertRaises(ValueError):aggregate_rows(rows,'2024-01-01')
    def test_invalid_value_and_timestamp_rejected(self):
        for r in [row(1,0,price=-1),row(1,0,size=float('nan')),row(1,86400000),row(1,0,maker='maybe')]:
            with self.assertRaises(ValueError):aggregate_rows([r],'2024-01-01')
    def test_missing_ids_counted(self):
        _,q=aggregate_rows([row(1,1),row(4,2)],'2024-01-01');self.assertEqual(q['missing_aggregate_ids'],2)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):aggregate_rows([],'2024-01-01')

class FeatureTests(unittest.TestCase):
    def test_features_do_not_change_when_future_changes(self):
        b=fixture();a=causal_features(b);b['close'][101:]*=3;b['qty'][101:]*=100;b['high'][101:]*=3
        np.testing.assert_allclose(a[:101],causal_features(b)[:101],equal_nan=True)
    def test_target_uses_correct_endpoint_and_class(self):
        b=fixture();x,y,r,t,end=make_features(b,threshold=2.)
        np.testing.assert_array_equal(end-t,60000);self.assertTrue((y==2).all());np.testing.assert_allclose(r,3.,atol=1e-9);self.assertEqual(x.shape[1],len(FEATURES))
    def test_tail_labels_are_purged(self):
        b=fixture();_,_,_,t,e=make_features(b);self.assertLess(e.max(),b['end'][-1]);self.assertEqual(t.min(),b['end'][30])
    def test_stale_current_or_target_excluded(self):
        b=fixture();b['age'][50]=31;_,_,_,t,_=make_features(b);self.assertNotIn(b['end'][50],t);self.assertNotIn(b['end'][44],t)
    def test_split_overlap_detected(self):
        data={'days':np.array([1,2,3,4]),'times':np.array([0,20,40,60]),'ends':np.array([30,35,55,75])}
        with self.assertRaises(ValueError):split_masks(data,{'train_end':1,'select':2,'calibrate':3,'test':[4]})
    def test_chronological_partitions_disjoint(self):
        data={'days':np.array([1,2,3,4]),'times':np.array([0,20,40,60]),'ends':np.array([10,30,50,70])}
        masks=split_masks(data,{'train_end':1,'select':2,'calibrate':3,'test':[4]});np.testing.assert_array_equal(sum(masks.values()),np.ones(4))

class MetricTests(unittest.TestCase):
    def test_perfect_metrics(self):
        y=np.array([0,1,2]);m=metrics(y,np.eye(3));self.assertEqual(m['accuracy'],1);self.assertEqual(m['brier'],0);self.assertEqual(m['ece'],0);self.assertEqual(sum(r['n'] for r in m['reliability']),3)
    def test_invalid_probabilities_rejected(self):
        for p in [np.ones((3,3)),np.full((3,3),np.nan),np.array([[1,-1,1]])]:
            with self.assertRaises(ValueError):losses(np.array([0,1,2]),p)
    def test_temperature_preserves_argmax_and_normalization(self):
        p=np.array([[.9,.06,.04],[.1,.3,.6]]);q=temperature_scale(p,2);np.testing.assert_allclose(q.sum(1),1);np.testing.assert_array_equal(p.argmax(1),q.argmax(1));self.assertTrue((q.max(1)<p.max(1)).all())
    def test_calibration_cannot_increase_fit_loss(self):
        p=np.tile([.98,.01,.01],(30,1));y=np.tile([0,1,2],10);t=fit_temperature(y,p);self.assertLessEqual(losses(y,temperature_scale(p,t)).mean(),losses(y,p).mean())
    def test_bootstrap_identity_zero_and_deterministic(self):
        y=np.tile([0,1,2],20);p=np.full((60,3),1/3);days=np.repeat([1,2],30)
        a=block_bootstrap_difference(y,p,p,days,20,5);self.assertEqual(a['low'],0);self.assertEqual(a['high'],0);self.assertEqual(a,block_bootstrap_difference(y,p,p,days,20,5))
    def test_psi_identity_and_shift(self):
        x=np.arange(1000);self.assertAlmostEqual(psi(x,x),0);self.assertGreater(psi(x,x+500),.2)

if __name__=='__main__':unittest.main()
