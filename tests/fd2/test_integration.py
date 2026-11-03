"""Fast CPU semantic tests using the real native FD2 scheduler and Wan blocks."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from methods.fd2.models.triangular_resampling import TriangularResamplingFD2
from methods.fd2.models.upstream import diffusion_forcing_wan as native

class TextStub(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.len,self.dim,self.cross_attn_norm,self.cross_rope=4,64,True,False
    def get_context(self,x,lengths,device,dtype,training=True):
        context=[torch.ones(4,64,device=device,dtype=dtype) for _ in lengths]
        return context,dict(cross_attn_mask=[torch.ones(n,4,device=device,dtype=torch.bool) for n in lengths])

def model(ratio=0,n=8,jitter=0,dim=263):
    with patch.object(native,'T5TextCrossModule',TextStub):
        return TriangularResamplingFD2(input_dim=dim,hidden_dim=64,ffn_dim=128,freq_dim=32,num_heads=4,num_layers=1,
             text_config={},attn_type='partial',prediction_type='vel',
             schedule_config=dict(steps=30,chunk_size=30,train_n_windows=n,random_epsilon=jitter,noise_type='linear',sigma_type='zero'),
             tr_config=dict(ratio=ratio,shift=.6,seed=1234)).train()

class Oracle(nn.Module):
    def __init__(self,velocity):
        super().__init__()
        self.velocity=velocity
        self.grad_flags=[]
    def forward(self,x,*args,**kwargs):
        self.grad_flags.append(torch.is_grad_enabled())
        return [self.velocity[:,:v.size(1)].clone() for v in x]

class BatchOracle(nn.Module):
    def __init__(self,velocities):
        super().__init__()
        self.velocities=velocities
    def forward(self,x,times,contexts,*args,**kwargs):
        assert not torch.is_grad_enabled()
        return [self.velocities[int(c[0,0])][:,:v.size(1)].clone() for v,c in zip(x,contexts)]

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        np.random.seed(1234)
    def test_native_off_parity(self):
        for n,jitter in [(1,0),(8,0),(8,.05)]:
            tr=model(0,n,jitter)
            baseline=copy.deepcopy(tr)
            baseline.__class__=native.DiffForcingWanModel
            batch=dict(feature=torch.randn(2,94,263),feature_length=torch.tensor([94,40]),text=['walk','dance'])
            state=torch.get_rng_state(),np.random.get_state()
            loss=baseline(batch)['total']
            loss.backward()
            torch.set_rng_state(state[0]);np.random.set_state(state[1])
            other=tr(batch)['total']
            other.backward()
            self.assertTrue(torch.equal(loss,other))
            for (name,a),(othername,b) in zip(baseline.named_parameters(),tr.named_parameters()):
                self.assertEqual(name,othername)
                if a.grad is not None: self.assertTrue(torch.equal(a.grad,b.grad),name)
    def test_mirrored_packing_parity(self):
        tr=model(1)
        tr.replay_sample=lambda *a,**k:None
        baseline=copy.deepcopy(tr)
        baseline.__class__=native.DiffForcingWanModel
        batch=dict(feature=torch.randn(2,200,263),feature_length=torch.tensor([200,94]),text=['walk','dance'])
        state=torch.get_rng_state(),np.random.get_state()
        expected=baseline(batch)['total']
        torch.set_rng_state(state[0]);np.random.set_state(state[1])
        actual=tr(batch)['total']
        self.assertTrue(torch.equal(actual,expected))
    def test_exact_flow_and_threshold_extremes(self):
        tr=model(1)
        for length in [1,29,30,94,200]:
            gt=torch.randn(263,length,1,1);noise=torch.randn_like(gt)
            tr.model=Oracle(gt-noise)
            for threshold in [0.,.45,1.]:
                tmax=tr.time_scheduler.get_t_max(length)
                times=[0.,tmax*.123,tmax*.451,tmax*.971]
                result=tr.replay_trajectory(gt,noise,times,torch.ones(4,64),torch.ones(length,4,dtype=torch.bool),threshold)
                for t,snap in zip(times,result['snapshots']):
                    alpha=(t-torch.arange(length)/30).clamp(0,1)
                    exact=gt*alpha[None,:,None,None]+noise*(1-alpha)[None,:,None,None]
                    torch.testing.assert_close(snap,exact,rtol=1e-5,atol=3e-6)
                self.assertFalse(any(tr.model.grad_flags))
                self.assertFalse(result['final'].requires_grad)
    def test_snapshot_branches_do_not_change_canonical_path(self):
        tr=model(1)
        gt=torch.randn(263,94,1,1);noise=torch.randn_like(gt)
        tr.model=Oracle(torch.full_like(gt,2.))
        ctx=torch.ones(4,64);mask=torch.ones(94,4,dtype=torch.bool)
        many=tr.replay_trajectory(gt,noise,[.131,.441,1.013,2.718],ctx,mask,.45)
        one=tr.replay_trajectory(gt,noise,[2.718],ctx,mask,.45)
        self.assertTrue(torch.equal(many['final'],one['final']))
    def test_masks_and_rope(self):
        tr=model(0)
        windows=[(0,30),(10,40),(20,50),(25,55)]
        mask=tr._build_multi_window_attn_mask(25,windows,0,'cpu')
        rope=tr._build_multi_window_rope_ids(25,windows,0,'cpu')[0]
        cursor=25
        for start,end in windows:
            size=end-start
            self.assertTrue(torch.equal(mask[cursor:cursor+size,:25],torch.arange(25)[None,:].expand(size,-1)<start))
            self.assertTrue(mask[cursor:cursor+size,cursor:cursor+size].all())
            self.assertFalse(mask[:25,cursor:cursor+size].any())
            self.assertTrue(torch.equal(rope[cursor:cursor+size],torch.arange(start,end)))
            other=mask[cursor:cursor+size,25:].clone()
            other[:,cursor-25:cursor-25+size]=False
            self.assertFalse(other.any())
            cursor+=size
    def test_batched_and_serial_replay_equivalent(self):
        tr=model(1)
        requests={}
        for i,length in enumerate([1,94,200]):
            gt=torch.randn(263,length,1,1);noise=torch.randn_like(gt)
            tmax=tr.time_scheduler.get_t_max(length)
            requests[i]=dict(gt=gt,noise=noise,times=[torch.tensor(tmax*f) for f in (.07,.413,.919)],
                            context=torch.full((4,64),float(i)),mask=torch.ones(length,4,dtype=torch.bool),threshold=.45,
                            key_rope=None,query_rope=None)
        tr.model=BatchOracle({i:r['gt']-r['noise'] for i,r in requests.items()})
        batched=tr.replay_batch(requests)
        for i,r in requests.items():
            tr.model=Oracle(r['gt']-r['noise'])
            serial=tr.replay_trajectory(r['gt'],r['noise'],r['times'],r['context'],r['mask'],r['threshold'])
            for a,b in zip(batched[i]['snapshots'],serial['snapshots']):
                torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-5)
    def test_rng_resume_and_validation_bypass(self):
        tr=model(1)
        tr.sample_threshold()
        sd=copy.deepcopy(tr.state_dict())
        expected=tr.sample_threshold()
        tr.load_state_dict(sd)
        self.assertEqual(expected,tr.sample_threshold())
        tr.eval()
        tr.replay_sample=lambda *a,**k: (_ for _ in ()).throw(AssertionError('replayed during eval'))
        tr(dict(feature=torch.randn(1,40,263),feature_length=torch.tensor([40]),text=['walk']))
    def test_reject_misaligned_jitter(self):
        with self.assertRaises(ValueError): model(1,8,.05)
    def test_text_microbatch_order_and_shapes(self):
        from methods.fd2.models.execution import MicrobatchedTextEncoder
        class Encoder:
            model='frozen'
            def __call__(self,texts,device): return [torch.full((len(t),2),float(ord(t[0]))) for t in texts]
        encoder=Encoder();split=MicrobatchedTextEncoder(encoder,2)
        texts=['abc','dddd','ee','fff','g']
        self.assertEqual(split.model,'frozen')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(encoder(texts,'cpu'),split(texts,'cpu'))))

if __name__ == '__main__':
    unittest.main(verbosity=2)
