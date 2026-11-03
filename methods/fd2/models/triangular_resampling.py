"""One canonical replay trajectory, exact branch snapshots, native FD2 packing."""
import math

import torch

from .packed_forward import PackedReplayForward
from .upstream.diffusion_forcing_wan import DiffForcingWanModel


class TriangularResamplingFD2(PackedReplayForward, DiffForcingWanModel):
    def __init__(self, tr_config=None, execution_config=None, **kwargs):
        super().__init__(**kwargs)
        if execution_config and execution_config.get('text_microbatch_size'):
            from .execution import MicrobatchedTextEncoder
            self.text_module.text_encoder=MicrobatchedTextEncoder(
                self.text_module.text_encoder,int(execution_config['text_microbatch_size']))
        self.tr_config = dict(tr_config or {})
        self.resampling_ratio = float(self.tr_config.get('ratio', 0.0))
        self.tr_shift = float(self.tr_config.get('shift', 0.6))
        self.tr_rng = torch.Generator(device='cpu')
        self.tr_rng.manual_seed(int(self.tr_config.get('seed', 1234)))
        self.replay_audit = None
        self._pending_replays = None
        if not 0 <= self.resampling_ratio <= 1 or self.tr_shift <= 0:
            raise ValueError('TR requires 0 <= ratio <= 1 and shift > 0')
        if self.resampling_ratio > 0:
            if self.attn_type != 'partial' or self.prediction_type != 'vel':
                raise ValueError('FD2 TR currently requires partial attention and velocity prediction')
            if self.time_scheduler.noise_type != 'linear' or self.time_scheduler.sigma_type != 'zero':
                raise ValueError('FD2 TR currently validates the deterministic linear flow schedule')
            if self.time_scheduler.random_epsilon != 0:
                raise ValueError('TR replay snapshots require random_epsilon=0; .05 is retained for native TR-off regression')
            if self.tr_config.get('clamp_mode', 'frontier') != 'frontier':
                raise ValueError('Only per-sample shared frontier clamp is implemented in FD2 TR')
            if self.time_scheduler.steps % self.time_scheduler.chunk_size:
                raise ValueError('The canonical inference grid must close: steps % chunk_size == 0')

    def get_extra_state(self):
        return dict(version=1, tr_rng=self.tr_rng.get_state())

    def set_extra_state(self, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported TR RNG checkpoint version')
        self.tr_rng.set_state(state['tr_rng'].cpu())

    def forward(self, x):
        # Absolutely no TR draws or replay under ratio=0, including native jitter.
        if self.resampling_ratio == 0 or not self.training:
            return super().forward(x)
        self._pending_replays = {}
        try:
            return self._forward_replayed(x)
        finally:
            self._pending_replays = None

    def sample_threshold(self):
        eta = torch.randn((), generator=self.tr_rng).item()
        return float(torch.sigmoid(torch.tensor(eta + math.log(self.tr_shift))))

    @torch.no_grad()
    def replay_sample(self, gt, noise, times, context, metadata, sample_index):
        if torch.rand((),generator=self.tr_rng).item() >= self.resampling_ratio:
            return None
        threshold = self.sample_threshold()
        mask = metadata['cross_attn_mask'][sample_index]
        key_rope = metadata.get('text_k_rope_ids')
        query_rope = metadata.get('text_q_rope_ids')
        if self._pending_replays is not None:
            self._pending_replays[sample_index] = dict(gt=gt,noise=noise,times=times,context=context,mask=mask,threshold=threshold,
                key_rope=None if key_rope is None else key_rope[sample_index],query_rope=None if query_rope is None else query_rope[sample_index])
            return None
        return self.replay_trajectory(
            gt, noise, times, context, mask, threshold,
            None if key_rope is None else key_rope[sample_index],
            None if query_rope is None else query_rope[sample_index],
        )

    @torch.no_grad()
    def apply_deferred_replay(self, all_xt, targets, clean_starts, clean_lens, windows):
        """Batch canonical grid forwards, then replace native packed inputs/targets."""
        if not self._pending_replays: return
        results=self.replay_batch(self._pending_replays)
        for index,result in results.items():
            request=self._pending_replays[index]
            gt=request['gt']
            cs,cl=clean_starts[index],clean_lens[index]
            parts=[result['final'][:,cs:cs+cl]]
            schedules,_=self.time_scheduler.get_time_schedules(gt.device,[gt.size(1)]*len(windows[index]),request['times'],training=False)
            for k,(start,end) in enumerate(windows[index]):
                state=result['snapshots'][k][:,start:end]
                parts.append(state)
                alpha=schedules[k][start:end][None,:,None,None]
                residual=(state-gt[:,start:end]*alpha)/(1-alpha).clamp(min=.05)
                old_target,a,b=targets[index][k]
                targets[index][k]=(gt[:,start:end]-residual,a,b)
            all_xt[index]=torch.cat(parts,dim=1)

    @torch.no_grad()
    def replay_batch(self, requests):
        """One model forward per grid point for all still-active training samples.

        Fractional snapshots reuse that grid point's velocity. Consequently
        each is exactly the same one-step Euler branch as serial replay, and
        no snapshot changes the canonical trajectory or commits future states.
        """
        states={i:r['noise'].clone() for i,r in requests.items()}
        values={i:[float(t) for t in r['times']] for i,r in requests.items()}
        snapshots={i:[] for i in requests}
        step=0
        dt=1/self.time_scheduler.steps
        while any(len(snapshots[i])<len(values[i]) for i in requests):
            t0=step*dt
            # First collect grid-aligned snapshots (no velocity is necessary).
            for i in requests:
                while len(snapshots[i])<len(values[i]) and values[i][len(snapshots[i])] <= t0+1e-8:
                    snapshots[i].append(states[i].clone())
            active=[i for i in requests if len(snapshots[i])<len(values[i])]
            if not active: break
            lengths=[states[i].size(1) for i in active]
            device=states[active[0]].device
            times=[torch.tensor(t0,device=device) for _ in active]
            is_,ie_,os_,oe_=self.time_scheduler.get_windows(lengths,times,training=False)
            schedules,_=self.time_scheduler.get_time_schedules(device,lengths,times,training=False)
            xt=[states[i][:,s:e] for i,s,e in zip(active,is_,ie_)]
            ctx=[requests[i]['context'] for i in active]
            crosses=[requests[i]['mask'][s:e] for i,s,e in zip(active,is_,ie_)]
            key_rope=[requests[i]['key_rope'] for i in active]
            query_rope=[requests[i]['query_rope'] for i in active]
            keys=None if all(r is None for r in key_rope) else self._shift_rope_ids(key_rope,is_)
            qs=[]
            for item,s,e in zip(query_rope,is_,ie_):
                if item is None:qs.append(None)
                else:
                    f,h,w=item
                    qs.append((None if f is None else f[s:e],h,w))
            qs=None if all(r is None for r in qs) else self._shift_rope_ids(qs,is_)
            predictions=self.model(xt,self._expand_schedule_to_patch([a[s:e]*self.time_embedding_scale for a,s,e in zip(schedules,is_,ie_)]),
                self._expand_context_to_patch(ctx),self._patch_seq_len(max(lengths)),
                attn_mask=self._build_partial_mask(is_,ie_,os_,oe_,device),text_k_rope_ids=keys,text_q_rope_ids=qs,
                cross_attn_mask=self._expand_mask_to_patch(crosses),text_seq_len=max(int(c.size(0)) for c in ctx),y=None)
            for j,i in enumerate(active):
                r=requests[i]
                a,b=os_[j],oe_[j]
                velocity=predictions[j][:,a-is_[j]:b-is_[j]].to(states[i].dtype)
                def advance(t1):
                    out=states[i].clone()
                    out[:,a:b]+=velocity*(t1-t0)
                    alpha=(t1-torch.arange(lengths[j],device=device)/self.time_scheduler.chunk_size).clamp(0,1)
                    teacher=r['gt'][:,a:b]*alpha[a:b][None,:,None,None]+r['noise'][:,a:b]*(1-alpha[a:b])[None,:,None,None]
                    out[:,a:b]=torch.where((alpha[a:b]<r['threshold'])[None,:,None,None],teacher,out[:,a:b])
                    return out
                while len(snapshots[i])<len(values[i]) and values[i][len(snapshots[i])] < (step+1)*dt-1e-8:
                    endpoint=values[i][len(snapshots[i])]
                    snapshots[i].append(advance(endpoint))
                if len(snapshots[i])<len(values[i]):states[i]=advance((step+1)*dt)
            step+=1
            if step>max(self.time_scheduler.get_total_steps(r['gt'].size(1)) for r in requests.values())+1:
                raise AssertionError('Replay batch failed to terminate')
        result={}
        for i,r in requests.items():
            final=snapshots[i][-1]
            for snap,t in zip(snapshots[i],r['times']):
                _,_,starts,_=self.time_scheduler.get_windows([r['gt'].size(1)],[t],training=True)
                if not torch.equal(snap[:,:starts[0]],final[:,:starts[0]]):
                    raise AssertionError('A committed frame changed in batched replay')
            result[i]=dict(final=final,snapshots=snapshots[i],threshold=r['threshold'])
        self.replay_audit=dict(canonical_steps=step,thresholds={i:r['threshold'] for i,r in requests.items()},samples=len(requests))
        return result

    @torch.no_grad()
    def replay_trajectory(self, gt, noise, times, context, text_mask, threshold,
                          text_k_rope=None, text_q_rope=None):
        """Snapshots branch from canonical grid states without altering that grid.

        alpha increases from noise (0) to clean (1). One Gaussian trajectory
        and one threshold are shared by all snapshots of a training sample.
        Commit freezes a frame; no target window ever sees later history.
        """
        length = gt.size(1)
        values = [float(t) for t in times]
        if not values or values != sorted(values):
            raise ValueError('Replay times must be nonempty and nondecreasing')
        t_max = self.time_scheduler.get_t_max(length)
        if values[0] < 0 or values[-1] > t_max + 1e-7:
            raise ValueError('Replay time outside FD2 schedule')
        steps = self.time_scheduler.steps
        device = gt.device
        main = noise.detach().clone()
        current_step = 0
        snapshots, audit = [], []
        for endpoint in values:
            scaled = endpoint * steps
            nearest = round(scaled)
            # Only account for floating representation of genuine grid points;
            # do not ceil continuous training times.
            grid = nearest if abs(scaled-nearest) < 1e-6 else math.floor(scaled)
            while current_step < grid:
                t0 = current_step / steps
                t1 = (current_step + 1) / steps
                main = self._replay_euler(main, gt, noise, t0, t1,
                                         context, text_mask, threshold,
                                         text_k_rope, text_q_rope)
                current_step += 1
            grid_time = current_step / steps
            if endpoint > grid_time + 1e-8:
                # A disposable fractional branch; never feeds the next grid step.
                snap = self._replay_euler(main, gt, noise, grid_time, endpoint,
                                         context, text_mask, threshold,
                                         text_k_rope, text_q_rope)
            else:
                snap = main.clone()
            snapshots.append(snap)
            audit.append(dict(t=endpoint, grid_time=grid_time, partial_dt=max(0.,endpoint-grid_time)))
        # The last snapshot contains the shared committed prefix. Each earlier
        # window's prefix must be bitwise identical to this final prefix.
        final = snapshots[-1]
        for snap,t in zip(snapshots,values):
            _,_,starts,_ = self.time_scheduler.get_windows(
                [length],[torch.tensor(t,device=device)],training=True)
            prefix = starts[0]
            if not torch.equal(snap[:,:prefix],final[:,:prefix]):
                raise AssertionError('A committed frame changed after a snapshot')
        self.replay_audit = dict(threshold=threshold, snapshots=audit, canonical_steps=current_step)
        return dict(final=final, snapshots=snapshots, threshold=threshold)

    @torch.no_grad()
    def _replay_euler(self, state, gt, noise, t0, t1, context, text_mask, threshold,
                      text_k_rope=None, text_q_rope=None):
        if t1 < t0 or t1-t0 > 1/self.time_scheduler.steps + 1e-7:
            raise ValueError('Invalid canonical/fractional replay step')
        length = state.size(1)
        device = state.device
        t = torch.tensor(t0,device=device,dtype=torch.float32)
        is_,ie_,os_,oe_ = self.time_scheduler.get_windows([length],[t],training=False)
        s,e,a,b = is_[0],ie_[0],os_[0],oe_[0]
        schedules,_ = self.time_scheduler.get_time_schedules(device,[length],[t],training=False)
        xt = [state[:,s:e]]
        ts = self._expand_schedule_to_patch([schedules[0][s:e]*self.time_embedding_scale])
        self_mask = self._build_partial_mask(is_,ie_,os_,oe_,device)
        cross = self._expand_mask_to_patch([text_mask[s:e]])
        k_rope = None if text_k_rope is None else self._shift_rope_ids([text_k_rope],[s])
        if text_q_rope is None:
            q_rope = None
        else:
            f,h,w=text_q_rope
            q_rope = self._shift_rope_ids([(None if f is None else f[s:e],h,w)],[s])
        prediction = self.model(
            xt,ts,self._expand_context_to_patch([context]),self._patch_seq_len(max(length,e-s)),
            attn_mask=self_mask,text_k_rope_ids=k_rope,text_q_rope_ids=q_rope,
            cross_attn_mask=cross,text_seq_len=int(context.size(0)),y=None,
        )[0][:,a-s:b-s]
        out = state.clone()
        out[:,a:b] = out[:,a:b] + prediction.to(out.dtype)*(t1-t0)
        alpha = torch.clamp(t1-torch.arange(length,device=device)/self.time_scheduler.chunk_size,0,1)
        teacher = gt[:,a:b]*alpha[a:b][None,:,None,None]+noise[:,a:b]*(1-alpha[a:b])[None,:,None,None]
        out[:,a:b] = torch.where((alpha[a:b]<threshold)[None,:,None,None],teacher,out[:,a:b])
        return out
