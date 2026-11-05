"""One canonical replay trajectory, exact branch snapshots, native FD2 packing."""
import copy
import math

import torch
import torch.distributed as dist

from .packed_forward import PackedReplayForward
from .upstream.diffusion_forcing_wan import DiffForcingWanModel
from .upstream.quadratic_loss import quadratic_error_sum


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
        self.target_mode = str(self.tr_config.get('target_mode', 'default'))
        self.disable_threshold_cut = bool(
            self.tr_config.get('disable_threshold_cut', self.target_mode == 'pure_dmd')
        )
        self.dmd_weight = float(self.tr_config.get('dmd_weight', 0.0))
        self.dmd_cfg_scale = float(self.tr_config.get('dmd_cfg_scale', 1.0))
        self.dmd_lr = float(self.tr_config.get('dmd_lr', 2.0e-5))
        self.dmd_warmup_steps = int(self.tr_config.get('dmd_warmup_steps', 20))
        self.dmd_critic_steps = int(self.tr_config.get('dmd_critic_steps', 5))
        self.dmd_critic_mb = int(self.tr_config.get('dmd_critic_mb', 4))
        self.dmd_min_alpha = float(self.tr_config.get('dmd_min_alpha', 0.02))
        self.dmd_max_alpha = float(self.tr_config.get('dmd_max_alpha', 0.98))
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
        if self.disable_threshold_cut:
            return 0.0
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
        self._rollout_v_targets={}
        for index,result in results.items():
            request=self._pending_replays[index]
            gt=request['gt']
            full_clean=result.get('full_clean',result['final'])
            cs,cl=clean_starts[index],clean_lens[index]
            parts=[full_clean[:,cs:cs+cl].detach() if self.target_mode == 'pure_dmd' else result['final'][:,cs:cs+cl]]
            schedules,_=self.time_scheduler.get_time_schedules(gt.device,[gt.size(1)]*len(windows[index]),request['times'],training=False)
            v_tgt_parts=[]
            for k,(start,end) in enumerate(windows[index]):
                if self.target_mode == 'pure_dmd':
                    x0_w_clean=full_clean[:,start:end].detach()
                    alpha_k=schedules[k][start:end][None,:,None,None].to(x0_w_clean.dtype)
                    eps_k=torch.randn_like(x0_w_clean)
                    parts.append(alpha_k*x0_w_clean+(1.0-alpha_k)*eps_k)
                    v_tgt_parts.append(x0_w_clean.float()-eps_k.float())
                    continue
                state=result['snapshots'][k][:,start:end]
                parts.append(state)
                alpha=schedules[k][start:end][None,:,None,None]
                residual=(state-gt[:,start:end]*alpha)/(1-alpha).clamp(min=.05)
                old_target,a,b=targets[index][k]
                targets[index][k]=(gt[:,start:end]-residual,a,b)
            all_xt[index]=torch.cat(parts,dim=1)
            if v_tgt_parts:
                self._rollout_v_targets[index]=torch.cat(v_tgt_parts,dim=1)

    @torch.no_grad()
    def replay_batch(self, requests):
        """One model forward per grid point for all still-active training samples.

        Fractional snapshots reuse that grid point's velocity. Consequently
        each is exactly the same one-step Euler branch as serial replay, and
        no snapshot changes the canonical trajectory or commits future states.
        """
        need_full_clean=(self.target_mode=='pure_dmd')
        states={i:r['noise'].clone() for i,r in requests.items()}
        values={
            i:[float(t) for t in r['times']]+([float(self.time_scheduler.get_t_max(r['gt'].size(1)))] if need_full_clean else [])
            for i,r in requests.items()
        }
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
                    if r['threshold'] > 0:
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
            full_clean=snapshots[i].pop() if need_full_clean else snapshots[i][-1]
            final=snapshots[i][-1]
            for snap,t in zip(snapshots[i],r['times']):
                _,_,starts,_=self.time_scheduler.get_windows([r['gt'].size(1)],[t],training=True)
                if not torch.equal(snap[:,:starts[0]],final[:,:starts[0]]):
                    raise AssertionError('A committed frame changed in batched replay')
            result[i]=dict(final=final,full_clean=full_clean,snapshots=snapshots[i],threshold=r['threshold'])
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

    def _ensure_dmd_bundle(self, device):
        bundle = getattr(self, '_dmd_bundle', None)
        if bundle is not None:
            return bundle
        real_m = copy.deepcopy(self.model).to(device).eval()
        real_m.requires_grad_(False)
        fake_m = copy.deepcopy(self.model).to(device).train()
        fake_m.requires_grad_(True)
        opt_fake = torch.optim.AdamW(
            fake_m.parameters(),
            lr=self.dmd_lr,
            betas=(0.0, 0.999),
            weight_decay=0.0,
        )
        enc_null = None
        if self.dmd_cfg_scale != 1.0:
            with torch.no_grad():
                enc_null = self.text_module.encode([''], device)[0].to(
                    device=device, dtype=self.param_dtype
                )
        bundle = dict(
            real_model=real_m,
            fake_model=fake_m,
            opt_fake=opt_fake,
            encoded_null=enc_null,
            step=0,
        )
        object.__setattr__(self, '_dmd_bundle', bundle)
        return bundle

    def compute_dmd_loss_and_step_critic(
        self,
        all_xt,
        all_t_raw,
        context,
        seq_len_for_model,
        final_attn_mask,
        text_k_rope_ids,
        text_q_rope_ids,
        all_cross_attn_mask,
        text_seq_len,
        all_rope,
        all_clean_lens,
        predicted_result,
    ):
        device = all_xt[0].device
        batch_size = len(all_xt)
        bundle = self._ensure_dmd_bundle(device)
        real_m = bundle['real_model']
        fake_m = bundle['fake_model']
        opt_fake = bundle['opt_fake']
        cur_step = int(bundle.get('step', 0))
        bundle['step'] = cur_step + 1

        all_xt_dmd = [xt.detach() for xt in all_xt]
        all_t_dmd = self._expand_schedule_to_patch(all_t_raw)
        ctx_detached = [c.detach() for c in context]
        alpha_dmd_wins = [
            all_t_raw[i][all_clean_lens[i]:] / float(self.time_embedding_scale)
            for i in range(batch_size)
        ]

        loss_fake_total = 0.0
        critic_mb = max(self.dmd_critic_mb, 1)
        n_critic = max(self.dmd_critic_steps, 1)
        rollout_v_targets = getattr(self, '_rollout_v_targets', None)

        for _ in range(n_critic):
            opt_fake.zero_grad(set_to_none=True)
            for mb_s in range(0, batch_size, critic_mb):
                mb_e = min(batch_size, mb_s + critic_mb)
                mb_idx = range(mb_s, mb_e)
                v_fake_mb = fake_m(
                    [all_xt_dmd[i] for i in mb_idx],
                    [all_t_dmd[i] for i in mb_idx],
                    [ctx_detached[i] for i in mb_idx],
                    seq_len_for_model,
                    attn_mask=[final_attn_mask[i] for i in mb_idx] if final_attn_mask is not None else None,
                    text_k_rope_ids=[text_k_rope_ids[i] for i in mb_idx] if text_k_rope_ids is not None else None,
                    text_q_rope_ids=[text_q_rope_ids[i] for i in mb_idx] if text_q_rope_ids is not None else None,
                    cross_attn_mask=[all_cross_attn_mask[i] for i in mb_idx] if all_cross_attn_mask is not None else None,
                    text_seq_len=text_seq_len,
                    y=None,
                    rope_ids=[all_rope[i] for i in mb_idx] if all_rope is not None else None,
                )
                loss_fake_mb = 0.0
                for local_j, i in enumerate(mb_idx):
                    c_len = all_clean_lens[i]
                    v_target_fake = rollout_v_targets[i].detach().float()
                    err_fake = v_fake_mb[local_j][:, c_len:].float() - v_target_fake
                    if self.loss_matrix is None:
                        se_fake = (err_fake ** 2).sum()
                    else:
                        se_fake = quadratic_error_sum(err_fake, self.loss_matrix.float())
                    loss_fake_mb = loss_fake_mb + se_fake / max(err_fake.numel(), 1)
                loss_fake_mb = loss_fake_mb / batch_size
                loss_fake_mb.backward()
                loss_fake_total = loss_fake_total + float(loss_fake_mb.detach()) / n_critic
            if dist.is_available() and dist.is_initialized():
                for p in fake_m.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(fake_m.parameters(), 1.0)
            opt_fake.step()
            opt_fake.zero_grad(set_to_none=True)

        warmup_scale = min(1.0, float(cur_step + 1) / float(max(self.dmd_warmup_steps, 1)))

        with torch.no_grad():
            v_real = real_m(
                all_xt_dmd,
                all_t_dmd,
                ctx_detached,
                seq_len_for_model,
                attn_mask=final_attn_mask,
                text_k_rope_ids=text_k_rope_ids,
                text_q_rope_ids=text_q_rope_ids,
                cross_attn_mask=all_cross_attn_mask,
                text_seq_len=text_seq_len,
                y=None,
                rope_ids=all_rope,
            )
            v_fake_ng = fake_m(
                all_xt_dmd,
                all_t_dmd,
                ctx_detached,
                seq_len_for_model,
                attn_mask=final_attn_mask,
                text_k_rope_ids=text_k_rope_ids,
                text_q_rope_ids=text_q_rope_ids,
                cross_attn_mask=all_cross_attn_mask,
                text_seq_len=text_seq_len,
                y=None,
                rope_ids=all_rope,
            )

        loss_dmd = 0.0
        for i in range(batch_size):
            c_len = all_clean_lens[i]
            alpha_win = alpha_dmd_wins[i].float()
            m_win = ((alpha_win >= self.dmd_min_alpha) & (alpha_win <= self.dmd_max_alpha)).float()[None, :, None, None]
            v_student_win = predicted_result[i][:, c_len:].float()
            v_real_win = v_real[i][:, c_len:].float()
            v_fake_win = v_fake_ng[i][:, c_len:].float()
            mask_denom = (m_win.sum() * v_student_win.size(0)).clamp(min=1.0)
            norm_s = ((v_student_win.detach() - v_real_win).abs() * m_win).sum() / mask_denom
            norm_f = ((v_fake_win - v_real_win).abs() * m_win).sum() / mask_denom
            norm_factor = norm_s + norm_f + 1e-4
            grad_v = torch.nan_to_num((v_fake_win - v_real_win) / norm_factor).clamp(-2.0, 2.0) * m_win
            dmd_target_v = (v_student_win.detach() - grad_v).detach()
            err_dmd = (v_student_win - dmd_target_v) * m_win
            if self.loss_matrix is None:
                se_dmd = (err_dmd ** 2).sum()
            else:
                se_dmd = quadratic_error_sum(err_dmd, self.loss_matrix.float())
            loss_dmd = loss_dmd + 0.5 * se_dmd / mask_denom
        loss_dmd = warmup_scale * (loss_dmd / batch_size)

        return loss_dmd, torch.tensor(loss_fake_total, device=device)
