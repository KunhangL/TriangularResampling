"""Mechanically adapted upstream forward; see build_integration.py and source manifest."""
import os
import numpy as np
import torch
from .upstream.diffusion_forcing_wan import EPSILON
from .upstream.quadratic_loss import quadratic_error_sum

class PackedReplayForward:
    def _forward_replayed(self, x):
        """Multi-window packed training.

        Per sample (valid_len = T_i), divide t-range [0, (T_i+c-1)/c) into
        `train_n_windows` equal buckets (default 10, configurable via
        schedule_config); each bucket yields one uniform t_k. Windows are
        derived from t_k via `time_scheduler.get_windows`, so the choice of
        n_windows stays consistent with the inference path's formula and
        training cost is independent of sequence length.
        The packed self-attn sequence is
            [clean feature | W_0 | W_1 | ... | W_{N-1}]
        wired together by `_build_multi_window_attn_mask` and per-token rope
        ids from `_build_multi_window_rope_ids`. Cross-attn is uniform (full
        True mask, no rope override, no cross-rope). Loss is per-token MSE
        averaged across all window tokens in the batch.

        When spatial > 1, the metadata (time / attn_mask / cross_attn_mask /
        seq_len) is expanded to patch granularity; rope_ids' f_ids stays at
        frame level since rope_apply broadcasts across H/W internally.
        """
        x = self._extract_inputs(x)
        feature_original = x["feature"]
        feature_length = x["feature_length"]
        feature_original = (feature_original - self.mean) / self.std
        batch_size = feature_original.shape[0]
        pad_seq_len = feature_original.shape[1]
        device = feature_original.device
        n_windows = int(self.schedule_config.get("train_n_windows", 10))

        valid_len = []
        feature = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), pad_seq_len)
            valid_len.append(length)
            feature.append(feature_original[i, :length, ...])
        feature = self.preprocess(feature)  # list of (C, T_i, 1, 1)

        # Text context + metadata. cross_attn_mask handles multi-segment Q->K
        # visibility; text_k_rope_ids / text_q_rope_ids carry per-segment rope
        # info when cross_rope=True (None otherwise).
        context, ctx_meta = self.text_module.get_context(
            x, valid_len, device, self.param_dtype, training=True,
        )
        cross_attn_mask_full = ctx_meta["cross_attn_mask"]
        text_k_rope_ids_full = ctx_meta.get("text_k_rope_ids")
        text_q_rope_ids_full = ctx_meta.get("text_q_rope_ids")

        all_xt = []
        all_t = []
        all_rope = []
        all_attn_mask = []
        all_cross_attn_mask = []
        all_targets = []  # per-sample list of (target_tensor, seq_start, seq_end)
        # Per-sample packed layout (used after the loop to gather Q-side text rope
        # and to shift K-side rope into the packed coordinate frame).
        all_clean_frame_starts = []
        all_clean_lens = []
        all_packed_windows = []

        for i in range(batch_size):
            T_i = valid_len[i]
            x0_i = feature[i]
            eps_i = torch.randn_like(x0_i)

            # Split t-range [0, t_max) into n_windows equal buckets; one
            # uniform sample per bucket. Bucket size scales with t_max so the
            # window count stays constant across sequence lengths.
            t_max = self.time_scheduler.get_t_max(T_i)
            bucket = t_max / n_windows
            t_values = [
                torch.tensor(
                    np.random.uniform(k * bucket, (k + 1) * bucket),
                    device=device,
                )
                for k in range(n_windows)
            ]
            input_start_all, _, output_start, output_end = self.time_scheduler.get_windows(
                [T_i] * n_windows, t_values, training=True,
            )
            windows = list(zip(output_start, output_end))

            # t_values is non-decreasing => window starts are
            # monotonically non-decreasing => clean section bounds:
            #   - clean_frame_start = min(input_start): the earliest window's
            #     history start
            #   - clean_frame_end   = max(output_start) = windows[-1][0]
            # Frames beyond clean_frame_end are not read by any W_k, and
            # frames before clean_frame_start are never visible to any W_k
            # (mask gates by frame index), so trimming the buffer to this
            # span is sufficient.
            clean_frame_start = min(input_start_all)
            clean_frame_end = windows[-1][0]
            clean_len = clean_frame_end - clean_frame_start

            # N schedules over T_i frames, one per t_k. Reuses existing
            # scheduler so noise_type / random_epsilon stay consistent with
            # generate.
            schedules, _ = self.time_scheduler.get_time_schedules(
                device, [T_i] * n_windows, t_values, training=True,
            )
            alphas, dalphas, betas, dbetas, _sg, _dla, _dlb = (
                self.time_scheduler.get_noise_levels(
                    device, [T_i] * n_windows, schedules,
                )
            )
            # Clean section: same α·x0 + β·noise noising path as every window,
            # so the latent agrees with the (possibly random_epsilon-jittered)
            # schedule that's also handed to the model as time embedding.
            clean_schedule, _ = self.time_scheduler.get_time_schedules(
                device, [T_i], [torch.tensor(t_max, device=device)],
                training=True,
            )
            clean_alphas, _, clean_betas, _, _, _, _ = (
                self.time_scheduler.get_noise_levels(
                    device, [T_i], clean_schedule,
                )
            )

            replay = self.replay_sample(x0_i, eps_i, t_values, context[i], ctx_meta, i)

            # Assemble packed sequence [clean | W_0 | ... | W_{N-1}].
            ca = clean_alphas[0][clean_frame_start:clean_frame_end]
            cb = clean_betas[0][clean_frame_start:clean_frame_end]
            x0_c = x0_i[:, clean_frame_start:clean_frame_end, ...]
            eps_c = eps_i[:, clean_frame_start:clean_frame_end, ...]
            parts_xt = [
                x0_c * ca[None, :, None, None]
                + eps_c * cb[None, :, None, None]
            ]
            if replay is not None:
                parts_xt = [replay["final"][:, clean_frame_start:clean_frame_end]]
            parts_time = [clean_schedule[0][clean_frame_start:clean_frame_end]]
            targets = []
            cursor = clean_len
            for k in range(n_windows):
                s, e = windows[k]
                L = e - s
                a = alphas[k][s:e]
                b = betas[k][s:e]
                x0_w = x0_i[:, s:e, ...]
                eps_w = eps_i[:, s:e, ...]
                xt_w = (
                    x0_w * a[None, :, None, None]
                    + eps_w * b[None, :, None, None]
                )
                if replay is not None:
                    xt_w = replay["snapshots"][k][:, s:e]
                    eps_w = (xt_w - x0_w * a[None, :, None, None]) / b.clamp(min=EPSILON)[None, :, None, None]
                parts_xt.append(xt_w)
                parts_time.append(schedules[k][s:e])

                if self.prediction_type == "vel":
                    da = dalphas[k][s:e]
                    db = dbetas[k][s:e]
                    tgt = (
                        x0_w * da[None, :, None, None]
                        + eps_w * db[None, :, None, None]
                    )
                elif self.prediction_type == "x0":
                    tgt = x0_w
                elif self.prediction_type == "eps":
                    tgt = eps_w
                else:
                    raise ValueError(
                        f"Unknown prediction_type: {self.prediction_type!r}"
                    )
                targets.append((tgt, cursor, cursor + L))
                cursor += L

            xt_i = torch.cat(parts_xt, dim=1)
            time_i = torch.cat(parts_time) * self.time_embedding_scale

            all_xt.append(xt_i)
            all_t.append(time_i)
            all_rope.append(
                self._build_multi_window_rope_ids(
                    clean_len, windows, clean_frame_start, device,
                )
            )
            all_attn_mask.append(
                self._build_multi_window_attn_mask(
                    clean_len, windows, clean_frame_start, device,
                )
            )
            # Cross-attn mask: gather per-frame rows from cross_attn_mask_full
            # to follow the packed [clean | W_k] layout. Frame-level here;
            # spatial expansion happens after the loop.
            all_cross_attn_mask.append(
                self._gather_for_packed(
                    cross_attn_mask_full[i], clean_frame_start, clean_len, windows,
                )
            )
            all_targets.append(targets)
            all_clean_frame_starts.append(clean_frame_start)
            all_clean_lens.append(clean_len)
            all_packed_windows.append(windows)

        self.apply_deferred_replay(all_xt, all_targets, all_clean_frame_starts, all_clean_lens, all_packed_windows)

        # Expand frame-level time / cross_attn_mask to patch-level
        # (no-ops when spatial_size == 1).
        all_t = self._expand_schedule_to_patch(all_t)
        all_cross_attn_mask = self._expand_mask_to_patch(all_cross_attn_mask)
        # seq_len = max(pad_seq_len, max_packed). In n=1 mode
        # packed_length <= T_i <= pad_seq_len so pad_seq_len dominates and
        # the model sees the same padded geometry as the inference path.
        # With n>1 (partial only) packed can exceed pad_seq_len; the max
        # keeps the upper bound correct.
        seq_len_for_model = self._patch_seq_len(
            max(pad_seq_len, max(xt.size(1) for xt in all_xt))
        )
        text_seq_len = max(int(ctx.size(0)) for ctx in context)

        # Cross-attn rope (only active when cross_rope=True): K rope encodes
        # per-token segment-start-frame in absolute coords, Q rope override
        # exists for null samples (zeros over T_i). Both must be expressed in
        # the same packed coordinate frame as self-attn rope_ids (which is
        # relative to clean_frame_start), so:
        #   - K rope: shift by clean_frame_start (subtract) per sample
        #   - Q rope: gather per-frame f_ids to packed layout, then shift
        text_k_rope_ids = self._shift_rope_ids(
            text_k_rope_ids_full, all_clean_frame_starts,
        )
        text_q_rope_ids = self._gather_text_q_rope_packed(
            text_q_rope_ids_full, all_clean_frame_starts, all_clean_lens,
            all_packed_windows,
        )
        text_q_rope_ids = self._shift_rope_ids(
            text_q_rope_ids, all_clean_frame_starts,
        )

        # The multi-window mask is only meaningful for attn_type='partial'.
        # Under full/causal (where __init__ already forced n_windows=1), the
        # packed sequence is a single window, so we let WanModel apply
        # default full attention or its internal causal mask.
        final_attn_mask = all_attn_mask if self.attn_type == "partial" else None
        if os.environ.get("FD2_AUDIT_SHAPES") == "1":
            packed_lens = [int(v.size(1)) for v in all_xt]
            clean_lens = [int(v) for v in all_clean_lens]
            window_lens = [
                [int(e - s) for s, e in windows]
                for windows in all_packed_windows
            ]
            mask_shapes = (
                [tuple(int(v) for v in m.shape) for m in all_attn_mask]
                if final_attn_mask is not None else None
            )
            print(
                "FD2_AUDIT_SHAPES "
                f"batch={batch_size} n_windows={n_windows} "
                f"pad_seq_len={pad_seq_len} seq_len_for_model={seq_len_for_model} "
                f"packed_lens_minmax=({min(packed_lens)},{max(packed_lens)}) "
                f"clean_lens_minmax=({min(clean_lens)},{max(clean_lens)}) "
                f"window_lens_first={window_lens[0]} "
                f"mask_shapes_first={mask_shapes[0] if mask_shapes else None} "
                f"attn_type={self.attn_type}",
                flush=True,
            )
        predicted_result = self.model(
            all_xt,
            all_t,
            context,
            seq_len_for_model,
            attn_mask=final_attn_mask,
            text_k_rope_ids=text_k_rope_ids,
            text_q_rope_ids=text_q_rope_ids,
            cross_attn_mask=all_cross_attn_mask,
            text_seq_len=text_seq_len,
            y=None,
            rope_ids=all_rope,
        )

        # Per-sample mean -> batch mean. Within each sample sum SE across
        # all W_k tokens and divide by the token count to get per-sample
        # MSE, then average across the batch.
        loss = 0.0
        for i in range(batch_size):
            pred_i = predicted_result[i]
            sample_se_sum = pred_i.new_zeros(())
            sample_n = 0
            for tgt, s, e in all_targets[i]:
                error = pred_i[:, s:e, ...] - tgt
                if self.loss_matrix is None:
                    sample_se_sum = sample_se_sum + (error ** 2).sum()
                else:
                    sample_se_sum = sample_se_sum + quadratic_error_sum(
                        error, self.loss_matrix
                    )
                sample_n += error.numel()
            loss = loss + sample_se_sum / max(sample_n, 1)
        loss = loss / batch_size
        return {"total": loss, "mse": loss}
