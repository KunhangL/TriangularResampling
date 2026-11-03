"""Resampling forcing dispatcher (streaming + mei138 + interleaved-PHC main path).

`sources` is a list and controls which stages run on the last `resample_len`
latents of each sample's clean context region:

* sources=["model"]: run streaming GT-clamp Euler rollout on the trailing
  `resample_len` latents and overwrite them in-place. Earlier context stays GT.
* sources=["phc"]: skip model rollout; route the GT tail through the PHC
  integrator (decoded_138 anchor). On failure the tail is left untouched.
* sources=["model","phc"]: streaming model rollout with n_replace=1 PHC
  corrections fired inside the Euler loop right after each latent commits,
  matching inference `2 fwd : 1 commit : 1 PHC` cadence (always interleaved).

Streaming Euler loop semantics:
    alpha_init = 0 全位置 (纯噪声起步, trajectory matches inference start),
    每步对 alpha(pos, step) < rt 的位置用 noised(GT, alpha) 钳制,
    否则保留 Euler 输出. rt per-rollout logit-normal 采样, 决定 GT 教师强制跨度.
    M_total = ceil(t_end/dt) per-sample, dt = 1/scheduler.steps batch-shared
    与 inference stream_generate_step 对齐.

PHC pipeline (`_phc_integrator_path`, mei138-only, decoded_138 anchor):
    VAE.decode(feature tail) → 138d → mei_138_to_phc_anchor_and_targets
    → PHC reset + rollout planning_horizon → re-encode → overwrite.

Note on conventions: Dev's TriangularTimeScheduler uses `schedule` values where
1 = clean and 0 = noise (i.e. `alpha = schedule`, `beta = 1 - schedule` for the
linear branch). The original FloodDiffusion resampling used the opposite
convention (`noise_level = 0` = clean). Inside this class, `resample_t` keeps
the original "noise strength" meaning (0 = clean, 1 = noise), and translates to
the Dev schedule via `schedule = 1 - resample_t` right before `get_noise_levels`.
"""
import math
import os
import time
import json
from collections import defaultdict

import numpy as np
import torch


_VALID_SOURCE_COMBOS = (["model"], ["phc"], ["model", "phc"])

# Prefix latents used to warm the streaming encoder cache before re-encoding a
# PHC-corrected window (must exceed the encoder's causal receptive field so the
# first_chunk=True seed effect washes out before the window boundary).
_ENC_BOOTSTRAP_TOKENS = 6


_PHC_PROFILE_KEYS = (
    "calls",
    "samples",
    "decode_ms",
    "anchor_target_ms",
    "refs_build_ms",
    "reset_ms",
    "action_ms",
    "sim_step_ms",
    "body_to_mei_ms",
    "encode_ms",
    "writeback_ms",
    "total_ms",
)


class Resampler:
    def __init__(
        self,
        sources=None,
        shift: float = 0.6,
        resample_len: int = 5,              # trailing N latents (PHC + replay window size). Must be > 0 and divisible by chunk_size.
        prediction_type: str = "vel",
        time_scheduler=None,
        time_embedding_scale: float = 1.0,
        clamp_mode: str = "frontier",
        # PHC-only (required when "phc" in sources, ignored otherwise):
        vae=None,
        phc_config=None,
        # Inference-only PHC replacement window. This revives the historical
        # test-time knob without changing training-side resample_len semantics.
        tt_phc_replace_len: int = None,
        time_alignment: str = "legacy_ceil",
    ):
        # Accept legacy string form; normalize to list.
        if sources is None:
            sources = ["model"]
        elif isinstance(sources, str):
            sources = [sources]
        else:
            sources = list(sources)
        assert sources in _VALID_SOURCE_COMBOS, (
            f"Invalid sources={sources}; must be one of {_VALID_SOURCE_COMBOS}"
        )
        assert time_scheduler is not None, "time_scheduler is required"
        self.sources = sources
        if time_alignment not in ("legacy_ceil", "exact"):
            raise ValueError(f"Unknown replay time alignment: {time_alignment!r}")
        if time_alignment == "exact" and sources != ["model"]:
            raise NotImplementedError("Exact replay time alignment is validated for model-only TR")
        self.time_alignment = time_alignment
        if clamp_mode not in ("frontier", "random", "none"):
            raise ValueError(
                "clamp_mode must be one of {'frontier', 'random', 'none'} "
                f"(got {clamp_mode!r})"
            )
        if shift <= 0:
            raise ValueError(f"shift must be positive (got {shift})")
        self.shift = shift
        self.clamp_mode = clamp_mode
        self.prediction_type = prediction_type
        self.time_scheduler = time_scheduler
        self.time_embedding_scale = time_embedding_scale

        # Streaming is the only mode after the rect cleanup.
        self._cs = int(time_scheduler.chunk_size)
        assert int(resample_len) > 0, (
            f"resample_len must be > 0 (got {resample_len})"
        )
        assert int(resample_len) % self._cs == 0, (
            f"resample_len ({resample_len}) must be a multiple of "
            f"time_scheduler.chunk_size ({self._cs})"
        )
        self.resample_len = int(resample_len)
        if tt_phc_replace_len is None:
            self._tt_phc_replace_len = 1
        else:
            self._tt_phc_replace_len = int(tt_phc_replace_len)
            if self._tt_phc_replace_len <= 0:
                raise ValueError(
                    "tt_phc_replace_len must be a positive integer "
                    f"(got {tt_phc_replace_len})"
                )
        # Multi-chunk replay + PHC source: PHC's planning_horizon is set to
        # resample_len at backend init in `_init_phc`, so PHC and model share
        # the same resample_len-token replay window.
        if self.resample_len > self._cs and "phc" in self.sources:
            print(
                f"[resampler] resample_len={self.resample_len} "
                f"(> chunk_size={self._cs}) with sources={self.sources}: model "
                f"and PHC both write the last {self.resample_len}-token replay "
                f"region (PHC planning_horizon set to {self.resample_len} latents)."
            )

        # Rolling per-step stats, reset by training loop after each forward pass.
        self._log_total = 0
        self._log_triggered = 0
        self._log_resample_t_sum = 0.0
        # PHC-only rolling stats.
        self._log_phc_fail = 0
        self._log_phc_latency_ms_sum = 0.0
        self._log_phc_profile = {k: 0.0 for k in _PHC_PROFILE_KEYS}
        self._phc_temp_buffers = {}
        self._phc_prefix_decode_cache = {}
        # Last resampled 20fps physical frame per stream. Carrying this across
        # PHC windows makes PHC->MEI rates and heading branch selection
        # continuous at block boundaries.
        self._phc_mei_boundary = {}
        self._phc_continue_ref_vel_blend_frames = 0
        # Persistent encoder cache for the streaming PHC-writeback re-encode
        # (inference stream; training keeps a per-sample state in `states`).
        self._infer_enc_state = self._new_phc_enc_state()

        self._phc_backend = None
        self._vae = None
        self._latent_mean = None
        self._latent_std = None
        self._phc_horizon_mode = None
        # Fixed anchor source (the gt_138 debug bypass was removed). Kept as a
        # constant attribute for debug-capture / log metadata.
        self.phc_anchor_source = "decoded_138"

        # Debug capture: when set to a dict (typically empty) externally,
        # resample_batch + PHC integrator populate keyed entries for full-
        # pipeline dbgviz. None disables capture. Schema (consumer is
        # `utils.debug_viz.render_phc_chain`):
        #   latent_pre_resample:   List[Tensor] clean clones at entry
        #   latent_post_resample:  List[Tensor] final clones at exit
        #   windows: List[dict], one entry per (chosen sample × PHC call)
        #     each entry: sample_i, rs_start, ctx_end, n_replace,
        #                 phc_init_state, phc_refs,
        #                 phc_post_reset_body_pos,
        #                 phc_rollout_body_pos, phc_rollout_body_rot,
        #                 mei138_pre_phc_decoded, mei138_phc_rollout,
        #                 mei138_post_phc_decoded
        self._dbg_train = None
        # Long-sequence inference trace capture. Debug-only side channel
        # used by `generate_long_sequences.main` when cfg.config.debug_chain
        # is enabled; None when disabled (zero-overhead fast path).
        # Keys: dir(str), window_limit(int), counter(int), records(list).
        self._dbg_infer = None

        # Continue-from-live-state bookkeeping. PHC re-anchors to the
        # controller's own tracked end pose every window except the first of a
        # stream (which seeds from the decoded anchor). Inference tracks one
        # stream; training resets per batch in _resample_with_optional_phc.
        self._infer_phc_started = False
        # Proposal-2 cache: continuous PHC-corrected mei138 per stream (the
        # physical-truth trajectory, never re-decoded). None outside inference;
        # populated only when reset_inference_state() arms it.
        self._phc_traj = None
        # Per-stream streaming decode cache for inference (mirror of training's
        # st['phc_decode_state']). Lazily created on the first correction so
        # prefix_tokens matches that window's rs_start; reset on buffer rollback.
        # When set, inference decodes only the new token per window (O(N), same
        # window-local path as training) instead of re-decoding [0:f_hi].
        self._infer_decode_state = None

        if "phc" in self.sources:
            assert vae is not None, "PHC source requires a VAE instance"
            assert phc_config is not None, "PHC source requires phc_config dict"
            self._init_phc(vae, phc_config)

    @property
    def inference_window_size(self) -> int:
        """Latents per PHC inference correction window.

        Defaults to 1 so existing inference keeps the interleaved training
        cadence. ``tt_phc_replace_len`` may override it for test-time block
        correction without changing training-side ``resample_len``.
        """
        return self._tt_phc_replace_len

    def _set_latent_norm_from_model(self, model):
        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
        if mean is None or std is None:
            raise AttributeError(
                f"{type(model).__name__} has no latent mean/std buffers"
            )
        self._latent_mean = mean.detach()
        self._latent_std = std.detach()

    def _denorm_latent_batch(self, latents):
        if self._latent_mean is None or self._latent_std is None:
            raise RuntimeError(
                "PHC VAE decode needs latent mean/std from the Wan model; "
                "call resample_batch(..., model=...) before PHC correction."
            )
        mean = self._latent_mean.to(device=latents.device, dtype=latents.dtype)
        std = self._latent_std.to(device=latents.device, dtype=latents.dtype)
        mean = mean.reshape(1, 1, -1)
        std = std.reshape(1, 1, -1)
        return latents * std + mean

    @staticmethod
    def _cache_clone(cache):
        cloned = []
        for item in cache:
            if torch.is_tensor(item):
                cloned.append(item.detach().clone())
            else:
                cloned.append(item)
        return cloned

    def _cache_snapshot(self, field="_feat_map"):
        model = getattr(self._vae, "model", None)
        cache = getattr(model, field, None)
        return None if cache is None else self._cache_clone(cache)

    def _cache_restore(self, cache, field="_feat_map", idx_field="_conv_idx"):
        model = getattr(self._vae, "model", None)
        if model is None or cache is None:
            return False
        setattr(model, field, self._cache_clone(cache))
        if hasattr(model, idx_field):
            setattr(model, idx_field, [0])
        return True

    @staticmethod
    def _cache_stack(caches):
        if not caches:
            return None
        if any(cache is None for cache in caches):
            return None
        first_len = len(caches[0])
        if any(len(cache) != first_len for cache in caches):
            return None
        stacked = []
        for items in zip(*caches):
            if all(torch.is_tensor(item) for item in items):
                base_shape = tuple(items[0].shape[1:])
                if any(tuple(item.shape[1:]) != base_shape for item in items):
                    return None
                stacked.append(torch.cat([
                    item.detach() for item in items
                ], dim=0))
            elif all(item is None for item in items):
                stacked.append(None)
            elif all(isinstance(item, str) and item == "Rep" for item in items):
                stacked.append("Rep")
            else:
                return None
        return stacked

    @staticmethod
    def _cache_split(cache, batch_size):
        if cache is None:
            return None
        split = [[] for _ in range(int(batch_size))]
        for item in cache:
            if torch.is_tensor(item):
                if int(item.shape[0]) != int(batch_size):
                    return None
                for b in range(int(batch_size)):
                    split[b].append(item[b:b + 1].detach().clone())
            else:
                for b in range(int(batch_size)):
                    split[b].append(item)
        return split

    @staticmethod
    def _vae_autocast(device):
        device = torch.device(device)
        enabled = (
            os.environ.get("PHYSFLOOD_PHC_VAE_BF16") == "1"
            and device.type == "cuda"
        )
        return torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        )

    def _vae_install_group_cache(self, streams, idxs, cache_key, first_chunk):
        """Install per-group VAE stream_decode cache before a decode call.

        When `first_chunk` is True, clears the VAE cache (no prior state to
        restore). Otherwise stacks each stream's cached `cache_key` field and
        writes it to `vae.model._feat_map` so the next `vae.stream_decode`
        call resumes from that combined state.
        """
        if first_chunk:
            self._vae.clear_cache()
            return
        stacked = self._cache_stack([streams[idx][cache_key] for idx in idxs])
        if stacked is None:
            raise RuntimeError(f"cannot stack {cache_key}")
        model = getattr(self._vae, "model", None)
        if model is None:
            raise RuntimeError("VAE has no .model attribute")
        model._feat_map = stacked
        if hasattr(model, "_conv_idx"):
            model._conv_idx = [0]

    def _cache_snapshot_split(self, n_g, label):
        """Snapshot current VAE cache and split into `n_g` per-stream pieces."""
        split = self._cache_split(self._cache_snapshot(), n_g)
        if split is None:
            raise RuntimeError(f"cannot split {label}")
        return split

    def _try_cached_mei138_prefix_decode(self, feat_full, sample_idx,
                                         rs_start_tok, decode_end_tok,
                                         vae_device):
        """Decode using a cached stable prefix before the replay window.

        Interleaved PHC rewrites tokens inside [rs_start, ...] after every
        correction, so replay-region decoder state cannot be safely reused.
        The prefix before rs_start is clean context and never changes within
        the resample pass; caching only that state keeps semantics identical
        while avoiding repeated decode of long stable prefixes.
        """
        if (
            self._vae is None
            or int(rs_start_tok) <= 0
            or int(decode_end_tok) <= int(rs_start_tok)
            or not hasattr(self._vae, "stream_decode")
        ):
            return None

        rs_start_tok = int(rs_start_tok)
        decode_end_tok = int(decode_end_tok)
        suffix_tok = decode_end_tok - rs_start_tok
        # stream_decode is per-sample here because each sample owns a different
        # decoder cache. It only wins when a long stable prefix is skipped and
        # the replay suffix is still short; otherwise the existing batched full
        # decode is faster. Thresholds tuned for the 138d main yaml.
        if rs_start_tok < 32 or suffix_tok > 16:
            return None
        key = (int(sample_idx), rs_start_tok)
        entry = self._phc_prefix_decode_cache.get(key)
        try:
            if entry is None:
                lat_prefix = (
                    feat_full[:, :rs_start_tok, 0, 0]
                    .transpose(0, 1)
                    .float()
                    .to(vae_device)
                )
                denorm_prefix = self._denorm_latent_batch(lat_prefix.unsqueeze(0))
                self._vae.clear_cache()
                with torch.no_grad(), self._vae_autocast(vae_device):
                    decoded_prefix = self._vae.stream_decode(
                        denorm_prefix, first_chunk=True,
                    )[0].float()
                # Full decode uses token*4 frame indexing downstream. If the
                # streaming decoder does not line up to that convention for a
                # given VAE variant, fall back to the exact full-decode path.
                expected_prefix_frames = self._mei138_frames_for_decode_tokens(
                    rs_start_tok,
                )
                if int(decoded_prefix.shape[0]) != expected_prefix_frames:
                    self._vae.clear_cache()
                    return None
                entry = {
                    "decoded_prefix": decoded_prefix.detach(),
                    "cache": self._cache_snapshot(),
                }
                self._phc_prefix_decode_cache[key] = entry

            if not self._cache_restore(entry["cache"]):
                return None
            lat_suffix = (
                feat_full[:, rs_start_tok:decode_end_tok, 0, 0]
                .transpose(0, 1)
                .float()
                .to(vae_device)
            )
            denorm_suffix = self._denorm_latent_batch(lat_suffix.unsqueeze(0))
            with torch.no_grad(), self._vae_autocast(vae_device):
                decoded_suffix = self._vae.stream_decode(
                    denorm_suffix, first_chunk=False,
                )[0].float()
            self._vae.clear_cache()
            full_dec = torch.cat(
                [entry["decoded_prefix"].to(decoded_suffix.device), decoded_suffix],
                dim=0,
            )
            expected_frames = self._mei138_frames_for_decode_tokens(decode_end_tok)
            if int(full_dec.shape[0]) < expected_frames:
                return None
            return full_dec
        except Exception:
            if hasattr(self._vae, "clear_cache"):
                self._vae.clear_cache()
            self._phc_prefix_decode_cache.pop(key, None)
            return None

    def _new_phc_decode_state(self, sample_idx, prefix_tokens):
        return {
            "sample_idx": int(sample_idx),
            "prefix_tokens": int(prefix_tokens),
            "tokens": None,
            "decoded": None,
            "cache": None,
            "cache_before_last": None,
            "frames_before_last": 0,
            "last_start": None,
            "last_end": None,
            "last_pre_cache": None,
            "last_pre_frames": None,
            "valid": True,
        }

    @staticmethod
    def _new_phc_enc_state():
        """Streaming encoder cache for the PHC-writeback re-encode.

        `tokens` = absolute token index the cache is positioned at (== the next
        window's rs_start when it can be reused); `cache` = `_enc_feat_map`
        snapshot at that boundary.
        """
        return {"tokens": None, "cache": None}

    def _decode_token_span_batched(
        self,
        streams,
        feat_fulls,
        end_toks,
        replace_start_toks=None,
        *,
        vae_device,
        frame_scale_20fps=4,
    ):
        """Incrementally decode interleaved-PHC mei138 frames for N streams.

        Groups streams by (initial, start_tok, end_tok, replace_mode) so VAE
        `stream_decode` can be called once per group on stacked latents. N=1
        groups go through the same code path (cache stack/split are trivial
        at batch=1, and VAE.stream_decode is fully homogeneous in batch dim).

        Each per-stream state dict tracks (tokens, decoded, cache,
        cache_before_last, frames_before_last, last_start, last_end,
        last_pre_cache, last_pre_frames) so a subsequent
        `_refresh_last_span_batched` call can replay the most-recently
        appended span with a new latent (PHC writeback).
        """
        n = len(streams)
        results = [None] * n
        if (
            n == 0
            or self._vae is None
            or not hasattr(self._vae, "stream_decode")
        ):
            return results
        if replace_start_toks is None:
            replace_start_toks = [max(0, int(tok) - 2) for tok in end_toks]
        frame_scale_20fps = int(frame_scale_20fps)

        groups = defaultdict(list)
        for idx, state in enumerate(streams):
            if (
                state is None
                or not bool(state.get("valid", False))
                or int(end_toks[idx]) < 1
            ):
                continue
            end_tok = int(end_toks[idx])
            replace_start = int(replace_start_toks[idx])
            initial = state.get("tokens") is None
            if initial:
                start_tok = int(state.get("prefix_tokens", 0))
                if start_tok < 0 or start_tok > end_tok:
                    continue
            else:
                start_tok = int(state["tokens"])
                if end_tok < start_tok:
                    continue
                if end_tok == start_tok:
                    results[idx] = state["decoded"]
                    continue
            if replace_start == start_tok:
                replace_mode = 0
            elif (not initial) and replace_start == start_tok - 1:
                # A changed token behind the installed decoder cache must be
                # replayed from its pre-token snapshot. The causal-span path
                # below is contiguous and should never need this legacy mode;
                # invalidate instead of silently decoding new tokens on top of
                # stale state. The caller will use an exact fresh-prefix decode.
                state["valid"] = False
                continue
            else:
                continue
            groups[(initial, start_tok, end_tok, replace_mode)].append(idx)

        for (initial, start_tok, end_tok, replace_mode), idxs in groups.items():
            try:
                self._decode_token_span_group(
                    streams=streams,
                    feat_fulls=feat_fulls,
                    idxs=idxs,
                    initial=initial,
                    start_tok=start_tok,
                    end_tok=end_tok,
                    replace_mode=replace_mode,
                    replace_start_toks=replace_start_toks,
                    vae_device=vae_device,
                    frame_scale_20fps=frame_scale_20fps,
                    results=results,
                )
            except Exception:
                for idx in idxs:
                    streams[idx]["valid"] = False
                if hasattr(self._vae, "clear_cache"):
                    self._vae.clear_cache()

        return results

    def _decode_token_span_group(
        self,
        *,
        streams,
        feat_fulls,
        idxs,
        initial,
        start_tok,
        end_tok,
        replace_mode,
        replace_start_toks,
        vae_device,
        frame_scale_20fps,
        results,
    ):
        """One VAE pass for streams sharing the same group key.

        Token span is decoded one token at a time so the per-token snapshots
        needed by refresh can be captured.
        """
        n_g = len(idxs)

        # 1. Initial prefix decode (only when state.tokens is None).
        if initial:
            self._vae.clear_cache()
            if start_tok > 0:
                lat_prefix = torch.stack([
                    feat_fulls[idx][:, :start_tok, 0, 0]
                    .transpose(0, 1)
                    .float()
                    .to(vae_device)
                    for idx in idxs
                ], dim=0)
                denorm_prefix = self._denorm_latent_batch(lat_prefix)
                with torch.no_grad(), self._vae_autocast(vae_device):
                    decoded_prefix = self._vae.stream_decode(
                        denorm_prefix, first_chunk=True,
                    ).float()
                expected = self._mei138_frames_for_decode_tokens(
                    start_tok, frame_scale_20fps=frame_scale_20fps,
                )
                if int(decoded_prefix.shape[1]) != int(expected):
                    raise RuntimeError(
                        f"prefix decode produced "
                        f"{int(decoded_prefix.shape[1])} frames, "
                        f"expected {int(expected)}"
                    )
                prefix_cache = self._cache_snapshot_split(n_g, "prefix cache")
            else:
                decoded_prefix = None
                empty_cache = self._cache_snapshot()
                prefix_cache = [self._cache_clone(empty_cache) for _ in idxs]

            for local_idx, idx in enumerate(idxs):
                state = streams[idx]
                decoded = (
                    torch.empty(0, 138, device=vae_device, dtype=torch.float32)
                    if decoded_prefix is None
                    else decoded_prefix[local_idx].detach()
                )
                state["tokens"] = int(start_tok)
                state["decoded"] = decoded
                state["cache"] = prefix_cache[local_idx]
                state["cache_before_last"] = None
                state["frames_before_last"] = 0

        # 2. Install pre-replay VAE cache (once per group).
        self._vae_install_group_cache(
            streams, idxs, "cache", first_chunk=(start_tok == 0),
        )

        # 3. Snapshot last_pre cache/frames per stream (consumed by refresh).
        if replace_mode == 0:
            last_pre_cache = (
                [None] * n_g if start_tok == 0
                else [self._cache_clone(streams[idx]["cache"]) for idx in idxs]
            )
            last_pre_frames = [
                int(streams[idx]["decoded"].shape[0]) for idx in idxs
            ]
        else:  # replace_mode == -1
            last_pre_cache = [
                streams[idx].get("cache_before_last") for idx in idxs
            ]
            last_pre_frames = [
                int(streams[idx].get("frames_before_last", 0)) for idx in idxs
            ]

        # 4. Decode tokens [start_tok, end_tok), one at a time.
        cache_before_last = [None] * n_g
        frames_before_last = [0] * n_g
        current_frames = [int(streams[idx]["decoded"].shape[0]) for idx in idxs]
        tail_chunks = []
        for tok in range(int(start_tok), int(end_tok)):
            if tok == 0:
                pre_cache_for_tok = [None] * n_g
            else:
                pre_cache_for_tok = self._cache_snapshot_split(
                    n_g, "pre-token cache",
                )
            pre_frames_for_tok = list(current_frames)
            lat_tail = torch.stack([
                feat_fulls[idx][:, tok:tok + 1, 0, 0]
                .transpose(0, 1)
                .float()
                .to(vae_device)
                for idx in idxs
            ], dim=0)
            denorm_tail = self._denorm_latent_batch(lat_tail)
            with torch.no_grad(), self._vae_autocast(vae_device):
                decoded_one = self._vae.stream_decode(
                    denorm_tail, first_chunk=(tok == 0),
                ).float()
            expected = self._mei138_frames_for_token_span(
                tok, tok + 1, frame_scale_20fps=frame_scale_20fps,
            )
            if int(decoded_one.shape[1]) != int(expected):
                raise RuntimeError(
                    f"token {tok} decode produced "
                    f"{int(decoded_one.shape[1])} frames, "
                    f"expected {int(expected)}"
                )
            tail_chunks.append(decoded_one.detach())
            cache_before_last = pre_cache_for_tok
            frames_before_last = pre_frames_for_tok
            current_frames = [f + int(expected) for f in current_frames]

        # 5. Write back per-stream state + populate results.
        final_cache = self._cache_snapshot_split(n_g, "final decode cache")
        decoded_tail = (
            torch.cat(tail_chunks, dim=1)
            if len(tail_chunks) > 1 else tail_chunks[0]
        )
        for local_idx, idx in enumerate(idxs):
            state = streams[idx]
            state["decoded"] = torch.cat(
                [state["decoded"], decoded_tail[local_idx].detach()], dim=0,
            )
            state["tokens"] = int(end_tok)
            state["cache"] = final_cache[local_idx]
            state["cache_before_last"] = cache_before_last[local_idx]
            state["frames_before_last"] = frames_before_last[local_idx]
            state["last_start"] = int(replace_start_toks[idx])
            state["last_end"] = int(end_tok)
            state["last_pre_cache"] = last_pre_cache[local_idx]
            state["last_pre_frames"] = last_pre_frames[local_idx]
            results[idx] = state["decoded"]

    def _refresh_last_span_batched(
        self,
        streams,
        corrected_latent,
        feat_fulls=None,
        *,
        vae_device,
        frame_scale_20fps=4,
    ):
        """Replay each stream's most-recently appended span with a new latent.

        Used by interleaved-PHC after `_decode_token_span_batched`: PHC produces
        a corrected version of the latest span's latent; this rolls each stream's
        VAE cache back to the snapshot captured before that span and re-decodes
        with the new latent. Streams sharing (first_chunk, span_len) are batched
        into one VAE call (N=1 groups go through the same code path).
        """
        n = len(streams)
        refreshed = [False] * n
        if (
            corrected_latent is None
            or int(corrected_latent.shape[0]) != n
            or self._vae is None
            or not hasattr(self._vae, "stream_decode")
        ):
            return refreshed
        if feat_fulls is None:
            feat_fulls = [None] * n
        frame_scale_20fps = int(frame_scale_20fps)

        groups = defaultdict(list)
        for idx, state in enumerate(streams):
            if (
                state is None
                or not bool(state.get("valid", False))
                or state.get("last_start") is None
            ):
                continue
            start_tok = int(state["last_start"])
            end_tok = int(state["last_end"])
            if end_tok <= start_tok:
                continue
            if start_tok > 0 and state.get("last_pre_cache") is None:
                continue
            if end_tok - start_tok > 1 and feat_fulls[idx] is None:
                continue
            # Same first_chunk + span_len → same VAE call shape; stream_decode
            # takes one first_chunk flag for the whole batch, and each token's
            # expected frame count only depends on (start_tok==0, span_len).
            groups[(start_tok == 0, end_tok - start_tok)].append(idx)

        for (first_chunk, span_len), idxs in groups.items():
            try:
                self._refresh_last_span_group(
                    streams=streams,
                    idxs=idxs,
                    first_chunk=first_chunk,
                    span_len=span_len,
                    corrected_latent=corrected_latent,
                    feat_fulls=feat_fulls,
                    vae_device=vae_device,
                    frame_scale_20fps=frame_scale_20fps,
                    refreshed=refreshed,
                )
            except Exception:
                for idx in idxs:
                    streams[idx]["valid"] = False
                if hasattr(self._vae, "clear_cache"):
                    self._vae.clear_cache()

        return refreshed

    def _refresh_last_span_group(
        self,
        *,
        streams,
        idxs,
        first_chunk,
        span_len,
        corrected_latent,
        feat_fulls,
        vae_device,
        frame_scale_20fps,
        refreshed,
    ):
        """One refresh pass for streams sharing (first_chunk, span_len)."""
        n_g = len(idxs)

        # 1. Install pre-span VAE cache (= state.last_pre_cache).
        self._vae_install_group_cache(
            streams, idxs, "last_pre_cache", first_chunk=first_chunk,
        )

        starts = [int(streams[idx]["last_start"]) for idx in idxs]
        pre_frames = [int(streams[idx]["last_pre_frames"]) for idx in idxs]
        tail_chunks = []
        cache_before_last = None
        frames_before_last = [0] * n_g

        for off in range(int(span_len)):
            if off < int(corrected_latent.shape[1]):
                # corrected_latent is already in VAE space. Replay one token
                # per stream_decode call; passing the whole N-token block at
                # off=0 breaks the per-token frame/cache contract for N>1.
                denorm_tail = corrected_latent[idxs, off:off + 1].to(vae_device)
            else:
                latents = []
                for idx, start_tok in zip(idxs, starts):
                    tok = start_tok + off
                    lat = (
                        feat_fulls[idx][:, tok:tok + 1, 0, 0]
                        .transpose(0, 1)
                        .float()
                        .to(vae_device)
                    )
                    latents.append(lat)
                denorm_tail = self._denorm_latent_batch(
                    torch.stack(latents, dim=0)
                )

            if off == span_len - 1:
                tok0 = starts[0] + off
                if tok0 == 0:
                    cache_before_last = [None] * n_g
                else:
                    cache_before_last = self._cache_snapshot_split(
                        n_g, "pre-last cache",
                    )
                for local_idx, start_tok in enumerate(starts):
                    prev_frames = 0
                    for prev_off in range(off):
                        prev_frames += self._mei138_frames_for_token_span(
                            start_tok + prev_off,
                            start_tok + prev_off + 1,
                            frame_scale_20fps=frame_scale_20fps,
                        )
                    frames_before_last[local_idx] = (
                        pre_frames[local_idx] + prev_frames
                    )

            with torch.no_grad(), self._vae_autocast(vae_device):
                decoded_one = self._vae.stream_decode(
                    denorm_tail, first_chunk=(first_chunk and off == 0),
                ).float()
            expected = self._mei138_frames_for_token_span(
                starts[0] + off, starts[0] + off + 1,
                frame_scale_20fps=frame_scale_20fps,
            )
            if int(decoded_one.shape[1]) != int(expected):
                raise RuntimeError(
                    f"refresh decode produced {int(decoded_one.shape[1])} "
                    f"frames, expected {int(expected)}"
                )
            tail_chunks.append(decoded_one.detach())

        final_cache = self._cache_snapshot_split(n_g, "refresh final cache")
        decoded_tail = (
            torch.cat(tail_chunks, dim=1)
            if len(tail_chunks) > 1 else tail_chunks[0]
        )
        for local_idx, idx in enumerate(idxs):
            state = streams[idx]
            pf = pre_frames[local_idx]
            state["decoded"] = torch.cat(
                [state["decoded"][:pf], decoded_tail[local_idx].detach()], dim=0,
            )
            state["tokens"] = int(state["last_end"])
            state["cache"] = final_cache[local_idx]
            state["cache_before_last"] = cache_before_last[local_idx]
            state["frames_before_last"] = frames_before_last[local_idx]
            state["last_pre_cache"] = None
            refreshed[idx] = True

    @staticmethod
    def _mei138_heading_com_at(mei, frame_idx, fps=20.0):
        """Recover MEI138 integrated heading and planar CoM at one frame."""
        mei = mei.float()
        T = int(mei.shape[0])
        idx = max(0, min(int(frame_idx), T - 1))
        if T <= 0:
            raise ValueError("cannot recover heading/com from empty MEI138")
        heading = torch.zeros(T, device=mei.device, dtype=mei.dtype)
        if T > 1:
            heading[1:] = torch.cumsum(mei[1:, 2] / float(fps), dim=0)
        heading = torch.remainder(heading + math.pi, 2.0 * math.pi) - math.pi

        vx = mei[:, 0]
        vz = mei[:, 1]
        cy = torch.cos(heading)
        sy = torch.sin(heading)
        vel_x = cy * vx + sy * vz
        vel_z = -sy * vx + cy * vz
        com = torch.zeros(T, 2, device=mei.device, dtype=mei.dtype)
        if T > 1:
            com[1:, 0] = torch.cumsum(vel_x[1:] / float(fps), dim=0)
            com[1:, 1] = torch.cumsum(vel_z[1:] / float(fps), dim=0)
        return heading[idx], com[idx]

    def _ensure_phc_horizon(self, n_replace=None):
        """Match PHC rollout length to the current correction window.

        Latents are 5fps so 1 token = 6 PhysX frames @30fps (CLoSD 200ms
        convention via `set_planning_horizon`). `n_replace` defaults to
        `self.resample_len` (training replay region); inference passes the
        chosen-driven window so PhysX horizon scales to the actual window.
        Cached per `n_replace` so identical-shape callers don't pay re-set cost.
        """
        if self._phc_backend is None:
            return
        if n_replace is None:
            n_replace = self.resample_len
        n_replace = int(n_replace)
        if self._phc_horizon_mode == n_replace:
            return
        self._phc_backend.set_planning_horizon(n_replace)
        horizon_30fps = self._phc_backend.planning_horizon_30fps
        if (
            getattr(self._phc_backend, "_episode_buf", None) is None
            or self._phc_backend._episode_buf.shape[1] < horizon_30fps + 64
        ):
            self._phc_backend.init_episode_buf(horizon_30fps + 64)
        self._phc_horizon_mode = n_replace

    # ------------------------------------------------------------------
    # PHC backend setup (one-shot, costly — ~30s cold init)
    # ------------------------------------------------------------------

    def _init_phc(self, vae, phc_config):
        """Boot PHCIsaacLabBackend (mei138 path).

        Only `phc_isaaclab` is supported after the IsaacGym → IsaacLab
        migration. sources=["phc"] / ["model","phc"] always route through
        `_phc_integrator_path` (mei138-only).
        """
        backend_flavor = phc_config.get("backend", "phc_isaaclab")
        if backend_flavor != "phc_isaaclab":
            raise ValueError(
                f"phc backend={backend_flavor!r} not supported; legacy "
                "IsaacGym backends were removed during the IsaacLab migration. "
                "Set phc.backend='phc_isaaclab' (default)."
            )
        from physics.phc_isaaclab_backend import PHCIsaacLabBackend

        self._phc_continue_ref_vel_blend_frames = int(
            phc_config.get("continue_ref_vel_blend_frames", 0)
        )
        if self._phc_continue_ref_vel_blend_frames < 0:
            raise ValueError(
                "phc.continue_ref_vel_blend_frames must be non-negative"
            )

        # Device string: respect explicit override, else cuda:device_id. Under
        # DDP the train_ldf.py top-level LOCAL_RANK clamp pins each rank to
        # cuda:0, so the per-rank device string is always "cuda:0" in practice.
        _dev_id = phc_config.get("device_id", 0)
        _dev_str = phc_config.get("device", f"cuda:{_dev_id}")
        self._phc_backend = PHCIsaacLabBackend(
            num_envs=phc_config.get("num_envs", 1),
            device=_dev_str,
            motion_file=phc_config.get("isaaclab_motion_file",
                "sample_data/synthetic_swing_squat.pkl"),
            num_motions=phc_config.get("num_motions", 1),
            policy_path=phc_config.get("isaaclab_policy_path",
                "physics/closd/output/CLoSD/CLoSD_no_finetune/Humanoid.pth"),
            hml_mean_path=phc_config["hml_mean_path"],
            hml_std_path=phc_config["hml_std_path"],
        )
        # PHC's replace window matches model's: last self.resample_len tokens.
        # Backend's planning_horizon is set once at boot (latent_per_chunk * 6
        # frames @30fps); larger resample_len → longer rollout per call.
        # Episode buffer below is sized to match.
        self._phc_backend.set_planning_horizon(self.resample_len)
        horizon_30fps = self._phc_backend.planning_horizon_30fps

        self._vae = vae

        # Episode buffer holds horizon_30fps + slack so PHC can step the full
        # horizon plus a few extra frames without a wrap-around.
        self._phc_backend.init_episode_buf(horizon_30fps + 64)
        self._phc_backend.reset()

    # ------------------------------------------------------------------
    # stats helpers (for wandb logging from the outer forward())
    # ------------------------------------------------------------------

    def reset_stats(self):
        self._log_total = 0
        self._log_triggered = 0
        self._log_resample_t_sum = 0.0
        self._log_phc_fail = 0
        self._log_phc_latency_ms_sum = 0.0
        self._log_phc_profile = {k: 0.0 for k in _PHC_PROFILE_KEYS}

    def _profile_add(self, key, value):
        if not hasattr(self, "_log_phc_profile"):
            self._log_phc_profile = {k: 0.0 for k in _PHC_PROFILE_KEYS}
        self._log_phc_profile[key] = self._log_phc_profile.get(key, 0.0) + float(value)

    def _profile_add_phc_call(self, n_chosen, decode_ms=0.0,
                              anchor_target_ms=0.0, refs_build_ms=0.0,
                              reset_ms=0.0, action_ms=0.0, sim_step_ms=0.0,
                              body_to_mei_ms=0.0, encode_ms=0.0,
                              writeback_ms=0.0, total_ms=0.0):
        self._profile_add("calls", 1)
        self._profile_add("samples", int(n_chosen))
        self._profile_add("decode_ms", decode_ms)
        self._profile_add("anchor_target_ms", anchor_target_ms)
        self._profile_add("refs_build_ms", refs_build_ms)
        self._profile_add("reset_ms", reset_ms)
        self._profile_add("action_ms", action_ms)
        self._profile_add("sim_step_ms", sim_step_ms)
        self._profile_add("body_to_mei_ms", body_to_mei_ms)
        self._profile_add("encode_ms", encode_ms)
        self._profile_add("writeback_ms", writeback_ms)
        self._profile_add("total_ms", total_ms)

    def _temp_zeros(self, name, shape, device, dtype):
        key = (name, torch.device(device), dtype)
        buf = self._phc_temp_buffers.get(key)
        if buf is None or tuple(buf.shape) != tuple(shape):
            buf = torch.empty(shape, device=device, dtype=dtype)
            self._phc_temp_buffers[key] = buf
        return buf.zero_()

    def reset_inference_state(self):
        """Reset per-stream inference state before a new generation run.

        Clears the streaming encoder cache so the first PHC writeback of the
        run bootstraps from the decoded context instead of stale state, marks
        the next correction as the stream's first (decoded-anchor seed), and
        arms the continuous PHC-corrected trajectory cache.
        """
        self._infer_enc_state = self._new_phc_enc_state()
        self._infer_phc_started = False
        self._phc_traj = {}
        self._phc_mei_boundary = {}
        self._infer_decode_state = None
        self._phc_prefix_decode_cache.clear()
        if self._vae is not None and hasattr(self._vae, "clear_cache"):
            self._vae.clear_cache()

    def reset_decode_state(self):
        """Drop the streaming decode cache so the next correction re-bootstraps.

        Called by the inference driver right after a buffer rollback: rollback
        shifts token indices, desyncing the cache's `tokens` counter, so we
        force a fresh prefix decode on the (shifted) buffer next window. The
        re-bootstrap is lossless for the window we use (causal receptive field
        is filled long before the window start)."""
        self._infer_decode_state = None
        self._phc_prefix_decode_cache.clear()
        if self._vae is not None and hasattr(self._vae, "clear_cache"):
            self._vae.clear_cache()

    def get_corrected_trajectory(self, i=0):
        """Concatenated PHC-corrected mei138 `(T, 138)` for stream `i`, or None.

        The continuous physical-truth trajectory (proposal-2 cache): exact PHC
        output stitched across windows, never re-decoded. Use for deployment /
        visualization instead of decoding the latent buffer.
        """
        if not self._phc_traj or int(i) not in self._phc_traj:
            return None
        return torch.cat(self._phc_traj[int(i)], dim=0)

    def enable_dbg_infer(self, trace_dir, window_limit=1):
        """Enable debug trace saving for upcoming inference PHC corrections."""
        if not trace_dir:
            self._dbg_infer = None
            return
        self._dbg_infer = {
            "dir": str(trace_dir),
            "window_limit": (None if window_limit is None
                             else max(0, int(window_limit))),
            "counter": 0,
            "records": [],
        }
        os.makedirs(self._dbg_infer["dir"], exist_ok=True)

    def pop_dbg_infer_records(self):
        if self._dbg_infer is None:
            return []
        records = list(self._dbg_infer["records"])
        self._dbg_infer["records"] = []
        return records

    def _dbg_infer_enabled(self):
        if self._dbg_infer is None:
            return False
        limit = self._dbg_infer["window_limit"]
        if limit is None:
            return True
        return int(self._dbg_infer["counter"]) < int(limit)

    def _maybe_log_phc_profile(self, label, n_chosen, horizon_30fps,
                                t_decode_ms, t_reset_ms, t_action_ms,
                                t_step_ms, interval=50):
        """Increment PHC rollout counter and emit a one-line profile log every
        `interval`-th call. Shared by 263 and 138 integrator paths so the
        cadence stays consistent across representations.
        """
        if not hasattr(self, "_phc_profile_count"):
            self._phc_profile_count = 0
        self._phc_profile_count += 1
        if self._phc_profile_count % int(interval) != 1:
            return
        h = max(int(horizon_30fps), 1)
        print(
            f"[{label}] n_chosen={n_chosen} horizon={horizon_30fps} "
            f"vae_decode={t_decode_ms:.0f}ms reset={t_reset_ms:.0f}ms "
            f"action_total={t_action_ms:.0f}ms ({t_action_ms/h:.1f}ms/step) "
            f"step_total={t_step_ms:.0f}ms ({t_step_ms/h:.1f}ms/step)",
            flush=True,
        )

    @staticmethod
    def _compute_dbg_infer_stats(capture):
        """Compute per-window stats from a finalized PHC capture dict.

        Reads `phc_post_reset_body_pos / phc_refs / phc_rollout_body_pos`
        (when present) and returns up to 2 metric groups:
        - `ref1_reset_body_{mean,max,root}_m`
            (refs[1] vs reset_pose body / root delta)
        - `raw_step_jump_body_{mean,max}_m` +
          `raw_vs_ref_body_{mean,max}_m` +
          `raw_root_z_{min,max}_m`
            (rollout consecutive-frame jumps, rollout vs refs tracking error,
             root z range)
        """
        stats = {}
        reset_pose = capture.get("phc_post_reset_body_pos")
        refs = capture.get("phc_refs")
        raw_episode = capture.get("phc_rollout_body_pos")
        if reset_pose is not None and refs is not None:
            delta = (refs[:, 1] - reset_pose).norm(dim=-1)
            stats["ref1_reset_body_mean_m"] = float(delta.mean().item())
            stats["ref1_reset_body_max_m"] = float(delta.max().item())
            stats["ref1_reset_root_m"] = float(delta[:, 0].mean().item())
        if raw_episode is not None and refs is not None:
            n_cmp = min(raw_episode.shape[1], refs.shape[1])
            if n_cmp > 1:
                jump = (
                    raw_episode[:, 1:n_cmp] - raw_episode[:, :n_cmp - 1]
                ).norm(dim=-1)
                err = (
                    raw_episode[:, :n_cmp] - refs[:, :n_cmp]
                ).norm(dim=-1)
                stats["raw_step_jump_body_mean_m"] = float(jump.mean().item())
                stats["raw_step_jump_body_max_m"] = float(jump.max().item())
                stats["raw_vs_ref_body_mean_m"] = float(err.mean().item())
                stats["raw_vs_ref_body_max_m"] = float(err.max().item())
                stats["raw_root_z_min_m"] = float(
                    raw_episode[..., 2].min().item())
                stats["raw_root_z_max_m"] = float(
                    raw_episode[:, :, 0, 2].max().item())
        return stats

    def _save_dbg_infer_window(self, capture):
        """Save one PHC window's capture dict as `arrays.npz` + `stats.json`.

        Array keys are saved if present and convertible (torch tensor / array);
        `phc_init_state` dict is flattened into `phc_init_state_<field>` npz keys.
        """
        if self._dbg_infer is None:
            return
        out_dir = os.path.join(
            self._dbg_infer["dir"], f"window_{int(capture['window_idx']):03d}",
        )
        os.makedirs(out_dir, exist_ok=True)

        def _to_np(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().float().cpu().numpy()
            return np.asarray(x, dtype=np.float32)

        arrays = {}
        for key in (
            "mei138_pre_phc_decoded",
            "phc_refs", "phc_ref_body_vel", "phc_live_body_vel",
            "phc_post_reset_body_pos",
            "phc_rollout_body_pos", "phc_rollout_body_rot",
            "mei138_phc_rollout", "mei138_post_phc_decoded",
        ):
            arr = _to_np(capture.get(key))
            if arr is not None:
                arrays[key] = arr
        for field, tensor in (capture.get("phc_init_state") or {}).items():
            arr = _to_np(tensor)
            if arr is not None:
                arrays[f"phc_init_state_{field}"] = arr
        arrays_path = os.path.join(out_dir, "arrays.npz")
        np.savez_compressed(arrays_path, **arrays)

        stats = dict(capture.get("stats", {}))
        stats.update({
            "window_idx": int(capture["window_idx"]),
            "chosen_index": int(capture.get("chosen_index", 0)),
            "rs_start": int(capture.get("rs_start", -1)),
            "ctx_end": int(capture.get("ctx_end", -1)),
            "end_index": int(capture.get("end_index",
                                           capture.get("ctx_end", -1))),
            "n_replace": int(capture.get("n_replace", -1)),
            "frame_scale_20fps": int(capture.get("frame_scale_20fps", -1)),
            "representation": str(capture.get("representation", "mei138")),
            "anchor_source": str(capture.get("anchor_source", "")),
            "decode_frame_lo": int(capture.get("decode_frame_lo", -1)),
            "decode_frame_hi": int(capture.get("decode_frame_hi", -1)),
            "decode_end_tok": int(capture.get("decode_end_tok", -1)),
        })
        stats_path = os.path.join(out_dir, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)

        self._dbg_infer["records"].append({
            "window_idx": stats["window_idx"],
            "dir": out_dir,
            "arrays": arrays_path,
            "stats": stats_path,
            "rs_start": stats["rs_start"],
            "ctx_end": stats["ctx_end"],
        })

    # ------------------------------------------------------------------
    # PHC debug capture: single source of truth across train / infer modes.
    # `_begin_phc_capture` builds the per-window dict (or None when neither
    # debug mode is active); all PHC pipeline phases write into it; the
    # `_phc_integrator_path` tail calls `_end_phc_capture` which dispatches
    # to the appropriate consumer (infer trace save / training _dbg_train).
    # ------------------------------------------------------------------

    def _begin_phc_capture(self, chosen, n_replace, frame_scale_20fps):
        """Allocate a per-window capture dict when either dbg mode is active.

        Both consumers see the SAME fields filled — capture is consumer-agnostic.
        Returns None when neither train nor infer dbg is enabled.
        """
        train_on = self._dbg_train is not None
        infer_on = self._dbg_infer_enabled()
        if not (train_on or infer_on):
            return None
        capture = {
            "_chosen": chosen,                       # popped at _end
            "rs_start": int(chosen[0]["rs_start"]),
            "ctx_end": int(chosen[0]["ctx_end"]),
            # end_index: train-side defaults from sub_chosen; infer-side
            # apply_inference_correction has no transition zone notion, so
            # fall back to ctx_end (no extra range past the PHC commit point).
            "end_index": int(chosen[0].get("end_index", chosen[0]["ctx_end"])),
            "n_replace": int(n_replace),
            "frame_scale_20fps": int(frame_scale_20fps),
            "representation": "mei138",
            "anchor_source": str(self.phc_anchor_source),
            "decode_frame_lo": int(self._mei138_frames_for_decode_tokens(
                chosen[0]["rs_start"], frame_scale_20fps=frame_scale_20fps,
            )),
            "decode_frame_hi": int(self._mei138_frames_for_decode_tokens(
                chosen[0]["ctx_end"], frame_scale_20fps=frame_scale_20fps,
            )),
            "decode_end_tok": int(chosen[0]["ctx_end"]),
        }
        if infer_on:
            capture["window_idx"] = int(self._dbg_infer["counter"])
            capture["chosen_index"] = int(chosen[0].get("i", 0))
            self._dbg_infer["counter"] += 1
        return capture

    def _end_phc_capture(self, capture):
        """Dispatch a finalized capture dict to enabled consumers.

        - infer: hand to `_save_dbg_infer_window` (npz + stats.json)
        - train: slice each capture tensor to per-sample numpy and append
                 to `self._dbg_train["windows"]` (one entry per chosen sample
                 per PHC integrator call). Consumer (`render_phc_chain`)
                 groups by `sample_i` to build one chain video per sample.
        """
        if capture is None:
            return
        chosen = capture.pop("_chosen", None)

        if self._dbg_infer is not None and capture.get("window_idx") is not None:
            capture["stats"] = self._compute_dbg_infer_stats(capture)
            self._save_dbg_infer_window(capture)

        if self._dbg_train is not None and chosen is not None:
            def _np(t, k):
                return t[k].detach().cpu().numpy() if t is not None else None
            phc_init_state = capture.get("phc_init_state")
            refs = capture.get("phc_refs")
            ref_body_vel = capture.get("phc_ref_body_vel")
            live_body_vel = capture.get("phc_live_body_vel")
            reset_pose = capture.get("phc_post_reset_body_pos")
            rollout_pos = capture.get("phc_rollout_body_pos")
            rollout_rot = capture.get("phc_rollout_body_rot")
            mei_pre = capture.get("mei138_pre_phc_decoded")
            mei_rollout = capture.get("mei138_phc_rollout")
            mei_post = capture.get("mei138_post_phc_decoded")
            windows = self._dbg_train.setdefault("windows", [])
            for k, c in enumerate(chosen):
                windows.append({
                    "sample_i": int(c["i"]),
                    "rs_start": int(c["rs_start"]),
                    "ctx_end": int(c["ctx_end"]),
                    "end_index": int(c.get("end_index", c["ctx_end"])),
                    "n_replace": int(c.get("n_replace", 1)),
                    "phc_init_state": (
                        {field: _np(t, k) for field, t in phc_init_state.items()}
                        if phc_init_state is not None else None
                    ),
                    "phc_refs": _np(refs, k),
                    "phc_ref_body_vel": _np(ref_body_vel, k),
                    "phc_live_body_vel": _np(live_body_vel, k),
                    "phc_post_reset_body_pos": _np(reset_pose, k),
                    "phc_rollout_body_pos": _np(rollout_pos, k),
                    "phc_rollout_body_rot": _np(rollout_rot, k),
                    "mei138_pre_phc_decoded": _np(mei_pre, k),
                    "mei138_phc_rollout": _np(mei_rollout, k),
                    "mei138_post_phc_decoded": _np(mei_post, k),
                })

    def get_stats(self):
        rate = self._log_triggered / max(self._log_total, 1)
        avg_t = self._log_resample_t_sum / max(self._log_triggered, 1)
        stats = {"triggered_rate": float(rate), "avg_resample_t": float(avg_t)}
        if "phc" in self.sources:
            # phc_fail_rate = PHC-stage failures / triggered samples. For
            # ["model","phc"] a failure means the tail kept the naive model
            # output; for ["phc"] a failure means the tail stayed as clean GT.
            stats["phc_fail_rate"] = float(
                self._log_phc_fail / max(self._log_triggered, 1)
            )
            stats["phc_avg_latency_ms"] = float(
                self._log_phc_latency_ms_sum / max(self._log_triggered, 1)
            )
            prof = getattr(self, "_log_phc_profile", {})
            calls = max(float(prof.get("calls", 0.0)), 1.0)
            samples = max(float(prof.get("samples", 0.0)), 1.0)
            stats["phc_call_count"] = float(prof.get("calls", 0.0))
            stats["phc_sample_count"] = float(prof.get("samples", 0.0))
            stats["phc_total_ms"] = float(prof.get("total_ms", 0.0))
            stats["phc_decode_ms"] = float(prof.get("decode_ms", 0.0))
            stats["phc_anchor_target_ms"] = float(
                prof.get("anchor_target_ms", 0.0)
            )
            stats["phc_refs_build_ms"] = float(prof.get("refs_build_ms", 0.0))
            stats["phc_encode_ms"] = float(prof.get("encode_ms", 0.0))
            stats["phc_body_to_mei_ms"] = float(prof.get("body_to_mei_ms", 0.0))
            stats["phc_action_ms"] = float(prof.get("action_ms", 0.0))
            stats["phc_sim_step_ms"] = float(prof.get("sim_step_ms", 0.0))
            stats["phc_reset_ms"] = float(prof.get("reset_ms", 0.0))
            stats["phc_writeback_ms"] = float(prof.get("writeback_ms", 0.0))
            stats["phc_total_ms_per_call"] = float(
                prof.get("total_ms", 0.0) / calls
            )
            stats["phc_total_ms_per_sample"] = float(
                prof.get("total_ms", 0.0) / samples
            )
        return stats

    # ------------------------------------------------------------------
    # Sampling + adaptive M helpers (Phase 1 streaming API).
    # ------------------------------------------------------------------

    def _sample_resample_t(self, batch_size, device):
        """Logit-normal shifted sampling for per-rollout GT-clamp threshold.

        Returns Tensor of shape (batch_size,) with values in [0, 1). rt is
        compared against alpha_next inside the streaming Euler loop; it never
        feeds the model's time embedding directly (alpha_current does).
        Exact alignment permits a fractional final Euler step.
        """
        z = torch.randn(batch_size, device=device)
        ts = torch.sigmoid(z)
        s = self.shift
        rt = s * ts / (1 + (s - 1) * ts)  # (B,)
        return rt

    # ------------------------------------------------------------------
    # main entry point, called from DiffForcingWanModel.forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def resample_batch(
        self,
        feature,
        context_end,
        output_end,
        time_steps,
        text_context,
        model,
        resampling_ratio: float,
    ):
        """Modify `feature` list in place; return the same list for convenience.

        Args:
            feature:         List of per-sample tensors, each shape (C, T_i, H, W).
                             Clean `feature[i][:, :output_end[i]]` is the region
                             this pass may overwrite (clean prefix + transition
                             zone; transition zone gets partial denoise + no PHC
                             because α<1 there).
            context_end:     List[int]. Per-sample index where the clean prefix
                             ends (= `output_start_index[i]` in the main forward).
                             Transition zone starts here.
            output_end:      List[int]. Per-sample index where the transition
                             zone ends (= `output_end_index[i]`). Rollout
                             extends spatially to here; loss is computed in
                             [context_end, output_end).
            time_steps:      List[Tensor scalar]. Per-sample t value from the
                             scheduler. Used as the temporal stopping criterion
                             for Euler (so transition zone reaches its schedule
                             alpha, not full commit).
            text_context:    List from `text_module.get_context`. Either
                             List[Tensor(L, dim)] (single-text samples) or
                             List[List[Tensor]] (per-frame multi-segment).
            model:           WanModel instance (= `DiffForcingWanModel.model`).
            resampling_ratio: per-sample Bernoulli probability of being touched.
        """
        self._phc_prefix_decode_cache = {}
        if self._dbg_train is not None:
            self._dbg_train["latent_pre_resample"] = [f.detach().clone() for f in feature]

        batch_size = len(feature)
        device = feature[0].device
        # latent_mean / latent_std must be set by callers before resample_batch:
        # train forward (diffusion_forcing_wan.py) and inference orchestrator
        # (generate_long_sequences.py) both call _set_latent_norm_from_model
        # explicitly. _phc_integrator_path asserts presence and fail-fasts
        # otherwise.
        # Feature is always a VAE-encoded 5fps latent (1 token = 4 frames
        # @20fps = 6 frames @30fps); 4× temporal upsample is WanVAE-specific
        # and hardcoded inside `_phc_integrator_path` / `_resample_with_optional_phc`.
        if "phc" in self.sources:
            self._ensure_phc_horizon()

        # Logit-normal shifted sampling
        resample_ts = self._sample_resample_t(batch_size, device)

        # Per-sample Bernoulli gate (one Tensor pull instead of N rand calls).
        gates = torch.rand(batch_size, device=device) < resampling_ratio
        gates_cpu = gates.detach().cpu().tolist()
        resample_ts_cpu = resample_ts.detach().cpu().tolist()

        # Pass 1: collect samples for model rollout. PHC fires inside the
        # Euler loop (interleaved per-pl commit), so we don't need a separate
        # phc_chosen list here. Long-sequence inference takes the
        # apply_inference_correction entry point and builds its own chosen
        # list, also bypassing this pass.
        model_chosen = []  # list of (i, ctx_end, end_index, resample_t, t_per_sample)
        for i in range(batch_size):
            self._log_total += 1
            ctx_end = int(context_end[i])
            if ctx_end <= 0 or not bool(gates_cpu[i]):
                continue
            resample_t = float(resample_ts_cpu[i])
            if "model" in self.sources:
                model_chosen.append((
                    i, ctx_end, int(output_end[i]),
                    resample_t, float(time_steps[i].item()),
                ))
            self._log_triggered += 1
            self._log_resample_t_sum += resample_t

        # Per-sample indices actually rolled out this step. The forward's
        # self-forcing loss reads this to decide which samples use xt=feature
        # directly (resampled) vs standard noised-GT (e.g. ctx_end<=0 samples).
        self._last_resampled_idx = {c[0] for c in model_chosen}

        # Pass 2: batched model rollout (+ interleaved PHC when "phc" in
        # sources). Streaming dynamic GT-clamp singlepass: alpha_init=0 全位置,
        # 每步 Euler 更新后对 alpha<rt 的 token 用 noised(GT, alpha) 钳制;
        # sources=["model","phc"] 时 Euler loop 内每个 commit (alpha 0→1) 立刻
        # fire n_replace=1 PHC 修正, 不再有 "model 跑完 → PHC" 的先后关系。
        if model_chosen and "model" in self.sources:
            self._resample_with_optional_phc(
                feature, model_chosen, text_context, model, device,
            )

        # Snapshot the final post-resample feature for debug viz, when
        # requested. In interleaved-PHC mode this IS the after-PHC state too —
        # model rollout and PHC writebacks are merged inside the streaming
        # Euler loop, so there's no separable "pre-PHC" snapshot point.
        if self._dbg_train is not None:
            self._dbg_train["latent_post_resample"] = [
                f.detach().clone() for f in feature
            ]
            # Per-sample (rs_start, ctx_end, end_index) for samples that
            # actually went through model rollout. Always populated; viz
            # orchestrator uses it as a fallback when PHC windows are empty
            # (e.g. no-PHC mode) to render a 2-column before/after diff.
            self._dbg_train["resample_ranges"] = [
                {"sample_i": int(t[0]),
                 "rs_start": max(0, int(t[1]) - self.resample_len),
                 "ctx_end": int(t[1]),
                 "end_index": int(t[2])}
                for t in model_chosen
            ]

        return feature

    # ------------------------------------------------------------------
    # sources=["model"] / ["model","phc"] 共用入口 — streaming dynamic
    # GT-clamp singlepass + 可选 interleaved PHC.
    #   - model only: alpha_init=0 全位置纯噪声起步, Euler 推进每步对
    #     alpha<rt 的 token 用 noised(GT, alpha) 钳制, 否则保留 Euler 输出
    #   - +phc:       Euler loop 内 alpha 0→1 commit 触发 n_replace=1 PHC,
    #                 写回 feature[abs_pos] 和 st['x'][pl], 然后继续下一
    #                 Euler step (graceful degrade on silent PHC fail)
    #   - rt 为 per-rollout logit-normal 标量, 决定 "GT 教师强制" 跨度
    # ------------------------------------------------------------------

    def _resample_with_optional_phc(
        self, feature, chosen, text_context, model, device,
    ):
        """Dynamic GT-clamp singlepass streaming over [rs_start, end_index).

        Spatial extent: rollout covers clean prefix [rs_start, ctx_end) and
        transition zone [ctx_end, end_index). Both use the same RT-GT-clamp
        + Euler math.

        Temporal stop: legacy_ceil preserves the historical global-time ceil.
        exact converts the endpoint to replay-local coordinates and shortens
        the final Euler update to match the sampled training time. The loss
        and training-time sampling are unchanged.

        Semantics:
          - alpha_init = 0 全位置 (纯噪声起步, x = eps), 跟 inference 起点严格对齐
          - rt per-rollout 标量 (logit-normal sample), 用作 GT-clamp 阈值
          - 每个 Euler step k 算完 x_{k+1} = x_k + v_pred * delta 后, 对每个
            token 位置 pos 检查 alpha_next(pos):
              · alpha_next < rt → 用 noised(GT, alpha_next) 覆盖该位置
                  (a * GT + (1-a) * fresh_eps, RF 线性 schedule)
              · alpha_next >= rt → 保留 Euler 输出
            钳制后 token 仍参与下一 step Euler 更新, alpha 继续推进
          - dt = 1/scheduler.steps batch-shared, 跟 inference stream_generate_step
            一致.

        Interleaved PHC (sources=["model","phc"], always on):
          每个 Euler step 末尾检测 alpha 刚 0→1 的位置, 按 pl 升序 fire
          n_replace=1 PHC, 写回 feature[abs_pos] + st['x'][pl] (graceful degrade
          on silent PHC fail). 因为全样本 t0=0 + dt 共享, alpha 在所有 sample
          同 pl 同 step 一致, 一次 PHC call batch=bs.

        `content_len` (time_scheduler.content_len) bounds the model input
        window to the last `content_len` tokens, matching inference
        `init_generated`. Must satisfy content_len >= resample_len + cs.
        """
        if not chosen:
            return
        if self.prediction_type != "vel":
            raise NotImplementedError(
                f"streaming mode currently only supports prediction_type='vel'; "
                f"got {self.prediction_type!r}"
            )
        cs = self._cs
        content_len = self.time_scheduler.content_len  # may be None
        # PHC is always interleaved into the Euler loop when "phc" is a source.
        interleaved_phc = "phc" in self.sources
        # Each rollout (this batch) re-seeds PHC: the first fire resets to the
        # decoded anchor, later fires continue from the live pose.
        self._phc_batch_started = False
        self._phc_mei_boundary = {}

        # Inference-aligned dt: TriangularTimeScheduler.get_time_steps advances
        # t by 1/scheduler.steps per call. Sharing dt across the batch keeps
        # alpha on the same discrete grid as inference, and (since alpha_init=0
        # uniformly) every sample's alpha at a given (step, pl) is identical
        # → interleaved-PHC group-by-pl collapses to one PHC call per pl.
        dt = 1.0 / float(self.time_scheduler.steps)

        states = []
        for (i, ctx_end, end_index, resample_t, t_per_sample) in chosen:
            rs_start = max(0, ctx_end - self.resample_len)
            # L spans clean prefix + transition zone; Euler stops at t_per_sample
            # so transition zone positions (pl >= ctx_end-rs_start) stay at α<1
            # and never commit / never fire PHC.
            L = end_index - rs_start
            rt = float(resample_t)
            t_end_local = t_per_sample - rs_start / float(cs)
            stop_time = (t_end_local if self.time_alignment == "exact"
                         else t_per_sample)
            M_total = int(math.ceil(stop_time / dt))

            if content_len is not None:
                seq_lo = max(0, end_index - content_len)
                assert seq_lo <= rs_start, (
                    f"content_len={content_len} truncates below replay start: "
                    f"end_index={end_index} rs_start={rs_start} seq_lo={seq_lo}; "
                    f"content_len must be >= resample_len + chunk_size = "
                    f"{self.resample_len + cs}"
                )
            else:
                seq_lo = 0

            dtype = feature[i].dtype
            pos = torch.arange(L, device=device, dtype=dtype)
            GT_clean = feature[i][:, rs_start:end_index, ...].clone()
            x = torch.randn_like(GT_clean)  # alpha_init=0 → x = pure noise
            alpha_current = torch.zeros(L, device=device, dtype=dtype)

            single_text = text_context[i]
            if isinstance(single_text, (list, tuple)):
                single_text = list(single_text[seq_lo:end_index])

            st = {
                'i': i, 'rs_start': rs_start, 'ctx_end': ctx_end,
                'end_index': end_index, 'L': L,
                'seq_lo': seq_lo, 'rt': rt, 'M_total': M_total,
                't_end_local': t_end_local,
                'pos': pos, 'alpha_current': alpha_current,
                'x': x, 'GT_clean': GT_clean,
                'text': single_text,
            }
            if interleaved_phc:
                # alpha_init=0 means no latent starts committed. Commit timing is
                # deterministic from (step, cs), so keep it as a Python index to
                # avoid per-step GPU->CPU sync from nonzero().tolist().
                st['committed_upto'] = -1
                st['phc_decode_state'] = self._new_phc_decode_state(
                    sample_idx=i, prefix_tokens=max(1, rs_start),
                )
                st['phc_enc_state'] = self._new_phc_enc_state()
            states.append(st)

        M_max = max(st['M_total'] for st in states)

        for step in range(M_max):
            active = [
                (k_idx, st) for k_idx, st in enumerate(states)
                if step < st['M_total']
            ]
            if not active:
                break
            seq_len = max(st['end_index'] - st['seq_lo'] for _, st in active)
            # Build batched model input: clean prefix [seq_lo, rs_start) +
            # replay region [rs_start, end_index) (clean prefix + transition
            # zone) with per-pos schedule. Transition zone positions follow the
            # same schedule as the rest of the replay region (alpha_current
            # covers all L positions).
            xt_fulls, schedule_inputs, text_ctxs = [], [], []
            for _, st in active:
                seq_lo = st['seq_lo']
                seq_T = st['end_index'] - seq_lo
                schedule_full = torch.ones(seq_T, device=device,
                                           dtype=feature[st['i']].dtype)
                schedule_full[st['rs_start'] - seq_lo:] = st['alpha_current']
                xt_full = feature[st['i']][:, seq_lo:st['end_index'], ...].clone()
                xt_full[:, st['rs_start'] - seq_lo:, ...] = st['x']
                xt_fulls.append(xt_full)
                schedule_inputs.append(schedule_full * self.time_embedding_scale)
                text_ctxs.append(st['text'])

            predicted = model(xt_fulls, schedule_inputs, text_ctxs, seq_len, y=None)

            new_commits = defaultdict(list) if interleaved_phc else None
            for pred_idx, (k_idx, st) in enumerate(active):
                t_step = dt * (step + 1)  # t0=0 全样本
                if self.time_alignment == "exact":
                    t_step = min(t_step, st['t_end_local'])
                alpha_next = (t_step - st['pos'] / cs).clamp(0.0, 1.0)

                # Euler update on replay region.
                seq_lo = st['seq_lo']
                v_tail = predicted[pred_idx][:, st['rs_start'] - seq_lo:, ...]
                delta = (alpha_next - st['alpha_current'])[None, :, None, None]
                x_euler = st['x'] + v_tail * delta

                # GT-clamp: alpha_next < rt 的位置覆盖为 noised(GT, alpha_next)
                # = alpha * GT + (1 - alpha) * fresh_eps (RF 线性 schedule;
                # 与 time_scheduler.add_noise 内核数学等价). 用 fresh_eps
                # 每 step 重采, 跟训练分布一致 (logit-normal rt 已涵盖各 noise
                # level, 这里只是把该位置拉回 alpha 对应的 noise manifold).
                frontier_mask = alpha_next < st['rt']  # (L,) bool
                if self.clamp_mode == "frontier":
                    clamp_mask = frontier_mask
                elif self.clamp_mode == "random":
                    # Keep the exact number of clamps produced by the
                    # triangular frontier, but place them uniformly at random.
                    # This isolates the value of the contiguous triangular
                    # structure from the amount of GT information supplied.
                    n_clamped = int(frontier_mask.sum().item())
                    clamp_mask = torch.zeros_like(frontier_mask)
                    if n_clamped > 0:
                        random_idx = torch.randperm(
                            int(st['L']), device=device
                        )[:n_clamped]
                        clamp_mask[random_idx] = True
                else:  # clamp_mode == "none": complete self-rollout
                    clamp_mask = torch.zeros_like(frontier_mask)

                if bool(clamp_mask.any()):
                    eps = torch.randn_like(st['GT_clean'])
                    a = alpha_next[None, :, None, None]
                    x_clamped = a * st['GT_clean'] + (1 - a) * eps
                    m = clamp_mask[None, :, None, None]
                    st['x'] = torch.where(m, x_clamped, x_euler)
                else:
                    st['x'] = x_euler
                st['alpha_current'] = alpha_next

                # Interleaved PHC: fire on alpha 0 → 1 commit transitions.
                # (alpha < rt 时 token 仍在 GT 流形上, 不需要 PHC 修正)
                if interleaved_phc:
                    commit_upto = int(math.floor(
                        float(cs) * (t_step - (1.0 - 1e-6))
                    ))
                    commit_upto = min(int(st['L']) - 1, commit_upto)
                    prev_upto = int(st['committed_upto'])
                    for pl in range(max(prev_upto + 1, 0), commit_upto + 1):
                        new_commits[pl].append(k_idx)
                    st['committed_upto'] = max(prev_upto, commit_upto)

            # Fire pl-by-pl in ascending order so each pl's vae.decode anchor
            # (per-sample feature[i][:rs_start+pl+1]) consumes the previous
            # pl's PHC writeback. Same logic as the legacy interleaved-PHC.
            if interleaved_phc and new_commits:
                for pl in sorted(new_commits):
                    ids = new_commits[pl]
                    # (1) Sync st['x'] → feature[abs_pos] so vae.decode sees
                    #     the freshly committed latent (not the stale GT).
                    for k in ids:
                        st = states[k]
                        abs_pos = st['rs_start'] + pl
                        feature[st['i']][:, abs_pos:abs_pos + 1, ...] = (
                            st['x'][:, pl:pl + 1, ...]
                        )
                    # Causal Wan VAE token 0 produces the single seed frame;
                    # it has no four-frame PHC span and must remain immutable.
                    phc_ids = [
                        k for k in ids
                        if states[k]['rs_start'] + pl >= 1
                    ]
                    if not phc_ids:
                        continue
                    # (2) Build sub_chosen and fire n_replace=1 PHC.
                    sub_chosen = [{
                        'i': states[k]['i'],
                        'rs_start': states[k]['rs_start'] + pl,
                        'ctx_end': states[k]['rs_start'] + pl + 1,
                        'end_index': states[k]['end_index'],
                        'n_replace': 1,
                        'decode_state': states[k].get('phc_decode_state'),
                        'enc_state': states[k].get('phc_enc_state'),
                        'is_first': not self._phc_batch_started,
                    } for k in phc_ids]
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        self._phc_integrator_path(feature, sub_chosen)
                    # After the first fire of this rollout the controller is
                    # placed; every later window continues from its live pose.
                    self._phc_batch_started = True
                    # (3) Pull PHC-corrected feature[abs_pos] back into st['x']
                    #     (graceful degrade if PHC silently failed).
                    for k in phc_ids:
                        st = states[k]
                        abs_pos = st['rs_start'] + pl
                        st['x'][:, pl:pl + 1, ...] = (
                            feature[st['i']][:, abs_pos:abs_pos + 1, ...]
                        )

        for st in states:
            feature[st['i']][:, st['rs_start']:st['end_index'], ...] = st['x']

    # ------------------------------------------------------------------
    # GT-RSI helpers — anchor integrator output at GT motion pose
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _finite_diff_body_vel_out(refs, out, fps=30.0):
        """In-place variant of _finite_diff_body_vel using a reusable buffer."""
        out.zero_()
        T = refs.shape[1]
        if T < 2:
            return out
        dt = 1.0 / fps
        out[:, 1:-1] = (refs[:, 2:] - refs[:, :-2]) / (2 * dt)
        out[:, 0] = (refs[:, 1] - refs[:, 0]) / dt
        out[:, -1] = (refs[:, -1] - refs[:, -2]) / dt
        return out

    @staticmethod
    def _prepend_mei_anchor_frame(seq):
        """Add a zero-velocity anchor frame before a MEI138 replacement stream."""
        if seq.shape[0] <= 0:
            raise ValueError("cannot prepend anchor to an empty MEI138 sequence")
        anchor = seq[:1].clone()
        anchor[..., 0:3] = 0.0
        return torch.cat([anchor, seq], dim=0)

    @staticmethod
    def _pad_or_crop_time(seq, length):
        length = int(length)
        if seq.shape[0] == length:
            return seq
        if seq.shape[0] > length:
            return seq[:length]
        if seq.shape[0] <= 0:
            raise ValueError("cannot pad an empty sequence")
        pad = seq[-1:].expand(length - seq.shape[0], *seq.shape[1:])
        return torch.cat([seq, pad], dim=0)

    @staticmethod
    def _mei138_frames_for_decode_tokens(num_tokens, frame_scale_20fps=4):
        """Decoded MEI138 frames produced by this causal Wan VAE prefix decode.

        The first latent token produces one frame; every later token contributes
        `frame_scale_20fps` frames. This matches `WanVAE_.decode()` /
        `stream_decode(first_chunk=True)` and is why the legacy path decoded
        through `ctx_end + 1` before slicing `rs_start * 4`.
        """
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            return 0
        frame_scale_20fps = max(1, int(frame_scale_20fps))
        return 1 + (num_tokens - 1) * frame_scale_20fps

    @staticmethod
    def _mei138_frames_for_token_span(start_tok, end_tok, frame_scale_20fps=4):
        start_tok = max(0, int(start_tok))
        end_tok = max(start_tok, int(end_tok))
        if end_tok <= start_tok:
            return 0
        frame_scale_20fps = max(1, int(frame_scale_20fps))
        if start_tok == 0:
            return 1 + (end_tok - 1) * frame_scale_20fps
        return (end_tok - start_tok) * frame_scale_20fps

    @staticmethod
    def _mei138_decode_tokens_for_frames(required_frames, total_tokens,
                                         frame_scale_20fps=4):
        """Minimum latent tokens needed for a MEI138 frame horizon."""
        required_frames = max(1, int(required_frames))
        frame_scale_20fps = max(1, int(frame_scale_20fps))
        total_tokens = max(1, int(total_tokens))
        need_tokens = 1 + int(math.ceil(
            max(0, required_frames - 1) / frame_scale_20fps
        ))
        return max(1, min(need_tokens, total_tokens))

    @staticmethod
    def _decode_length_bucket(length):
        """Coarse power-of-two bucket for PHC VAE fallback decoding."""
        length = max(1, int(length))
        if length <= 16:
            return 16
        if length <= 32:
            return 32
        if length <= 64:
            return 64
        if length <= 128:
            return 128
        return int(2 ** math.ceil(math.log2(length)))

    def _build_mei_seq_from_full(self, full, f_lo, f_hi, need_frames):
        """Slice + prepend-anchor + pad/crop one sample's MEI138 sequence.

        Used by `_phc_integrator_path`; returns (seq, anchor_idx). Anchor
        frame is the **last frame before the replacement window** (`f_lo - 1`),
        so seq covers `[0, f_hi)`. The f_lo == 0 boundary case prepends a
        zero-velocity anchor at index 0 instead.
        """
        if f_lo == 0:
            seq = self._prepend_mei_anchor_frame(full[:f_hi])
            anchor_idx = 0
        else:
            seq = full[:f_hi]
            anchor_idx = f_lo - 1
        seq = self._pad_or_crop_time(seq, int(anchor_idx) + need_frames + 1)
        return seq, anchor_idx

    @staticmethod
    def _align_refs_to_pose_inplace(refs, target_pose, n_chosen, start_frame=1):
        """SE(2)+height align refs[start_frame:] to target_pose in z-up MUJOCO.

        Continue-mode (`_phc_integrator_path`, non-first window) re-anchors the
        decoded targets to the controller's live body pose with start_frame=0,
        so the window starts exactly where tracking ended without changing the
        relative target motion.
        """
        if n_chosen <= 0 or refs.shape[1] <= start_frame:
            return
        src = refs[:n_chosen, start_frame].clone()
        tgt = target_pose[:n_chosen].to(device=refs.device, dtype=refs.dtype)
        src_root_xy = src[:, 0, :2]
        tgt_root_xy = tgt[:, 0, :2]
        src_dir = src[:, 1, :2] - src_root_xy
        tgt_dir = tgt[:, 1, :2] - tgt_root_xy
        src_yaw = torch.atan2(src_dir[:, 1], src_dir[:, 0])
        tgt_yaw = torch.atan2(tgt_dir[:, 1], tgt_dir[:, 0])
        delta = tgt_yaw - src_yaw
        cy = torch.cos(delta).view(n_chosen, 1, 1)
        sy = torch.sin(delta).view(n_chosen, 1, 1)

        chunk = refs[:n_chosen, start_frame:]
        chunk[..., :2] = chunk[..., :2] - src_root_xy[:, None, None, :]
        xp = chunk[..., 0].clone()
        yp = chunk[..., 1].clone()
        chunk[..., 0] = cy * xp - sy * yp
        chunk[..., 1] = sy * xp + cy * yp
        chunk[..., :2] = chunk[..., :2] + tgt_root_xy[:, None, None, :]
        chunk[..., 2] = chunk[..., 2] + (
            tgt[:, 0, 2] - src[:, 0, 2]
        ).view(n_chosen, 1, 1)
        refs[:n_chosen, start_frame:] = chunk
        refs[:n_chosen, 0] = tgt
        refs[:n_chosen, start_frame] = tgt

    def _phc_compute_anchor_and_targets(
        self, mei_raw, anchor_idx_t,
        initial_heading_t, initial_com_t,
        horizon_30fps, sk_tree, task_device, backend, capture,
    ):
        """Compute PHC init state + `refs_chunk` and install init state.

        Writes the full init state dict into `capture["phc_init_state"]` when
        capture is enabled. Returns `refs_chunk`.
        """
        from physics.motion_utils import mei_138_to_phc_anchor_and_targets
        init_state, refs_chunk = mei_138_to_phc_anchor_and_targets(
            mei_raw,
            anchor_indices=anchor_idx_t,
            horizon_30fps=horizon_30fps,
            fps=20.0,
            initial_heading=initial_heading_t,
            initial_com=initial_com_t,
            device=task_device,
            skeleton_tree=sk_tree,
        )
        # Diagnostic env-var hooks for tracking-failure debugging:
        #   PHYSFLOOD_PHC_ZERO_DOF_VEL=1     → zero per-joint dof_vel
        #   PHYSFLOOD_PHC_ZERO_ROOT_VEL=1    → also zero root_vel/ang_vel
        # Helps isolate whether init-state velocity noise (from VAE roundtrip
        # + R1·R0ᵀ finite-diff) drives the first-frame body jump.
        if os.environ.get("PHYSFLOOD_PHC_ZERO_DOF_VEL") == "1":
            init_state["dof_vel"] = torch.zeros_like(init_state["dof_vel"])
        if os.environ.get("PHYSFLOOD_PHC_ZERO_ROOT_VEL") == "1":
            init_state["root_vel"] = torch.zeros_like(init_state["root_vel"])
            init_state["root_ang_vel"] = torch.zeros_like(init_state["root_ang_vel"])
        backend.set_init_state_override(init_state)
        if hasattr(backend, 'set_init_motion_time'):
            backend.set_init_motion_time(0.0)
        if capture is not None:
            capture["phc_init_state"] = init_state
        return refs_chunk

    def _compute_ref_vels(self, refs_full, vels_full, n_chosen, T_full):
        """30fps finite-diff body velocities for the ref callback, in place.

        Used both when first building the targets and when re-anchoring them to
        the live pose in continue mode (after alignment the vels are stale)."""
        self._finite_diff_body_vel_out(refs_full, vels_full, fps=30.0)
        if T_full >= 3:
            dt_30 = 1.0 / 30.0
            vels_full[:n_chosen, 1] = (
                refs_full[:n_chosen, 2] - refs_full[:n_chosen, 1]
            ) / dt_30
            vels_full[:n_chosen, 0] = vels_full[:n_chosen, 1]

    def _blend_continue_ref_vels(
        self, vels_full, live_body_vel, n_chosen, T_full,
    ):
        """Blend a continued window's first target velocities from live state.

        The PHC callback first consumes frame 1. With M=3 this installs
        [live, 0.5*live+0.5*reference, reference] at frames 1..3, removing the
        deterministic velocity reset while returning to the decoded target in
        100 ms. Positions remain the aligned physical targets.
        """
        m = min(
            int(self._phc_continue_ref_vel_blend_frames),
            max(0, int(T_full) - 1),
        )
        if m <= 0 or n_chosen <= 0:
            return
        live = live_body_vel[:n_chosen].to(
            device=vels_full.device, dtype=vels_full.dtype,
        )
        vels_full[:n_chosen, 0] = live
        for frame_idx in range(1, m + 1):
            alpha = 1.0 if m == 1 else float(frame_idx - 1) / float(m - 1)
            vels_full[:n_chosen, frame_idx] = torch.lerp(
                live, vels_full[:n_chosen, frame_idx], alpha,
            )

    def _phc_setup_ref_callback(
        self, refs_chunk, num_envs, n_chosen, T_full,
        task_device, backend, capture,
    ):
        """Build `refs_full` / `vels_full` from `refs_chunk` and register the
        backend's absolute-ref callback. Returns `(refs_full, vels_full)`; the
        continue-mode branch re-anchors them in place to the live pose."""
        refs_full = self._temp_zeros(
            "mei138_refs_full",
            (num_envs, T_full, 24, 3),
            device=task_device,
            dtype=refs_chunk.dtype,
        )
        refs_full[:n_chosen] = refs_chunk
        if n_chosen < num_envs:
            refs_full[n_chosen:] = refs_full[n_chosen - 1:n_chosen]
        vels_full = self._temp_zeros(
            "mei138_vels_full",
            refs_full.shape,
            device=task_device,
            dtype=refs_full.dtype,
        )
        self._compute_ref_vels(refs_full, vels_full, n_chosen, T_full)

        env_ids_full = torch.arange(num_envs, device=task_device)

        def _cb(env):
            f = (env.episode_length_buf + 1).clamp(max=T_full - 1).long()
            return {
                'rg_pos': refs_full[env_ids_full, f, :, :],
                'body_vel': vels_full[env_ids_full, f, :, :],
            }
        backend.set_absolute_ref_callback(_cb)
        return refs_full, vels_full

    def _phc_decode_full_mei_batch(
        self, feature, chosen, vae_device,
        need_frames, frame_scale_20fps,
    ):
        """Decode the full MEI138 sequence for each chosen sample.

        Stateful batched `_decode_token_span_batched` (training) +
        prefix-cache fallback (`_try_cached_mei138_prefix_decode`) +
        bucketed `vae.decode` fallback for samples missed by the first two.
        Inference passes no decode_state, so it goes straight to the fallbacks.

        Returns `(full_decs, decode_ms)` where `full_decs[k]` is a
        `(T_motion, 138)` tensor for chosen sample `k`.
        """
        t0 = time.time()
        n_chosen = len(chosen)
        full_decs = [None] * n_chosen
        fallback_by_bucket = defaultdict(list)
        decode_specs = []
        for k, c in enumerate(chosen):
            feat_full = feature[c['i']]
            f_lo = self._mei138_frames_for_decode_tokens(
                c['rs_start'], frame_scale_20fps=frame_scale_20fps,
            )
            f_hi = self._mei138_frames_for_decode_tokens(
                c['ctx_end'], frame_scale_20fps=frame_scale_20fps,
            )
            if f_hi - f_lo != int(need_frames):
                raise RuntimeError(
                    "PHC window must exclude causal seed token 0: "
                    f"tokens [{c['rs_start']}, {c['ctx_end']}) map to "
                    f"{f_hi - f_lo} frames, expected {need_frames}"
                )
            # Decode through the committed boundary only. The legacy formula
            # decoded ctx_end+1 and made three of four target frames depend on
            # an uncommitted future token.
            decode_end_tok = int(c['ctx_end'])
            decode_specs.append((feat_full, c, decode_end_tok))

        cached_list = self._decode_token_span_batched(
            [spec[1].get('decode_state') for spec in decode_specs],
            [spec[0] for spec in decode_specs],
            [spec[2] for spec in decode_specs],
            [spec[1]['rs_start'] for spec in decode_specs],
            vae_device=vae_device,
            frame_scale_20fps=frame_scale_20fps,
        )
        for k, (feat_full, c, decode_end_tok) in enumerate(decode_specs):
            cached = cached_list[k]
            if cached is None:
                # Final safety net: stateful path skipped this sample
                # (state invalid / disabled / mismatched replace_mode).
                cached = self._try_cached_mei138_prefix_decode(
                    feat_full,
                    sample_idx=c['i'],
                    rs_start_tok=c['rs_start'],
                    decode_end_tok=decode_end_tok,
                    vae_device=vae_device,
                )
            if cached is not None:
                full_decs[k] = cached
            else:
                bucket = self._decode_length_bucket(decode_end_tok)
                fallback_by_bucket[bucket].append((k, c, decode_end_tok))

        for bucket in sorted(fallback_by_bucket):
            fallback = fallback_by_bucket[bucket]
            latents = []
            valid_token_lengths = []
            for _k, c, decode_end_tok in fallback:
                feat_full = feature[c['i']]
                lat = (
                    feat_full[:, :decode_end_tok, 0, 0]
                    .transpose(0, 1)
                    .float()
                )
                latents.append(lat.to(vae_device))
                valid_token_lengths.append(int(lat.shape[0]))
            if not latents:
                continue
            max_T = max(lat.shape[0] for lat in latents)
            z_dim = latents[0].shape[1]
            batched_lat = torch.zeros(
                len(latents), max_T, z_dim,
                device=vae_device, dtype=latents[0].dtype,
            )
            for j, lat in enumerate(latents):
                batched_lat[j, :lat.shape[0]] = lat
            batched_lat = self._denorm_latent_batch(batched_lat)

            with torch.no_grad(), self._vae_autocast(vae_device):
                full_dec_batch = self._vae.decode(batched_lat)

            for j, (k, _c, _decode_end_tok) in enumerate(fallback):
                valid_frames = min(
                    int(full_dec_batch.shape[1]),
                    max(1, self._mei138_frames_for_decode_tokens(
                        valid_token_lengths[j],
                        frame_scale_20fps=frame_scale_20fps,
                    )),
                )
                full_decs[k] = full_dec_batch[j, :valid_frames].float()
        return full_decs, (time.time() - t0) * 1000

    def _phc_run_step_loop(self, backend, horizon_30fps):
        """Run PhysX action+step loop for `horizon_30fps` steps.

        Returns `(action_ms_total, step_ms_total)` on success, or `None` on
        action-NaN / env-done failure (with the failure cause already printed).
        Caller should treat `None` as fail and bail with `_fail() + return`.
        """
        action_ms = 0.0
        step_ms = 0.0
        for _step in range(horizon_30fps):
            ta = time.time()
            action = backend.get_action()
            action_ms += (time.time() - ta) * 1000
            if not torch.isfinite(action).all():
                print(f"[integrator138] FAIL: action NaN at step {_step}",
                      flush=True)
                return None
            ts = time.time()
            done, _ = backend.step(action)
            step_ms += (time.time() - ts) * 1000
            if done:
                print(f"[integrator138] FAIL: env done at step {_step}",
                      flush=True)
                return None
        return action_ms, step_ms

    @torch.no_grad()
    def _encode_corrected_tail(self, tracked_mei, chosen, full_decs, vae_device,
                               frame_scale_20fps=4):
        """Re-encode each PHC-corrected mei window as a streaming continuation.

        Non-streaming `vae.encode` of a `4*n_replace`-frame window keeps only
        the seed frame and drops the rest, so a fall developing mid-window
        never enters the latent. Here each window is encoded with
        `stream_encode(first_chunk=False)` continuing from a per-stream encoder
        cache, so all frames contribute. The cache (`enc_state`) is reused
        across contiguous corrections; when it cannot be reused (first call /
        gap / buffer rollback) a bounded prefix from the decoded context warms
        it. Returns VAE-space latents `(n_chosen, n_replace, z_dim)` — same
        space as `vae.encode`.
        """
        boot_frames = 1 + frame_scale_20fps * _ENC_BOOTSTRAP_TOKENS
        outs = []
        with torch.no_grad():
            for k, c in enumerate(chosen):
                es = c.get("enc_state")
                rs_start, ctx_end = int(c["rs_start"]), int(c["ctx_end"])
                win = tracked_mei[k:k + 1]
                reuse = (
                    es is not None
                    and es.get("tokens") == rs_start
                    and es.get("cache") is not None
                )
                with self._vae_autocast(vae_device):
                    if reuse:
                        self._cache_restore(
                            es["cache"], field="_enc_feat_map",
                            idx_field="_enc_conv_idx")
                        mu = self._vae.stream_encode(
                            win, first_chunk=False, keep_cache=True)
                    else:
                        rs_f = min(
                            self._mei138_frames_for_decode_tokens(
                                rs_start, frame_scale_20fps=frame_scale_20fps),
                            int(full_decs[k].shape[0]),
                        )
                        self._vae.clear_cache()
                        if rs_f > 0:
                            prefix = full_decs[k][max(0, rs_f - boot_frames):rs_f]
                            self._vae.stream_encode(
                                prefix.unsqueeze(0).to(vae_device),
                                first_chunk=True, keep_cache=True)
                            mu = self._vae.stream_encode(
                                win, first_chunk=False, keep_cache=True)
                        else:
                            mu = self._vae.stream_encode(
                                win, first_chunk=True, keep_cache=True)
                if es is not None:
                    es["cache"] = self._cache_snapshot(field="_enc_feat_map")
                    es["tokens"] = ctx_end
                outs.append(mu[:, -int(c["n_replace"]):])
        return torch.cat(outs, dim=0)

    def _phc_integrator_path(self, feature, chosen):
        """PHC correction path for MEI138 decoded/latent features."""
        if not chosen:
            return
        if self._latent_mean is None or self._latent_std is None:
            raise RuntimeError(
                "PHC integrator (mei138) needs model latent mean/std before "
                "correction. Training: diffusion_forcing_wan._step calls "
                "_set_latent_norm_from_model when 'phc' is in resample_source. "
                "Inference: build_phc_corrector calls it after construction."
            )
        backend = self._phc_backend
        num_envs = backend.num_envs
        if len(chosen) > num_envs:
            for cs in range(0, len(chosen), num_envs):
                self._phc_integrator_path(
                    feature, chosen[cs:cs + num_envs]
                )
            return

        n_chosen = len(chosen)
        # See _phc_integrator_path docstring: caller fills chosen[i]['n_replace']
        # (train: self.resample_len; infer: self.inference_window_size).
        n_replace = int(chosen[0]['n_replace'])
        # First window of a stream seeds PHC from the decoded anchor (reset);
        # later windows continue from the live PhysX state (set per-stream by
        # the train/infer caller). Synchronized commits keep a batched call
        # homogeneous, so reading chosen[0] is enough (mirrors n_replace).
        is_first = bool(chosen[0].get('is_first', True))
        # Feature is always VAE-encoded 5fps latent → 1 token = 4 mei138 frames.
        frame_scale_20fps = 4
        self._ensure_phc_horizon(n_replace=n_replace)
        t0 = time.time()

        def _fail():
            self._log_phc_fail += n_chosen
            self._log_phc_latency_ms_sum += (time.time() - t0) * 1000 * n_chosen

        try:
            need_frames = n_replace * frame_scale_20fps
            horizon_30fps = backend.planning_horizon_30fps
            T_full = horizon_30fps + 1
            task_device = backend.task.device
            mei_seq_list = []
            anchor_idx_list = []
            initial_heading_values = None
            initial_com_values = None

            vae_device = self._vae.mean.device
            full_decs, _t_decode_ms = self._phc_decode_full_mei_batch(
                feature, chosen, vae_device,
                need_frames, frame_scale_20fps,
            )

            # Window-local heading/CoM recovery is only safe when the streaming
            # decode cache covered the window (training); inference has no
            # decode_state so this stays False and falls back to the global
            # anchor path below.
            use_window_local = all(
                c.get('decode_state') is not None
                and bool(c['decode_state'].get('valid', False))
                and c['decode_state'].get('tokens') is not None
                and int(c['decode_state'].get('tokens')) >= int(c['ctx_end'])
                for c in chosen
            )
            if use_window_local:
                initial_heading_values = []
                initial_com_values = []

            for k, c in enumerate(chosen):
                f_lo = self._mei138_frames_for_decode_tokens(
                    c['rs_start'], frame_scale_20fps=frame_scale_20fps,
                )
                f_hi = self._mei138_frames_for_decode_tokens(
                    c['ctx_end'], frame_scale_20fps=frame_scale_20fps,
                )
                if f_hi - f_lo != need_frames:
                    raise RuntimeError(
                        f"invalid causal PHC frame span [{f_lo}, {f_hi}) "
                        f"for n_replace={n_replace}"
                    )
                full = self._pad_or_crop_time(full_decs[k], max(1, f_hi))
                if use_window_local and f_lo > 0:
                    win_anchor_idx = f_lo - 1
                    heading, com = self._mei138_heading_com_at(
                        full, win_anchor_idx, fps=20.0,
                    )
                    initial_heading_values.append(heading)
                    initial_com_values.append(com)
                    seq = self._pad_or_crop_time(
                        full[win_anchor_idx:f_hi], need_frames + 1,
                    )
                    anchor_idx = 0
                else:
                    if use_window_local:
                        initial_heading_values.append(
                            torch.zeros((), device=full.device, dtype=full.dtype)
                        )
                        initial_com_values.append(
                            torch.zeros(2, device=full.device, dtype=full.dtype)
                        )
                    seq, anchor_idx = self._build_mei_seq_from_full(
                        full, f_lo, f_hi, need_frames)
                mei_seq_list.append(seq)
                anchor_idx_list.append(anchor_idx)

            max_mei_T = max(int(seq.shape[0]) for seq in mei_seq_list)
            mei_raw = torch.stack([
                self._pad_or_crop_time(seq, max_mei_T)
                for seq in mei_seq_list
            ], dim=0)
            if not torch.isfinite(mei_raw).all():
                print("[integrator138] FAIL: mei_raw NaN after feature decode",
                      flush=True)
                _fail()
                return
            anchor_idx_t = torch.as_tensor(
                anchor_idx_list, device=mei_raw.device, dtype=torch.long,
            )
            initial_heading_t = (
                torch.stack(initial_heading_values).to(
                    device=mei_raw.device, dtype=torch.float32,
                )
                if initial_heading_values is not None else None
            )
            initial_com_t = (
                torch.stack(initial_com_values).to(
                    device=mei_raw.device, dtype=torch.float32,
                )
                if initial_com_values is not None else None
            )

            capture = self._begin_phc_capture(
                chosen, n_replace, frame_scale_20fps)
            if capture is not None:
                capture["mei138_pre_phc_decoded"] = mei_raw[:n_chosen].detach().clone()

            sk_tree = getattr(backend.task, '_motion_lib_sk_tree', None)
            _t_anchor_target0 = time.time()
            refs_chunk = self._phc_compute_anchor_and_targets(
                mei_raw, anchor_idx_t,
                initial_heading_t, initial_com_t,
                horizon_30fps, sk_tree, task_device, backend, capture,
            )
            if refs_chunk is None:
                _fail()
                return
            _t_anchor_target_ms = (time.time() - _t_anchor_target0) * 1000

            _t_refs_build0 = time.time()
            refs_full, vels_full = self._phc_setup_ref_callback(
                refs_chunk, num_envs, n_chosen, T_full,
                task_device, backend, capture,
            )
            _t_refs_build_ms = (time.time() - _t_refs_build0) * 1000

            # The callback still falls back to motion_lib for motion_bodies /
            # body metadata. In generated-init mode all envs can safely
            # reference the backend's default sample motion.
            backend._env._motion_id[:num_envs] = 0

            _t_reset0 = time.time()
            if is_first:
                # First window: place the controller at the decoded anchor
                # (one-time VAE-roundtrip seed).
                backend.reset()
            else:
                # Continue from the live pose: re-anchor the new targets to
                # where tracking actually ended (SE(2)+height align refs[0] to
                # the controller's current body pose), refresh their velocities,
                # then restart the ref timeline without teleporting. The
                # previous window's tracked end pose IS the anchor — no decoded
                # anchor, no first-frame velocity jump.
                live_body_pos = backend._rigid_body_pos[:n_chosen].detach().clone()
                live_body_vel = backend._rigid_body_vel[:n_chosen].detach().clone()
                self._align_refs_to_pose_inplace(
                    refs_full, live_body_pos, n_chosen, start_frame=0)
                self._compute_ref_vels(refs_full, vels_full, n_chosen, T_full)
                self._blend_continue_ref_vels(
                    vels_full, live_body_vel, n_chosen, T_full,
                )
                backend.continue_episode()
            _t_reset_ms = (time.time() - _t_reset0) * 1000

            if capture is not None:
                # Save the references exactly as consumed by the controller:
                # continue-mode alignment and velocity blending are complete.
                capture["phc_refs"] = (
                    refs_full[:n_chosen].detach().clone()
                )
                capture["phc_ref_body_vel"] = (
                    vels_full[:n_chosen].detach().clone()
                )
                capture["phc_live_body_vel"] = (
                    backend._rigid_body_vel[:n_chosen].detach().clone()
                )
                capture["phc_post_reset_body_pos"] = (
                    backend._episode_buf[:n_chosen, 0].detach().clone()
                )

            step_result = self._phc_run_step_loop(backend, horizon_30fps)
            if step_result is None:
                _fail()
                return
            _t_action_ms, _t_step_ms = step_result

            self._maybe_log_phc_profile(
                "PHC138 profile", n_chosen, horizon_30fps,
                _t_decode_ms, _t_reset_ms, _t_action_ms, _t_step_ms,
            )

            if capture is not None:
                horizon_30 = backend.planning_horizon_30fps
                end = backend._frame_idx
                start = max(0, end - (horizon_30 + 1))
                capture["phc_rollout_body_pos"] = (
                    backend._episode_buf[:n_chosen, start:end].detach().clone()
                )
                if getattr(backend, "_episode_rot_buf", None) is not None:
                    capture["phc_rollout_body_rot"] = (
                        backend._episode_rot_buf[:n_chosen, start:end].detach().clone()
                    )

            body_pos_30, body_rot_30 = backend.get_last_horizon_body_state()
            if body_pos_30.shape[1] <= 0 or body_rot_30.shape[1] <= 0:
                print("[integrator138] FAIL: empty PHC body-state horizon",
                      flush=True)
                _fail()
                return
            from physics.motion_utils import phc_body_state_to_mei_138
            _t_body_to_mei0 = time.time()
            prev_boundary = None
            if not is_first:
                previous = [
                    self._phc_mei_boundary.get(int(c['i'])) for c in chosen
                ]
                if any(state is None for state in previous):
                    raise RuntimeError(
                        "continued PHC window is missing its MEI boundary state"
                    )
                prev_boundary = {
                    "body_rot_20": torch.stack([
                        state["body_rot_20"] for state in previous
                    ], dim=0),
                    "joints_20": torch.stack([
                        state["joints_20"] for state in previous
                    ], dim=0),
                }
            tracked_mei, _, next_boundary = phc_body_state_to_mei_138(
                body_rot_30[:n_chosen],
                body_pos_30[:n_chosen],
                getattr(backend.task, '_motion_lib_sk_tree', None),
                prev_boundary=prev_boundary,
                return_boundary=True,
            )
            _t_body_to_mei_ms = (time.time() - _t_body_to_mei0) * 1000
            if tracked_mei.shape[1] < need_frames:
                pad = tracked_mei[:, -1:].expand(
                    tracked_mei.shape[0], need_frames - tracked_mei.shape[1], 138)
                tracked_mei = torch.cat([tracked_mei, pad], dim=1)
            elif tracked_mei.shape[1] > need_frames:
                tracked_mei = tracked_mei[:, :need_frames]
            if not torch.isfinite(tracked_mei).all():
                print("[integrator138] FAIL: tracked_mei has NaN", flush=True)
                _fail()
                return

            if capture is not None:
                capture["mei138_phc_rollout"] = tracked_mei[:n_chosen].detach().clone()

            tracked_mei_gpu = tracked_mei.to(self._vae.mean.device)
            _t_encode0 = time.time()
            corrected_latent = self._encode_corrected_tail(
                tracked_mei_gpu, chosen, full_decs, vae_device,
                frame_scale_20fps=frame_scale_20fps,
            )
            _t_encode_ms = (time.time() - _t_encode0) * 1000
            if (
                corrected_latent.shape[1] != n_replace
                or not torch.isfinite(corrected_latent).all()
            ):
                print(
                    "[integrator138] FAIL: corrected_latent shape="
                    f"{tuple(corrected_latent.shape)} expected "
                    f"(*, {n_replace}, *)",
                    flush=True,
                )
                _fail()
                return
            corrected_latent_vae = corrected_latent
            if capture is not None:
                # Decode the unnormalized latent back to mei138 to capture what
                # the integrator handed PHC, before model-latent norm.
                with torch.no_grad(), self._vae_autocast(self._vae.mean.device):
                    writeback_decoded = self._vae.decode(corrected_latent)
                capture["mei138_post_phc_decoded"] = (
                    writeback_decoded[:n_chosen].detach().clone()
                )
            # Match model latent normalization (see _phc_integrator_path).
            _m = self._latent_mean.to(
                device=corrected_latent.device, dtype=corrected_latent.dtype
            ).reshape(1, 1, -1)
            _s = self._latent_std.to(
                device=corrected_latent.device, dtype=corrected_latent.dtype
            ).reshape(1, 1, -1)
            corrected_latent = (corrected_latent - _m) / _s

            self._end_phc_capture(capture)

            latency_per_sample_ms = (time.time() - t0) * 1000 / max(n_chosen, 1)
            _t_writeback0 = time.time()
            target0 = feature[chosen[0]['i']]
            corrected_all = corrected_latent.to(
                device=target0.device, dtype=target0.dtype,
            )
            corrected_4d_all = (
                corrected_all
                .transpose(1, 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            for k, c in enumerate(chosen):
                feature[c['i']][:, c['rs_start']:c['ctx_end'], ...] = (
                    corrected_4d_all[k]
                )
                self._log_phc_latency_ms_sum += latency_per_sample_ms
                stream_i = int(c['i'])
                self._phc_mei_boundary[stream_i] = {
                    "body_rot_20": next_boundary["body_rot_20"][k].clone(),
                    "joints_20": next_boundary["joints_20"][k].clone(),
                }
                # Proposal-2 cache: exact PHC physical output, committed only
                # after its corrected latent has been validated and written.
                if self._phc_traj is not None:
                    self._phc_traj.setdefault(stream_i, []).append(
                        tracked_mei[k].detach().cpu()
                    )
            # n_replace is always 1 (one-latent-at-a-time correction), so the
            # most-recently appended decode span always needs a refresh with the
            # corrected latent. No-op in inference (decode_state is None there).
            refresh_states = [c.get('decode_state') for c in chosen]
            refresh_feat_fulls = [feature[c['i']] for c in chosen]
            self._refresh_last_span_batched(
                refresh_states,
                corrected_latent_vae,
                refresh_feat_fulls,
                vae_device=self._vae.mean.device,
                frame_scale_20fps=frame_scale_20fps,
            )
            _t_writeback_ms = (time.time() - _t_writeback0) * 1000
            self._profile_add_phc_call(
                n_chosen,
                decode_ms=_t_decode_ms,
                anchor_target_ms=_t_anchor_target_ms,
                refs_build_ms=_t_refs_build_ms,
                reset_ms=_t_reset_ms,
                action_ms=_t_action_ms,
                sim_step_ms=_t_step_ms,
                body_to_mei_ms=_t_body_to_mei_ms,
                encode_ms=_t_encode_ms,
                writeback_ms=_t_writeback_ms,
                total_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            import traceback
            print(
                f"[resampler._phc_integrator_path] FAILED on "
                f"{n_chosen} samples: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            _fail()

    # ------------------------------------------------------------------
    # PHC source: GT-RSI batched path. Each rollout: per-env RSI to (motion,
    # time) drawn from the dataset crop the sample came from, integrator-
    # anchored ref window from VAE.decode, single PhysX rollout for all envs
    # in parallel. Splits into chunks of phc.num_envs if batch is larger.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Test-time PHC correction (inference)
    # ------------------------------------------------------------------

    def apply_inference_correction(
        self,
        model_generated_0,
        ctx_end: int,
        min_rs_start: int = 0,
        n_replace: int = None,
    ) -> bool:
        """Apply PHC correction to model.generated[0] in-place during inference.

        model_generated_0 is model.generated[0]: shape (C, T, 1, 1), same
        format and normalization as training's feature[i]. Delegates directly
        to _phc_integrator_path so train and inference use the same code path.

        Args:
            model_generated_0: model.generated[0] tensor (C, buf_len, 1, 1).
            ctx_end: model.current_commit — index of last committed token.
            min_rs_start: earliest token index PHC may overwrite. Used by
                default-stance-prefixed long generation to keep the fixed prefix
                immutable. Causal seed token 0 is always immutable as well.
            n_replace: optional per-call window override. Full inference blocks
                use ``inference_window_size``; rollback/tail partial blocks pass
                their actual positive length here.

        Returns True if correction was applied, False if ctx_end < window
        (not enough context yet; caller should skip silently).
        """
        # Test-time correction granularity is independent of self.resample_len
        # (the training replay region). A per-call override is used only for
        # rollback/tail partial blocks.
        n_replace = (
            self.inference_window_size
            if n_replace is None else int(n_replace)
        )
        if n_replace <= 0:
            raise ValueError(f"n_replace must be positive (got {n_replace})")
        rs_start = ctx_end - n_replace
        min_rs_start = max(1, int(min_rs_start))
        if rs_start < min_rs_start:
            return False
        # Lazily create the streaming decode cache with prefix_tokens=rs_start
        # so the initial bootstrap's replace_mode matches (replace_start==
        # start_tok). After this, tokens==rs_start each window (commit advances
        # by 1), so decoding continues incrementally — only the new token.
        # PHYSFLOOD_NO_STREAM_DECODE=1 disables it (one-shot decode = old path),
        # for A/B timing the streaming-decode speedup.
        use_stream_dec = os.environ.get("PHYSFLOOD_NO_STREAM_DECODE") != "1"
        if use_stream_dec and self._infer_decode_state is None:
            self._infer_decode_state = self._new_phc_decode_state(
                sample_idx=0, prefix_tokens=rs_start)
        chosen = [{'i': 0, 'rs_start': rs_start, 'ctx_end': ctx_end,
                   'n_replace': n_replace, 'enc_state': self._infer_enc_state,
                   'decode_state': (self._infer_decode_state
                                    if use_stream_dec else None),
                   'is_first': not self._infer_phc_started}]
        fail_before = self._log_phc_fail
        self._phc_integrator_path([model_generated_0], chosen)
        ok = self._log_phc_fail == fail_before
        if ok:
            # Controller is now placed; subsequent windows continue from its
            # live pose instead of re-seeding from a decoded anchor.
            self._infer_phc_started = True
        return ok
