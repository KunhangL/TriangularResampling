"""Long-sequence N-way comparison for LDF checkpoints.

Generates the same sequence from N checkpoints using identical seed, text and
streaming schedule, renders each to mp4, then composites them side-by-side
(col0 | col1 | ... | colN) and uploads to wandb.

Optionally applies test-time PHC correction to one column (phc_col=<idx>),
using the same Resampler._phc_integrator_path code path as training.

The training config (configs/138d/ldf_138d_streaming_model_phc.yaml) is the
single source of truth: its model / VAE / normalization / content_len drive
both train_ldf.py and inference, and its `inference:` block holds the
generate-only switches (phc_col, phc_interval, prompt_set, lengths_sec,
debug_chain). generate reads inference.<key> first, then the OmegaConf root,
so CLI `--override <key>=...` still works.

Usage (PHC long-seq inference; defaults from the config's inference: block):
    python generate_long_sequences.py \\
        --config configs/138d/ldf_138d_streaming_model_phc.yaml \\
        --override test_ckpt=outputs/<run>/step_step=NNNNN.ckpt out_dir=<dir>

Multi-way comparison / quick debug (override switches as needed):
    ... --override \\
        ckpt_0=outputs/.../ldf_138d_streaming_model/step_step=50000.ckpt \\
        ckpt_1=outputs/.../ldf_138d_streaming_model_phc/step_step=50000.ckpt \\
        label_0=streaming label_1="streaming+PHC" \\
        phc_col=1 prompt_set=loopable only_lengths=120s

Smoke (first prompt × 60s only):
    ... smoke=true
"""

import json
import os
import re
import time
from datetime import datetime

import imageio
import numpy as np
import torch
import wandb
from lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch_ema import ExponentialMovingAverage

from common.utils.initialize import instantiate, load_config
from methods.fd1.utils.visualize import render_single_video

os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROMPT_SETS = {
    # Original set: naturally loopable locomotion / upright motion.
    "loopable": [
        ("walk_circle", "a person walks in a circle at a steady pace"),
        ("run_forward", "a person runs forward continuously"),
        ("jumping_jacks", "a person is doing jumping jacks repeatedly"),
        ("march_place", "a person marches in place, lifting knees high"),
        ("sway_dance", "a person sways side to side while dancing with their arms"),
    ],
    # Diverse set: non-locomotion, covers sitting, upper body only, flowing, etc.
    "diverse": [
        ("sit_stand", "a person sits down and then stands up repeatedly"),
        ("sit_stand_standstart",
         "starting from a standing position, repeatedly sit down and stand up"),
        ("squat", "a person squats up and down repeatedly"),
        ("boxing", "a person throws punches with both hands, shadow boxing"),
        ("taichi", "a person performs slow and flowing tai chi movements"),
        ("kick_cycle", "a person kicks forward with alternating legs"),
    ],
}
# Combined set of all 11 prompts (loopable + diverse).
PROMPT_SETS["all"] = PROMPT_SETS["loopable"] + PROMPT_SETS["diverse"]

LENGTHS_SEC = {"60s": 60, "90s": 90, "120s": 120, "600s": 600}
DEFAULT_FPS = 20
VAE_TIME_COMPRESSION = 4
SEED = 42


# ---------------------------------------------------------------------------
# Model / VAE loading helpers
# ---------------------------------------------------------------------------

def load_vae_from_cfg(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False,
                      **cfg.test_vae.params)
    ck = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(ck["state_dict"], strict=True)
    if "ema_state" in ck:
        ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        ema.load_state_dict(ck["ema_state"])
        ema.copy_to(vae.parameters())
        print(f"[vae] loaded {cfg.test_vae_ckpt} with EMA", flush=True)
    else:
        print(f"[vae] loaded {cfg.test_vae_ckpt} (no EMA)", flush=True)
    vae.to(device)
    vae.eval()
    return vae


def build_model_from_cfg(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model_params.pop("phc", None)
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False,
                        **model_params)
    model.to(device)
    model.eval()
    return model


def load_model_ckpt_into(model, ckpt_path, ema_decay, ckpt_format=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ck["state_dict"]
    if ckpt_format == "flooddiffusion_official":
        state_dict = {
            key.replace(".norm3.", ".cross_attn_norm_layer."): value
            for key, value in state_dict.items()
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != {"mean", "std"} or unexpected:
            raise RuntimeError(
                "Official FloodDiffusion checkpoint conversion produced "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        print(
            "[model] converted official FloodDiffusion keys "
            "(.norm3 -> .cross_attn_norm_layer); mean/std remain identity",
            flush=True,
        )
    else:
        model.load_state_dict(state_dict, strict=True)
    if "ema_state" in ck:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        ema.load_state_dict(ck["ema_state"])
        ema.copy_to(model.parameters())
        print(f"[model] loaded {ckpt_path} with EMA", flush=True)
    else:
        print(f"[model] loaded {ckpt_path} (no EMA)", flush=True)


# ---------------------------------------------------------------------------
# PHC corrector construction
# ---------------------------------------------------------------------------

def build_phc_corrector(cfg, model):
    """Construct a Resampler(sources=['phc']) for test-time PHC correction.

    Loads a SEPARATE VAE instance so that PHC's internal decode() calls do not
    corrupt the streaming _feat_map cache used by the main generation loop's
    stream_decode(). wan_vae.decode() calls clear_cache() at start and end,
    which would wipe the streaming state if it shared the same VAE object.
    """
    from methods.fd1.models.resampler import Resampler
    print("[phc] loading separate VAE for PHC (avoids streaming cache conflict)...",
          flush=True)
    phc_vae = load_vae_from_cfg(cfg)

    mp = cfg.model.params
    phc_cfg = dict(mp.get("phc", {}))
    # anchor_source is fixed to decoded_138; the live configs carry no
    # motion_pkl_dirs/motion_lib_cache_paths, so the backend boots in non-GT-RSI
    # mode from the lightweight synthetic motion. (GT-RSI lives on only for the
    # standalone physics/scripts/gt_track.py tool, not this inference path.)

    # Inference: single env is sufficient and much faster to boot than 64.
    phc_cfg["num_envs"] = 1
    # Under the generation script there is no DDP rank pinning; the visible
    # GPU is whatever CUDA_VISIBLE_DEVICES selects, always cuda:0.
    phc_cfg["device"] = "cuda:0"
    phc_cfg["device_id"] = 0

    # Restore the historical inference-only replacement-window knob. It is
    # intentionally decoupled from training resample_len so block-size
    # ablations do not alter the trained replay recipe.
    train_resample_len = int(mp.get("resample_len", 5))
    tt_phc_replace_len = int(OmegaConf.select(
        cfg.config, "tt_phc_replace_len", default=1,
    ))
    if tt_phc_replace_len <= 0:
        raise ValueError(
            "tt_phc_replace_len must be a positive integer "
            f"(got {tt_phc_replace_len})"
        )
    corrector = Resampler(
        sources=["phc"],
        shift=float(mp.get("resample_shift", 0.6)),
        resample_len=train_resample_len,
        tt_phc_replace_len=tt_phc_replace_len,
        prediction_type=str(mp.get("prediction_type", "vel")),
        time_scheduler=model.time_scheduler,
        time_embedding_scale=float(getattr(model, "time_embedding_scale", 1.0)),
        vae=phc_vae,
        phc_config=phc_cfg,
    )
    print(f"[phc] test-time PHC corrector ready "
          f"(inference_window_size={corrector.inference_window_size}, "
          f"tt_phc_replace_len={tt_phc_replace_len}, "
          f"train_resample_len={train_resample_len})",
          flush=True)
    return corrector


def prime_generation_prefix(model, caption, prefix_feature):
    """Insert an already-normalized latent prefix into the streaming state."""
    if prefix_feature is None:
        return 0
    prefix_feature = prefix_feature.to(
        device=model.generated[0].device,
        dtype=model.generated[0].dtype,
    )
    prefix_tokens = int(prefix_feature.shape[0])
    if prefix_tokens <= 0:
        return 0
    if prefix_tokens >= model.buf_len:
        raise ValueError(
            f"prefix_tokens={prefix_tokens} must be < model.buf_len={model.buf_len}")

    model.generated[0][:, :prefix_tokens, ...] = (
        prefix_feature.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
    )

    # The streaming scheduler commits one token after chunk_size conditions.
    # Prime the text stream far enough that `current_commit=prefix_tokens`
    # is a consistent already-committed history.
    device = next(model.parameters()).device
    cond_len = prefix_tokens + int(model.time_scheduler.chunk_size) - 1
    x = {model.input_keys["text"]: [caption]}
    for _ in range(cond_len):
        model.text_module.update_stream(x, device, model.param_dtype)
    model.condition_frames = cond_len
    model.current_commit = prefix_tokens
    _, model.current_step = model.time_scheduler.get_committable(cond_len)
    return prefix_tokens


# ---------------------------------------------------------------------------
# Single-sequence generation
# ---------------------------------------------------------------------------

def _stream_decode_window(vae, model, model_generated_0, lo, hi, first_chunk):
    """Slice [lo, hi) from model.generated[0], denormalize, stream-decode."""
    with torch.no_grad():
        latent_slice = model_generated_0[:, lo:hi, ...]
        latent_slice = model.postprocess([latent_slice])[0]   # (T, C)
        latent_slice = latent_slice * model.std + model.mean   # denorm
        decoded = vae.stream_decode(
            latent_slice[None, :].to(vae.mean.device),
            first_chunk=first_chunk,
        )[0]
    return decoded


@torch.no_grad()
def _decode_committed_for_debug(decode_vae, model, commit_tokens):
    """Snapshot full decode of `model.generated[0][:, :commit_tokens]`.

    Used by debug chain to capture the model's pre/post-PHC commit decode
    on the first PHC window. Pass `corrector._vae` (the separately-loaded
    PHC-side VAE) so this one-shot decode does NOT touch the streaming
    decoder's temporal cache used by run-time `_stream_decode_window`.
    """
    commit_tokens = int(commit_tokens)
    committed = model.generated[0][:, :commit_tokens, ...]
    committed = model.postprocess([committed])[0]
    committed = committed * model.std + model.mean
    return decode_vae.decode(committed[None, :])[0].detach().float().cpu()


def generate_one(vae, model, caption, num_iters, target_frames, out_mp4, label,
                 phc_corrector=None, phc_interval=1, prefix_feature=None,
                 debug_chain_dir=None, capture_first_window=False,
                 debug_trace_windows=1, stream_seq_len=100,
                 fps=DEFAULT_FPS, seed=SEED, render=True,
                 strict_target_frames=False, text_segments=None):
    """Streaming-generate one sequence and optionally render it.

    When phc_corrector is not None, decoding is deferred per PHC window so the
    streamed VAE only sees post-correction latents. Without a corrector, every
    fresh commit decodes immediately. The renderer concatenates the streamed
    chunks - we never re-decode the buffer.

    Debug-only args (used by the chain visualization side-channel):
    * `debug_chain_dir`: when set + corrector available, enable corrector
      trace dump (writes phc_trace/window_NNN/{arrays.npz, stats.json}).
    * `capture_first_window`: when True, snapshot `decode_committed` before
      and after the first PHC correction (uses corrector._vae to avoid
      mid-stream cache pollution).
    * `debug_trace_windows`: corrector trace window_limit (passed through).

    Returns dict with {ok, path, total_frames, elapsed_s, phc_corrections,
    phc_corrected_latents, motion, motion_dim, first_window,
    dbg_infer_records}.
    """
    if render:
        if not out_mp4:
            raise ValueError("out_mp4 is required when render=true")
        os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    vae.clear_cache()
    # Reset at the sample boundary so the same prompt/seed is paired across
    # every checkpoint even though checkpoints are evaluated in outer loops.
    seed_everything(int(seed), workers=True)
    # Fixed model window (default 100, aligned with generate_ldf.py) so attention
    # cost stays bounded regardless of target length. The model auto-rolls its
    # buffer when conditions exceed 2 * stream_seq_len; we predict each rollback
    # before stream_generate_step fires it and shift the local indices below
    # accordingly, so decoded chunks stay contiguous and PHC windows stay aligned.
    model.init_generated(stream_seq_len)
    prefix_caption = (
        str(text_segments[0]["caption"]) if text_segments else caption
    )
    prefix_tokens = prime_generation_prefix(model, prefix_caption, prefix_feature)
    if phc_corrector is not None and hasattr(phc_corrector, "reset_inference_state"):
        phc_corrector.reset_inference_state()

    if (phc_corrector is not None and debug_chain_dir
            and hasattr(phc_corrector, "enable_dbg_infer")):
        phc_corrector.enable_dbg_infer(
            debug_chain_dir, window_limit=debug_trace_windows)

    # PHC inference correction window. Decoupled from phc_corrector.resample_len
    # (= K*chunk_size, training replay region) so test-time window can be any
    # positive int without disturbing training-side semantics.
    phc_window = int(phc_corrector.inference_window_size) if phc_corrector is not None else 0
    # Token 0 is the causal VAE seed (one frame, not a four-frame motion span),
    # so PHC starts at token 1 even for generated-from-scratch sequences.
    # These indices are LOCAL to model.generated[0]'s current buffer view and
    # shift down by model.seq_len on a real rollback.
    phc_base = max(prefix_tokens, 1) if phc_corrector is not None else prefix_tokens
    next_phc_ctx_end = phc_base + phc_window
    prefix_tokens_local = phc_base

    stream_decoded = []
    t_start = time.time()
    nan_detected = False
    first_chunk = True
    phc_corrections = 0
    phc_corrected_latents = 0
    prev_decoded_end = 0  # exclusive token index up to which we've stream-decoded
    first_window = None
    snap_vae = (
        phc_corrector._vae
        if (phc_corrector is not None and getattr(phc_corrector, "_vae", None) is not None)
        else vae
    )

    # Decode the prefix first so the VAE temporal cache is primed before any
    # generated/corrected token enters.
    if prefix_tokens > 0:
        decoded_prefix = _stream_decode_window(
            vae, model, model.generated[0], 0, prefix_tokens, first_chunk
        )
        first_chunk = False
        if torch.isnan(decoded_prefix).any():
            print(f"  [ABORT] NaN in prefix decode ({label})", flush=True)
            return {"ok": False, "reason": "nan_prefix", "phc_corrections": 0}
        stream_decoded.append(decoded_prefix)
        prev_decoded_end = prefix_tokens

    def _apply_phc_window(ctx_end, n_replace, where):
        """Apply one contiguous PHC block and account for its actual length."""
        nonlocal phc_corrections, phc_corrected_latents, first_window
        n_replace = int(n_replace)
        rs_start = int(ctx_end) - n_replace
        if capture_first_window and first_window is None:
            first_window = {
                "rs_start": rs_start,
                "ctx_end": int(ctx_end),
                "model_before_phc_hml": _decode_committed_for_debug(
                    snap_vae, model, ctx_end),
            }
        ok = phc_corrector.apply_inference_correction(
            model.generated[0],
            int(ctx_end),
            min_rs_start=max(0, prefix_tokens_local),
            n_replace=n_replace,
        )
        if ok:
            phc_corrections += 1
            phc_corrected_latents += n_replace
            if (capture_first_window and first_window is not None
                    and "model_after_phc_hml" not in first_window):
                first_window["model_after_phc_hml"] = (
                    _decode_committed_for_debug(snap_vae, model, ctx_end))
        else:
            raise RuntimeError(
                f"PHC correction failed at {where} block "
                f"[{rs_start}, {ctx_end}); aborting sample to avoid an "
                "uncorrected latent gap"
            )
        return ok

    def _try_phc(commit_now, step_idx):
        """Trigger every PHC window whose [rs_start, ctx_end) is now committed.
        Loops because commit may run ahead of PHC by more than one window
        (single-token-per-step commit + phc_window>1 is the common case)."""
        nonlocal next_phc_ctx_end
        if phc_corrector is None or step_idx % phc_interval != 0:
            return
        while commit_now >= next_phc_ctx_end:
            _apply_phc_window(
                next_phc_ctx_end, phc_window, "full",
            )
            next_phc_ctx_end += phc_window

    def _flush_partial_phc(commit_end, where):
        """Correct a committed partial block at rollback or sequence tail."""
        nonlocal next_phc_ctx_end
        if phc_corrector is None or phc_window <= 1:
            return
        last_phc_end = next_phc_ctx_end - phc_window
        commit_end = int(commit_end)
        n_partial = commit_end - last_phc_end
        if n_partial <= 0:
            return
        if n_partial >= phc_window:
            raise RuntimeError(
                f"{where} partial PHC block has invalid size {n_partial} "
                f"for configured window {phc_window}"
            )
        _apply_phc_window(commit_end, n_partial, where)
        # Restart full-block scheduling immediately after the forced partial.
        next_phc_ctx_end = commit_end + phc_window

    def _flush_decode(safe_end, where):
        """Stream-decode [prev_decoded_end, safe_end). Returns False on NaN."""
        nonlocal prev_decoded_end, first_chunk
        if safe_end <= prev_decoded_end:
            return True
        decoded = _stream_decode_window(
            vae, model, model.generated[0],
            prev_decoded_end, safe_end, first_chunk,
        )
        first_chunk = False
        if torch.isnan(decoded).any():
            print(f"  [ABORT] NaN at {where} ({label})", flush=True)
            return False
        stream_decoded.append(decoded)
        prev_decoded_end = safe_end
        return True

    for i in range(num_iters):
        # Predict rollback BEFORE stream_generate_step fires it.
        # stream_generate_step does: condition_frames += 1; if cf > buf_len: rollback().
        will_rollback = (model.condition_frames + 1 > model.buf_len)

        if will_rollback:
            # Once rollback runs, indices [0, seq_len) get overwritten by the
            # back half — every ready PHC window must trigger and every
            # final-committed token in the front half must be decoded NOW,
            # else we lose data and produce a video gap.
            cur_commit = int(model.current_commit)
            _try_phc(cur_commit, i)
            if phc_corrector is not None:
                pending_start = next_phc_ctx_end - phc_window
                if pending_start < model.seq_len:
                    raise RuntimeError(
                        "pending PHC block would be discarded by rollback: "
                        f"start={pending_start}, seq_len={model.seq_len}"
                    )
            if phc_corrector is None:
                pre_safe = min(cur_commit, model.seq_len)
            else:
                pre_safe = min(cur_commit,
                               next_phc_ctx_end - phc_window,
                               model.seq_len)
            if not _flush_decode(pre_safe, "pre-rollback flush"):
                nan_detected = True
                break
            if prev_decoded_end < model.seq_len:
                # PHC pending window straddles the rollback edge -> tokens at
                # [prev_decoded_end, seq_len) would be discarded mid-correction.
                raise RuntimeError(
                    f"pre-rollback flush stopped at {prev_decoded_end}/"
                    f"{model.seq_len}; PHC pending window likely straddles "
                    f"rollback edge. Increase stream_seq_len so the "
                    f"(single-latent) PHC window fits in one half-buffer."
                )

        step_caption = caption
        if text_segments:
            step_time_sec = i * VAE_TIME_COMPRESSION / float(fps)
            step_caption = str(text_segments[-1]["caption"])
            for segment in text_segments:
                if (float(segment["start_sec"]) <= step_time_sec
                        < float(segment["end_sec"])):
                    step_caption = str(segment["caption"])
                    break
        x = {model.input_keys["text"]: [step_caption]}
        output = model.stream_generate_step(x)["generated"]

        if will_rollback:
            shift = model.seq_len
            prev_decoded_end -= shift
            next_phc_ctx_end -= shift
            prefix_tokens_local = max(1, prefix_tokens_local - shift)
            if prev_decoded_end < 0:
                raise RuntimeError(
                    "prev_decoded_end went negative after rollback")
            if phc_corrector is not None and next_phc_ctx_end - phc_window < 0:
                raise RuntimeError(
                    "next_phc_ctx_end - phc_window went negative after rollback")
            # Buffer indices just shifted; drop the corrector's streaming decode
            # cache so the next correction re-bootstraps on the shifted buffer.
            if phc_corrector is not None and hasattr(
                    phc_corrector, "reset_decode_state"):
                phc_corrector.reset_decode_state()

        if output[0].shape[0] == 0:
            continue
        new_commit = int(model.current_commit)

        _try_phc(new_commit, i)

        # safe_end: largest commit index whose tokens are final right now.
        # With PHC: tokens past (next_phc_ctx_end - phc_window) are inside the
        # still-pending correction window and must wait.
        if phc_corrector is None:
            safe_end = new_commit
        else:
            safe_end = min(new_commit, next_phc_ctx_end - phc_window)

        if not _flush_decode(safe_end, f"step {i}"):
            nan_detected = True
            break

        if i % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            decoded_frames = sum(d.shape[0] for d in stream_decoded)
            phc_info = f"  phc={phc_corrections}" if phc_corrector else ""
            print(
                f"  [{label}] step {i:4d}/{num_iters}  "
                f"elapsed={elapsed:6.1f}s  rate={rate:.2f} step/s  "
                f"frames={decoded_frames}/{target_frames}{phc_info}",
                flush=True,
            )

    # Finish every full block, then PHC-correct the final short block so the
    # corrected trajectory covers the whole generated latent timeline.
    if not nan_detected:
        final_commit = int(model.current_commit)
        _try_phc(final_commit, num_iters)
        _flush_partial_phc(final_commit, "tail")
        if not _flush_decode(final_commit, "tail flush"):
            nan_detected = True

    vae.clear_cache()

    if nan_detected or len(stream_decoded) == 0:
        return {"ok": False, "reason": "nan" if nan_detected else "empty",
                "phc_corrections": phc_corrections}

    motion = torch.cat(stream_decoded, dim=0)
    if strict_target_frames and motion.shape[0] < target_frames:
        return {
            "ok": False,
            "reason": f"short_motion:{motion.shape[0]}<{target_frames}",
            "phc_corrections": phc_corrections,
            "phc_corrected_latents": phc_corrected_latents,
        }
    # Streaming VAE decoding can emit one excess frame after compensating for
    # its first-chunk temporal warm-up. Metric artifacts have an exact contract.
    if strict_target_frames and motion.shape[0] > target_frames:
        motion = motion[:target_frames]
    total_frames = motion.shape[0]
    elapsed = time.time() - t_start
    phc_info = (
        f"  phc_corrections={phc_corrections}"
        f"  phc_corrected_latents={phc_corrected_latents}"
        if phc_corrector else ""
    )
    print(
        f"  [{label}] DONE frames={total_frames} ({total_frames / fps:.1f}s)  "
        f"elapsed={elapsed:.1f}s{phc_info}",
        flush=True,
    )

    motion_np = motion.cpu().numpy()
    motion_dim = int(motion_np.shape[-1])
    if motion_dim not in (138, 263):
        raise ValueError(
            f"generate_one: unsupported motion dim {motion_dim} (got "
            f"shape {tuple(motion_np.shape)}); expected 263 or 138")
    if render:
        print(f"  [{label}] rendering -> {out_mp4}", flush=True)
        # 263 → existing skeleton render path; 138 → mesh render path.
        if motion_dim == 138:
            render_single_video(motion=motion_np, save_path=out_mp4, dim=138,
                                render_setting={"render_mode": "mesh", "fps": fps})
        else:
            render_single_video(motion=motion_np, save_path=out_mp4, dim=263,
                                render_setting={"fps": fps})

    # PHC-raw (proposal-2 cache): the controller's own tracked motion, exact
    # physical output never re-encoded/decoded. Tail-aligned onto the decoded
    # timeline (corrections cover the committed tail; the prefix stays decoded)
    # so the raw and writeback-decoded columns share one timeline.
    phc_motion_np = None
    if (phc_corrector is not None and motion_dim == 138
            and hasattr(phc_corrector, "get_corrected_trajectory")):
        traj = phc_corrector.get_corrected_trajectory(0)
        if traj is not None:
            phc_motion_np = motion_np.copy()
            m = min(int(traj.shape[0]), phc_motion_np.shape[0])
            phc_motion_np[-m:] = traj[-m:].cpu().numpy()

    # Pop trace records (only populated when debug_chain_dir was set).
    dbg_infer_records = []
    if (phc_corrector is not None
            and hasattr(phc_corrector, "pop_dbg_infer_records")):
        dbg_infer_records = phc_corrector.pop_dbg_infer_records()

    return {"ok": True, "path": out_mp4, "total_frames": total_frames,
            "elapsed_s": elapsed, "phc_corrections": phc_corrections,
            "phc_corrected_latents": phc_corrected_latents,
            "motion": motion_np, "motion_dim": motion_dim,
            "phc_motion": phc_motion_np,
            "first_window": first_window,
            "dbg_infer_records": dbg_infer_records}


# ---------------------------------------------------------------------------
# Composite helpers
# ---------------------------------------------------------------------------

def _get_font(size=28):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def composite_n_way(mp4_paths, out_mp4, labels, caption=None,
                    default_fps=DEFAULT_FPS):
    """Stack N videos horizontally with text labels burned in.

    Optional caption string is rendered as a centered text banner below the frames.
    """
    assert len(mp4_paths) == len(labels), "mp4_paths and labels must have same length"
    readers = [imageio.get_reader(p) for p in mp4_paths]
    meta = readers[0].get_meta_data()
    fps = meta.get("fps", default_fps)
    n_frames = min(r.count_frames() for r in readers)

    font_label = _get_font(28)
    font_caption = _get_font(22)
    pad = 8

    writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=1,
                                codec="libx264", quality=8)
    try:
        for idx in range(n_frames):
            frames = [r.get_data(idx) for r in readers]
            # Normalise heights
            h = max(f.shape[0] for f in frames)
            w = frames[0].shape[1]
            normed = []
            for f in frames:
                if f.shape[0] != h or f.shape[1] != w:
                    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                    canvas[:f.shape[0], :f.shape[1]] = f
                    f = canvas
                normed.append(f)
            combo = np.concatenate(normed, axis=1)
            total_w = combo.shape[1]

            # Optional caption banner below frames
            if caption:
                banner_h = 36
                banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
                combo = np.concatenate([combo, banner], axis=0)

            img = Image.fromarray(combo)
            draw = ImageDraw.Draw(img)

            # Column labels at top
            for col_idx, text in enumerate(labels):
                text_x = col_idx * w + pad
                bbox = draw.textbbox((text_x, pad), text, font=font_label)
                draw.rectangle(
                    [(bbox[0] - 4, bbox[1] - 2), (bbox[2] + 4, bbox[3] + 2)],
                    fill=(0, 0, 0),
                )
                draw.text((text_x, pad), text, fill=(255, 255, 255), font=font_label)

            # Prompt caption centered at bottom
            if caption:
                cap_y = h + 6
                bbox = draw.textbbox((0, cap_y), caption, font=font_caption)
                text_w = bbox[2] - bbox[0]
                cap_x = (total_w - text_w) // 2
                draw.text((cap_x, cap_y), caption, fill=(200, 200, 200),
                          font=font_caption)

            writer.append_data(np.array(img))
    finally:
        writer.close()
        for r in readers:
            r.close()
    return out_mp4


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _as_list(v):
    if v is None:
        return None
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    try:
        return list(v)
    except TypeError:
        return [v]


def _as_int_list(v, default=None):
    values = _as_list(v)
    if values is None:
        return list(default or [])
    parsed = [int(x) for x in values]
    if not parsed:
        raise ValueError("seeds must contain at least one integer")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"seeds contains duplicates: {parsed}")
    return parsed


def _normalize_caption(caption):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", caption.casefold()).split())


def _load_prompt_manifest(path):
    prompts = []
    seen_ids = set()
    seen_captions = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            missing = {"prompt_id", "caption", "text_tokens"} - set(item)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {sorted(missing)}")
            prompt_id = str(item["prompt_id"]).strip()
            caption = str(item["caption"]).strip()
            tokens = item["text_tokens"]
            if not prompt_id or not caption:
                raise ValueError(f"{path}:{line_no}: empty prompt_id or caption")
            if not isinstance(tokens, list) or not all(
                    isinstance(token, str) and token for token in tokens):
                raise ValueError(
                    f"{path}:{line_no}: text_tokens must be a non-empty string list")
            segments = item.get("segments")
            if segments is not None:
                if not isinstance(segments, list) or not segments:
                    raise ValueError(
                        f"{path}:{line_no}: segments must be a non-empty list")
                previous_end = 0.0
                for segment_index, segment in enumerate(segments):
                    if not isinstance(segment, dict):
                        raise ValueError(
                            f"{path}:{line_no}: segment {segment_index} is not an object")
                    segment_caption = str(segment.get("caption", "")).strip()
                    segment_tokens = segment.get("text_tokens")
                    start_sec = float(segment.get("start_sec", -1))
                    end_sec = float(segment.get("end_sec", -1))
                    if (not segment_caption
                            or not isinstance(segment_tokens, list)
                            or not all(isinstance(token, str) and token
                                       for token in segment_tokens)
                            or start_sec != previous_end
                            or end_sec <= start_sec):
                        raise ValueError(
                            f"{path}:{line_no}: invalid segment {segment_index}")
                    previous_end = end_sec
            normalized = _normalize_caption(caption)
            if prompt_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate prompt_id {prompt_id!r}")
            if normalized in seen_captions:
                raise ValueError(
                    f"{path}:{line_no}: duplicate normalized caption {caption!r}")
            seen_ids.add(prompt_id)
            seen_captions.add(normalized)
            prompts.append({
                "prompt_id": prompt_id,
                "caption": caption,
                "text_tokens": list(tokens),
                **({"segments": segments} if segments is not None else {}),
            })
    if not prompts:
        raise ValueError(f"prompt manifest is empty: {path}")
    return prompts


def _safe_path_component(value):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not safe:
        raise ValueError(f"value cannot form a safe path component: {value!r}")
    return safe


def _save_motion_npz(path, motion, *, fps, seed, target_seconds,
                     prompt_id, caption, text_tokens, method, checkpoint):
    """Atomically persist the metric-mode long-motion interchange format."""
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] not in (138, 263):
        raise ValueError(
            "metric motion must have shape (T, 138) or (T, 263), "
            f"got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError("metric motion contains non-finite values")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        motion=motion,
        representation_dim=np.asarray(int(motion.shape[1]), dtype=np.int64),
        fps=np.asarray(int(fps), dtype=np.int64),
        seed=np.asarray(int(seed), dtype=np.int64),
        target_seconds=np.asarray(int(target_seconds), dtype=np.int64),
        prompt_id=np.asarray(str(prompt_id)),
        caption=np.asarray(str(caption)),
        text_tokens=np.asarray(list(text_tokens), dtype=np.str_),
        method=np.asarray(str(method)),
        checkpoint=np.asarray(str(checkpoint)),
    )
    os.replace(tmp_path, path)


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp_path, path)


# Sentinel distinguishing "key absent" from "key present but null", so an
# explicit `null` (e.g. debug_chain.trace_windows: null = dump every window)
# is preserved instead of falling through to the default.
_MISSING = object()


def _isel(cfg_config, key, default=None):
    """Read a generate_long_sequences switch.

    Priority: OmegaConf root first (so CLI `--override <key>=...` and legacy
    root-level configs win), then the `inference:` namespace block (shared-
    config defaults, since training and inference use one file), then `default`.
    """
    val = OmegaConf.select(cfg_config, key, default=_MISSING)
    if val is _MISSING:
        val = OmegaConf.select(cfg_config, f"inference.{key}", default=_MISSING)
    return default if val is _MISSING else val


def _resolve_lengths_sec(cfg_config):
    lengths = dict(LENGTHS_SEC)
    override = _isel(cfg_config, "lengths_sec", default=None)
    if override is None:
        return lengths

    if isinstance(override, str):
        parsed = {}
        for item in override.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                key, value = item.split(":", 1)
            elif "=" in item:
                key, value = item.split("=", 1)
            else:
                raise ValueError(
                    f"Bad lengths_sec item {item!r}; expected tag:seconds")
            parsed[key.strip()] = int(value)
        return parsed

    raw = OmegaConf.to_container(override, resolve=True) if OmegaConf.is_config(override) else override
    if not isinstance(raw, dict):
        raise ValueError("lengths_sec must be a dict or 'tag:seconds,...' string")
    return {str(k): int(v) for k, v in raw.items()}


def _resolve_ckpt_list(cfg_config, test_ckpt):
    """Collect ckpt_0, ckpt_1, ... from override keys.

    Also supports legacy ckpt_pre + test_ckpt (2-way) for backward compat.
    Returns list of ckpt paths and list of default labels.
    """
    ckpts, labels = [], []

    # New N-way format: ckpt_0, ckpt_1, ...
    for i in range(64):
        v = OmegaConf.select(cfg_config, f"ckpt_{i}", default=None)
        if v is None:
            break
        ckpts.append(str(v))
        lv = OmegaConf.select(cfg_config, f"label_{i}", default=None)
        labels.append(str(lv) if lv is not None else f"col{i}")

    if ckpts:
        return ckpts, labels

    # Legacy 2-way format: ckpt_pre + test_ckpt
    ckpt_pre = OmegaConf.select(cfg_config, "ckpt_pre", default=None)
    if ckpt_pre is None:
        raise ValueError(
            "No checkpoints found. Provide ckpt_0/ckpt_1/... or legacy ckpt_pre+test_ckpt."
        )
    ckpts = [str(ckpt_pre), str(test_ckpt)]
    labels = ["pre-finetune", "post-finetune"]
    return ckpts, labels


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")

    test_fps = OmegaConf.select(cfg.config, "test_setting.fps",
                                default=DEFAULT_FPS)
    fps = int(_isel(cfg.config, "fps", default=test_fps))
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    smoke = bool(OmegaConf.select(cfg.config, "smoke", default=False))

    # Auto-build the dual-pass ckpt list when debug_chain is enabled and the
    # user only provided `test_ckpt` (preset configs use this shortcut so the
    # CLI override is a single key). Use OmegaConf.select with default=None
    # so missing/mandatory `test_ckpt` doesn't bomb access.
    _debug_enable_for_ckpt = bool(_isel(cfg.config, "debug_chain.enable", default=False))
    _explicit_ckpt0 = OmegaConf.select(
        cfg.config, "ckpt_0", default=None,
        throw_on_missing=False, throw_on_resolution_failure=False)
    _test_ckpt_val = OmegaConf.select(
        cfg.config, "test_ckpt", default=None,
        throw_on_missing=False, throw_on_resolution_failure=False)
    if (_debug_enable_for_ckpt and _explicit_ckpt0 is None
            and _test_ckpt_val not in (None, "")):
        OmegaConf.update(cfg.config, "ckpt_0", str(_test_ckpt_val))
        OmegaConf.update(cfg.config, "ckpt_1", str(_test_ckpt_val))
        # Labels are filled by the PHC-aware default below (model_only /
        # model_phc), so no need to set label_0/label_1 here.
        print(
            f"[main] debug_chain auto-fill: ckpt_0=ckpt_1={_test_ckpt_val}",
            flush=True,
        )

    # Resolve checkpoint list (new N-way or legacy 2-way).
    _legacy_test_ckpt = OmegaConf.select(
        cfg.config, "test_ckpt", default=None,
        throw_on_missing=False, throw_on_resolution_failure=False)
    ckpt_list, default_labels = _resolve_ckpt_list(cfg.config, _legacy_test_ckpt)
    ckpt_formats = [
        OmegaConf.select(cfg.config, f"ckpt_format_{i}", default=None)
        for i in range(len(ckpt_list))
    ]

    # phc_cols: set of column indices that get test-time PHC correction.
    # Accepts a single int (-1 = none) or comma-separated ints (e.g. "2,4").
    _phc_col_raw = _isel(cfg.config, "phc_col", default=-1)
    if isinstance(_phc_col_raw, str) and "," in _phc_col_raw:
        phc_cols = set(int(x.strip()) for x in _phc_col_raw.split(",") if x.strip())
    else:
        _v = int(_phc_col_raw)
        phc_cols = set() if _v < 0 else {_v}

    # Auto-extend ckpt_list if any phc_col points beyond current length.
    for pc in sorted(phc_cols):
        while len(ckpt_list) <= pc:
            ckpt_list.append(ckpt_list[-1])
            default_labels.append(f"col{len(ckpt_list) - 1}")

    # debug_chain: opt-in chain visualization side-channel for PHC inspection.
    # When enabled, generate_one dumps per-window arrays.npz/stats.json under
    # combo_dir/phc_trace/, snapshots model commit decode before/after the
    # first PHC correction, and main runs render_chain after composite step.
    # Constraint: requires exactly N=2 columns with phc_col={1} so col 0
    # (model only) and col 1 (model+PHC) form a clean dual-pass.
    debug_enable = bool(_isel(cfg.config, "debug_chain.enable", default=False))
    _dtw_raw = _isel(cfg.config, "debug_chain.trace_windows", default=1)
    # null → unlimited (trace every PHC window). int → cap.
    debug_trace_windows = None if _dtw_raw is None else int(_dtw_raw)
    if debug_enable:
        if len(ckpt_list) != 2 or phc_cols != {1}:
            raise ValueError(
                "debug_chain.enable=true requires exactly N=2 columns with "
                "phc_col=1 (col 0 = model only, col 1 = model+PHC). Got "
                f"n_cols={len(ckpt_list)} phc_cols={sorted(phc_cols)}")

    # Per-column labels: explicit label_i wins, else default by PHC membership
    # (model_phc for corrected columns, model_only otherwise) so the common
    # same-model ±PHC comparison is self-labeling without passing label_0/_1.
    # For genuine multi-model A/B (multiple non-PHC columns) pass label_i to
    # disambiguate, else they all read "model_only".
    for i in range(len(ckpt_list)):
        lv = OmegaConf.select(cfg.config, f"label_{i}", default=None)
        if lv is not None:
            default_labels[i] = str(lv)
        else:
            default_labels[i] = "model_phc" if i in phc_cols else "model_only"
    labels = default_labels

    n_cols = len(ckpt_list)
    if len(set(labels)) != len(labels):
        raise ValueError(f"column labels/method names must be unique, got {labels}")
    method_files = [_safe_path_component(label) for label in labels]
    if len(set(method_files)) != len(method_files):
        raise ValueError(
            f"column labels collapse to duplicate output names: {method_files}")

    prompt_manifest = _isel(cfg.config, "prompt_manifest", default=None)
    if prompt_manifest:
        prompt_manifest = str(prompt_manifest)
        prompts = _load_prompt_manifest(prompt_manifest)
        prompt_set_name = os.path.splitext(os.path.basename(prompt_manifest))[0]
    else:
        prompt_set_name = _isel(cfg.config, "prompt_set", default="loopable")
        if prompt_set_name not in PROMPT_SETS:
            raise ValueError(
                f"prompt_set={prompt_set_name!r} not in {list(PROMPT_SETS)}")
        prompts = [
            {"prompt_id": prompt_id, "caption": caption, "text_tokens": []}
            for prompt_id, caption in PROMPT_SETS[prompt_set_name]
        ]

    seeds = _as_int_list(_isel(cfg.config, "seeds", default=None), default=[SEED])
    save_motion = bool(_isel(cfg.config, "save_motion", default=False))
    render = bool(_isel(cfg.config, "render", default=True))
    metric_mode = save_motion and not render
    if debug_enable and not render:
        raise ValueError("debug_chain.enable=true requires render=true")
    lengths_sec = _resolve_lengths_sec(cfg.config)

    out_dir_override = OmegaConf.select(cfg.config, "out_dir", default=None)
    if out_dir_override is None:
        out_dir_override = OmegaConf.select(cfg.config, "out_root", default=None)
    if out_dir_override:
        out_root = str(out_dir_override)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_ckpt_dir = os.path.dirname(ckpt_list[-1])
        out_root = os.path.join(last_ckpt_dir, "long_sequences", ts)
    os.makedirs(out_root, exist_ok=True)

    for i, (ckpt, lbl) in enumerate(zip(ckpt_list, labels)):
        phc_marker = " [+TT-PHC]" if i in phc_cols else ""
        print(f"[main] col{i}: {lbl}{phc_marker}  ckpt={ckpt}", flush=True)

    wandb_key = cfg.logger.wandb.wandb_key
    if wandb_key and wandb_key.strip():
        os.environ["WANDB_API_KEY"] = wandb_key

    last_tag = os.path.basename(ckpt_list[-1]).replace(".ckpt", "")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"long_seq_{n_cols}way_{last_tag}_{run_stamp}_{prompt_set_name}"
    if phc_cols:
        run_name += "_ttphc"
    if smoke:
        run_name += "_smoke"

    wandb_mode = str(OmegaConf.select(
        cfg.logger.wandb, "mode", default="online"))
    if metric_mode:
        wandb_mode = "disabled"
    wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        mode=wandb_mode,
        name=run_name,
        config={
            "ckpt_list": ckpt_list,
            "ckpt_formats": ckpt_formats,
            "labels": labels,
            "phc_cols": sorted(phc_cols),
            "vae_ckpt": cfg.test_vae_ckpt,
            "prompt_set": prompt_set_name,
            "prompt_manifest": prompt_manifest,
            "prompts": [
                {"id": p["prompt_id"], "caption": p["caption"]} for p in prompts
            ],
            "lengths_sec": lengths_sec,
            "fps": fps,
            "vae_time_compression": VAE_TIME_COMPRESSION,
            "seeds": seeds,
            "save_motion": save_motion,
            "render": render,
            "smoke": smoke,
            "out_root": out_root,
        },
        dir=out_root,
    )
    if wandb_mode != "disabled":
        print(f"[main] wandb run: {wandb.run.url}", flush=True)
    else:
        print("[main] wandb mode=disabled (no upload)", flush=True)

    print("[main] loading VAE (shared)...", flush=True)
    vae = load_vae_from_cfg(cfg)
    print("[main] building model skeleton...", flush=True)
    model = build_model_from_cfg(cfg)
    ema_decay = cfg.model.ema_decay

    # Build PHC corrector once if needed (expensive: ~30s IsaacLab boot).
    phc_corrector = None
    if phc_cols:
        mp = cfg.model.params
        if not mp.get("phc", None):
            raise ValueError(
                "phc_col requires model.params.phc config block in the YAML."
            )
        print("[main] booting test-time PHC corrector (IsaacLab, ~30s)...", flush=True)
        phc_corrector = build_phc_corrector(cfg, model)

    # Always start from FloodDiffusion's self-generated init (no PHC default-stance
    # prefix). PHC first window uses decoded_138 (gt_138 is a debug-only path that
    # requires raw GT MEI138 from the dataset, unavailable at long-seq inference).
    prefix_feature = None
    prefix_tokens = 0
    if phc_cols and phc_corrector is None:
        raise ValueError("phc_col requires a PHC corrector")

    # phc_interval: apply PHC correction every N steps (1 = every step, default).
    phc_interval = int(_isel(cfg.config, "phc_interval", default=1))
    if phc_interval <= 0:
        raise ValueError(f"phc_interval must be positive (got {phc_interval})")
    if (phc_corrector is not None
            and phc_interval > 1
            and int(phc_corrector.inference_window_size) > 1):
        raise ValueError(
            "phc_interval>1 cannot be combined with tt_phc_replace_len>1: "
            "use phc_interval=1 for contiguous block correction"
        )
    if phc_interval > 1:
        print(f"[main] phc_interval={phc_interval} (PHC every {phc_interval} steps)",
              flush=True)

    only_lengths = _as_list(_isel(cfg.config, "only_lengths", default=None))
    only_prompts = _as_list(_isel(cfg.config, "only_prompts", default=None))
    if only_lengths is not None:
        bad = [L for L in only_lengths if L not in lengths_sec]
        if bad:
            raise ValueError(f"only_lengths contains unknown tags: {bad}")
    if only_prompts is not None:
        known = {p["prompt_id"] for p in prompts}
        bad = [p for p in only_prompts if p not in known]
        if bad:
            raise ValueError(f"only_prompts not in prompt_set={prompt_set_name}: {bad}")
        prompts = [p for p in prompts if p["prompt_id"] in only_prompts]

    # max_iters: cap iters per combo (quick pipeline validation).
    max_iters = OmegaConf.select(cfg.config, "max_iters", default=None)
    max_iters = int(max_iters) if max_iters is not None else None

    # stream_seq_len: model window size (latent tokens). Buffer = 2 * stream_seq_len;
    # generation rolls when commit exceeds buffer. Default 100 (aligned with
    # generate_ldf.py) keeps attention cost bounded for any target length.
    stream_seq_len = int(OmegaConf.select(cfg.config, "stream_seq_len", default=100))
    print(f"[main] stream_seq_len={stream_seq_len} (buf_len={stream_seq_len * 2})",
          flush=True)

    if smoke:
        first_len_tag = next(iter(lengths_sec))
        combos = [(prompts[0], first_len_tag, lengths_sec[first_len_tag], seeds[0])]
    else:
        combos = [
            (prompt, len_tag, secs, seed)
            for prompt in prompts
            for len_tag, secs in lengths_sec.items()
            for seed in seeds
            if only_lengths is None or len_tag in only_lengths
        ]

    # Shard combos for parallel multi-process runs.
    # Launch N workers with num_workers=N worker_idx=0..N-1; each handles its slice.
    num_workers = int(OmegaConf.select(cfg.config, "num_workers", default=1))
    worker_idx = int(OmegaConf.select(cfg.config, "worker_idx", default=0))
    if num_workers > 1:
        combos = combos[worker_idx::num_workers]
        print(f"[main] worker {worker_idx}/{num_workers}: {len(combos)} combos assigned",
              flush=True)

    print(
        f"[main] prompt_set={prompt_set_name}  {n_cols} columns  "
        f"{len(combos)} combos  seeds={seeds}  render={render}  "
        f"save_motion={save_motion}  smoke={smoke}",
        flush=True,
    )

    # ckpt-first generation: load each checkpoint once and generate all assigned
    # combos before moving to the next checkpoint. This avoids reloading the same
    # large checkpoint once per prompt.
    # col_results_map[(sample_key, col_idx)] = (mp4_path, res)
    col_results_map = {}
    attempts = []
    worker_summary_path = os.path.join(
        out_root,
        "generation_summary.json" if num_workers == 1
        else f"worker_{worker_idx:03d}_summary.json",
    )

    for col_idx, (ckpt, lbl) in enumerate(zip(ckpt_list, labels)):
        corrector = phc_corrector if col_idx in phc_cols else None
        print(f"\n[main] ===== col{col_idx}: {lbl}  ckpt={ckpt} =====", flush=True)
        load_model_ckpt_into(
            model, ckpt, ema_decay, ckpt_format=ckpt_formats[col_idx])
        if corrector is not None and hasattr(corrector, "_set_latent_norm_from_model"):
            corrector._set_latent_norm_from_model(model)

        for prompt, len_tag, secs, seed in combos:
            pid = prompt["prompt_id"]
            caption = prompt["caption"]
            text_tokens = prompt["text_tokens"]
            sample_key = (pid, len_tag, int(seed))
            metric_layout = save_motion or prompt_manifest or len(seeds) > 1
            tag = f"{pid}_{len_tag}" if not metric_layout else f"{pid}_{len_tag}_seed{seed}"
            combo_dir = os.path.join(out_root, len_tag, pid)
            if metric_layout:
                combo_dir = os.path.join(combo_dir, f"seed_{seed}")
            os.makedirs(combo_dir, exist_ok=True)
            target_frames = secs * fps
            target_latents = target_frames // VAE_TIME_COMPRESSION
            num_iters = (
                max(0, target_latents - prefix_tokens)
                if prefix_feature is not None else target_latents
            )
            if save_motion:
                # The streaming VAE's first five-latent chunk emits one frame;
                # subsequent latents emit four. Five extra inference steps turn
                # 600 latent steps from 2381 frames into 2401, which is then
                # deterministically cropped to the exact 2400-frame contract.
                metric_warmup_tokens = int(_isel(
                    cfg.config, "metric_decode_warmup_tokens",
                    default=VAE_TIME_COMPRESSION + 1))
                if metric_warmup_tokens < 0:
                    raise ValueError("metric_decode_warmup_tokens must be non-negative")
                num_iters += metric_warmup_tokens
            if max_iters is not None:
                num_iters = min(num_iters, max_iters)

            print(
                f"  [{lbl}] {tag} : '{caption}' ({num_iters} iters, seed={seed})",
                flush=True,
            )
            mp4_path = (
                os.path.join(combo_dir, f"{tag}_col{col_idx}.mp4")
                if render else None
            )

            # debug_chain: only the column with PHC writes a trace dump
            # (col 0 model-only has no corrector; trace would be empty).
            col_debug_dir = None
            col_capture_first = False
            if debug_enable and corrector is not None:
                col_debug_dir = os.path.join(combo_dir, "phc_trace")
                col_capture_first = True

            npz_path = None
            try:
                res = generate_one(
                    vae, model, caption, num_iters, target_frames,
                    mp4_path, lbl, phc_corrector=corrector,
                    phc_interval=phc_interval,
                    prefix_feature=prefix_feature,
                    debug_chain_dir=col_debug_dir,
                    capture_first_window=col_capture_first,
                    debug_trace_windows=debug_trace_windows,
                    stream_seq_len=stream_seq_len, fps=fps,
                    seed=seed, render=render,
                    strict_target_frames=save_motion,
                    text_segments=prompt.get("segments"),
                )
                if res["ok"] and save_motion:
                    npz_path = os.path.join(
                        combo_dir, f"{method_files[col_idx]}.npz")
                    _save_motion_npz(
                        npz_path,
                        res["motion"],
                        fps=fps,
                        seed=seed,
                        target_seconds=secs,
                        prompt_id=pid,
                        caption=caption,
                        text_tokens=text_tokens,
                        method=lbl,
                        checkpoint=ckpt,
                    )
                    print(f"  [{lbl}] motion -> {npz_path}", flush=True)
            except Exception as exc:
                import traceback
                res = {
                    "ok": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "phc_corrections": 0,
                }
                traceback.print_exc()

            col_results_map[(sample_key, col_idx)] = (mp4_path, res)
            attempt = {
                "prompt_id": pid,
                "length": len_tag,
                "target_seconds": int(secs),
                "seed": int(seed),
                "method": lbl,
                "checkpoint": str(ckpt),
                "ok": bool(res["ok"]),
                "reason": res.get("reason"),
                "npz": npz_path,
                "mp4": mp4_path,
                "frames": int(res.get("total_frames", 0)),
                "elapsed_s": float(res.get("elapsed_s", 0.0)),
            }
            attempts.append(attempt)
            _write_json_atomic(worker_summary_path, {
                "worker_idx": worker_idx,
                "num_workers": num_workers,
                "out_root": out_root,
                "attempts": attempts,
                "num_ok": sum(item["ok"] for item in attempts),
                "num_fail": sum(not item["ok"] for item in attempts),
            })
            if not res["ok"]:
                print(f"  [FAIL] {tag} col{col_idx} reason={res['reason']}", flush=True)
                wandb.run.summary[f"fail/{tag}_col{col_idx}"] = res["reason"]
            elif metric_mode and not debug_enable:
                # NPZ is already durable; do not retain 120s arrays until all
                # checkpoints/prompts have completed.
                res.pop("motion", None)
                res.pop("phc_motion", None)

    # Composite and log per combo
    summary = []
    for prompt, len_tag, secs, seed in combos:
        pid = prompt["prompt_id"]
        caption = prompt["caption"]
        sample_key = (pid, len_tag, int(seed))
        metric_layout = save_motion or prompt_manifest or len(seeds) > 1
        tag = f"{pid}_{len_tag}" if not metric_layout else f"{pid}_{len_tag}_seed{seed}"
        combo_dir = os.path.join(out_root, len_tag, pid)
        if metric_layout:
            combo_dir = os.path.join(combo_dir, f"seed_{seed}")

        col_entries = [
            (labels[i], *col_results_map[(sample_key, i)]) for i in range(n_cols)
        ]
        # col_entries[i] = (label, mp4_path, res)

        failed = [i for i, (_, _, res) in enumerate(col_entries) if not res["ok"]]
        if failed:
            summary.append({"tag": tag, "ok": False, "failed_cols": failed})
            continue

        # Skip compositing for single-column runs: a 1-way composite is
        # degenerate (just re-encodes col0) and would write an unwanted
        # *_composite.mp4. Multi-column behavior below is unchanged.
        if render and n_cols >= 2:
            # Insert a middle column = raw PHC motion (exact controller output, no
            # VAE roundtrip) just before each PHC column whose res carries it.
            comp_paths, comp_labels = [], []
            for col_idx, (lbl, mp4_path, res) in enumerate(col_entries):
                if res.get("phc_motion") is not None:
                    raw_mp4 = os.path.join(combo_dir, f"{tag}_col{col_idx}_phcraw.mp4")
                    render_single_video(motion=res["phc_motion"], save_path=raw_mp4,
                                        dim=138, render_setting={"render_mode": "mesh",
                                                                 "fps": fps})
                    comp_paths.append(raw_mp4)
                    comp_labels.append(f"{lbl}_raw")
                comp_paths.append(mp4_path)
                comp_labels.append(lbl)

            mp4_comp = os.path.join(combo_dir, f"{tag}_composite.mp4")
            print(f"\n  compositing {len(comp_paths)} cols → {mp4_comp}", flush=True)
            t0 = time.time()
            composite_n_way(comp_paths, mp4_comp, comp_labels, caption=caption,
                            default_fps=fps)
            print(f"  composited in {time.time() - t0:.1f}s", flush=True)

            wandb.log({f"long_seq/{tag}_composite": wandb.Video(mp4_comp, format="gif")})

        entry = {
            "tag": tag,
            "ok": True,
            "caption": caption,
            "prompt_id": pid,
            "seed": int(seed),
        }
        for col_idx, (lbl, _, res) in enumerate(col_entries):
            entry[f"frames_col{col_idx}"] = res["total_frames"]
            entry[f"elapsed_col{col_idx}_s"] = res["elapsed_s"]
            if res.get("phc_corrections", 0):
                entry[f"phc_corrections_col{col_idx}"] = res["phc_corrections"]
            if res.get("phc_corrected_latents", 0):
                entry[f"phc_corrected_latents_col{col_idx}"] = (
                    res["phc_corrected_latents"]
                )
        summary.append(entry)

    # debug_chain: render PHC inspection grid per combo using col 0 (model only)
    # and col 1 (model+PHC) results plus the trace dump under combo_dir/phc_trace/.
    if debug_enable:
        from methods.fd1.utils.debug_viz import render_phc_chain, load_window_from_npz
        from methods.fd1.utils.render_mesh import SMPLMeshRenderer
        chain_mesh_renderer = SMPLMeshRenderer(gender="neutral")
        print("\n[main] === debug_chain rendering ===", flush=True)
        for prompt, len_tag, secs, seed in combos:
            pid = prompt["prompt_id"]
            caption = prompt["caption"]
            sample_key = (pid, len_tag, int(seed))
            metric_layout = save_motion or prompt_manifest or len(seeds) > 1
            tag = f"{pid}_{len_tag}" if not metric_layout else f"{pid}_{len_tag}_seed{seed}"
            combo_dir = os.path.join(out_root, len_tag, pid)
            if metric_layout:
                combo_dir = os.path.join(combo_dir, f"seed_{seed}")
            mo_entry = col_results_map.get((sample_key, 0))
            phc_entry = col_results_map.get((sample_key, 1))
            if mo_entry is None or phc_entry is None:
                continue
            _, mo_res = mo_entry
            _, phc_res = phc_entry
            if not mo_res.get("ok") or not phc_res.get("ok"):
                print(f"  [debug_chain] skip {tag}: a column failed", flush=True)
                continue
            try:
                trace_records = phc_res.get("dbg_infer_records", [])
                windows = [load_window_from_npz(rec) for rec in trace_records]
                # Long-seq chains have ~146 windows; mesh-rendering each row is
                # ~90s, so the full chain is impractical. Cap to the first
                # `chain_window_limit` windows (default 10) and give the renderer
                # enough wall-clock budget to finish those without truncation.
                chain_window_limit = int(_isel(
                    cfg.config, "debug_chain.window_limit", default=10))
                chain_summary = render_phc_chain(
                    out_mp4=os.path.join(combo_dir, "window_chain.mp4"),
                    windows=windows[:chain_window_limit],
                    col_left_motion_138=mo_res["motion"],
                    col_left_label="model only",
                    col_right_motion_138=phc_res["motion"],
                    col_right_label="model + PHC",
                    mesh_renderer=chain_mesh_renderer,
                    caption=caption,
                    max_render_seconds=1800,
                )
                if chain_summary:
                    print(
                        f"  [debug_chain] {tag}: {chain_summary['video']} "
                        f"({chain_summary['rows']} windows)",
                        flush=True,
                    )
                else:
                    print(
                        f"  [debug_chain] {tag}: no PHC trace produced "
                        f"(window not triggered?)",
                        flush=True,
                    )
            except Exception as e:
                import traceback
                print(
                    f"  [debug_chain] FAIL {tag}: {type(e).__name__}: {e}",
                    flush=True,
                )
                traceback.print_exc()

    print("\n[main] ===== summary =====", flush=True)
    for s in summary:
        print(f"  {s}", flush=True)
    _write_json_atomic(worker_summary_path, {
        "worker_idx": worker_idx,
        "num_workers": num_workers,
        "out_root": out_root,
        "attempts": attempts,
        "samples": summary,
        "num_ok": sum(item["ok"] for item in attempts),
        "num_fail": sum(not item["ok"] for item in attempts),
    })
    wandb.run.summary["num_ok"] = sum(1 for s in summary if s["ok"])
    wandb.run.summary["num_fail"] = sum(1 for s in summary if not s["ok"])

    run_id = wandb.run.id
    wandb.finish()
    print(f"[main] done. wandb: https://wandb.ai/{cfg.logger.wandb.entity}/"
          f"{cfg.logger.wandb.project}/runs/{run_id}", flush=True)


if __name__ == "__main__":
    main()
