"""Unified train + infer PHC debug visualization (N×5 chain).

Each row = one PHC integrator call (== one PHC window). 5 columns:
    1. col_left  motion (138d, mesh)  - train: clean vae (latent_pre_resample)
                                        infer: model-only generation
    2. model intent (mei138_pre_phc_decoded, mesh) - placed in window region;
       zero-velocity hold outside.
    3. PHC refs (24-body z-up → 22-joint y-up SMPL, skel) - integrator refs;
       hold reset pose outside.
    4. PHC rollout (mei138_phc_rollout, mesh; IFK already in capture) -
       PhysX-tracked motion, same window placement as col 2.
    5. col_right motion (138d, mesh) - train: after resample (latent_post_resample)
                                       infer: model+PHC generation

Train caller: models.diffusion_forcing_wan._step decodes the two batch-level
latent snapshots and groups per-sample windows from resampler._dbg_train.
Infer caller: generate_long_sequences.main runs PHC-off + PHC-on generation,
loads per-window npz dumps.
"""
import os
import json
import time

import numpy as np
import torch
import imageio
from PIL import Image


_FPS = 20
_FRAME_SCALE = 4   # 1 latent token = 4 motion frames @ 20fps
_BODY_COLOR = (0.78, 0.72, 0.64, 1.0)


# -----------------------------------------------------------------------------
# Train-only: latent → 138d via VAE
# -----------------------------------------------------------------------------

def _denorm_latent(feat_4d, model):
    """(C, T, 1, 1) latent in diff-norm space → (T, C) latent in VAE-input space."""
    lat = feat_4d[:, :, 0, 0].permute(1, 0)
    return lat * model.std + model.mean


@torch.no_grad()
def decode_latent_to_138(feat_4d, model, vae, device=None):
    """Train-side latent → (T_motion, 138) numpy.

    Force fp32 (autocast disabled) to match resampler's decode precision.
    """
    if vae is None:
        raise ValueError("decode_latent_to_138 requires a VAE")
    if device is None:
        device = feat_4d.device
    z = _denorm_latent(feat_4d, model).to(device).unsqueeze(0).float()
    with torch.amp.autocast(device_type='cuda', enabled=False):
        out = vae.decode(z)
    return out[0].float().cpu().numpy()


# -----------------------------------------------------------------------------
# Coordinate conversion: 24-body z-up MUJOCO → 22-joint y-up SMPL canonical
# -----------------------------------------------------------------------------

def _zup_mujoco_to_yup_smpl_22(motion_zup_mujoco, mujoco_2_smpl):
    """(..., 24, 3) z-up mujoco → (..., 22, 3) y-up SMPL.

    Bone reorder MUJOCO → SMPL canonical via mujoco_2_smpl, take first 22.
    Axis chain inverse: (x, y, z) → (-x, z, y).
    """
    m = np.asarray(motion_zup_mujoco)
    m22 = m[..., mujoco_2_smpl, :][..., :22, :]
    x = m22[..., 0]
    y = m22[..., 1]
    z = m22[..., 2]
    return np.stack([-x, z, y], axis=-1).astype(np.float32)


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def _phc_30_to_20_window(motion_30, frames_20, start_index=1):
    """Resample a 30fps PHC sequence to a `frames_20`-long 20fps window
    (nearest-neighbor, skips the reset frame at idx 0 by default)."""
    motion = np.asarray(motion_30, dtype=np.float32)
    if motion.ndim == 4:
        motion = motion[0]
    out = []
    for k in range(int(frames_20)):
        idx = int(start_index) + int(round(k * 30.0 / 20.0))
        idx = max(0, min(idx, motion.shape[0] - 1))
        out.append(motion[idx])
    return np.stack(out, axis=0)


# -----------------------------------------------------------------------------
# Frame post-process
# -----------------------------------------------------------------------------

def _apply_red_interval(frames, start, end, alpha=0.38):
    """Tint frames in [start, end) red. Half-open interval."""
    start = max(0, min(int(start), len(frames)))
    end = max(start, min(int(end), len(frames)))
    if end <= start:
        return frames
    red = np.array([255, 40, 40], dtype=np.float32)
    chunk = frames[start:end].astype(np.float32)
    frames[start:end] = ((1.0 - alpha) * chunk + alpha * red).clip(0, 255).astype(
        np.uint8
    )
    return frames


def _draw_label(frames, label):
    if not label:
        return frames
    import cv2
    for img in frames:
        cv2.rectangle(img, (8, 8), (8 + max(90, 13 * len(label)), 38),
                      (0, 0, 0), -1)
        cv2.putText(img, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frames


# -----------------------------------------------------------------------------
# Panel renderers
# -----------------------------------------------------------------------------

def _render_mesh_panel(motion_138, label, mesh_renderer,
                       red_intervals=None, image_size=480, fps=_FPS):
    """(T, 138) MEI → mesh + FK skeleton overlay. Returns (T, H, W, 3) uint8."""
    from methods.fd1.utils.render_mesh import render_mei138_mesh_frames
    rendered = render_mei138_mesh_frames(
        motion_138, fps=fps, renderer=mesh_renderer,
        mesh_color=_BODY_COLOR, image_size=image_size,
    )
    frames = np.stack(rendered, axis=0)
    if red_intervals:
        for (s, e) in red_intervals:
            _apply_red_interval(frames, s, e)
    return _draw_label(frames, label)


def _render_skel_panel(joints_22_yup, label, camera, red_intervals=None):
    """(T, 22, 3) y-up SMPL joints → matplotlib 2D stick figure frames."""
    from methods.fd1.utils.render_skeleton import get_humanml3d_chains, render_skeleton_frames
    frames = render_skeleton_frames(
        data=np.asarray(joints_22_yup, dtype=np.float32),
        chains=get_humanml3d_chains(),
        camera_params=camera,
    )
    if red_intervals:
        for (s, e) in red_intervals:
            _apply_red_interval(frames, s, e)
    return _draw_label(frames, label)


# -----------------------------------------------------------------------------
# Window timeline assembly (skel side: 24-body z-up; mesh side: 138d)
# -----------------------------------------------------------------------------

def _build_skel_timelines(windows, total_frames):
    """For each window build a refs 24-body z-up timeline that holds the
    window's own first / last frame pose outside the red interval (mirrors
    the per-window zero-velocity hold used by `_build_mei138_timelines` for
    cols 2 / 5, so col 3 doesn't snap back to a global reset pose).

    Returns: List[dict] with refs_timeline ((total_frames, 24, 3)) and
    red_interval.
    """
    out = []
    for w in windows:
        red_start = int(w["rs_start"]) * _FRAME_SCALE
        red_end = int(w["ctx_end"]) * _FRAME_SCALE
        n_target = red_end - red_start
        if n_target <= 0:
            continue
        refs = _phc_30_to_20_window(w["phc_refs"], n_target)
        end_clamp = min(red_end, total_frames)
        n_active = max(0, end_clamp - red_start)
        if n_active <= 0:
            continue
        rrefs = np.empty((total_frames, 24, 3), dtype=np.float32)
        rrefs[:red_start] = refs[0]                              # window-start pose
        rrefs[red_start:red_start + n_active] = refs[:n_active]
        rrefs[red_start + n_active:] = refs[n_active - 1]        # window-end pose
        out.append({
            "refs_timeline": rrefs,
            "red_interval": (red_start, red_start + n_active),
        })
    return out


def _build_mei138_timelines(windows, total_frames, key):
    """For each window build a (total_frames, 138) timeline placing the
    last `(ctx_end - rs_start) * 4` frames of window[key] inside the red
    interval. Outside the window, hold a zero-velocity version of the
    first / last window frame respectively (MEI138 indices 0:3 are root
    velocity — zeroing them freezes the root, otherwise the renderer
    integrates non-zero velocity → fixed pose drift).
    """
    out = []
    for w in windows:
        mei = np.asarray(w[key], dtype=np.float32)
        if mei.ndim == 3:
            mei = mei[0]
        if mei.shape[0] < 1:
            continue
        red_start = max(0, int(w["rs_start"]) * _FRAME_SCALE)
        red_end = min(int(w["ctx_end"]) * _FRAME_SCALE, total_frames)
        window_len = red_end - red_start
        if window_len <= 0:
            continue
        if mei.shape[0] >= window_len:
            window_data = mei[-window_len:].copy()
        else:
            pad = np.repeat(mei[-1:], window_len - mei.shape[0], axis=0)
            window_data = np.concatenate([mei, pad], axis=0)

        def _zero_vel(frame):
            f = frame.copy()
            if f.shape[-1] == 138:
                f[0:3] = 0.0
            return f
        hold_pre = _zero_vel(window_data[0])
        hold_post = _zero_vel(window_data[-1])
        timeline = np.zeros((total_frames, mei.shape[-1]), dtype=np.float32)
        if red_start > 0:
            timeline[:red_start] = hold_pre
        timeline[red_start:red_end] = window_data
        if red_end < total_frames:
            timeline[red_end:] = hold_post
        out.append({
            "motion_138": timeline,
            "red_interval": (red_start, red_end),
        })
    return out


# -----------------------------------------------------------------------------
# Grid writer
# -----------------------------------------------------------------------------

# libx264 rejects dimensions over 16384 px, and even a legal-but-tall grid
# encodes painfully slowly (a ~15000px clip took ~11 min, which blew past the
# NCCL collective timeout and aborted DDP training). Cap grid height low so the
# stacked chain stays small and fast to encode (~1-2 min); proportionally
# downscale the whole frame when the raw stack is taller.
_MAX_GRID_HEIGHT = 6000


def _write_grid(rows, out_mp4, fps=_FPS, max_height=_MAX_GRID_HEIGHT):
    """Write a 2D grid of (T, H, W, 3) panels to one mp4 (row major).

    When the stacked grid is taller than `max_height`, every frame is
    proportionally downscaled to a uniform safe size so the clip still encodes
    (keeping all rows, just smaller) instead of producing a 0-byte file."""
    n_frames = min(panel.shape[0] for row in rows for panel in row)
    writer = imageio.get_writer(
        out_mp4, fps=fps, codec="libx264", macro_block_size=1,
        ffmpeg_params=["-preset", "ultrafast", "-crf", "28", "-threads", "1"],
    )
    out_size = None        # (W, H) downscale target; None = keep native size
    size_resolved = False
    try:
        for t in range(n_frames):
            row_imgs = [np.concatenate([p[t] for p in row], axis=1) for row in rows]
            frame = np.concatenate(row_imgs, axis=0)
            if not size_resolved:
                size_resolved = True
                h, w = frame.shape[:2]
                if h > max_height:
                    scale = max_height / h
                    new_h = (int(h * scale) // 2) * 2   # even dims for yuv420p
                    new_w = (int(w * scale) // 2) * 2
                    out_size = (new_w, new_h)
                    print(f"[debug_viz] grid height {h}px > {max_height} "
                          f"({len(rows)} rows); downscaling frames to "
                          f"{new_w}x{new_h}", flush=True)
            if out_size is not None:
                frame = np.asarray(
                    Image.fromarray(frame).resize(out_size, Image.BILINEAR))
            writer.append_data(frame)
    finally:
        writer.close()


# -----------------------------------------------------------------------------
# Caption banner (optional, train-side has motion_id; infer leaves None)
# -----------------------------------------------------------------------------

def _load_motion_caption(text_dir, motion_id):
    """First-line caption from `<text_dir>/<motion_id>.txt` (HumanML3D format)."""
    if not text_dir or not motion_id:
        return None
    path = os.path.join(text_dir, f"{motion_id}.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            line = f.readline().strip()
    except Exception:
        return None
    if not line:
        return None
    return line.split("#", 1)[0].strip() or None


def _wrap_caption_lines(text, max_chars=80):
    """Greedy word-boundary wrap to <= max_chars per line."""
    lines = []
    words = text.split()
    cur = []
    cur_len = 0
    for word in words:
        add = len(word) + (1 if cur else 0)
        if cur_len + add > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [word]
            cur_len = len(word)
        else:
            cur.append(word)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def _add_caption_banner(in_mp4, out_mp4, caption, fps=20, banner_height=72,
                       max_chars=78, font_size=20):
    """Append a black banner under each frame carrying `caption`."""
    from PIL import Image, ImageDraw, ImageFont
    if not caption:
        return False
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    lines = _wrap_caption_lines(caption, max_chars=max_chars)
    line_h = font_size + 4
    needed_h = max(banner_height, line_h * len(lines) + 16)
    reader = imageio.get_reader(in_mp4)
    try:
        meta = reader.get_meta_data()
        try:
            in_fps = float(meta.get("fps", fps))
        except Exception:
            in_fps = fps
        writer = imageio.get_writer(
            out_mp4, fps=in_fps, codec="libx264", macro_block_size=1,
            ffmpeg_params=["-preset", "ultrafast", "-crf", "28", "-threads", "1",
                           "-pix_fmt", "yuv420p"],
        )
        try:
            for frame in reader:
                arr = np.asarray(frame, dtype=np.uint8)
                h, w = arr.shape[:2]
                banner = np.zeros((needed_h, w, 3), dtype=np.uint8)
                pil = Image.fromarray(banner)
                draw = ImageDraw.Draw(pil)
                y0 = (needed_h - line_h * len(lines)) // 2
                for k, line in enumerate(lines):
                    try:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        text_w = bbox[2] - bbox[0]
                    except Exception:
                        text_w = font_size * len(line) // 2
                    x = max(8, (w - text_w) // 2)
                    draw.text((x, y0 + k * line_h), line,
                              fill=(255, 255, 255), font=font)
                composed = np.concatenate([arr, np.asarray(pil)], axis=0)
                writer.append_data(composed)
        finally:
            writer.close()
    finally:
        reader.close()
    return True


# -----------------------------------------------------------------------------
# Infer trace npz → window dict
# -----------------------------------------------------------------------------

def load_window_from_npz(record):
    """Load one infer-side window dump as the same `window` dict shape
    consumed by `render_phc_chain`.

    `record` is one entry of resampler.pop_dbg_infer_records():
       {"window_idx", "dir", "arrays": <npz path>, "stats": <json path>, ...}
    """
    arrays = dict(np.load(record["arrays"]))
    with open(record["stats"], "r", encoding="utf-8") as f:
        stats = json.load(f)
    phc_init_state = {}
    for k, v in list(arrays.items()):
        if k.startswith("phc_init_state_"):
            phc_init_state[k[len("phc_init_state_"):]] = v
    if not phc_init_state:
        phc_init_state = None

    def _take0(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        return arr[0] if arr.ndim >= 3 else arr

    return {
        "sample_i": int(stats.get("chosen_index", 0)),
        "rs_start": int(stats["rs_start"]),
        "ctx_end": int(stats["ctx_end"]),
        "end_index": int(stats.get("end_index", stats["ctx_end"])),
        "n_replace": int(stats.get("n_replace", 1)),
        "phc_init_state": phc_init_state,
        "phc_refs": _take0(arrays.get("phc_refs")),
        "phc_post_reset_body_pos": _take0(arrays.get("phc_post_reset_body_pos")),
        "phc_rollout_body_pos": _take0(arrays.get("phc_rollout_body_pos")),
        "phc_rollout_body_rot": (
            _take0(arrays.get("phc_rollout_body_rot"))
            if "phc_rollout_body_rot" in arrays else None
        ),
        "mei138_pre_phc_decoded": _take0(arrays.get("mei138_pre_phc_decoded")),
        "mei138_phc_rollout": _take0(arrays.get("mei138_phc_rollout")),
        "mei138_post_phc_decoded": _take0(arrays.get("mei138_post_phc_decoded")),
        "stats": stats,
    }


# -----------------------------------------------------------------------------
# Main entry: render N×6 chain
# -----------------------------------------------------------------------------

def render_phc_chain(
    out_mp4,
    windows,
    col_left_motion_138,
    col_left_label,
    col_right_motion_138,
    col_right_label,
    mesh_renderer,
    caption=None,
    text_dir=None,
    motion_id=None,
    fps=_FPS,
    image_size=480,
    max_render_seconds=180,
):
    """Render N×6 PHC chain mp4. Returns dict with 'video' / 'rows' / 'total_frames'
    on success, or None when nothing to render.

    Args:
        out_mp4:                 output mp4 path (parent dirs created).
        windows:                 list of per-window dicts (see resampler
                                 `_end_phc_capture` train branch or
                                 `load_window_from_npz`).
        col_left_motion_138:     (T, 138) full-length context motion for col 1.
                                 train: clean vae; infer: model only.
        col_left_label:          panel title for col 1.
        col_right_motion_138:    (T, 138) full-length context motion for col 6.
                                 train: after resample; infer: model + PHC.
        col_right_label:         panel title for col 6.
        mesh_renderer:           shared SMPLMeshRenderer instance.
        caption:                 optional explicit caption string.
        text_dir + motion_id:    optional caption source (used iff caption None).
    """
    from physics.closd.utils.rep_util import mujoco_2_smpl
    from methods.fd1.utils.render_skeleton import compute_camera_params

    if not windows:
        return None

    col_left = np.asarray(col_left_motion_138, dtype=np.float32)
    col_right = np.asarray(col_right_motion_138, dtype=np.float32)
    # Crop col_left / col_right to the actually-trained range (clean prefix +
    # transition zone, = max end_index across windows × _FRAME_SCALE). For
    # infer, max end_index ≈ generation length so this is a no-op.
    crop_end_tokens = max(
        int(w.get("end_index", w["ctx_end"])) for w in windows
    )
    crop_end_frames = crop_end_tokens * _FRAME_SCALE
    total_frames = int(min(col_left.shape[0], col_right.shape[0],
                            crop_end_frames))
    if total_frames < 1:
        return None
    col_left = col_left[:total_frames]
    col_right = col_right[:total_frames]

    skel_per_window = _build_skel_timelines(windows, total_frames)
    intent_per_window = _build_mei138_timelines(
        windows, total_frames, "mei138_pre_phc_decoded")
    rollout_per_window = _build_mei138_timelines(
        windows, total_frames, "mei138_phc_rollout")
    n = min(len(skel_per_window), len(intent_per_window),
            len(rollout_per_window))
    if n == 0:
        return None
    skel_per_window = skel_per_window[:n]
    intent_per_window = intent_per_window[:n]
    rollout_per_window = rollout_per_window[:n]

    # Shared skel camera across all skel rows (single refs panel only).
    pool = np.concatenate(
        [w["refs_timeline"] for w in skel_per_window], axis=0,
    )
    pool_22 = _zup_mujoco_to_yup_smpl_22(pool, mujoco_2_smpl)
    skel_camera = compute_camera_params(pool_22)

    rows = []
    t_start = time.monotonic()
    for k in range(n):
        # Wall-clock budget: render time scales as rows × frames, so a near-full
        # (105-token) chain can take ~1h, blowing the NCCL collective timeout and
        # aborting DDP. Stop once over budget (keeping >=1 row) and write what we
        # have. Checked per-row so the cap holds regardless of sequence length.
        if rows and (time.monotonic() - t_start) > max_render_seconds:
            print(f"[debug_viz] render budget {max_render_seconds}s exceeded at "
                  f"row {len(rows)}/{n}; truncating chain", flush=True)
            break
        sw = skel_per_window[k]
        intent = intent_per_window[k]
        rollout = rollout_per_window[k]
        red_this = [sw["red_interval"]]
        rows.append([
            _render_mesh_panel(col_left, col_left_label, mesh_renderer,
                               red_intervals=red_this,
                               image_size=image_size, fps=fps),
            _render_mesh_panel(intent["motion_138"], "model intent",
                               mesh_renderer,
                               red_intervals=[intent["red_interval"]],
                               image_size=image_size, fps=fps),
            _render_skel_panel(
                _zup_mujoco_to_yup_smpl_22(sw["refs_timeline"], mujoco_2_smpl),
                "PHC refs", skel_camera, red_intervals=red_this),
            _render_mesh_panel(rollout["motion_138"], "PHC rollout",
                               mesh_renderer,
                               red_intervals=[rollout["red_interval"]],
                               image_size=image_size, fps=fps),
            _render_mesh_panel(col_right, col_right_label, mesh_renderer,
                               red_intervals=red_this,
                               image_size=image_size, fps=fps),
        ])

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    _write_grid(rows, out_mp4, fps=fps)

    if caption is None and text_dir and motion_id:
        caption = _load_motion_caption(text_dir, motion_id)
    if caption:
        tmp_mp4 = out_mp4 + ".__captioning.mp4"
        try:
            if _add_caption_banner(out_mp4, tmp_mp4, caption, fps=fps):
                os.replace(tmp_mp4, out_mp4)
        except Exception as e:
            print(f"[debug_viz] caption banner failed: {e}", flush=True)
        finally:
            if os.path.exists(tmp_mp4):
                try:
                    os.remove(tmp_mp4)
                except OSError:
                    pass

    return {"video": out_mp4, "rows": len(rows), "total_frames": total_frames}


# -----------------------------------------------------------------------------
# No-PHC: 1×2 before/after-resample motion comparison
# -----------------------------------------------------------------------------

def render_resample_diff(
    out_mp4,
    sample_range,
    col_left_motion_138,
    col_left_label,
    col_right_motion_138,
    col_right_label,
    mesh_renderer,
    caption=None,
    text_dir=None,
    motion_id=None,
    fps=_FPS,
    image_size=480,
):
    """Render a 1×2 before/after-resample motion comparison (no-PHC mode).

    For no-PHC training, there are no PHC windows — only the latent before
    and after a single resample call. This entry point produces a 2-panel
    mp4 with the replay region (`rs_start*4 .. end_index*4`) highlighted in
    red on both panels, cropped to `end_index*4` frames (the trained range).

    Args:
        out_mp4:                 output mp4 path (parent dirs created).
        sample_range:            dict {"rs_start", "ctx_end", "end_index"}
                                 (token indices; from `_dbg_train["resample_ranges"]`).
        col_left_motion_138:     (T, 138) clean-vae motion (pre-resample decode).
        col_left_label:          panel title for left col.
        col_right_motion_138:    (T, 138) post-resample motion.
        col_right_label:         panel title for right col.
        mesh_renderer:           shared SMPLMeshRenderer instance.
        caption / text_dir / motion_id / fps / image_size: same as render_phc_chain.
    """
    if sample_range is None:
        return None
    rs_start_f = int(sample_range["rs_start"]) * _FRAME_SCALE
    end_idx_f = int(sample_range["end_index"]) * _FRAME_SCALE
    col_left = np.asarray(col_left_motion_138, dtype=np.float32)
    col_right = np.asarray(col_right_motion_138, dtype=np.float32)
    total_frames = int(min(col_left.shape[0], col_right.shape[0], end_idx_f))
    if total_frames < 1:
        return None
    col_left = col_left[:total_frames]
    col_right = col_right[:total_frames]
    red = [(rs_start_f, min(end_idx_f, total_frames))]

    rows = [[
        _render_mesh_panel(col_left, col_left_label, mesh_renderer,
                           red_intervals=red, image_size=image_size, fps=fps),
        _render_mesh_panel(col_right, col_right_label, mesh_renderer,
                           red_intervals=red, image_size=image_size, fps=fps),
    ]]

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    _write_grid(rows, out_mp4, fps=fps)

    if caption is None and text_dir and motion_id:
        caption = _load_motion_caption(text_dir, motion_id)
    if caption:
        tmp_mp4 = out_mp4 + ".__captioning.mp4"
        try:
            if _add_caption_banner(out_mp4, tmp_mp4, caption, fps=fps):
                os.replace(tmp_mp4, out_mp4)
        except Exception as e:
            print(f"[debug_viz] caption banner failed: {e}", flush=True)
        finally:
            if os.path.exists(tmp_mp4):
                try:
                    os.remove(tmp_mp4)
                except OSError:
                    pass

    return {"video": out_mp4, "rows": 1, "total_frames": total_frames}
