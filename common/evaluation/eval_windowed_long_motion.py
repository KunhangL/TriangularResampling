#!/usr/bin/env python3
"""Build dataset prompt manifests and evaluate long motions by fixed windows."""

import argparse
import csv
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# ``python common/evaluation/eval_windowed_long_motion.py`` sets sys.path[0] to
# this directory; explicitly expose the repository root for local imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.metrics.t2m import T2MMetrics
from common.metrics.tools.utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
    calculate_jerk,
    calculate_kid,
)
from common.utils.initialize import Config
from common.utils.motion_process import recover_joint_positions_263


def convert_motion_138_to_263(*args, **kwargs):
    """138D (MEI) -> 263D transcoding is not part of this release."""
    raise NotImplementedError(
        "138D->263D conversion needs the SMPL-H FK / physics tree, which is not included; "
        "train and evaluate with 263D features.")


CACHE_VERSION = 5
REQUIRED_NPZ_KEYS = {
    "motion",
    "fps",
    "seed",
    "target_seconds",
    "prompt_id",
    "caption",
    "text_tokens",
    "method",
    "checkpoint",
}


def normalize_caption(caption):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", caption.casefold()).split())


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_manifest(path):
    records = []
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
            normalized = normalize_caption(caption)
            if prompt_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate prompt_id {prompt_id!r}")
            if normalized in seen_captions:
                raise ValueError(
                    f"{path}:{line_no}: duplicate normalized caption {caption!r}")
            seen_ids.add(prompt_id)
            seen_captions.add(normalized)
            records.append({
                "prompt_id": prompt_id,
                "caption": caption,
                "text_tokens": list(tokens),
                **{key: value for key, value in item.items()
                   if key not in {"prompt_id", "caption", "text_tokens"}},
            })
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _humanml_spec(config_path):
    cfg = Config(config_path).config
    # Training configs use both forms in this repository: the historical
    # multi-dataset wrapper (``data.datasets``) and a direct single
    # HumanML3DDataset (SEED_138). Normalize both without changing the legacy
    # selection behavior.
    nested_datasets = cfg.data.get("datasets", None)
    candidates = nested_datasets if nested_datasets is not None else [cfg.data]
    dataset_cfg = None
    for candidate in candidates:
        if str(candidate.target).endswith("HumanML3DDataset"):
            dataset_cfg = candidate
            break
    if dataset_cfg is None:
        raise ValueError(f"no HumanML3DDataset found in {config_path}")

    split_paths = [str(path) for path in dataset_cfg.val_meta_paths]
    feature_path = str(dataset_cfg.get(
        "val_feature_path", dataset_cfg.get("feature_path", "")))
    text_path = str(dataset_cfg.get("text_path", "texts"))
    if not feature_path:
        raise ValueError("HumanML3D validation feature_path is missing")
    t2m_cfg = cfg.metrics.t2m
    return cfg, dataset_cfg, split_paths, feature_path, text_path, t2m_cfg


def _babel_spec(config_path):
    cfg = Config(config_path).config
    nested_datasets = cfg.data.get("datasets", None)
    candidates = nested_datasets if nested_datasets is not None else [cfg.data]
    dataset_cfg = None
    for candidate in candidates:
        if str(candidate.target).endswith("BabelDataset"):
            dataset_cfg = candidate
            break
    if dataset_cfg is None:
        raise ValueError(f"no BabelDataset found in {config_path}")
    split_paths = []
    for key in ("train_meta_paths", "val_meta_paths", "test_meta_paths"):
        split_paths.extend(str(path) for path in dataset_cfg.get(key, []))
    roots = []
    for split_path in split_paths:
        root = Path(split_path).parent
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("BABEL dataset has no split roots")
    return {
        "roots": roots,
        "feature_path": str(dataset_cfg.get("feature_path", "new_joint_vecs")),
        "text_path": str(dataset_cfg.get("text_path", "texts")),
        "fps": int(dataset_cfg.get("feature_fps", 20)),
    }


def _active_segment(item, window_index, window_seconds):
    segments = item.get("segments")
    if not segments:
        raise ValueError(
            f"manifest item {item['prompt_id']!r} has no source segments")
    center_sec = window_index * window_seconds + window_seconds / 2.0
    for segment in segments:
        if float(segment["start_sec"]) <= center_sec < float(segment["end_sec"]):
            return segment
    raise ValueError(
        f"no active segment for {item['prompt_id']!r} at {center_sec}s")


def _locate_babel_source(spec, motion_id):
    matches = []
    for root in spec["roots"]:
        feature = root / spec["feature_path"] / f"{motion_id}.npy"
        text = root / spec["text_path"] / f"{motion_id}.txt"
        if feature.is_file() and text.is_file():
            matches.append((feature, text))
    if not matches:
        raise FileNotFoundError(
            f"BABEL source motion {motion_id!r} is unavailable under {spec['roots']}")
    unique = {(str(feature.resolve()), str(text.resolve())) for feature, text in matches}
    if len(unique) != 1:
        raise ValueError(f"BABEL source motion {motion_id!r} resolves ambiguously")
    feature, text = matches[0]
    return feature.resolve(), text.resolve()


def _source_annotation(text_path, segment):
    """Recover the first source annotation selected by the legacy builder."""
    wanted_caption = str(segment["caption"]).strip()
    wanted_tokens = list(segment["text_tokens"])
    with Path(text_path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("#")
            if len(fields) < 4:
                continue
            if fields[0].strip() != wanted_caption or fields[1].split() != wanted_tokens:
                continue
            start_sec = float(fields[2])
            end_sec = float(fields[3])
            if (not math.isfinite(start_sec) or not math.isfinite(end_sec)
                    or end_sec <= start_sec):
                continue
            return start_sec, end_sec, line_no
    raise ValueError(
        f"cannot recover BABEL annotation {wanted_caption!r} from {text_path}")


def collect_manifest_source_references(config_path, manifest,
                                       expected_windows, window_seconds,
                                       min_frames=40, max_frames=200):
    """Build paired, window-specific real BABEL references.

    The legacy stream manifest records the source motion but not annotation
    timestamps. The original builder selected the first globally unique
    caption/token occurrence, so the first exact source-text match recovers the
    selected annotation deterministically. Very short action annotations are
    expanded around their midpoint to the evaluator's 40-frame minimum.
    """
    spec = _babel_spec(config_path)
    fps = int(spec["fps"])
    references = []
    annotation_cache = {}
    for window_index in range(expected_windows):
        window_refs = []
        for item in manifest:
            segment = _active_segment(item, window_index, window_seconds)
            motion_id = str(segment["source_motion"])
            feature_path, text_path = _locate_babel_source(spec, motion_id)
            key = (str(text_path), str(segment["caption"]),
                   tuple(segment["text_tokens"]))
            if key not in annotation_cache:
                annotation_cache[key] = _source_annotation(text_path, segment)
            start_sec, end_sec, line_no = annotation_cache[key]
            with feature_path.open("rb") as handle:
                motion = np.load(handle, allow_pickle=False)
                if (motion.ndim != 2 or motion.shape[1] != 263
                        or len(motion) < 4
                        or not np.isfinite(motion).all()):
                    raise ValueError(
                        f"invalid BABEL 263D source motion: {feature_path} {motion.shape}")
                annotated_start = max(0, min(len(motion) - 1,
                    int(round(start_sec * fps))))
                annotated_end = max(annotated_start + 1, min(len(motion),
                    int(round(end_sec * fps))))
                desired = min(len(motion), min(max_frames, max(min_frames,
                    annotated_end - annotated_start)))
                center = (annotated_start + annotated_end) // 2
                crop_start = max(0, min(len(motion) - desired,
                    center - desired // 2))
                clip = np.array(
                    motion[crop_start:crop_start + desired],
                    dtype=np.float32, copy=True)
            window_refs.append({
                "prompt_id": item["prompt_id"],
                "motion_id": motion_id,
                "path": str(feature_path),
                "text_path": str(text_path),
                "text_line": int(line_no),
                "annotation_start_sec": float(start_sec),
                "annotation_end_sec": float(end_sec),
                "crop_start": int(crop_start),
                "crop_frames": int(len(clip)),
                "motion": clip,
            })
        references.append(window_refs)
    return references, spec


def _split_ids(split_paths):
    names = set()
    for split_path in split_paths:
        with open(split_path, "r", encoding="utf-8") as handle:
            names.update(line.strip() for line in handle if line.strip())
    return sorted(names)


def _full_motion_caption(text_file):
    with open(text_file, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("#")
            if len(fields) < 4:
                continue
            try:
                from_tag = float(fields[2])
                to_tag = float(fields[3])
            except ValueError:
                continue
            if (not math.isfinite(from_tag) or from_tag == 0.0) and (
                    not math.isfinite(to_tag) or to_tag == 0.0):
                tokens = [token for token in fields[1].split() if token]
                if fields[0].strip() and tokens:
                    return fields[0].strip(), tokens, line_no
    return None


def collect_reference_motions(config_path, window_frames=200):
    _, _, split_paths, feature_dir, _, _ = _humanml_spec(config_path)
    name_to_root = {}
    for split_path in split_paths:
        root = Path(split_path).parent
        with open(split_path, "r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if name:
                    if name in name_to_root and name_to_root[name] != root:
                        raise ValueError(f"duplicate motion id across roots: {name}")
                    name_to_root[name] = root

    references = []
    representation_dim = None
    for name in sorted(name_to_root):
        motion_path = name_to_root[name] / feature_dir / f"{name}.npy"
        if not motion_path.exists():
            continue
        with open(motion_path, "rb") as handle:
            motion = np.load(handle, allow_pickle=False)
        if motion.ndim == 2 and motion.shape[1] in (138, 263) and len(motion) > 0:
            if not np.isfinite(motion).all():
                print(f"skipping non-finite reference motion: {name}")
                continue
            current_dim = int(motion.shape[1])
            if representation_dim is None:
                representation_dim = current_dim
            elif current_dim != representation_dim:
                raise ValueError(
                    "reference bank mixes motion representations: "
                    f"{representation_dim} and {current_dim}")
            clip_frames = min(len(motion), int(window_frames))
            start = (len(motion) - clip_frames) // 2
            references.append((name, str(motion_path), start, clip_frames))
    if len(references) < 2:
        raise ValueError(
            f"only {len(references)} valid reference motions found")
    return references, split_paths, feature_dir, representation_dim


def build_manifest(config_path, output, limit=512, window_frames=200,
                   sample_seed=None):
    _, _, split_paths, feature_dir, text_dir, _ = _humanml_spec(config_path)
    name_to_root = {}
    for split_path in split_paths:
        root = Path(split_path).parent
        with open(split_path, "r", encoding="utf-8") as handle:
            for line in handle:
                name = line.strip()
                if name:
                    name_to_root[name] = root

    prompts = []
    seen_captions = set()
    for name in sorted(name_to_root):
        root = name_to_root[name]
        motion_path = root / feature_dir / f"{name}.npy"
        text_path = root / text_dir / f"{name}.txt"
        if not motion_path.exists() or not text_path.exists():
            continue
        motion = np.load(motion_path, mmap_mode="r")
        if motion.ndim != 2 or motion.shape[1] not in (138, 263) or len(motion) == 0:
            continue
        full_caption = _full_motion_caption(text_path)
        if full_caption is None:
            continue
        caption, tokens, _ = full_caption
        normalized = normalize_caption(caption)
        if normalized in seen_captions:
            continue
        seen_captions.add(normalized)
        prompts.append({
            "prompt_id": name,
            "caption": caption,
            "text_tokens": tokens,
            "source_motion": str(motion_path),
            "source_window_start": max(0, (len(motion) - window_frames) // 2),
            "source_window_frames": min(len(motion), int(window_frames)),
        })
    if sample_seed is not None:
        rng = np.random.default_rng(int(sample_seed))
        order = rng.permutation(len(prompts))
        prompts = [prompts[index] for index in order]
    if limit:
        prompts = prompts[:int(limit)]

    if not prompts:
        raise ValueError("no eligible unique HumanML3D prompts found")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    os.replace(tmp, output)
    # Re-read through the strict validator before declaring success.
    load_manifest(output)
    print(
        f"wrote {len(prompts)} unique prompts to {output} "
        f"from {len(split_paths)} validation split file(s); "
        f"sample_seed={sample_seed}",
        flush=True,
    )
    return prompts


def slice_motion_windows(motion, window_frames=200, expected_windows=None):
    motion = np.asarray(motion)
    if motion.ndim != 2 or motion.shape[1] not in (138, 263):
        raise ValueError(
            f"motion must have shape (T, 138) or (T, 263), got {motion.shape}")
    n_full, tail = divmod(len(motion), int(window_frames))
    if tail:
        warnings.warn(
            f"skipping partial tail of {tail} frame(s)", RuntimeWarning,
            stacklevel=2,
        )
    if expected_windows is not None:
        if n_full < expected_windows:
            warnings.warn(
                f"only {n_full}/{expected_windows} complete windows available",
                RuntimeWarning,
                stacklevel=2,
            )
        elif n_full > expected_windows:
            warnings.warn(
                f"ignoring {n_full - expected_windows} complete extra window(s)",
                RuntimeWarning,
                stacklevel=2,
            )
        n_full = min(n_full, int(expected_windows))
    return [
        np.asarray(motion[i * window_frames:(i + 1) * window_frames], dtype=np.float32)
        for i in range(n_full)
    ]


def load_generated_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_NPZ_KEYS - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing NPZ keys {sorted(missing)}")
        record = {
            "path": str(path),
            "motion": np.asarray(archive["motion"], dtype=np.float32),
            "fps": int(np.asarray(archive["fps"]).item()),
            "seed": int(np.asarray(archive["seed"]).item()),
            "target_seconds": int(np.asarray(archive["target_seconds"]).item()),
            "prompt_id": str(np.asarray(archive["prompt_id"]).item()),
            "caption": str(np.asarray(archive["caption"]).item()),
            "text_tokens": [str(x) for x in np.asarray(archive["text_tokens"]).tolist()],
            "method": str(np.asarray(archive["method"]).item()),
            "checkpoint": str(np.asarray(archive["checkpoint"]).item()),
            "representation_dim": int(np.asarray(
                archive["representation_dim"]
                if "representation_dim" in archive.files
                else archive["motion"].shape[1]).item()),
        }
    if (record["motion"].ndim != 2
            or record["motion"].shape[1] not in (138, 263)):
        raise ValueError(
            f"{path}: motion shape is {record['motion'].shape}, "
            "expected (T, 138) or (T, 263)")
    if record["representation_dim"] != record["motion"].shape[1]:
        raise ValueError(
            f"{path}: representation_dim={record['representation_dim']} "
            f"does not match motion shape {record['motion'].shape}")
    if not np.isfinite(record["motion"]).all():
        raise ValueError(f"{path}: motion contains non-finite values")
    return record


def frechet_distance(real_embeddings, generated_embeddings):
    real_stats = calculate_activation_statistics_np(np.asarray(real_embeddings))
    gen_stats = calculate_activation_statistics_np(np.asarray(generated_embeddings))
    value = float(calculate_frechet_distance_np(*real_stats, *gen_stats))
    return 0.0 if abs(value) < 1e-9 else max(0.0, value)


def r_precision_at_k(text_embeddings, motion_embeddings, top_k=3,
                     group_size=32, permutation=None):
    text_embeddings = np.asarray(text_embeddings, dtype=np.float64)
    motion_embeddings = np.asarray(motion_embeddings, dtype=np.float64)
    if text_embeddings.shape != motion_embeddings.shape or text_embeddings.ndim != 2:
        raise ValueError("paired text/motion embeddings must have matching N x D shape")
    n = len(text_embeddings)
    if n < 2:
        raise ValueError("R-precision requires at least two prompt-motion pairs")
    if permutation is None:
        permutation = np.arange(n)
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(n)):
        raise ValueError("permutation must contain each pair index exactly once")

    official = n >= group_size
    effective_group = group_size if official else n
    n_used = (n // effective_group) * effective_group
    correct = 0
    for start in range(0, n_used, effective_group):
        idx = permutation[start:start + effective_group]
        texts = text_embeddings[idx]
        motions = motion_embeddings[idx]
        squared = (
            np.sum(texts * texts, axis=1, keepdims=True)
            - 2.0 * texts.dot(motions.T)
            + np.sum(motions * motions, axis=1)[None, :]
        )
        ranking = np.argsort(np.maximum(squared, 0.0), axis=1)
        k = min(int(top_k), effective_group)
        correct += int(sum(row in ranking[row, :k] for row in range(effective_group)))
    return correct / n_used, n_used, official


def summarize_curve(times_s, values, r_precision=False):
    times_s = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times_s) < 2 or len(times_s) != len(values):
        raise ValueError("curve summary requires at least two aligned points")
    span = float(times_s[-1] - times_s[0])
    if span <= 0:
        raise ValueError("curve timestamps must be increasing")
    slope_per_second = float(np.polyfit(times_s, values, deg=1)[0])
    summary = {
        "first": float(values[0]),
        "last": float(values[-1]),
        "delta": float(values[-1] - values[0]),
        "normalized_auc": float(np.trapz(values, times_s) / span),
    }
    if r_precision:
        summary["slope_percentage_points_per_minute"] = slope_per_second * 6000.0
    else:
        summary["slope_per_minute"] = slope_per_second * 60.0
    return summary


def _bootstrap_curve_summary(times_s, curves, r_precision=False):
    """Vectorized counterpart of :func:`summarize_curve` for B curves."""
    times_s = np.asarray(times_s, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[1] != len(times_s):
        raise ValueError("bootstrap curves must have shape (B, num_windows)")
    centered_t = times_s - times_s.mean()
    slopes = (curves @ centered_t) / float(np.dot(centered_t, centered_t))
    scale = 6000.0 if r_precision else 60.0
    return {
        "first": curves[:, 0],
        "last": curves[:, -1],
        "delta": curves[:, -1] - curves[:, 0],
        "normalized_auc": np.trapz(curves, times_s, axis=1)
        / float(times_s[-1] - times_s[0]),
        ("slope_percentage_points_per_minute" if r_precision
         else "slope_per_minute"): slopes * scale,
    }


def _percentile_intervals(values_by_name, confidence):
    tail = (1.0 - float(confidence)) * 50.0
    return {
        name: {
            "lower": float(np.percentile(values, tail)),
            "upper": float(np.percentile(values, 100.0 - tail)),
        }
        for name, values in values_by_name.items()
    }


def _rprecision_prompt_indicators(text, motion, permutation, group_size=32,
                                  top_k=3):
    """Return one fixed-retrieval correctness indicator per prompt."""
    text = np.asarray(text, dtype=np.float64)
    motion = np.asarray(motion, dtype=np.float64)
    permutation = np.asarray(permutation, dtype=np.int64)
    n = len(text)
    effective_group = group_size if n >= group_size else n
    n_used = (n // effective_group) * effective_group
    indicators = np.full(n, np.nan, dtype=np.float64)
    for start in range(0, n_used, effective_group):
        idx = permutation[start:start + effective_group]
        texts = text[idx]
        motions = motion[idx]
        squared = (
            np.sum(texts * texts, axis=1, keepdims=True)
            - 2.0 * texts.dot(motions.T)
            + np.sum(motions * motions, axis=1)[None, :]
        )
        ranking = np.argsort(np.maximum(squared, 0.0), axis=1)
        k = min(int(top_k), effective_group)
        local = np.asarray(
            [row in ranking[row, :k] for row in range(effective_group)],
            dtype=np.float64,
        )
        indicators[idx] = local
    if np.isnan(indicators).any():
        raise ValueError("bootstrap R-precision requires every prompt to be evaluated")
    return indicators


def _batched_bootstrap_fid(reference_embeddings, cluster_embeddings, weights,
                           device, chunk_size=32):
    """Cluster-bootstrap FID while keeping the reference bank fixed.

    ``cluster_embeddings`` is P x S x D.  A bootstrap draw resamples the P
    prompts and keeps all S generations belonging to each selected prompt.
    """
    reference = np.asarray(reference_embeddings, dtype=np.float32)
    clusters = np.asarray(cluster_embeddings, dtype=np.float32)
    if clusters.ndim != 3:
        raise ValueError("cluster embeddings must have shape (P, S, D)")
    p, s, d = clusters.shape
    if weights.shape[1] != p:
        raise ValueError("bootstrap weight width does not match prompt count")

    torch_device = torch.device(device)
    ref = torch.from_numpy(reference).to(torch_device)
    ref_mean = ref.mean(dim=0)
    ref_centered = ref - ref_mean
    ref_cov = ref_centered.T @ ref_centered / float(len(ref) - 1)
    ref_cov = (ref_cov + ref_cov.T) * 0.5
    evals, evecs = torch.linalg.eigh(ref_cov)
    ref_sqrt = (evecs * evals.clamp_min(0).sqrt().unsqueeze(0)) @ evecs.T
    ref_trace = torch.trace(ref_cov)

    cluster = torch.from_numpy(clusters).to(torch_device)
    cluster_sum = cluster.sum(dim=1)
    cluster_outer = torch.einsum("psd,pse->pde", cluster, cluster)
    cluster_outer = cluster_outer.reshape(p, d * d)
    n = float(p * s)

    outputs = []
    for start in range(0, len(weights), chunk_size):
        weight = torch.from_numpy(
            np.asarray(weights[start:start + chunk_size], dtype=np.float32)
        ).to(torch_device)
        sums = weight @ cluster_sum
        means = sums / n
        seconds = (weight @ cluster_outer).reshape(-1, d, d)
        cov = (seconds - n * means.unsqueeze(2) * means.unsqueeze(1)) / (n - 1.0)
        cov = (cov + cov.transpose(1, 2)) * 0.5
        middle = ref_sqrt.unsqueeze(0) @ cov @ ref_sqrt.unsqueeze(0)
        middle = (middle + middle.transpose(1, 2)) * 0.5
        middle_evals = torch.linalg.eigvalsh(middle).clamp_min(0)
        fid = (
            ((means - ref_mean.unsqueeze(0)) ** 2).sum(dim=1)
            + ref_trace + cov.diagonal(dim1=1, dim2=2).sum(dim=1)
            - 2.0 * middle_evals.sqrt().sum(dim=1)
        ).clamp_min(0)
        outputs.append(fid.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _batched_paired_bootstrap_fid(reference_embeddings, cluster_embeddings,
                                  weights, device, chunk_size=32):
    """Paired prompt bootstrap of both real and generated distributions."""
    reference = np.asarray(reference_embeddings, dtype=np.float32)
    clusters = np.asarray(cluster_embeddings, dtype=np.float32)
    if reference.ndim != 2 or clusters.ndim != 3:
        raise ValueError("paired FID expects P x D references and P x S x D generations")
    p, s, d = clusters.shape
    if reference.shape != (p, d) or weights.shape[1] != p:
        raise ValueError("paired FID reference/generation/weight shapes do not align")
    torch_device = torch.device(device)
    real = torch.from_numpy(reference).to(torch_device)
    gen = torch.from_numpy(clusters).to(torch_device)
    real_outer = torch.einsum("pd,pe->pde", real, real).reshape(p, d * d)
    gen_sum = gen.sum(dim=1)
    gen_outer = torch.einsum("psd,pse->pde", gen, gen).reshape(p, d * d)
    outputs = []
    for start in range(0, len(weights), chunk_size):
        weight = torch.from_numpy(np.asarray(
            weights[start:start + chunk_size], dtype=np.float32)).to(torch_device)
        real_n = float(p)
        gen_n = float(p * s)
        real_mean = weight @ real / real_n
        gen_mean = weight @ gen_sum / gen_n
        real_second = (weight @ real_outer).reshape(-1, d, d)
        gen_second = (weight @ gen_outer).reshape(-1, d, d)
        real_cov = (
            real_second - real_n * real_mean.unsqueeze(2) * real_mean.unsqueeze(1)
        ) / (real_n - 1.0)
        gen_cov = (
            gen_second - gen_n * gen_mean.unsqueeze(2) * gen_mean.unsqueeze(1)
        ) / (gen_n - 1.0)
        real_cov = (real_cov + real_cov.transpose(1, 2)) * 0.5
        gen_cov = (gen_cov + gen_cov.transpose(1, 2)) * 0.5
        real_evals, real_evecs = torch.linalg.eigh(real_cov)
        real_sqrt = (
            real_evecs * real_evals.clamp_min(0).sqrt().unsqueeze(1)
        ) @ real_evecs.transpose(1, 2)
        middle = real_sqrt @ gen_cov @ real_sqrt
        middle = (middle + middle.transpose(1, 2)) * 0.5
        middle_evals = torch.linalg.eigvalsh(middle).clamp_min(0)
        fid = (
            ((real_mean - gen_mean) ** 2).sum(dim=1)
            + real_cov.diagonal(dim1=1, dim2=2).sum(dim=1)
            + gen_cov.diagonal(dim1=1, dim2=2).sum(dim=1)
            - 2.0 * middle_evals.sqrt().sum(dim=1)
        ).clamp_min(0)
        outputs.append(fid.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


def _paired_prompt_bootstrap(reference_embeddings, records, methods,
                             manifest, prompt_order, seeds_by_method,
                             generated_embeddings, text_embeddings,
                             expected_windows, times_s, metric_seed,
                             replicates, confidence, device):
    """Paired cluster bootstrap over prompts/streams, grouping all seeds."""
    prompt_ids = [item["prompt_id"] for item in manifest]
    p = len(prompt_ids)
    rng = np.random.default_rng(metric_seed + 9173)
    sampled = rng.integers(0, p, size=(replicates, p), endpoint=False)
    weights = np.zeros((replicates, p), dtype=np.int16)
    rows = np.arange(replicates)[:, None]
    np.add.at(weights, (rows, sampled), 1)
    weight_float = weights.astype(np.float64)

    output = {}
    for method in methods:
        seeds = seeds_by_method[method]
        fid_curves = np.empty((replicates, expected_windows), dtype=np.float64)
        mm_curves = np.empty_like(fid_curves)
        r_curves = np.empty_like(fid_curves)
        for window_index in range(expected_windows):
            clusters = np.stack([
                np.stack([
                    generated_embeddings[(method, seed, prompt_id, window_index)]
                    for seed in seeds
                ], axis=0)
                for prompt_id in prompt_ids
            ], axis=0)
            if np.asarray(reference_embeddings).ndim == 2:
                fid_curves[:, window_index] = _batched_bootstrap_fid(
                    reference_embeddings, clusters, weights, device=device)
            else:
                fid_curves[:, window_index] = _batched_paired_bootstrap_fid(
                    np.asarray(reference_embeddings)[window_index], clusters,
                    weights, device=device)

            texts = np.stack([
                text_embeddings[(prompt_id, window_index)]
                for prompt_id in prompt_ids
            ], axis=0)
            per_prompt_mm = np.linalg.norm(
                clusters - texts[:, None, :], axis=2).mean(axis=1)
            mm_curves[:, window_index] = weight_float @ per_prompt_mm / float(p)

            r_by_seed = []
            for seed_index, seed in enumerate(seeds):
                permutation = np.random.default_rng(
                    metric_seed + int(seed)).permutation(p)
                r_by_seed.append(_rprecision_prompt_indicators(
                    texts, clusters[:, seed_index, :], permutation,
                    group_size=32, top_k=3))
            per_prompt_r = np.stack(r_by_seed, axis=1).mean(axis=1)
            r_curves[:, window_index] = weight_float @ per_prompt_r / float(p)

        output[method] = {
            "unit": "prompt_or_stream_with_all_generation_seeds",
            "fid": _percentile_intervals(
                _bootstrap_curve_summary(times_s, fid_curves), confidence),
            "mm_dist": _percentile_intervals(
                _bootstrap_curve_summary(times_s, mm_curves), confidence),
            "r_at_3": _percentile_intervals(
                _bootstrap_curve_summary(times_s, r_curves, r_precision=True),
                confidence),
        }
    return output


def _file_identity(path):
    path = Path(str(path)).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _reference_cache_metadata(config_path, split_paths, feature_dir,
                              t2m_cfg, window_frames, conversion_device,
                              representation_dim):
    return {
        "cache_version": CACHE_VERSION,
        "config": str(Path(config_path).resolve()),
        "split_files": [_file_identity(path) for path in split_paths],
        "feature_path": feature_dir,
        "representation_dim": int(representation_dim),
        "window_frames": int(window_frames),
        "window_selection": "full_motion_or_deterministic_center_crop",
        "conversion_device": str(conversion_device),
        "metric_mean": _file_identity(t2m_cfg.metric_mean_path),
        "metric_std": _file_identity(t2m_cfg.metric_std_path),
        "movement_encoder": _file_identity(t2m_cfg.moveencoder.ckpt),
        "motion_encoder": _file_identity(t2m_cfg.motionencoder.ckpt),
    }


def _encode_motion_batch(motions, evaluator, device, conversion_device,
                         fps, batch_size=32):
    converted = []
    for index, motion in enumerate(motions):
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] not in (138, 263):
            raise ValueError(f"motion {index} has unsupported shape {motion.shape}")
        if motion.shape[1] == 138:
            motion_263 = convert_motion_138_to_263(
                motion, fps=fps, device=str(conversion_device))
            if motion_263.shape != (motion.shape[0] - 1, 263):
                raise ValueError(
                    f"138->263 conversion {index} returned {motion_263.shape}")
        else:
            motion_263 = motion
        if not np.isfinite(motion_263).all():
            raise ValueError(f"motion {index} contains non-finite values")
        converted.append(np.asarray(motion_263, dtype=np.float32))

    embeddings = []
    for start in range(0, len(converted), batch_size):
        items = converted[start:start + batch_size]
        lengths = np.asarray([len(item) for item in items], dtype=np.int64)
        order = np.argsort(-lengths, kind="stable")
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        items = [items[index] for index in order]
        lengths = lengths[order].tolist()
        max_length = max(lengths)
        batch = np.zeros(
            (len(items), max_length, items[0].shape[1]), dtype=np.float32)
        for index, item in enumerate(items):
            batch[index, :len(item)] = item
        tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            encoded = evaluator.get_motion_embeddings(tensor, lengths)
        embeddings.append(encoded.float().cpu().numpy()[inverse_order])
    return np.concatenate(embeddings, axis=0)


def _encode_text_batch(token_lists, evaluator, batch_size=64):
    embeddings = []
    for start in range(0, len(token_lists), batch_size):
        encoded = evaluator.get_text_embeddings(token_lists[start:start + batch_size])
        embeddings.append(encoded.float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _load_or_build_reference_cache(cache_path, metadata, references, evaluator,
                                   device, conversion_device, fps, batch_size):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_metadata = json.loads(str(archive["metadata_json"].item()))
            if cached_metadata != metadata:
                raise RuntimeError(
                    f"reference cache metadata mismatch: {cache_path}; "
                    "use a fresh output directory or remove this cache")
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            ids = [str(value) for value in archive["motion_ids"].tolist()]
        if len(embeddings) != len(references):
            raise RuntimeError(f"reference cache count mismatch: {cache_path}")
        print(f"loaded {len(embeddings)} reference embeddings from {cache_path}")
        return embeddings, ids, True

    window_frames = int(metadata["window_frames"])
    representation_dim = int(metadata["representation_dim"])
    motions = []
    for name, path, start, clip_frames in references:
        # Open each reference only for the duration of this read. Keeping
        # memmap-backed slices in ``motions`` also keeps one file descriptor
        # alive per reference and exhausts the default 1024-descriptor limit
        # on full long-horizon evaluation sets.
        with open(path, "rb") as handle:
            motion = np.load(handle, allow_pickle=False)
            clip = np.array(
                motion[start:start + clip_frames], dtype=np.float32, copy=True)
        if clip.shape != (clip_frames, representation_dim):
            raise ValueError(
                f"reference {name!r} center crop returned {clip.shape}")
        motions.append(clip)
    embeddings = _encode_motion_batch(
        motions, evaluator, device=device, conversion_device=conversion_device,
        fps=fps, batch_size=batch_size)
    ids = [
        f"{name}#start={start}#frames={clip_frames}"
        for name, _, start, clip_frames in references
    ]
    _atomic_npz(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        motion_ids=np.asarray(ids, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"cached {len(embeddings)} reference embeddings at {cache_path}")
    return embeddings, ids, False


def _manifest_reference_cache_metadata(config_path, manifest_path, spec,
                                       expected_windows, window_seconds,
                                       min_frames, max_frames, t2m_cfg):
    return {
        "cache_version": CACHE_VERSION,
        "reference_mode": "manifest_sources",
        "config": str(Path(config_path).resolve()),
        "manifest": _file_identity(manifest_path),
        "roots": [str(path.resolve()) for path in spec["roots"]],
        "feature_path": spec["feature_path"],
        "text_path": spec["text_path"],
        "fps": int(spec["fps"]),
        "expected_windows": int(expected_windows),
        "window_seconds": float(window_seconds),
        "min_reference_frames": int(min_frames),
        "max_reference_frames": int(max_frames),
        "annotation_selection": "first_exact_caption_and_token_match",
        "short_annotation_policy": "midpoint_expand_to_min_or_full_motion",
        "metric_mean": _file_identity(t2m_cfg.metric_mean_path),
        "metric_std": _file_identity(t2m_cfg.metric_std_path),
        "movement_encoder": _file_identity(t2m_cfg.moveencoder.ckpt),
        "motion_encoder": _file_identity(t2m_cfg.motionencoder.ckpt),
    }


def _load_or_build_manifest_reference_cache(cache_path, metadata, references,
                                            evaluator, device, batch_size):
    cache_path = Path(cache_path)
    expected_shape = (len(references), len(references[0]))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_metadata = json.loads(str(archive["metadata_json"].item()))
            if cached_metadata != metadata:
                raise RuntimeError(
                    f"manifest reference cache metadata mismatch: {cache_path}")
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            ids = np.asarray(archive["reference_ids"], dtype=np.str_)
        if embeddings.shape[:2] != expected_shape or ids.shape != expected_shape:
            raise RuntimeError(f"manifest reference cache shape mismatch: {cache_path}")
        print(f"loaded paired BABEL references from {cache_path}")
        return embeddings, ids.tolist(), True

    unique = {}
    for window_refs in references:
        for ref in window_refs:
            key = (ref["path"], ref["crop_start"], ref["crop_frames"])
            unique.setdefault(key, ref["motion"])
    unique_keys = sorted(unique)
    unique_embeddings = _encode_motion_batch(
        [unique[key] for key in unique_keys], evaluator,
        device=device, conversion_device="cpu", fps=int(metadata["fps"]),
        batch_size=batch_size)
    embedding_by_key = dict(zip(unique_keys, unique_embeddings))
    embeddings = np.stack([
        np.stack([
            embedding_by_key[(ref["path"], ref["crop_start"], ref["crop_frames"])]
            for ref in window_refs
        ], axis=0)
        for window_refs in references
    ], axis=0).astype(np.float32)
    ids = np.asarray([
        [
            f"{ref['motion_id']}#line={ref['text_line']}#start={ref['crop_start']}"
            f"#frames={ref['crop_frames']}"
            for ref in window_refs
        ]
        for window_refs in references
    ], dtype=np.str_)
    _atomic_npz(
        cache_path,
        embeddings=embeddings,
        reference_ids=ids,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"cached {len(unique)} unique BABEL source clips as "
        f"{embeddings.shape[0]}x{embeddings.shape[1]} paired references at {cache_path}")
    return embeddings, ids.tolist(), False


def _discover_generated(generated_root, manifest_by_id, fps, target_seconds,
                        expected_windows, window_frames):
    records = {}
    methods = set()
    checkpoints = defaultdict(set)
    for path in sorted(Path(generated_root).rglob("*.npz")):
        try:
            record = load_generated_npz(path)
        except ValueError as exc:
            # Ignore unrelated caches; malformed files in the generation tree
            # that look like metric artifacts still fail loudly.
            if "missing NPZ keys" in str(exc):
                continue
            raise
        prompt_id = record["prompt_id"]
        if prompt_id not in manifest_by_id:
            raise ValueError(f"{path}: prompt_id {prompt_id!r} is not in manifest")
        expected = manifest_by_id[prompt_id]
        if record["caption"] != expected["caption"]:
            raise ValueError(f"{path}: caption does not match manifest")
        if record["text_tokens"] != expected["text_tokens"]:
            raise ValueError(f"{path}: text_tokens do not match manifest")
        if record["fps"] != fps or record["target_seconds"] != target_seconds:
            raise ValueError(
                f"{path}: expected {target_seconds}s@{fps}fps, got "
                f"{record['target_seconds']}s@{record['fps']}fps")
        record["windows"] = slice_motion_windows(
            record["motion"], window_frames=window_frames,
            expected_windows=expected_windows)
        key = (record["method"], record["seed"], prompt_id)
        if key in records:
            raise ValueError(f"duplicate generated sample for {key}: {path}")
        records[key] = record
        methods.add(record["method"])
        checkpoints[record["method"]].add(record["checkpoint"])
    if not records:
        raise ValueError(f"no metric NPZ files found under {generated_root}")
    for method, values in checkpoints.items():
        if len(values) != 1:
            raise ValueError(f"method {method!r} maps to multiple checkpoints: {values}")
    return records, sorted(methods), {
        method: next(iter(values)) for method, values in checkpoints.items()
    }


def _write_curve_csv(path, rows, include_kid):
    fields = [
        "method", "window_index", "window_start_s", "window_end_s",
        "window_center_s", "fid", "mm_dist",
        "r_at_3", "n_generated", "n_prompts", "n_seeds",
        "r3_n_evaluated", "r3_official",
    ]
    if include_kid:
        fields[6:6] = ["kid_mean", "kid_std"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(path, rows, methods, include_kid):
    specs = [
        ("fid", "Windowed FID"),
        ("mm_dist", "Windowed MM-Dist"),
        ("r_at_3", "Windowed R@3"),
    ]
    if include_kid:
        specs.insert(1, ("kid_mean", "Windowed KID"))
    fig, axes = plt.subplots(
        1, len(specs), figsize=(4 * len(specs), 4), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (key, title) in zip(axes, specs):
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            axis.plot(
                [row["window_center_s"] for row in method_rows],
                [row[key] for row in method_rows],
                marker="o", linewidth=1.8, markersize=4, label=method,
            )
            if key == "kid_mean":
                values = np.asarray([row[key] for row in method_rows])
                stds = np.asarray([row["kid_std"] for row in method_rows])
                xs = np.asarray([row["window_center_s"] for row in method_rows])
                axis.fill_between(xs, values - stds, values + stds, alpha=0.12)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Window center (s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _reference_transition_jerk_curves(manifest, manifest_references, fps,
                                      transition_frames=30):
    """Derive one real BABEL jerk curve per stream from its source actions."""
    if manifest_references is None or not any(item.get("segments") for item in manifest):
        return None, None
    curves = []
    provenance = []
    for prompt_index, item in enumerate(manifest):
        prompt_clips = []
        prompt_sources = []
        # A 20-second segment occupies two 10-second metric windows. At each
        # generated transition, use the center of the newly active real BABEL
        # source action as the paired natural-motion reference.
        for segment_index in range(1, len(item["segments"])):
            window_index = segment_index * 2
            ref = manifest_references[window_index][prompt_index]
            motion = np.asarray(ref["motion"], dtype=np.float32)
            if motion.shape[1] != 263:
                raise ValueError("BABEL reference jerk requires 263D source motion")
            if len(motion) < transition_frames:
                pad_left = (transition_frames - len(motion)) // 2
                pad_right = transition_frames - len(motion) - pad_left
                motion = np.pad(motion, ((pad_left, pad_right), (0, 0)), mode="edge")
            start = (len(motion) - transition_frames) // 2
            clip = motion[start:start + transition_frames]
            prompt_clips.append(recover_joint_positions_263(clip, 22))
            prompt_sources.append({
                "motion_id": ref["motion_id"],
                "text_line": ref["text_line"],
                "crop_start": ref["crop_start"],
                "crop_frames": ref["crop_frames"],
            })
        stacked = np.stack(prompt_clips, axis=0)
        lengths = np.full(len(stacked), transition_frames, dtype=np.int64)
        curves.append(calculate_jerk(stacked, lengths).mean(axis=0))
        provenance.append({"prompt_id": item["prompt_id"], "sources": prompt_sources})
    return np.stack(curves, axis=0).astype(np.float32), provenance


def _generated_transition_jerk_artifacts(records, manifest, methods,
                                          seeds_by_method, fps,
                                          conversion_device,
                                          transition_frames=30):
    if not any(item.get("segments") for item in manifest):
        return None, None
    left = transition_frames // 2
    right = transition_frames - left
    keys = []
    curves = []
    for method in methods:
        for seed in seeds_by_method[method]:
            for item in manifest:
                motion = records[(method, seed, item["prompt_id"])]["motion"]
                if motion.shape[1] == 138:
                    motion = convert_motion_138_to_263(
                        motion, fps=fps, device=str(conversion_device))
                joints = recover_joint_positions_263(motion, 22)
                transitions = []
                for segment in item["segments"][:-1]:
                    boundary = int(round(float(segment["end_sec"]) * fps))
                    if boundary - left < 0 or boundary + right > len(joints):
                        raise ValueError(
                            f"{method}/{seed}/{item['prompt_id']}: invalid transition")
                    transitions.append(joints[boundary - left:boundary + right])
                stacked = np.stack(transitions, axis=0).astype(np.float32)
                lengths = np.full(len(stacked), transition_frames, dtype=np.int64)
                curves.append(calculate_jerk(stacked, lengths).mean(axis=0))
                keys.append(f"{method}|{seed}|{item['prompt_id']}")
    return np.asarray(keys, dtype=np.str_), np.asarray(curves, dtype=np.float32)


def _transition_jerk_from_artifacts(keys, curves, methods, reference_curves=None):
    if curves is None:
        return {}
    keys = [str(value) for value in keys.tolist()]
    output = {}
    for method in methods:
        selected = np.stack([
            curve for key, curve in zip(keys, curves)
            if key.split("|", 1)[0] == method
        ], axis=0)
        mean_curve = selected.mean(axis=0)
        result = {
            "peak_jerk": float(mean_curve.max()),
            "num_transitions": int(len(selected) * 5),
            "transition_frames": 30,
        }
        if reference_curves is not None:
            reference_curve = np.asarray(reference_curves).mean(axis=0)
            result.update({
                "area_under_jerk": float(np.abs(mean_curve - reference_curve).sum()),
                "reference": "paired_real_babel_source_action_curves",
                "reference_peak_jerk": float(reference_curve.max()),
                "reference_mean_jerk": float(reference_curve.mean()),
            })
        output[method] = result
    return output


def _bootstrap_transition_jerk_from_artifacts(
        keys, curves, manifest, methods, seeds_by_method, reference_curves,
        metric_seed, replicates, confidence):
    if curves is None:
        return {}
    prompt_ids = [item["prompt_id"] for item in manifest]
    p = len(prompt_ids)
    curve_by_key = {
        tuple(str(key).split("|", 2)): curve
        for key, curve in zip(keys.tolist(), curves)
    }
    rng = np.random.default_rng(metric_seed + 9173)
    sampled = rng.integers(0, p, size=(replicates, p), endpoint=False)
    weights = np.zeros((replicates, p), dtype=np.int16)
    rows = np.arange(replicates)[:, None]
    np.add.at(weights, (rows, sampled), 1)
    weights = weights.astype(np.float64)
    output = {}
    for method in methods:
        prompt_curves = np.stack([
            np.stack([
                curve_by_key[(method, str(seed), prompt_id)]
                for seed in seeds_by_method[method]
            ], axis=0).mean(axis=0)
            for prompt_id in prompt_ids
        ], axis=0)
        generated = weights @ prompt_curves / float(p)
        method_output = {
            "peak_jerk": _percentile_intervals(
                {"value": generated.max(axis=1)}, confidence)["value"]
        }
        if reference_curves is not None:
            real = weights @ np.asarray(reference_curves) / float(p)
            auj = np.abs(generated - real).sum(axis=1)
            method_output["area_under_jerk"] = _percentile_intervals(
                {"value": auj}, confidence)["value"]
        output[method] = method_output
    return output


def _transition_jerk_summaries(records, manifest_by_id, methods, fps,
                               conversion_device, reference_curves=None,
                               transition_frames=30):
    """Compute FlowMDM/MotionStreamer PJ and AUJ for prompt boundaries."""
    results = {}
    left = transition_frames // 2
    right = transition_frames - left
    for method in methods:
        transitions = []
        for (record_method, _, prompt_id), record in records.items():
            if record_method != method:
                continue
            segments = manifest_by_id[prompt_id].get("segments")
            if not segments or len(segments) <= 1:
                continue
            motion = record["motion"]
            if motion.shape[1] == 138:
                motion = convert_motion_138_to_263(
                    motion, fps=fps, device=str(conversion_device))
            joints = recover_joint_positions_263(motion, 22)
            for segment in segments[:-1]:
                boundary = int(round(float(segment["end_sec"]) * fps))
                if boundary - left < 0 or boundary + right > len(joints):
                    continue
                transitions.append(
                    np.asarray(joints[boundary - left:boundary + right],
                               dtype=np.float32))
        if not transitions:
            continue
        stacked = np.stack(transitions, axis=0)
        lengths = np.full(len(stacked), transition_frames, dtype=np.int64)
        jerk_by_time = calculate_jerk(stacked, lengths).mean(axis=0)
        method_result = {
            "peak_jerk": float(jerk_by_time.max()),
            "num_transitions": int(len(stacked)),
            "transition_frames": int(transition_frames),
        }
        if reference_curves is not None:
            reference_curve = np.asarray(reference_curves).mean(axis=0)
            method_result.update({
                "area_under_jerk": float(np.abs(
                    jerk_by_time - reference_curve).sum()),
                "reference": "paired_real_babel_source_action_curves",
                "reference_peak_jerk": float(reference_curve.max()),
                "reference_mean_jerk": float(reference_curve.mean()),
            })
        results[method] = method_result
    return results


def _bootstrap_transition_jerk(records, manifest, methods, seeds_by_method,
                               fps, conversion_device, metric_seed, replicates,
                               confidence, reference_curves=None,
                               transition_frames=30):
    if not any(item.get("segments") for item in manifest):
        return {}
    prompt_ids = [item["prompt_id"] for item in manifest]
    manifest_by_id = {item["prompt_id"]: item for item in manifest}
    p = len(prompt_ids)
    rng = np.random.default_rng(metric_seed + 9173)
    sampled = rng.integers(0, p, size=(replicates, p), endpoint=False)
    weights = np.zeros((replicates, p), dtype=np.int16)
    rows = np.arange(replicates)[:, None]
    np.add.at(weights, (rows, sampled), 1)
    weights = weights.astype(np.float64)
    left = transition_frames // 2
    right = transition_frames - left
    output = {}
    for method in methods:
        prompt_curves = []
        for prompt_id in prompt_ids:
            transitions = []
            segments = manifest_by_id[prompt_id].get("segments", [])
            for seed in seeds_by_method[method]:
                motion = records[(method, seed, prompt_id)]["motion"]
                if motion.shape[1] == 138:
                    motion = convert_motion_138_to_263(
                        motion, fps=fps, device=str(conversion_device))
                joints = recover_joint_positions_263(motion, 22)
                for segment in segments[:-1]:
                    boundary = int(round(float(segment["end_sec"]) * fps))
                    if boundary - left < 0 or boundary + right > len(joints):
                        continue
                    transitions.append(np.asarray(
                        joints[boundary - left:boundary + right],
                        dtype=np.float32))
            if not transitions:
                raise ValueError(
                    f"{method}/{prompt_id}: no valid transition for jerk bootstrap")
            stacked = np.stack(transitions, axis=0)
            lengths = np.full(len(stacked), transition_frames, dtype=np.int64)
            prompt_curves.append(calculate_jerk(stacked, lengths).mean(axis=0))
        prompt_curves = np.stack(prompt_curves, axis=0)
        curves = weights @ prompt_curves / float(p)
        peak = curves.max(axis=1)
        method_output = {
            "peak_jerk": _percentile_intervals({"value": peak}, confidence)["value"],
        }
        if reference_curves is not None:
            real_curves = weights @ np.asarray(reference_curves) / float(p)
            auj = np.abs(curves - real_curves).sum(axis=1)
            method_output["area_under_jerk"] = _percentile_intervals(
                {"value": auj}, confidence)["value"]
        output[method] = method_output
    return output


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    manifest_by_id = {item["prompt_id"]: item for item in manifest}
    prompt_order = {item["prompt_id"]: i for i, item in enumerate(manifest)}

    window_frames = int(args.fps * args.window_seconds)
    expected_windows = int(args.target_seconds // args.window_seconds)
    if args.target_seconds % args.window_seconds:
        raise ValueError("target_seconds must be divisible by window_seconds")
    if window_frames < 2:
        raise ValueError(f"window must contain at least two frames, got {window_frames}")

    cfg, _, split_paths, feature_dir, _, t2m_cfg = _humanml_spec(args.config)
    del cfg
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    conversion_device = torch.device(args.conversion_device)
    evaluator = T2MMetrics(t2m_cfg).to(device)
    evaluator.eval()

    cache_path = Path(args.reference_cache) if args.reference_cache else (
        output_dir / "reference_embeddings.npz")
    manifest_references = None
    if args.reference_mode == "humanml_test":
        references, _, _, reference_dim = collect_reference_motions(
            args.config, window_frames=window_frames)
        cache_metadata = _reference_cache_metadata(
            args.config, split_paths, feature_dir, t2m_cfg, window_frames,
            conversion_device, reference_dim)
        reference_embeddings, reference_ids, cache_hit = _load_or_build_reference_cache(
            cache_path, cache_metadata, references, evaluator,
            device=device, conversion_device=conversion_device, fps=args.fps,
            batch_size=args.batch_size)
    else:
        manifest_references, babel_spec = collect_manifest_source_references(
            args.config, manifest, expected_windows=expected_windows,
            window_seconds=args.window_seconds,
            min_frames=args.reference_min_frames,
            max_frames=window_frames)
        cache_metadata = _manifest_reference_cache_metadata(
            args.config, args.manifest, babel_spec, expected_windows,
            args.window_seconds, args.reference_min_frames, window_frames,
            t2m_cfg)
        reference_embeddings, reference_ids, cache_hit = (
            _load_or_build_manifest_reference_cache(
                cache_path, cache_metadata, manifest_references, evaluator,
                device=device, batch_size=args.batch_size))

    # Fixed-text manifests reuse one embedding at every window. For a
    # time-varying manifest, select the segment active at the window center so
    # semantic metrics follow the prompt stream instead of the composite label.
    text_requests = []
    text_request_keys = []
    for item in manifest:
        segments = item.get("segments")
        for window_index in range(expected_windows):
            tokens = item["text_tokens"]
            if segments:
                center_sec = (
                    window_index * args.window_seconds
                    + args.window_seconds / 2.0
                )
                tokens = segments[-1]["text_tokens"]
                for segment in segments:
                    if (float(segment["start_sec"]) <= center_sec
                            < float(segment["end_sec"])):
                        tokens = segment["text_tokens"]
                        break
            text_request_keys.append((item["prompt_id"], window_index))
            text_requests.append(tokens)
    text_embeddings_arr = _encode_text_batch(
        text_requests, evaluator, batch_size=args.text_batch_size)
    text_embeddings = {
        key: text_embeddings_arr[index]
        for index, key in enumerate(text_request_keys)
    }

    records, methods, checkpoints = _discover_generated(
        args.generated_root,
        manifest_by_id,
        fps=args.fps,
        target_seconds=args.target_seconds,
        expected_windows=expected_windows,
        window_frames=window_frames,
    )
    seeds_by_method = {
        method: sorted({key[1] for key in records if key[0] == method})
        for method in methods
    }

    missing_samples = []
    for method in methods:
        for seed in seeds_by_method[method]:
            for prompt in manifest:
                key = (method, seed, prompt["prompt_id"])
                if key not in records:
                    missing_samples.append({
                        "method": method, "seed": seed,
                        "prompt_id": prompt["prompt_id"],
                    })

    # Convert every generated window once, then index embeddings by sample/window.
    generated_keys = []
    generated_windows = []
    for key in sorted(records, key=lambda item: (
            item[0], item[1], prompt_order[item[2]])):
        for window_index, motion_window in enumerate(records[key]["windows"]):
            generated_keys.append((*key, window_index))
            generated_windows.append(motion_window)
    generated_embedding_arr = _encode_motion_batch(
        generated_windows, evaluator, device=device,
        conversion_device=conversion_device, fps=args.fps,
        batch_size=args.batch_size)
    generated_embeddings = {
        key: generated_embedding_arr[index]
        for index, key in enumerate(generated_keys)
    }

    rows = []
    for method in methods:
        for window_index in range(expected_windows):
            pairs = []
            for (record_method, seed, prompt_id), record in records.items():
                if record_method != method or window_index >= len(record["windows"]):
                    continue
                pairs.append((seed, prompt_id, generated_embeddings[
                    (method, seed, prompt_id, window_index)]))
            pairs.sort(key=lambda item: (item[0], prompt_order[item[1]]))
            if len(pairs) < 2:
                raise ValueError(
                    f"{method} window {window_index}: fewer than two samples")
            generated = np.stack([item[2] for item in pairs], axis=0)
            window_references = (
                reference_embeddings if np.asarray(reference_embeddings).ndim == 2
                else np.asarray(reference_embeddings)[window_index]
            )
            fid = frechet_distance(window_references, generated)
            kid_values = {}
            if not args.skip_kid:
                kid_mean, kid_std = calculate_kid(
                    reference_embeddings,
                    generated,
                    n_subsets=args.kid_subsets,
                    subset_size=args.kid_subset_size,
                    seed=args.metric_seed + window_index,
                )
                kid_values = {
                    "kid_mean": float(kid_mean),
                    "kid_std": float(kid_std),
                }
            paired_text = np.stack(
                [text_embeddings[(item[1], window_index)] for item in pairs], axis=0)
            mm_dist = float(np.linalg.norm(paired_text - generated, axis=1).mean())

            seed_scores = []
            r3_count = 0
            official_flags = []
            for seed in seeds_by_method[method]:
                seed_pairs = [item for item in pairs if item[0] == seed]
                if len(seed_pairs) < 2:
                    continue
                seed_text = np.stack(
                    [text_embeddings[(item[1], window_index)]
                     for item in seed_pairs], axis=0)
                seed_motion = np.stack([item[2] for item in seed_pairs], axis=0)
                permutation = np.random.default_rng(
                    args.metric_seed + int(seed)).permutation(len(seed_pairs))
                score, count, official = r_precision_at_k(
                    seed_text, seed_motion, top_k=3, group_size=32,
                    permutation=permutation)
                seed_scores.append((score, count))
                r3_count += count
                official_flags.append(official)
            if not seed_scores:
                raise ValueError(f"{method} window {window_index}: R@3 has no valid seed")
            r_at_3 = float(sum(score * count for score, count in seed_scores) / r3_count)
            unique_prompts = len({item[1] for item in pairs})
            start_s = window_index * args.window_seconds
            rows.append({
                "method": method,
                "window_index": window_index,
                "window_start_s": start_s,
                "window_end_s": start_s + args.window_seconds,
                "window_center_s": start_s + args.window_seconds / 2.0,
                "fid": float(fid),
                "mm_dist": mm_dist,
                "r_at_3": r_at_3,
                "n_generated": len(pairs),
                "n_prompts": unique_prompts,
                "n_seeds": len({item[0] for item in pairs}),
                "r3_n_evaluated": r3_count,
                "r3_official": bool(official_flags and all(official_flags)),
                **kid_values,
            })

    summaries = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        times = [row["window_center_s"] for row in method_rows]
        method_summary = {
            "checkpoint": checkpoints[method],
            "num_prompts": len(manifest),
            "seeds": seeds_by_method[method],
            "fid": summarize_curve(times, [row["fid"] for row in method_rows]),
            "mm_dist": summarize_curve(
                times, [row["mm_dist"] for row in method_rows]),
            "r_at_3": summarize_curve(
                times, [row["r_at_3"] for row in method_rows], r_precision=True),
            "r3_official": bool(all(row["r3_official"] for row in method_rows)),
        }
        if not args.skip_kid:
            method_summary["kid"] = summarize_curve(
                times, [row["kid_mean"] for row in method_rows])
        summaries[method] = method_summary

    reference_jerk_curves, reference_jerk_provenance = (
        _reference_transition_jerk_curves(
            manifest, manifest_references, fps=args.fps)
    )
    generated_jerk_keys, generated_jerk_curves = (
        _generated_transition_jerk_artifacts(
            records, manifest, methods, seeds_by_method, fps=args.fps,
            conversion_device=conversion_device)
    )
    transition_jerk = _transition_jerk_from_artifacts(
        generated_jerk_keys, generated_jerk_curves, methods,
        reference_curves=reference_jerk_curves)
    for method, jerk_summary in transition_jerk.items():
        summaries[method]["transition_jerk"] = jerk_summary

    bootstrap = {}
    if args.bootstrap_replicates > 0:
        times = [
            i * args.window_seconds + args.window_seconds / 2.0
            for i in range(expected_windows)
        ]
        bootstrap = _paired_prompt_bootstrap(
            reference_embeddings=reference_embeddings,
            records=records,
            methods=methods,
            manifest=manifest,
            prompt_order=prompt_order,
            seeds_by_method=seeds_by_method,
            generated_embeddings=generated_embeddings,
            text_embeddings=text_embeddings,
            expected_windows=expected_windows,
            times_s=times,
            metric_seed=args.metric_seed,
            replicates=args.bootstrap_replicates,
            confidence=args.bootstrap_confidence,
            device=device,
        )
        jerk_bootstrap = _bootstrap_transition_jerk_from_artifacts(
            keys=generated_jerk_keys,
            curves=generated_jerk_curves,
            manifest=manifest,
            methods=methods,
            seeds_by_method=seeds_by_method,
            reference_curves=reference_jerk_curves,
            metric_seed=args.metric_seed,
            replicates=args.bootstrap_replicates,
            confidence=args.bootstrap_confidence,
        )
        for method, intervals in jerk_bootstrap.items():
            bootstrap[method]["transition_jerk"] = intervals
        for method in methods:
            summaries[method]["bootstrap_confidence_intervals"] = bootstrap[method]

    _write_curve_csv(
        output_dir / "metrics_by_window.csv", rows,
        include_kid=not args.skip_kid)
    _plot_curves(
        output_dir / "windowed_metrics.png", rows, methods,
        include_kid=not args.skip_kid)
    artifact_keys = []
    artifact_embeddings = []
    for key in sorted(generated_embeddings, key=lambda item: (
            item[0], item[1], prompt_order[item[2]], item[3])):
        artifact_keys.append(f"{key[0]}|{key[1]}|{key[2]}|{key[3]}")
        artifact_embeddings.append(generated_embeddings[key])
    artifact_payload = {
        "generated_keys": np.asarray(artifact_keys, dtype=np.str_),
        "generated_embeddings": np.asarray(artifact_embeddings, dtype=np.float32),
        "reference_embeddings": np.asarray(reference_embeddings, dtype=np.float32),
        "reference_ids": np.asarray(reference_ids, dtype=np.str_),
        "method_names": np.asarray(methods, dtype=np.str_),
        "prompt_ids": np.asarray([item["prompt_id"] for item in manifest], dtype=np.str_),
        "text_embeddings": np.stack([
            np.stack([
                text_embeddings[(item["prompt_id"], window_index)]
                for window_index in range(expected_windows)
            ], axis=0)
            for item in manifest
        ], axis=0).astype(np.float32),
        "metadata_json": np.asarray(json.dumps({
            "artifact_version": 1,
            "reference_mode": args.reference_mode,
            "reference_definition": cache_metadata,
            "manifest": str(Path(args.manifest).resolve()),
            "target_seconds": args.target_seconds,
            "fps": args.fps,
            "window_seconds": args.window_seconds,
            "seeds_by_method": seeds_by_method,
        }, sort_keys=True)),
    }
    if reference_jerk_curves is not None:
        artifact_payload["reference_jerk_curves"] = np.asarray(
            reference_jerk_curves, dtype=np.float32)
        artifact_payload["reference_jerk_provenance_json"] = np.asarray(
            json.dumps(reference_jerk_provenance, sort_keys=True))
    if generated_jerk_curves is not None:
        artifact_payload["generated_jerk_keys"] = generated_jerk_keys
        artifact_payload["generated_jerk_curves"] = generated_jerk_curves
    _atomic_npz(output_dir / "evaluation_artifacts.npz", **artifact_payload)

    payload = {
        "config": {
            "config_path": str(Path(args.config).resolve()),
            "manifest": str(Path(args.manifest).resolve()),
            "generated_root": str(Path(args.generated_root).resolve()),
            "target_seconds": args.target_seconds,
            "fps": args.fps,
            "window_seconds": args.window_seconds,
            "window_frames": window_frames,
            "window_centers_s": [
                i * args.window_seconds + args.window_seconds / 2.0
                for i in range(expected_windows)
            ],
            "reference_mode": args.reference_mode,
            "reference_definition": cache_metadata,
            "fixed_reference_bank": args.reference_mode == "humanml_test",
            "reference_count": (
                len(reference_embeddings) if np.asarray(reference_embeddings).ndim == 2
                else int(np.asarray(reference_embeddings).shape[1])),
            "reference_ids_count": int(np.asarray(reference_ids).size),
            "reference_cache_hit": cache_hit,
            "kid_enabled": not args.skip_kid,
            "metric_seed": args.metric_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_confidence": args.bootstrap_confidence,
            "bootstrap_unit": "prompt_or_stream_with_all_generation_seeds",
            "conversion_device": str(conversion_device),
            "phc_enabled": False,
            "fid_bootstrap_includes_mean_difference": True,
            "auj_reference": (
                "paired_real_babel_source_action_curves"
                if reference_jerk_curves is not None else None),
        },
        "missing_samples": missing_samples,
        "methods": summaries,
    }
    _atomic_json(output_dir / "summary.json", payload)
    print(
        f"wrote {len(rows)} curve rows for {len(methods)} method(s) to {output_dir}; "
        f"missing samples={len(missing_samples)}",
        flush=True,
    )
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "build-manifest", help="build an exact-window-length prompt JSONL")
    manifest_parser.add_argument("--config", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--limit", type=int, default=256)
    manifest_parser.add_argument("--window-frames", type=int, default=200)
    manifest_parser.add_argument(
        "--sample-seed", type=int, default=None,
        help="uniformly shuffle all eligible prompts before applying --limit")

    eval_parser = subparsers.add_parser("evaluate", help="compute windowed metrics")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--manifest", required=True)
    eval_parser.add_argument("--generated-root", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--target-seconds", type=int, default=120)
    eval_parser.add_argument("--fps", type=int, default=20)
    eval_parser.add_argument("--window-seconds", type=int, default=10)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument(
        "--conversion-device", default="cpu",
        help="device for deterministic 138->263 SMPL-H conversion (default: cpu)")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--text-batch-size", type=int, default=64)
    eval_parser.add_argument("--kid-subsets", type=int, default=100)
    eval_parser.add_argument("--kid-subset-size", type=int, default=1000)
    eval_parser.add_argument(
        "--skip-kid", action="store_true",
        help="omit KID computation and all KID fields/plots")
    eval_parser.add_argument("--metric-seed", type=int, default=1234)
    eval_parser.add_argument(
        "--reference-mode", choices=("humanml_test", "manifest_sources"),
        default="humanml_test")
    eval_parser.add_argument(
        "--reference-cache", default=None,
        help="optional shared immutable reference-embedding cache path")
    eval_parser.add_argument("--reference-min-frames", type=int, default=40)
    eval_parser.add_argument(
        "--bootstrap-replicates", type=int, default=0,
        help="paired prompt/stream cluster-bootstrap replicates (0 disables)")
    eval_parser.add_argument(
        "--bootstrap-confidence", type=float, default=0.95)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "build-manifest":
        build_manifest(
            args.config, args.output, limit=args.limit,
            window_frames=args.window_frames,
            sample_seed=args.sample_seed)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
