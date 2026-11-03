#!/usr/bin/env python3
"""Summarize short-evaluation repeats and cross-repeat multimodality."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


KEY_MAP = {
    "FID": "fid",
    "Matching_score": "matching_distance",
    "R_precision_top_3": "r_precision",
    "Diversity": "diversity",
}


def parse_dataloader_zero(log_path):
    rows = []
    current = None
    header = re.compile(r"Validate metric\s+DataLoader\s+(\d+)")
    metric = re.compile(
        r"^\s*metrics/t2m_metrics/([A-Za-z0-9_]+)\s+"
        r"([-+0-9.eE]+)(?:\s+[-+0-9.eE]+)?\s*$")
    for line in Path(log_path).read_text(errors="replace").splitlines():
        match = header.search(line)
        if match:
            if current and all(key in current for key in KEY_MAP.values()):
                rows.append(current)
            current = {} if int(match.group(1)) == 0 else None
            continue
        if current is None:
            continue
        match = metric.match(line)
        if match and match.group(1) in KEY_MAP:
            current[KEY_MAP[match.group(1)]] = float(match.group(2))
    if current and all(key in current for key in KEY_MAP.values()):
        rows.append(current)
    return rows


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "lower": mean, "upper": mean, "n": len(values)}
    sem = values.std(ddof=1) / math.sqrt(len(values))
    radius = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * sem)
    return {
        "mean": mean,
        "lower": mean - radius,
        "upper": mean + radius,
        "n": int(len(values)),
    }


def multimodality_ci(embedding_dir, repeats, confidence=0.95,
                     pair_count=10, bootstrap_replicates=2000):
    files = sorted(Path(embedding_dir).glob("repeat_*.npz"))
    if len(files) < repeats:
        raise ValueError(f"expected at least {repeats} embedding dumps, found {len(files)}")
    # Some evaluations may have produced more repeats before the requested
    # budget was reduced. Use the first repeats so every method is summarized
    # with the same generation seeds.
    files = files[:repeats]
    generated = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            generated.append(np.asarray(archive["generated"], dtype=np.float32))
    shapes = {array.shape for array in generated}
    if len(shapes) != 1:
        raise ValueError(f"embedding shapes differ across repeats: {shapes}")
    activation = np.stack(generated, axis=1)
    rng = np.random.default_rng(1234)
    pair_count = min(pair_count, repeats)
    first = rng.choice(repeats, pair_count, replace=False)
    second = rng.choice(repeats, pair_count, replace=False)
    per_prompt = np.linalg.norm(
        activation[:, first, :] - activation[:, second, :], axis=2).mean(axis=1)
    sample_indices = rng.integers(
        0, len(per_prompt), size=(bootstrap_replicates, len(per_prompt)))
    bootstrap = per_prompt[sample_indices].mean(axis=1)
    tail = (1.0 - confidence) * 50.0
    return {
        "mean": float(per_prompt.mean()),
        "lower": float(np.percentile(bootstrap, tail)),
        "upper": float(np.percentile(bootstrap, 100.0 - tail)),
        "n_prompts": int(len(per_prompt)),
        "generation_repeats": int(repeats),
        "pairs_per_prompt": int(pair_count),
        "bootstrap_replicates": int(bootstrap_replicates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    rows = parse_dataloader_zero(args.log)
    if len(rows) < args.repeats:
        raise SystemExit(f"expected {args.repeats} DataLoader-0 tables, found {len(rows)}")
    # Keep the first repeats for seed-aligned comparisons when a log already
    # contains more completed evaluations than the requested budget.
    rows = rows[:args.repeats]
    summary = {
        key: mean_ci([row[key] for row in rows], args.confidence)
        for key in KEY_MAP.values()
    }
    summary["multimodality"] = multimodality_ci(
        args.embedding_dir, args.repeats, args.confidence)
    payload = {
        "confidence": args.confidence,
        "short_evaluation_repeats": args.repeats,
        "metrics": summary,
        "repeat_values": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(f"SHORT_SUMMARY_OK output={output} repeats={len(rows)}")


if __name__ == "__main__":
    main()
