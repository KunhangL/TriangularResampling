#!/usr/bin/env python3
"""Build deterministic 6x20s BABEL text streams for long-motion evaluation."""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils.initialize import Config


def normalize_caption(caption):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", caption.casefold()).split())


def babel_spec(config_path):
    cfg = Config(config_path).config
    for dataset in cfg.data.datasets:
        if str(dataset.target).endswith("BabelDataset"):
            return [Path(str(path)) for path in dataset.val_meta_paths], str(
                dataset.get("text_path", "texts")
            )
    raise ValueError(f"no BabelDataset found in {config_path}")


def collect_segments(config_path):
    split_paths, text_dir = babel_spec(config_path)
    candidates = []
    seen = set()
    for split_path in split_paths:
        root = split_path.parent
        with split_path.open("r", encoding="utf-8") as handle:
            motion_ids = [line.strip() for line in handle if line.strip()]
        for motion_id in motion_ids:
            text_path = root / text_dir / f"{motion_id}.txt"
            if not text_path.exists():
                continue
            with text_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("#")
                    if len(fields) < 2:
                        continue
                    caption = fields[0].strip()
                    tokens = [token for token in fields[1].split() if token]
                    key = (normalize_caption(caption), tuple(tokens))
                    if not caption or not tokens or key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "caption": caption,
                        "text_tokens": tokens,
                        "source_motion": motion_id,
                    })
    if len(candidates) < 6:
        raise ValueError(f"only {len(candidates)} usable BABEL text segments")
    return candidates


def build_manifest(config_path, output, streams, segments_per_stream,
                   segment_seconds, seed):
    candidates = collect_segments(config_path)
    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    required = streams * segments_per_stream
    if len(order) < required:
        repeats = int(math.ceil(required / len(order)))
        expanded = []
        for _ in range(repeats):
            block = list(order)
            rng.shuffle(block)
            expanded.extend(block)
        order = expanded

    records = []
    for stream_index in range(streams):
        chosen = [
            candidates[order[stream_index * segments_per_stream + j]]
            for j in range(segments_per_stream)
        ]
        segments = []
        for segment_index, item in enumerate(chosen):
            start_sec = segment_index * segment_seconds
            segments.append({
                "caption": item["caption"],
                "text_tokens": item["text_tokens"],
                "start_sec": start_sec,
                "end_sec": start_sec + segment_seconds,
                "source_motion": item["source_motion"],
            })
        records.append({
            "prompt_id": f"babel_stream_{stream_index:03d}",
            "caption": " | ".join(item["caption"] for item in chosen),
            "text_tokens": [
                token for item in chosen for token in item["text_tokens"]
            ],
            "segments": segments,
        })

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(
        f"wrote {len(records)} streams x {segments_per_stream} segments x "
        f"{segment_seconds}s to {output} from {len(candidates)} unique BABEL captions"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--streams", type=int, default=256)
    parser.add_argument("--segments-per-stream", type=int, default=6)
    parser.add_argument("--segment-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    build_manifest(
        args.config,
        args.output,
        streams=args.streams,
        segments_per_stream=args.segments_per_stream,
        segment_seconds=args.segment_seconds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

