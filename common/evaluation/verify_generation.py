#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root)
    files = list(root.rglob("*.npz"))
    if len(files) != args.expected:
        raise SystemExit(
            f"generation count mismatch: {len(files)} != {args.expected} under {root}")
    for worker in range(args.workers):
        summary_path = root / f"worker_{worker:03d}_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"missing worker summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if int(summary.get("num_fail", -1)) != 0:
            raise SystemExit(f"worker {worker} reports failures: {summary_path}")
    print(f"GENERATION_OK root={root} npz={len(files)} workers={args.workers}")


if __name__ == "__main__":
    main()
