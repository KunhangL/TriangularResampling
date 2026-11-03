"""Freeze the HumanML3D + BABEL membership used for FD2 post-training.

FD2 post-training consumes exactly the 263D samples that the FD1 dataset
readers load (same split files and length filters), without VAE tokens. This
script instantiates the FD1 datasets from an FD1 config, records every
sample's feature/text path and length, adds the complete HumanML3D test split
as the validation set, checks that every feature is a finite (T, 263) array,
and writes the JSON manifest consumed by
``methods.fd2.training.data.FrozenMotionDataset``.

Run from the repository root after the FD1 data is in place::

    python -m methods.fd2.prepare_data \
        --fd1-config methods/fd1/configs/263d/humanml_babel.yaml \
        --raw-data raw_data --output raw_data/manifests/humanml_babel.json

Paths are stored as given (relative to the working directory by default), so
run training from the same directory. ``--hash`` additionally stores SHA-256
digests of every feature and text file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from common.utils.initialize import Config, instantiate


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_groups(dataset, with_hash):
    """Flatten an FD1 (Multi)Dataset into membership groups without token fields."""
    children = getattr(dataset, 'datasets', [dataset])
    groups = []
    for child in children:
        samples = []
        for item in child.dataset:
            record = dict(dataset=str(item.get('dataset', '')), name=str(item['name']),
                          feature_length=int(item.get('feature_length', -1)),
                          feature_path=str(item['feature_path']), text_path=str(item['text_path']))
            if with_hash:
                record['feature_sha256'] = sha256(record['feature_path'])
                record['text_sha256'] = sha256(record['text_path'])
            samples.append(record)
        groups.append(dict(count=len(samples), samples=samples))
    return groups


def humanml3d_split(raw_data, split_file, with_hash):
    root = Path(raw_data) / 'HumanML3D'
    names = (root / split_file).read_text().splitlines()
    names = [n for n in names if n.strip()]
    assert len(names) == len(set(names)), f'duplicate ids in {root / split_file}'
    samples = []
    for name in names:
        feature = root / 'new_joint_vecs' / f'{name}.npy'
        text = root / 'texts' / f'{name}.txt'
        length = len(np.load(feature, mmap_mode='r', allow_pickle=False))
        record = dict(dataset='HumanML3D', name=name, feature_length=int(length),
                      feature_path=str(feature), text_path=str(text))
        if with_hash:
            record['feature_sha256'] = sha256(feature)
            record['text_sha256'] = sha256(text)
        samples.append(record)
    return [dict(count=len(samples), samples=samples)]


def check_features(splits, dims):
    problems = []
    for split, groups in splits.items():
        for group in groups:
            for row in group['samples']:
                x = np.load(row['feature_path'], mmap_mode='r', allow_pickle=False)
                if x.shape != (row['feature_length'], dims):
                    problems.append(f'{split}:{row["feature_path"]} shape {x.shape}')
                elif not np.isfinite(x).all():
                    problems.append(f'{split}:{row["feature_path"]} non-finite values')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fd1-config', required=True, help='FD1 263D training config that defines the datasets')
    parser.add_argument('--raw-data', default='raw_data', help='directory holding HumanML3D/ and BABEL/')
    parser.add_argument('--output', required=True, help='membership JSON to write')
    parser.add_argument('--dims', type=int, default=263)
    parser.add_argument('--hash', action='store_true', help='store SHA-256 of every feature/text file')
    parser.add_argument('--allow-problems', action='store_true',
                        help='write the manifest even if some features are malformed (they are listed)')
    args = parser.parse_args()

    cfg = Config(args.fd1_config).config
    splits = {}
    for split in ('train', 'test'):
        splits[split] = dataset_groups(instantiate(cfg.data.target, cfg=cfg, split=split), args.hash)
    splits['val'] = humanml3d_split(args.raw_data, 'test.txt', args.hash)

    problems = check_features(splits, args.dims)
    for line in problems:
        print('PROBLEM', line)
    if problems and not args.allow_problems:
        raise SystemExit(f'{len(problems)} malformed features; fix them or pass --allow-problems')

    payload = dict(source=str(Path(args.fd1_config)), source_sha256=sha256(args.fd1_config),
                   feature_fps=int(cfg.data.datasets[0].get('feature_fps', 20)), splits=splits,
                   problems=problems)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, output)
    counts = {k: sum(g['count'] for g in v) for k, v in splits.items()}
    print(f'MEMBERSHIP_OK output={output} counts={counts} sha256={sha256(output)}')


if __name__ == '__main__':
    main()
