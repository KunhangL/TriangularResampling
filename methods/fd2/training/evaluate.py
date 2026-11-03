"""Bridge an FD2 training config to the shared long-horizon evaluator.

The windowed evaluator (``common.evaluation.eval_windowed_long_motion``) reads
reference motions and texts through the FD1 dataset readers. This bridge adds a
``data.datasets`` block that points those readers at the same frozen 263D
features the FD2 run trained on, writes the resulting evaluation config to
``--eval-config-dir`` (default ``outputs/fd2_eval_configs``) and forwards every
other argument to the evaluator unchanged. Metric, curve, AUC and bootstrap code
is shared with FD1 and is not modified here.

Example::

    python -m methods.fd2.training.evaluate evaluate \
        --config methods/fd2/configs/humanml_babel_tr.yaml \
        --manifest outputs/manifests/humanml3d_prompts_256.jsonl \
        --generated-root outputs/fd2_eval/fixed/generated \
        --output-dir outputs/fd2_eval/fixed/metrics ...
"""
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from common.evaluation import eval_windowed_long_motion as shared
from .io_utils import atomic_json, sha

DATASET_READERS = {
    'HumanML3D': 'methods.fd1.datasets.humanml3d.HumanML3DDataset',
    'BABEL': 'methods.fd1.datasets.babel.BabelDataset',
}


def prepare_evaluation_config(path, eval_config_dir):
    cfg = OmegaConf.load(path)
    membership_path = Path(cfg.data.membership_manifest)
    manifest = json.loads(membership_path.read_text())
    raw_data = Path(cfg.dirs.raw_data)
    specifications = []
    for name, target in DATASET_READERS.items():
        record = next(r for group in manifest['splits']['train'] for r in group['samples'] if r['dataset'] == name)
        root = raw_data / name
        # The FD1 readers join these with the split-file directory (root), so store
        # them relative to it; works for relative and absolute membership paths.
        feature_dir = os.path.relpath(Path(record['feature_path']).parent, root)
        text_dir = os.path.relpath(Path(record['text_path']).parent, root)
        specifications.append(dict(
            target=target, feature_path=feature_dir, text_path=text_dir, feature_fps=20,
            train_meta_paths=[str(root / 'train.txt')],
            val_meta_paths=[str(root / 'test.txt')] if name == 'HumanML3D' else [],
            test_meta_paths=[str(root / 'test_min.txt')]))
    cfg.data = OmegaConf.create(dict(datasets=specifications))
    output = Path(eval_config_dir) / (Path(path).stem + '.eval.yaml')
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output, resolve=True)
    atomic_json(output.with_suffix('.provenance.json'), dict(
        source_config=str(Path(path).resolve()), source_sha256=sha(path),
        membership_manifest=str(membership_path), membership_sha256=sha(membership_path),
        shared_evaluator_sha256=sha(shared.__file__),
        change='reference-location metadata only; metric/curve/AUC/bootstrap code unchanged'))
    return output


def main():
    args = sys.argv[1:]
    eval_config_dir = 'outputs/fd2_eval_configs'
    if '--eval-config-dir' in args:
        pos = args.index('--eval-config-dir')
        eval_config_dir = args[pos + 1]
        del args[pos:pos + 2]
    if '--config' not in args:
        raise ValueError('--config is required')
    pos = args.index('--config') + 1
    args[pos] = str(prepare_evaluation_config(args[pos], eval_config_dir))
    sys.argv = [sys.argv[0], *args]
    shared.main()


if __name__ == '__main__':
    main()
