"""Download the official FloodDiffusion 2 assets used for FD2 + TR post-training.

Fetches the text-conditioned HumanML3D+BABEL release checkpoint
(``checkpoints/humanml3d_babel_fk_200k``: ``model.ckpt``, ``config.yaml`` and the
``assets/{Mean,Std,W}.npy`` normalisation / FK matrices) from the Hugging Face
repository ``AlayaLab/FloodDiffusion2`` at the revision used for the paper, and
optionally the shared dependencies (UMT5 encoder, HumanML3D T2M evaluator,
GloVe) and our data release (263D BABEL and the frozen FD2 training membership,
see ``common/download_data.py``). Files are placed under ``--root`` (default:
current directory, i.e. the repository root) in the layout expected by
``methods/fd2/configs/humanml_babel_tr.yaml``. HumanML3D comes with the
FloodDiffusion release (``python -m methods.fd1.download_assets --with-dataset``).

    python -m methods.fd2.download_assets              # checkpoint only
    python -m methods.fd2.download_assets --with-deps  # also deps/ (same layout as FD1)
    python -m methods.fd2.download_assets --with-data  # also raw_data/BABEL/ and raw_data/manifests/
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from common.download_data import download as download_tr_data

REPO_ID = 'AlayaLab/FloodDiffusion2'
REVISION = '2402033aff49850b31c1c5be9179c2fd7cb476cb'
CHECKPOINT = 'checkpoints/humanml3d_babel_fk_200k'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='.', help='destination root (repository root)')
    parser.add_argument('--revision', default=REVISION, help='Hugging Face revision to pin')
    parser.add_argument('--with-deps', action='store_true', help='also download deps/ (UMT5, T2M evaluator, GloVe)')
    parser.add_argument('--with-data', action='store_true',
                        help='also download our BABEL release and the FD2 training membership into raw_data/')
    args = parser.parse_args()
    patterns = [f'{CHECKPOINT}/**']
    if args.with_deps:
        patterns.append('deps/**')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, revision=args.revision, allow_patterns=patterns, local_dir=str(root))
    for name in ('model.ckpt', 'assets/Mean.npy', 'assets/Std.npy', 'assets/W.npy'):
        path = root / CHECKPOINT / name
        if not path.is_file():
            raise SystemExit(f'missing after download: {path}')
    print(f'ASSETS_OK {root / CHECKPOINT} (revision {args.revision})')
    if args.with_data:
        download_tr_data(root / 'raw_data', items=('babel', 'manifests'))


if __name__ == '__main__':
    main()
