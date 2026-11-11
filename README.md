# Triangular Resampling for Long-Horizon Motion Generation

[![arXiv](https://img.shields.io/badge/arXiv-2609.34697-b31b1b.svg)](https://arxiv.org/abs/2609.34697)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Official implementation of **Triangular Resampling (TR)**, a post-training method that
reduces error accumulation in long-horizon streaming motion generation.

Kunhang Li, Yiyi Cai, Xiangyue Zhang, Fangyuan Tu, Yuhan Wu, Zhixiang Wang, Kaipeng Zhang, Haiyang Liu

**Paper:** https://arxiv.org/abs/2609.34697

TR builds on the triangular denoising schedule of the FloodDiffusion family. During
post-training, every training sample is replayed through the model's own streaming
rollout; frames whose denoising progress is below a per-sample random threshold are
clamped to their ground-truth interpolation, and the model is supervised on the
resulting partially denoised states instead of on clean ground-truth context only.
Inference is unchanged. This repository provides TR for two backbones, each in its own
self-contained package, plus the shared 120-second evaluation protocol:

| Line | Backbone | Data | Package |
| --- | --- | --- | --- |
| FD1 | [FloodDiffusion](https://shandaai.github.io/FloodDiffusion/) (latent diffusion forcing on VAE tokens) | HumanML3D + BABEL, 263D, 20 fps | [`methods/fd1/`](methods/fd1/README.md) |
| FD2 | [FloodDiffusion 2](https://github.com/caiyy17/FloodDiffusion2) (feature-space diffusion forcing, partial attention) | HumanML3D + BABEL, 263D, 20 fps | [`methods/fd2/`](methods/fd2/README.md) |

## Repository layout

```
common/                      shared by both lines
  evaluation/                120 s windowed long-horizon evaluation, prompt-manifest builders
  metrics/                   HumanML3D text-to-motion evaluator (FID, matching distance, R-precision, ...)
  utils/                     config loading, 263D joint recovery, rotation / quaternion math
methods/
  fd1/                       FloodDiffusion + TR: models, datasets, configs, training, generation
  fd2/                       FloodDiffusion 2 + TR: models, training, configs, data preparation,
                             and the pinned upstream FloodDiffusion 2 source (upstream/FloodDiffusion2/)
tests/                       CPU unit tests for common/, fd1/ and fd2/
requirements.txt             Python dependencies
THIRD_PARTY_LICENSES.md      attribution for FloodDiffusion, FloodDiffusion 2, Wan, HumanML3D, ...
```

`methods/fd1` and `methods/fd2` do not import each other. Both import `common/`. The only
cross-line coupling is in evaluation: the shared evaluator loads reference motions through
the FD1 dataset readers, so the FD2 evaluation bridge (`methods/fd2/training/evaluate.py`)
points them at the same frozen 263D features the FD2 run trained on.

All commands below are run from the repository root with the root on `PYTHONPATH`
(`python -m ...` does this automatically).

## Installation

```bash
conda create -n tr python=3.11 -y && conda activate tr
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Videos need `ffmpeg`/`ffprobe` on `PATH`; mesh rendering uses an OpenGL/EGL runtime
(`PYOPENGL_PLATFORM=egl` for headless GPUs). `flash-attn` is optional for FD1 (PyTorch SDPA
is used otherwise); FD2 uses SDPA only.

## Assets and data

Both lines read the same directories at the repository root:

```
deps/                        t5_umt5-xxl-enc-bf16/   UMT5-XXL text encoder (+ google/umt5-xxl tokenizer)
                             t2m/{humanml3d,meta}/   HumanML3D text-to-motion evaluator and statistics
                             glove/                  GloVe word embeddings for the evaluator
                             smplh/                  SMPL-H body model (FD2 mesh rendering only)
raw_data/                    HumanML3D/              new_joint_vecs/, joints/, texts/, VAE tokens, split files, Mean/Std.npy
                             BABEL/                  new_joint_vecs/, texts/, VAE tokens, train|val|test_min.txt (our release)
                             manifests/              FD2 training membership (our release)
checkpoints/                 humanml3d_babel_fk_200k/ official FD2 initialisation (model.ckpt, assets/{Mean,Std,W}.npy)
outputs/                     official FD1 checkpoints and every run directory
```

- **FD1 release** (deps and official FD1 checkpoints; `--with-dataset` adds HumanML3D and our
  BABEL release): `python -m methods.fd1.download_assets --with-dataset`
  (Hugging Face [ShandaAI/FloodDiffusionDownloads](https://huggingface.co/ShandaAI/FloodDiffusionDownloads)).
- **FD2 release** (official `humanml3d_babel_fk_200k` checkpoint; `--with-deps` adds the same
  deps, `--with-data` our BABEL release and the FD2 manifests):
  `python -m methods.fd2.download_assets --with-deps --with-data`
  (Hugging Face [AlayaLab/FloodDiffusion2](https://huggingface.co/AlayaLab/FloodDiffusion2), pinned revision).
  HumanML3D comes with the FD1 command above. `methods/fd2/upstream/FloodDiffusion2/setup_project.py`
  is the upstream installer for the remaining FD2 checkpoints and the SMPL-H import (`--smplh`).
- **Our data release** (Hugging Face [KunhangLI/TriangularResampling](https://huggingface.co/KunhangLI/TriangularResampling)):
  `python -m common.download_data` downloads and checks (SHA-256) two archives, which the two
  commands above also fetch:
  - `BABEL.zip` → `raw_data/BABEL/`: all 5,341 [BABEL](https://babel.is.tue.mpg.de/) sequences
    in the 263D HumanML3D representation at 20 fps (`new_joint_vecs/<id>.npy`), HumanML3D-style
    annotations for the 4,877 labelled ones (`texts/<id>.txt`, one `caption#tokens#start#end`
    line per segment), the FloodDiffusion VAE tokens used by FD1
    (`TOKENS_20251030_085836_vae_wan_z4/`), and the splits `train.txt` (3,985), `val.txt`
    (1,356) and `test_min.txt` (10 validation sequences used for rendering). Training uses
    the 3,640 labelled training sequences; validation sequences are the references of the
    time-varying evaluation.
  - `manifests.zip` → `raw_data/manifests/`: the FD2 training membership `humanml_babel.json`
    (pinned by SHA-256 in `methods/fd2/configs/humanml_babel_tr.yaml`).

  These are the exact files of the paper's runs. The features are derived from AMASS and
  BABEL and are provided for non-commercial research under their licenses; see the dataset
  card. Datasets and body models are subject to their own licenses.
- **SMPL-X** (FD1 mesh rendering only): set `SMPLX_MODEL_DIR` or place the models under
  `deps/smplx_models`.

## Quick start

FD1 (details in [`methods/fd1/README.md`](methods/fd1/README.md)):

```bash
# TR post-training from the official FloodDiffusion streaming checkpoint (paper setting)
python -m methods.fd1.train_ldf --config methods/fd1/configs/263d/humanml_babel.yaml \
    --override exp_name=tr263_frontier_s0p6
# matched baseline without replay (TR off)
python -m methods.fd1.train_ldf --config methods/fd1/configs/263d/humanml_babel.yaml \
    --override exp_name=tr263_rfoff model.params.resampling_ratio=0
```

FD2 (details in [`methods/fd2/README.md`](methods/fd2/README.md)):

```bash
python -m methods.fd2.download_assets --with-deps --with-data   # plus HumanML3D from the FD1 command
# FD2 + TR (official reproduction, global batch 128)
bash methods/fd2/scripts/run_train_ddp.sh outputs/fd2_humanml_babel_tr
# FD2 + TR + DMD (self-generated prefix + DMD distribution matching, global batch 4, lr 2e-5)
torchrun --standalone --nproc_per_node=4 -m methods.fd2.training.train_ddp \
    --config methods/fd2/configs/humanml_babel_tr_dmd.yaml \
    --output outputs/fd2_humanml_babel_tr_dmd
```

## Released FD2 Checkpoints & Official 120s Benchmark Scores

Pre-trained post-training checkpoints for **FD2 + TR** (`step 200`) and **FD2 + TR + DMD** (`step 400`) are publicly available on Hugging Face at [**`KunhangLI/TriangularResampling`**](https://huggingface.co/KunhangLI/TriangularResampling), next to our data release (the repository is gated: run `hf auth login` and accept its terms once):

```bash
# Download both checkpoints into checkpoints/
hf download KunhangLI/TriangularResampling \
    checkpoints/fd2_tr_step200.ckpt \
    checkpoints/fd2_tr_dmd_step400.ckpt \
    --local-dir .
```

| Model | Step | Fixed First→Last (`0-10s`→`110-120s`) | Fixed ΔFID ↓ | Fixed FID AUC ↓ | Fixed R@3 ↑ | Varying FID AUC ↓ | Varying R@3 ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD2 Pretrain** | `200k` | `1.69 → 5.72` | `+4.03` | `4.33` | `53.7%` | `4.31` | `36.3%` |
| **FD2 + TR-off** | `200` | `1.60 → 5.43` | `+3.83` | `4.61` | `50.3%` | `5.74` | `35.5%` |
| **FD2 + TR** | `200` | **`0.75 → 1.43`** | **`+0.68`** | **`1.17`** | **`70.2%`** | **`3.63`** | **`40.7%`** |
| **FD2 + TR + DMD** | `400` | `1.69 → 3.76` | `+2.06` | `3.18` | `57.1%` | `5.56` | `35.0%` |

## Long-horizon evaluation

Both lines are evaluated with the same protocol in `common/evaluation/`: 120-second
generation for every prompt of a manifest, then metrics on fixed 10-second windows
(FID, matching distance, R-precision, their AUC over time, the FID degradation slope,
jerk) with bootstrap confidence intervals.

The paper's prompt manifests are frozen in [`common/evaluation/manifests/`](common/evaluation/manifests):

| Manifest | Protocol | Prompts |
| --- | --- | --- |
| `humanml3d_prompts_256_uniform_seed1234.jsonl` | fixed text | 256 unique HumanML3D test captions, one per 120 s sequence |
| `babel_text_streams_256.jsonl` | time-varying text | 256 streams of six 20-second BABEL validation captions |

```bash
FIXED=common/evaluation/manifests/humanml3d_prompts_256_uniform_seed1234.jsonl
VARYING=common/evaluation/manifests/babel_text_streams_256.jsonl
# FD1: generate + verify + evaluate, all checkpoints with identical seeds and text
bash methods/fd1/scripts/run_long_eval.sh $FIXED outputs/eval/fd1_fixed humanml_test outputs/eval/ref_fixed.npz \
    ckpt_0=<tr.ckpt> label_0=tr ckpt_1=<off.ckpt> label_1=tr_off
bash methods/fd1/scripts/run_long_eval.sh $VARYING outputs/eval/fd1_varying manifest_sources outputs/eval/ref_varying.npz \
    ckpt_0=<tr.ckpt> label_0=tr ckpt_1=<off.ckpt> label_1=tr_off
# FD2
bash methods/fd2/scripts/run_long_eval.sh fd2_tr <ckpt> $FIXED outputs/eval/fd2_fixed humanml_test outputs/eval/ref_fixed.npz
bash methods/fd2/scripts/run_long_eval.sh fd2_tr <ckpt> $VARYING outputs/eval/fd2_varying manifest_sources outputs/eval/ref_varying.npz
```

The fixed-text protocol compares every window with the HumanML3D test set
(`humanml_test`). The time-varying protocol compares each window with the BABEL
validation motion its active caption was taken from (`manifest_sources`), so it needs the
BABEL validation sequences under `raw_data/BABEL/`. Keep one reference cache per protocol.

The fixed-text manifest was built with
`python -m common.evaluation.eval_windowed_long_motion build-manifest --limit 256 --sample-seed 1234`
over the HumanML3D test split. The time-varying one samples 256 caption streams (seed 1234)
from 1,076 BABEL validation sequences that passed a motion-tracking quality filter; it is
released as a frozen file only. Use the frozen files for results comparable with the paper.

## Tests

CPU-only unit tests (no data or checkpoints needed): TR semantics on the real FD2
scheduler and attention blocks, the windowed evaluator's metrics, and the FD1 shift schedule.

```bash
python -m unittest discover -s tests -t . -v
```

## TODO

- [x] Our processed BABEL dataset (263D features, annotations, FD1 VAE tokens and splits)
      and the FD2 training membership, on [Hugging Face](https://huggingface.co/KunhangLI/TriangularResampling).
- [x] Trained checkpoints: FD2 + TR (step 200) and FD2 + TR + DMD (step 400),
      on [Hugging Face](https://huggingface.co/KunhangLI/TriangularResampling).
- [ ] Project page.

## Citation

```bibtex
@article{li2026triangular,
  title   = {Triangular Resampling for Long-Horizon Motion Generation},
  author  = {Li, Kunhang and Cai, Yiyi and Zhang, Xiangyue and Tu, Fangyuan and Wu, Yuhan and Wang, Zhixiang and Zhang, Kaipeng and Liu, Haiyang},
  journal = {arXiv preprint arXiv:2609.34697},
  year    = {2026}
}
```

## License and acknowledgements

This repository is released under the Apache License 2.0 (see [`LICENSE`](LICENSE)). It
builds on [FloodDiffusion](https://shandaai.github.io/FloodDiffusion/) (Apache-2.0) and
[FloodDiffusion 2](https://github.com/caiyy17/FloodDiffusion2) (MIT), and vendors components
from Wan 2.2, HumanML3D / text-to-motion and VideoPose3D; see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
