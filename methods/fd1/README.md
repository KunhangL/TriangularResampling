# FD1: FloodDiffusion + Triangular Resampling

Triangular Resampling (TR) applied to [FloodDiffusion](https://shandaai.github.io/FloodDiffusion/),
a latent diffusion-forcing model that streams HumanML3D 263D motion as VAE tokens
(4 frames per token, 20 fps). Everything in this directory is specific to the FD1 line;
shared code lives in `common/`.

## How TR is implemented

| File | Role |
| --- | --- |
| `models/diffusion_forcing_wan.py` | `DiffForcingWanModel`: the FloodDiffusion model with the TR training options below |
| `models/resampler.py` | `Resampler`: replays the trailing `resample_len` tokens of each sample's clean context through the model's own streaming Euler rollout (`sources=["model"]`) with the ground-truth clamp |
| `models/tr_schedule.py`, `models/tr_scheduled_wan.py` | optional shift curriculum (`ScheduledTRWan`, `resampling_schedule`); the paper uses a constant shift |
| `models/vae_wan_1d.py`, `models/vae_wan.py`, `models/tools/` | FloodDiffusion VAE and Wan transformer blocks |
| `datasets/` | HumanML3D / BABEL token readers and the multi-dataset wrapper |
| `train_ldf.py`, `train_vae.py`, `pretokenize_vae.py` | training entries |
| `generate_ldf.py`, `generate_long_sequences.py` | streaming inference and multi-checkpoint long-sequence generation |

When `resampling_ratio > 0` and the model is in training mode, each forward pass
(a) draws, per sample, one Gaussian noise trajectory and one threshold
`r = sigmoid(eta + log(shift))`, `eta ~ N(0, 1)`; (b) rolls the last `resample_len`
latents of the clean context forward with the model's own Euler steps on the
triangular schedule, clamping every latent whose denoising progress is below `r`
to its ground-truth interpolation ("frontier" clamp); (c) builds the noisy training
input from the replayed state while keeping the original ground truth as the loss
target. Validation and inference never replay.

Model options (`model.params`):

| Key | Paper setting | Meaning |
| --- | --- | --- |
| `resampling_ratio` | `1.0` | fraction of training samples that are replayed; `0` disables TR (matched baseline) |
| `resample_len` | `100` | number of trailing context tokens replayed (100 tokens = 20 s) |
| `resample_shift` | `0.6` | threshold shift |
| `resample_clamp_mode` | `frontier` | clamp rule (only `frontier` is supported) |
| `resample_source` | `[model]` | replay source; the physics-based option of the research code is not included |
| `warmup_steps` | `0` | first global step at which replay starts |
| `history_augmentation` | `none` | must stay `none` with TR |

## Configurations

| Config | Use |
| --- | --- |
| `configs/263d/humanml_babel.yaml` | TR post-training on HumanML3D + BABEL from the official FloodDiffusion streaming checkpoint (`outputs/20251107_021814_ldf_stream/step_step=240000.ckpt`), 30k steps (paper setting); also defines the dataset membership used by the FD2 line and by the evaluation manifests |
| `configs/263d/vae_official_263.yaml` | the official FloodDiffusion VAE (`outputs/vae_1d_z4_step=300000.ckpt`) used for tokenisation and decoding |
| `configs/paths_default.yaml` | `deps/`, `raw_data/`, `outputs/` locations and W&B settings; copy to `methods/fd1/configs/paths.yaml` (git-ignored) to override |

Paths inside the configs are relative to the directory you run from (the repository
root). Any key can be overridden on the command line with `--override key=value ...`.

## Data

`python -m methods.fd1.download_assets --with-dataset` fetches `deps/`, the official
checkpoints (`outputs/`), the HumanML3D data (`raw_data/HumanML3D/`) and our BABEL release
(`raw_data/BABEL/`, see the top-level README). Both datasets come with the VAE tokens the
configs read, `raw_data/<dataset>/TOKENS_20251030_085836_vae_wan_z4/` (4 frames per token,
encoded with the official FloodDiffusion VAE). For other data or another VAE, create tokens
with:

```bash
python -m methods.fd1.pretokenize_vae --config methods/fd1/configs/263d/vae_official_263.yaml \
    --datasets HumanML3D,BABEL --token_tag <TOKENS_tag>
```

(`--raw_data_path` defaults to `raw_data`.) BABEL sequences are read whole (`max_length`
unbounded) and the training loader crops `window_length: 200` frame windows from them.

## Training

```bash
# TR post-training (paper setting: ratio 1, shift 0.6, frontier clamp)
python -m methods.fd1.train_ldf --config methods/fd1/configs/263d/humanml_babel.yaml \
    --override exp_name=tr263_frontier_s0p6
# matched baseline: same schedule without replay
python -m methods.fd1.train_ldf --config methods/fd1/configs/263d/humanml_babel.yaml \
    --override exp_name=tr263_rfoff model.params.resampling_ratio=0
# resume
python -m methods.fd1.train_ldf --config ... --override exp_name=tr263_frontier_s0p6 resume_ckpt=outputs/<run>/last.ckpt
```

Runs are written to `outputs/<timestamp>_<exp_name>/` with checkpoints every
`validation.save_every_n_steps` steps. Multi-GPU training uses Lightning DDP
(`--override trainer.devices=N`). Set `WANDB_API_KEY` to log to Weights & Biases, or
pass `logger.wandb.mode=disabled`.

## Evaluation

**Short sequences** (standard HumanML3D text-to-motion metrics on the test split, one pass
per seed):

```bash
T2M_EMBEDDING_DUMP_DIR=outputs/eval/short_tr/embeddings \
python -m methods.fd1.train_ldf --config methods/fd1/configs/263d/humanml_babel.yaml \
    --override train=false test_ckpt=outputs/<run>/step_step=30000.ckpt seed=42 \
    exp_name=short_tr run_dir=outputs/eval/short_tr pre_fit_validate=false \
    test_setting.render=false logger.wandb.mode=disabled > outputs/eval/short_tr/evaluation.log
python -m common.evaluation.summarize_short_eval --log outputs/eval/short_tr/evaluation.log \
    --embedding-dir outputs/eval/short_tr/embeddings --output outputs/eval/short_tr/summary.json --repeats 1
```

**Long sequences** (120 s, fixed or time-varying text) with the paper's frozen prompt
manifests in `common/evaluation/manifests/` (see the top-level README):

```bash
bash methods/fd1/scripts/run_long_eval.sh common/evaluation/manifests/humanml3d_prompts_256_uniform_seed1234.jsonl \
    outputs/eval/fd1_fixed humanml_test outputs/eval/ref_fixed.npz \
    ckpt_0=outputs/<tr>/step_step=30000.ckpt label_0=tr ckpt_1=outputs/<off>/step_step=30000.ckpt label_1=tr_off
bash methods/fd1/scripts/run_long_eval.sh common/evaluation/manifests/babel_text_streams_256.jsonl \
    outputs/eval/fd1_varying manifest_sources outputs/eval/ref_varying.npz \
    ckpt_0=outputs/<tr>/step_step=30000.ckpt label_0=tr ckpt_1=outputs/<off>/step_step=30000.ckpt label_1=tr_off
```

The script runs `generate_long_sequences.py` for all checkpoints with identical seeds and
text, verifies the output count, and evaluates with
`common/evaluation/eval_windowed_long_motion.py`. `generate_long_sequences.py` can also be
called directly; its generation-only switches are read from the config's `inference:`
block or from `--override` (`ckpt_<i>` / `label_<i>`, `prompt_manifest`, `seeds`,
`lengths_sec`, `save_motion`, `render`, `out_dir`, `num_workers` / `worker_idx` for
sharding). `generate_ldf.py` renders a single streaming generation and compares it with
non-streaming generation of the same checkpoint.
