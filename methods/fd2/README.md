# FD2: FloodDiffusion 2 + Triangular Resampling

Triangular Resampling (TR) applied to [FloodDiffusion 2](https://github.com/caiyy17/FloodDiffusion2),
which runs diffusion forcing directly on 263D motion features with partial attention,
shared-history multi-window training and an FK-weighted velocity loss. Everything in this
directory is specific to the FD2 line; shared code lives in `common/`.

## Layout

| Path | Role |
| --- | --- |
| `models/triangular_resampling.py` | `TriangularResamplingFD2`: TR as a mixin on the official FD2 `DiffForcingWanModel` |
| `models/packed_forward.py` | the official multi-window packed forward, adapted to take replayed window states |
| `models/execution.py` | micro-batched frozen UMT5 encoding (bounds memory, does not change the loss) |
| `models/upstream/` | copy of the upstream FD2 `models/` package |
| `training/train.py` | single-process Lightning trainer: EMA, full-state checkpoints, deterministic resumable data stream |
| `training/train_ddp.py` | multi-rank DDP trainer with a fixed global batch and rank-aware full-state resume (used for the released checkpoints, 4 ranks) |
| `training/data.py` | frozen-membership feature/text dataset (no VAE tokens) built on the upstream crop and text processors (`native_datasets/`) |
| `training/generate.py`, `training/evaluate.py` | 120 s streaming generation and the bridge to the shared long-horizon evaluator |
| `training/native/` | copy of the upstream FD2 metrics, visualization and utility packages |
| `upstream/FloodDiffusion2/` | the complete upstream repository at commit `9761eb1c` (reference, licenses, asset installer) |
| `configs/humanml_babel_tr.yaml` | the paper configuration |
| `prepare_data.py`, `download_assets.py` | data membership and asset preparation |

## How TR is implemented

`TriangularResamplingFD2.forward` calls the official forward unchanged when `tr_config.ratio`
is `0` or the model is not in training mode (bitwise parity is covered by the tests).
Otherwise, for every sample in the batch:

1. One Gaussian noise trajectory and one threshold `r = sigmoid(eta + log(shift))`,
   `eta ~ N(0, 1)`, are drawn from a dedicated TR random generator (stored in checkpoints,
   so resumed runs reproduce the same draws).
2. The canonical replay is the model's own Euler rollout on the inference grid
   (`steps = chunk_size = 30`); after each step, frames whose progress `alpha` is below
   `r` are clamped to `alpha * GT + (1 - alpha) * noise` ("frontier" clamp). Committed
   frames are frozen, so no training window sees future history.
3. Each of the `train_n_windows` training windows receives an exact snapshot of the
   canonical trajectory at its sampled time; a snapshot at a fractional time is a
   disposable one-step Euler branch from the preceding grid state and never changes the
   canonical path. All active samples share one model forward per grid point
   (`replay_batch`), which is equivalent to serial replay.
4. The snapshots replace the native packed window inputs. The target is still the
   original ground truth; the residual implied by the replayed state is recovered with
   the `beta >= 0.05` guard, and the official FK-weighted velocity loss is applied.

TR requires `attn_type: partial`, `prediction_type: vel`, the linear deterministic
schedule and `random_epsilon: 0` (the upstream `0.05` jitter is only valid for TR-off
training).

## Configuration

`configs/humanml_babel_tr.yaml`, with every path relative to the repository root:

| Setting | Value |
| --- | --- |
| initialisation | official text-conditioned `humanml3d_babel_fk_200k` (parameters and EMA; optimizer, schedule and step count start fresh) |
| data | HumanML3D + BABEL 263D features at 20 fps, 200-frame windows, frozen membership manifest |
| schedule | 30,000 updates, AdamW lr 2e-4, cosine to 1e-6 over 30k, bf16 mixed precision |
| batch | global batch 128, `train_n_windows: 8`, `chunk_size: 30`, `steps: 30` |
| TR | `ratio: 1.0`, `shift: 0.6`, `clamp_mode: frontier`, `seed: 1234` |
| CFG at validation | `text_scale: 4.0`, `null_scale: -3.0` |
| checkpoints | full state every 1,000 steps; validation (HumanML3D FID etc.) and rendering every 5,000 steps |

For the matched baseline set `tr_config.ratio: 0` (optionally with the upstream
`random_epsilon: 0.05`). The FD2 trainers take the config file as is; copy and edit it
for variants.

## Assets and data

```bash
# checkpoints/humanml3d_babel_fk_200k, deps/, raw_data/BABEL/ and raw_data/manifests/
python -m methods.fd2.download_assets --with-deps --with-data
# HumanML3D (raw_data/HumanML3D/) comes with the FloodDiffusion release
python -m methods.fd1.download_assets --with-dataset
```

`raw_data/manifests/humanml_babel.json` is the frozen training membership of the paper's run:
the HumanML3D + BABEL samples that the FD1 dataset readers load for `humanml_babel.yaml`
(same split files and length filters), plus the full HumanML3D test split for validation.
`training/data.py` reads it and checks it against `data.membership_sha256` in the config.
Two HumanML3D test features contain non-finite values (`007975`, `M007975`); the membership
lists them under `problems`, and validation skips them (4,382 of 4,384 samples). The paper's
run replaced them with features recomputed from the joint positions instead; validation
metrics are only logged and affect neither training nor the saved checkpoints.
`prepare_data.py` rebuilds the membership from the raw data; run from the repository root,

```bash
python -m methods.fd2.prepare_data --fd1-config methods/fd1/configs/263d/humanml_babel.yaml \
    --raw-data raw_data --output raw_data/manifests/humanml_babel.json --allow-problems
```

reproduces the released file (`--allow-problems` keeps the two flagged validation
entries, as in the paper's run). `test_setting.render: true` additionally needs the SMPL-H model
under `deps/smplh/` (see `upstream/FloodDiffusion2/setup_project.py --smplh`); set it to
`false` to validate without mesh rendering.

## Training

```bash
# paper setting: 4 ranks, global batch 128
bash methods/fd2/scripts/run_train_ddp.sh outputs/fd2_humanml_babel_tr
# resume from a full-state checkpoint (same world size and config)
bash methods/fd2/scripts/run_train_ddp.sh outputs/fd2_humanml_babel_tr outputs/fd2_humanml_babel_tr/last.ckpt
# single-process trainer (reduce data.train_bs to fit one GPU)
python -m methods.fd2.training.train --config methods/fd2/configs/humanml_babel_tr.yaml --output outputs/fd2_single
# quick smoke test: 20 steps with batch size 2
python -m methods.fd2.training.train --config methods/fd2/configs/humanml_babel_tr.yaml \
    --output outputs/fd2_smoke --smoke-stop 20 --smoke-batch-size 2
```

Both trainers save `last.ckpt` with model, EMA, optimizer, scheduler, Python / NumPy /
Torch / CUDA and TR random states and the consumed-data cursor, so an interrupted run
continues exactly. `--max-seconds` (DDP) / `--max-time` (single process) stop cleanly
before a wall-clock budget. The DDP checkpoints carry a `wisteria_ddp` record (world
size and config fingerprint); the name is a format tag kept for compatibility with the
released checkpoints.

## Generation and evaluation

```bash
# 120 s streaming generation for every prompt of a manifest (EMA weights, 2400 x 263 NPZ)
python -m methods.fd2.training.generate --config methods/fd2/configs/humanml_babel_tr.yaml \
    --checkpoint outputs/fd2_humanml_babel_tr/last.ckpt \
    --manifest common/evaluation/manifests/humanml3d_prompts_256_uniform_seed1234.jsonl \
    --output outputs/eval/fd2_fixed/generated --method fd2_tr --seed 42
# generation + windowed metrics in one go
bash methods/fd2/scripts/run_long_eval.sh fd2_tr outputs/fd2_humanml_babel_tr/last.ckpt \
    common/evaluation/manifests/humanml3d_prompts_256_uniform_seed1234.jsonl outputs/eval/fd2_fixed \
    humanml_test outputs/eval/ref_fixed.npz
bash methods/fd2/scripts/run_long_eval.sh fd2_tr outputs/fd2_humanml_babel_tr/last.ckpt \
    common/evaluation/manifests/babel_text_streams_256.jsonl outputs/eval/fd2_varying \
    manifest_sources outputs/eval/ref_varying.npz
```

`training/evaluate.py` forwards all arguments to
`common/evaluation/eval_windowed_long_motion.py evaluate` after adding the reference-data
locations; see the top-level README for building manifests and for the time-varying-text
protocol.

## Tests

```bash
python -m unittest tests.fd2.test_integration -v
```

CPU-only semantic tests on the real FD2 scheduler and attention blocks: TR-off parity
with the official forward, packing parity, exact flow at threshold extremes, snapshot
branches not altering the canonical path, masks and RoPE, batched versus serial replay,
RNG resume and validation bypass, rejection of the misaligned jitter, and text
micro-batching.
