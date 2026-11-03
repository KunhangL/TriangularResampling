#!/usr/bin/env bash
# Multi-GPU FD2 + TR post-training (the setting used for the released FD2 checkpoints:
# four ranks, global batch 128). Run from the repository root.
#
#   bash methods/fd2/scripts/run_train_ddp.sh [OUTPUT_DIR] [RESUME_CKPT]
#
# Environment: NGPU (default 4), CONFIG (default methods/fd2/configs/humanml_babel_tr.yaml),
# MAX_SECONDS (optional wall-clock budget; the run saves a full-state checkpoint and
# stops cleanly before it elapses, resume with RESUME_CKPT=<output>/last.ckpt).
set -euo pipefail
NGPU="${NGPU:-4}"
CONFIG="${CONFIG:-methods/fd2/configs/humanml_babel_tr.yaml}"
OUTPUT="${1:-outputs/fd2_humanml_babel_tr}"
RESUME="${2:-}"
export PYTHONPATH="${PYTHONPATH:-.}" TOKENIZERS_PARALLELISM=false PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
args=(--config "${CONFIG}" --output "${OUTPUT}")
[[ -n "${RESUME}" ]] && args+=(--resume "${RESUME}")
[[ -n "${MAX_SECONDS:-}" ]] && args+=(--max-seconds "${MAX_SECONDS}")
exec torchrun --standalone --nnodes=1 --nproc-per-node="${NGPU}" \
    -m methods.fd2.training.train_ddp "${args[@]}"
