#!/usr/bin/env bash
# 120-second long-horizon protocol for FD1 checkpoints: generate every prompt of a manifest
# from one or more checkpoints with identical seeds, verify the outputs, then compute windowed
# FID / matching distance / R-precision / AUC / degradation slope / jerk with bootstrap
# confidence intervals. Run from the repository root.
#
#   bash methods/fd1/scripts/run_long_eval.sh MANIFEST OUT_DIR REFERENCE_MODE REFERENCE_CACHE \
#        ckpt_0=<ckpt> label_0=<name> [ckpt_1=<ckpt> label_1=<name> ...]
#
#   MANIFEST        prompt manifest (.jsonl); the paper's are in common/evaluation/manifests/
#                   (humanml3d_prompts_256_uniform_seed1234.jsonl: fixed text,
#                   babel_text_streams_256.jsonl: time-varying text). New fixed-text ones can be built
#                   with `python -m common.evaluation.eval_windowed_long_motion build-manifest`.
#   REFERENCE_MODE  humanml_test (fixed-text protocol) | manifest_sources (varying-text protocol)
#   REFERENCE_CACHE path of the cached reference statistics (.npz; created on first use)
# Environment: CONFIG (default methods/fd1/configs/263d/humanml_babel.yaml), GPU (default 0),
# SEED (default 42).
set -euo pipefail
MANIFEST="$1"; OUT="$2"; REF_MODE="$3"; REF_CACHE="$4"; shift 4
CONFIG="${CONFIG:-methods/fd1/configs/263d/humanml_babel.yaml}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONPATH="${PYTHONPATH:-.}" TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT}/generated" "${OUT}/metrics"
n_ckpt=$(printf '%s\n' "$@" | grep -c '^ckpt_[0-9]*=' || true)
n_prompt=$(grep -c . "${MANIFEST}")
python -u -m methods.fd1.generate_long_sequences --config "${CONFIG}" --override "$@" \
    phc_col=-1 "prompt_manifest=${MANIFEST}" "seeds=[${SEED:-42}]" \
    save_motion=true render=false lengths_sec=120s:120 "out_dir=${OUT}/generated" \
    logger.wandb.mode=disabled num_workers=1 worker_idx=0
# The verifier expects per-worker summaries; the single-worker generator writes one summary.
[[ -f "${OUT}/generated/generation_summary.json" ]] && \
    cp "${OUT}/generated/generation_summary.json" "${OUT}/generated/worker_000_summary.json"
python -m common.evaluation.verify_generation --root "${OUT}/generated" \
    --expected $((n_ckpt * n_prompt)) --workers 1
python -u -m common.evaluation.eval_windowed_long_motion evaluate \
    --config "${CONFIG}" --manifest "${MANIFEST}" \
    --generated-root "${OUT}/generated" --output-dir "${OUT}/metrics" \
    --target-seconds 120 --fps 20 --window-seconds 10 --device cuda --conversion-device cpu \
    --batch-size 32 --text-batch-size 64 --skip-kid --metric-seed 1235 \
    --bootstrap-replicates 200 --bootstrap-confidence 0.95 \
    --reference-mode "${REF_MODE}" --reference-cache "${REF_CACHE}"
echo "DONE metrics in ${OUT}/metrics/summary.json"
