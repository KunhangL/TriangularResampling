#!/usr/bin/env bash
# 120-second long-horizon protocol for an FD2 checkpoint: stream-generate every prompt of a
# manifest, verify the outputs, then compute windowed FID / matching distance / R-precision /
# AUC / degradation slope / jerk with bootstrap confidence intervals. Run from the repo root.
#
#   bash methods/fd2/scripts/run_long_eval.sh METHOD CKPT MANIFEST OUT_DIR REFERENCE_MODE REFERENCE_CACHE
#
#   METHOD          label stored in the generated files (e.g. fd2_tr_30k)
#   MANIFEST        prompt manifest (.jsonl); the paper's are in common/evaluation/manifests/
#                   (humanml3d_prompts_256_uniform_seed1234.jsonl: fixed text,
#                   babel_text_streams_256.jsonl: time-varying text). New fixed-text ones can be built
#                   with `python -m common.evaluation.eval_windowed_long_motion build-manifest`.
#   REFERENCE_MODE  humanml_test (fixed-text protocol) | manifest_sources (varying-text protocol)
#   REFERENCE_CACHE path of the cached reference statistics (.npz; created on first use)
# Environment: CONFIG, GPU (default 0), SEED (default 42).
set -euo pipefail
METHOD="$1"; CKPT="$2"; MANIFEST="$3"; OUT="$4"; REF_MODE="$5"; REF_CACHE="$6"
CONFIG="${CONFIG:-methods/fd2/configs/humanml_babel_tr.yaml}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONPATH="${PYTHONPATH:-.}" TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT}/generated" "${OUT}/metrics"
python -u -m methods.fd2.training.generate --config "${CONFIG}" --checkpoint "${CKPT}" \
    --manifest "${MANIFEST}" --output "${OUT}/generated" --method "${METHOD}" --seed "${SEED:-42}"
python -u -m methods.fd2.training.evaluate evaluate \
    --config "${CONFIG}" --manifest "${MANIFEST}" \
    --generated-root "${OUT}/generated" --output-dir "${OUT}/metrics" \
    --eval-config-dir "${OUT}/eval_config" \
    --target-seconds 120 --fps 20 --window-seconds 10 --device cuda --conversion-device cpu \
    --batch-size 32 --text-batch-size 64 --skip-kid --metric-seed 1235 \
    --bootstrap-replicates 200 --bootstrap-confidence 0.95 \
    --reference-mode "${REF_MODE}" --reference-cache "${REF_CACHE}"
echo "DONE metrics in ${OUT}/metrics/summary.json"
