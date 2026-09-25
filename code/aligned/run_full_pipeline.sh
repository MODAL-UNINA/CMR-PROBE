#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SC_REPRESENTATION="${SC_REPRESENTATION:-aligned}"
case "${SC_REPRESENTATION}" in
  aligned) DEFAULT_PROBE_TAG="aligned_last_v13" ;;
  mean) DEFAULT_PROBE_TAG="legacy_mean_v13" ;;
  *) echo "SC_REPRESENTATION must be 'aligned' or 'mean' (got: ${SC_REPRESENTATION})" >&2; exit 2 ;;
esac
PROBES_DIR="${PROBES_DIR:-${SCRIPT_DIR}/artifacts/probes/${DEFAULT_PROBE_TAG}}"
DELTA_DIR="${DELTA_DIR:-${PROBES_DIR}/delta}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MSTS_MANIFEST="${MSTS_MANIFEST:-${PROJECT_ROOT}/data/manifests/legacy_v13_msts_split_200_100_100.json}"
RECOMPUTE="${RECOMPUTE:-1}"

export PROBES_DIR DELTA_DIR GPU_ID PYTHON_BIN MSTS_MANIFEST SC_REPRESENTATION RECOMPUTE

bash "${SCRIPT_DIR}/run_core_pipeline.sh" "${PROJECT_ROOT}"

RECOMPUTE_ARG=()
if [[ "${RECOMPUTE}" == "1" ]]; then
  RECOMPUTE_ARG=(--recompute)
fi

# Frozen MM-SafetyBench/TextVQA calibration and held-out test evaluation.
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_mmsafety_textvqa_scores.py" \
  --probes "${PROBES_DIR}" \
  --delta "${DELTA_DIR}" \
  --gpu "${GPU_ID}" \
  --seed 42 \
  --n_cal_unsafe 100 \
  --n_test_unsafe 200 \
  --n_cal_safe 400 \
  --n_test_safe 1100 \
  --sc_representation "${SC_REPRESENTATION}" \
  "${RECOMPUTE_ARG[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/calibrate_mmsafety_textvqa_common_fpr.py" \
  --delta "${DELTA_DIR}" \
  --source_seed 42 \
  --split_seed 2026 \
  --fusion_fit_fraction 0.5

"${PYTHON_BIN}" "${SCRIPT_DIR}/6_evaluate_common_fpr_mmsafety_textvqa.py" \
  --delta "${DELTA_DIR}"

# Frozen zero-shot SIUO vs MMStar plus the existing repeated few-shot analysis.
"${PYTHON_BIN}" "${SCRIPT_DIR}/6_evaluate_pipeline_siuo.py" \
  --probes "${PROBES_DIR}" \
  --delta "${DELTA_DIR}" \
  --gpu "${GPU_ID}" \
  --calibration_cache "${DELTA_DIR}/scores_cal_seed42_msts100_mmstar400.pkl" \
  --mmstar_eval_cache "${DELTA_DIR}/eval_results_cache_v13_internal.pkl" \
  --fewshot_n_pos 30 \
  --fewshot_n_safe 30 \
  --fewshot_repeats 20 \
  --fewshot_seed 42 \
  --sc_representation "${SC_REPRESENTATION}" \
  "${RECOMPUTE_ARG[@]}"

echo "Full v13 rerun completed with S_C=${SC_REPRESENTATION}. Outputs are in ${DELTA_DIR}."
