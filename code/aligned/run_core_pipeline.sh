#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE_DIR="${SCRIPT_DIR}"
if [[ -z "${DATASET_DIR:-}" ]]; then
  if [[ -f "${PROJECT_ROOT}/data/legacy_v13/probe_train.json" ]]; then
    DATASET_DIR="${PROJECT_ROOT}/data/legacy_v13"
  fi
fi
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/legacy_v13}"
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
CALIBRATION_CACHE="${CALIBRATION_CACHE:-${DELTA_DIR}/scores_cal_seed42_msts100_mmstar400.pkl}"
RECOMPUTE="${RECOMPUTE:-1}"

RECOMPUTE_ARG=()
if [[ "${RECOMPUTE}" == "1" ]]; then
  RECOMPUTE_ARG=(--recompute)
fi

if [[ ! -f "${DATASET_DIR}/probe_train.json" ]]; then
  echo "Missing ${DATASET_DIR}/probe_train.json" >&2
  echo "Prepare the internal v13 dataset before running this pipeline." >&2
  exit 2
fi

DATASET_DIR="$(cd "${DATASET_DIR}" && pwd)"
DATASET_PROJECT_ROOT="${DATASET_PROJECT_ROOT:-$(cd "${DATASET_DIR}/../.." && pwd)}"
if [[ ! -f "${MSTS_MANIFEST}" && -f "${DATASET_DIR}/msts_split_200_100_100.json" ]]; then
  MSTS_MANIFEST="${DATASET_DIR}/msts_split_200_100_100.json"
fi
if [[ ! -f "${MSTS_MANIFEST}" ]]; then
  echo "Missing MSTS v13 manifest: ${MSTS_MANIFEST}" >&2
  exit 2
fi
echo "Using v13 dataset: ${DATASET_DIR}"
echo "Resolving v13 image paths from: ${DATASET_PROJECT_ROOT}"
echo "Using MSTS v13 manifest: ${MSTS_MANIFEST}"

mkdir -p "${PROBES_DIR}" "${DELTA_DIR}"
export PYTHONPATH="${PIPELINE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RECOMPUTE}" == "0" \
      && -f "${PROBES_DIR}/probe_text.pkl" \
      && -f "${PROBES_DIR}/probe_visual.pkl" \
      && -f "${PROBES_DIR}/probe_visual_orthogonalized.pkl" ]]; then
  echo "[resume] Reusing the completed S_T/S_V probes."
else
  "${PYTHON_BIN}" "${PIPELINE_DIR}/2_train_probes.py" \
    --dataset "${DATASET_DIR}" \
    --base_dir "${DATASET_PROJECT_ROOT}" \
    --out "${PROBES_DIR}" \
    --gpu "${GPU_ID}" \
    --clip_layer 21 \
    --llama_layer 17
fi

COMBO_SUMMARY="${PROBES_DIR}/combo/probe_combo_results.json"
if [[ "${RECOMPUTE}" == "0" \
      && -f "${COMBO_SUMMARY}" \
      && -f "${PROBES_DIR}/combo/probe_combo_1_linear_delta_h_orthogonalized.pkl" \
      && -f "${PROBES_DIR}/combo/probe_combo_2_svm_rbf_delta_h.pkl" \
      && -f "${PROBES_DIR}/combo/probe_combo_3_mlp_pair.pkl" ]] \
      && grep -q "\"sc_representation\": \"${SC_REPRESENTATION}\"" "${COMBO_SUMMARY}"; then
  echo "[resume] Reusing the completed S_C=${SC_REPRESENTATION} probes."
else
  "${PYTHON_BIN}" "${PIPELINE_DIR}/3.1_train_probe_combo.py" \
    --dataset "${DATASET_DIR}" \
    --base_dir "${DATASET_PROJECT_ROOT}" \
    --probes "${PROBES_DIR%/*}" \
    --tag "${PROBES_DIR##*/}" \
    --gpu "${GPU_ID}" \
    --llama_layer 17 \
    --sc_representation "${SC_REPRESENTATION}"
fi

"${PYTHON_BIN}" "${PIPELINE_DIR}/4.1_calibrate_tau_c.py" \
  --probes "${PROBES_DIR}" \
  --out "${DELTA_DIR}" \
  --msts_manifest "${MSTS_MANIFEST}" \
  --gpu "${GPU_ID}" \
  --n_cal_mmstar 400 \
  --sc_representation "${SC_REPRESENTATION}" \
  "${RECOMPUTE_ARG[@]}"

"${PYTHON_BIN}" "${PIPELINE_DIR}/4.2_calibrate_tau_tv.py" \
  --probes "${PROBES_DIR}" \
  --out "${DELTA_DIR}" \
  --gpu "${GPU_ID}" \
  --n_cal 400 \
  --harmful_splits "${DELTA_DIR}/harmful_contents_reserved.json" \
  "${RECOMPUTE_ARG[@]}"

if [[ ! -f "${DELTA_DIR}/eval_splits.json" ]] \
   || ! cmp -s "${DELTA_DIR}/eval_splits_v13.json" "${DELTA_DIR}/eval_splits.json"; then
  echo "The downstream split alias is missing or differs from eval_splits_v13.json." >&2
  exit 2
fi
if [[ ! -f "${DELTA_DIR}/sv_mmstar_cal_n400.pkl" ]]; then
  echo "Missing sv_mmstar_cal_n400.pkl: tau_V was not calibrated with the v13 MMStar controls." >&2
  exit 2
fi

"${PYTHON_BIN}" "${PIPELINE_DIR}/5_calibrate_common_fpr.py" \
  --delta "${DELTA_DIR}" \
  --cal_cache "${CALIBRATION_CACHE}" \
  --source_seed 42 \
  --split_seed 2026 \
  --fusion_fit_fraction 0.5

"${PYTHON_BIN}" "${PIPELINE_DIR}/6_evaluate_pipeline.py" \
  --probes "${PROBES_DIR}" \
  --delta "${DELTA_DIR}" \
  --msts_manifest "${MSTS_MANIFEST}" \
  --gpu "${GPU_ID}" \
  --sc_representation "${SC_REPRESENTATION}" \
  --use_internal_probes \
  "${RECOMPUTE_ARG[@]}"

if [[ ! -f "${DELTA_DIR}/eval_results_cache_v13_internal.pkl" ]]; then
  echo "Missing eval_results_cache_v13_internal.pkl after the v13 evaluation." >&2
  exit 2
fi

"${PYTHON_BIN}" "${PIPELINE_DIR}/6_evaluate_common_fpr.py" \
  --delta "${DELTA_DIR}" \
  --config "${DELTA_DIR}/common_fpr_config.json" \
  --test_cache "${DELTA_DIR}/eval_results_cache_v13_internal.pkl" \
  --learned_model "${DELTA_DIR}/learned_fusion_common_fpr.pkl" \
  --unsafe_name MSTS_test \
  --safe_name MMStar_test \
  --suffix msts_mmstar

echo "Core v13 pipeline completed with S_C=${SC_REPRESENTATION}. Outputs are in ${DELTA_DIR}."
