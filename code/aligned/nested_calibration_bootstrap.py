#!/usr/bin/env python3
"""Nested calibration/test bootstrap for the v13 paper comparison.

For every bootstrap replicate and benchmark this script:

1. resamples the same 200 benign ``operating_cal`` examples for every method;
2. recomputes the common-FPR threshold tau independently for each method;
3. resamples the unsafe and benign test sets (stratified, with replacement);
4. evaluates TPR, FPR, and F1 with the newly calibrated threshold.

The learned-fusion model remains frozen: it was fitted on the disjoint
``fusion_fit`` subset.  Only its operating threshold is recalibrated, exactly
as requested by the nested calibration bootstrap protocol.

With the v13 files in their standard locations the script can be run without
arguments from any directory::

    python 23_nested_calibration_bootstrap.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


OPS = (("fpr_05", 0.05), ("fpr_01", 0.01))
METHOD_ORDER = (
    "S_C aligned",
    "LEARNED fusion",
    "Qwen S_C aligned",
    "LlavaGuard",
    "GuardReasoner-style",
    "ShieldVLM",
    "Granite-Guardian-style",
)
METHOD_IDS = {
    "S_C aligned": "sc_aligned",
    "LEARNED fusion": "learned_fusion",
    "Qwen S_C aligned": "qwen_sc_aligned",
    "LlavaGuard": "llavaguard",
    "GuardReasoner-style": "guardreasoner_style",
    "ShieldVLM": "shieldvlm",
    "Granite-Guardian-style": "granite_guardian_style",
}

BENCHMARKS = {
    "msts_mmstar": {
        "split_file": "eval_splits.json",
        "cal_file": "common_fpr_calibration_scores.pkl",
        "test_file": "common_fpr_test_scores_msts_mmstar.pkl",
        "config_file": "common_fpr_config.json",
        "cal_safe_key": "mmstar_cal",
        "test_unsafe_key": "msts_test",
        "test_safe_key": "mmstar_test",
    },
    "mmsafety_textvqa": {
        "subdir": "mmsafety_textvqa_common_fpr",
        "split_file": "eval_splits_mmsafety_textvqa.json",
        "cal_file": "common_fpr_calibration_scores_mmsafety_textvqa.pkl",
        "test_file": "common_fpr_test_scores_mmsafety_textvqa.pkl",
        "config_file": "common_fpr_config_mmsafety_textvqa.json",
        "cal_safe_key": "textvqa_cal",
        "test_unsafe_key": "mmsafety_test",
        "test_safe_key": "textvqa_test",
    },
}

BASELINE_FILES = {
    "msts_mmstar": {
        "LlavaGuard": (
            "llamaguard_common_fpr/llavaguard_common_fpr_msts_mmstar.pkl",
            "llamaguard_common_fpr/llavaguard_common_fpr_msts_mmstar.json",
        ),
        "GuardReasoner-style": (
            "guardreasoner_style_common_fpr/guardreasoner_style_common_fpr_msts_mmstar.pkl",
            "guardreasoner_style_common_fpr/guardreasoner_style_common_fpr_msts_mmstar.json",
        ),
        "ShieldVLM": (
            "shieldvlm_common_fpr/shieldvlm_common_fpr_msts_mmstar.pkl",
            "shieldvlm_common_fpr/shieldvlm_common_fpr_msts_mmstar.json",
        ),
        "Granite-Guardian-style": (
            "graniteguardian_style_common_fpr/graniteguardian_style_common_fpr_msts_mmstar.pkl",
            "graniteguardian_style_common_fpr/graniteguardian_style_common_fpr_msts_mmstar.json",
        ),
    },
    "mmsafety_textvqa": {
        "LlavaGuard": (
            "mmsafety_textvqa_common_fpr/llamaguard_common_fpr/llamaguard_common_fpr_mmsafety_textvqa.pkl",
            "mmsafety_textvqa_common_fpr/llamaguard_common_fpr/llavaguard_common_fpr_mmsafety_textvqa.json",
        ),
        "GuardReasoner-style": (
            "mmsafety_textvqa_common_fpr/guardreasoner_style_common_fpr/guardreasoner_style_common_fpr_mmsafety_textvqa_uniform_v2.pkl",
            "mmsafety_textvqa_common_fpr/guardreasoner_style_common_fpr/guardreasoner_style_common_fpr_mmsafety_textvqa_uniform_v2.json",
        ),
        "ShieldVLM": (
            "mmsafety_textvqa_common_fpr/shieldvlm_common_fpr/shieldvlm_common_fpr_mmsafety_textvqa.pkl",
            "mmsafety_textvqa_common_fpr/shieldvlm_common_fpr/shieldvlm_common_fpr_mmsafety_textvqa.json",
        ),
        "Granite-Guardian-style": (
            "mmsafety_textvqa_common_fpr/graniteguardian_style_common_fpr/graniteguardian_style_common_fpr_mmsafety_textvqa.pkl",
            "mmsafety_textvqa_common_fpr/graniteguardian_style_common_fpr/graniteguardian_style_common_fpr_mmsafety_textvqa.json",
        ),
    },
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def finite_1d(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        raise ValueError(f"{label}: empty array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: non-finite values")
    return array


def ordered_scores(indices, scores, expected_indices, label: str) -> np.ndarray:
    indices = [int(value) for value in indices]
    expected = [int(value) for value in expected_indices]
    scores = finite_1d(scores, label)
    if len(indices) != len(scores):
        raise ValueError(f"{label}: indices/scores mismatch")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{label}: duplicate indices")
    by_index = dict(zip(indices, scores))
    if set(by_index) != set(expected):
        raise RuntimeError(f"{label}: cache does not match the frozen v13 split")
    return np.asarray([by_index[index] for index in expected], dtype=np.float64)


def modern_work_dir(delta: Path, benchmark: str) -> Path:
    subdir = BENCHMARKS[benchmark].get("subdir")
    return delta / subdir if subdir else delta


def tau_from_modern_config(config: dict, method: str, op_name: str) -> float:
    try:
        return float(config["methods"][method]["operating_points"][op_name]["tau"])
    except KeyError as exc:
        raise KeyError(f"Missing tau for {method}/{op_name}") from exc


def tau_from_baseline_config(config: dict, op_name: str) -> float:
    if "results" in config and op_name in config["results"]:
        return float(config["results"][op_name]["tau"])
    if op_name in config and isinstance(config[op_name], dict):
        return float(config[op_name]["tau"])
    raise KeyError(f"Missing baseline tau for {op_name}")


def load_modern_method(
    delta: Path,
    benchmark: str,
    cache_method: str,
    display_name: str,
) -> dict:
    spec = BENCHMARKS[benchmark]
    work_dir = modern_work_dir(delta, benchmark)
    split_path = work_dir / spec["split_file"]
    cal_path = work_dir / spec["cal_file"]
    test_path = work_dir / spec["test_file"]
    config_path = work_dir / spec["config_file"]
    split = load_json(split_path)
    cal = load_pickle(cal_path)
    test = load_pickle(test_path)
    config = load_json(config_path)

    if cal.get("sc_representation") != "aligned":
        raise RuntimeError(
            f"{display_name}/{benchmark}: expected aligned S_C, found "
            f"{cal.get('sc_representation')!r}"
        )
    cal_positions = [int(value) for value in cal["split"]["safe_operating"]]
    full_cal_ids = [int(value) for value in split[spec["cal_safe_key"]]]
    cal_ids = [full_cal_ids[position] for position in cal_positions]
    cal_safe = finite_1d(
        cal["scores_safe_operating"][cache_method],
        f"{display_name}/{benchmark}/cal_safe",
    )
    if len(cal_ids) != len(cal_safe):
        raise ValueError(f"{display_name}/{benchmark}: calibration mismatch")

    expected_unsafe = [int(value) for value in split[spec["test_unsafe_key"]]]
    expected_safe = [int(value) for value in split[spec["test_safe_key"]]]
    if benchmark == "msts_mmstar":
        unsafe_indices = [int(row["idx"]) for row in test["results_unsafe"]]
        safe_indices = [int(row["idx"]) for row in test["results_safe"]]
        unsafe_scores = test["method_scores_unsafe"][cache_method]
        safe_scores = test["method_scores_safe"][cache_method]
    else:
        method = test["methods"][cache_method]
        unsafe_indices = [int(value) for value in test.get("idx_unsafe", expected_unsafe)]
        safe_indices = [int(value) for value in test.get("idx_safe", expected_safe)]
        unsafe_scores = method["scores_unsafe"]
        safe_scores = method["scores_safe"]

    return {
        "cal_ids": cal_ids,
        "cal_safe": cal_safe,
        "test_unsafe": ordered_scores(
            unsafe_indices,
            unsafe_scores,
            expected_unsafe,
            f"{display_name}/{benchmark}/test_unsafe",
        ),
        "test_safe": ordered_scores(
            safe_indices,
            safe_scores,
            expected_safe,
            f"{display_name}/{benchmark}/test_safe",
        ),
        "taus_frozen": {
            op_name: tau_from_modern_config(config, cache_method, op_name)
            for op_name, _ in OPS
        },
        "sources": {
            "calibration_scores": str(cal_path),
            "test_scores": str(test_path),
            "threshold_config": str(config_path),
            "split": str(split_path),
        },
    }


def load_baseline_method(
    baseline_delta: Path,
    benchmark: str,
    display_name: str,
    expected_cal_ids: list[int],
    expected_unsafe_ids: list[int],
    expected_safe_ids: list[int],
) -> dict:
    cache_rel, config_rel = BASELINE_FILES[benchmark][display_name]
    cache_path = baseline_delta / cache_rel
    config_path = baseline_delta / config_rel
    cache = load_pickle(cache_path)
    config = load_json(config_path)

    cal_rows = cache["cal_safe"]
    cal_ids = [int(row["idx"]) for row in cal_rows]
    cal_scores = [float(row["score"]) for row in cal_rows]
    cal_safe = ordered_scores(
        cal_ids,
        cal_scores,
        expected_cal_ids,
        f"{display_name}/{benchmark}/cal_safe",
    )
    unsafe_rows = cache["test_unsafe"]
    safe_rows = cache["test_safe"]
    test_unsafe = ordered_scores(
        [row["idx"] for row in unsafe_rows],
        [row["score"] for row in unsafe_rows],
        expected_unsafe_ids,
        f"{display_name}/{benchmark}/test_unsafe",
    )
    test_safe = ordered_scores(
        [row["idx"] for row in safe_rows],
        [row["score"] for row in safe_rows],
        expected_safe_ids,
        f"{display_name}/{benchmark}/test_safe",
    )
    return {
        "cal_ids": list(expected_cal_ids),
        "cal_safe": cal_safe,
        "test_unsafe": test_unsafe,
        "test_safe": test_safe,
        "taus_frozen": {
            op_name: tau_from_baseline_config(config, op_name)
            for op_name, _ in OPS
        },
        "sources": {
            "calibration_and_test_scores": str(cache_path),
            "threshold_config": str(config_path),
        },
    }


def threshold_at_fpr(scores_safe: np.ndarray, target_fpr: float) -> float:
    """Exact rule used by 5_calibrate_common_fpr.py."""
    scores = finite_1d(scores_safe, "scores_safe")
    candidates = np.unique(scores)
    candidates = np.concatenate(
        [candidates, [np.nextafter(float(scores.max()), np.inf)]]
    )
    for tau in candidates:
        if float(np.mean(scores >= tau)) <= target_fpr + 1e-15:
            return float(tau)
    raise AssertionError("unreachable")


def thresholds_from_sorted(sorted_scores: np.ndarray, target_fpr: float) -> np.ndarray:
    """Vectorized equivalent of :func:`threshold_at_fpr`, including ties."""
    n_cal = sorted_scores.shape[1]
    max_fp = int(math.floor((target_fpr + 1e-15) * n_cal))
    maxima = sorted_scores[:, -1]
    if max_fp == 0:
        return np.nextafter(maxima, np.inf)

    candidate = sorted_scores[:, n_cal - max_fp]
    n_at_or_above = np.count_nonzero(
        sorted_scores >= candidate[:, None], axis=1
    )
    valid = n_at_or_above <= max_fp
    greater = np.where(sorted_scores > candidate[:, None], sorted_scores, np.inf)
    next_greater = np.min(greater, axis=1)
    fallback = np.nextafter(maxima, np.inf)
    next_greater = np.where(np.isfinite(next_greater), next_greater, fallback)
    return np.where(valid, candidate, next_greater)


def metrics_from_counts(tp: np.ndarray, fp: np.ndarray, n_unsafe: int, n_safe: int):
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = n_unsafe - tp
    tpr = tp / n_unsafe
    fpr = fp / n_safe
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0,
    )
    return tpr, fpr, f1


def point_metrics(method: dict, tau: float) -> dict:
    pred_unsafe = method["test_unsafe"] >= tau
    pred_safe = method["test_safe"] >= tau
    tp = int(pred_unsafe.sum())
    fp = int(pred_safe.sum())
    n_unsafe = len(pred_unsafe)
    n_safe = len(pred_safe)
    tpr, fpr, f1 = metrics_from_counts(
        np.asarray([tp]), np.asarray([fp]), n_unsafe, n_safe
    )
    return {
        "tau": float(tau),
        "tpr": float(tpr[0]),
        "fpr": float(fpr[0]),
        "f1": float(f1[0]),
        "tp": tp,
        "fp": fp,
        "fn": n_unsafe - tp,
        "tn": n_safe - fp,
    }


def percentile_summary(values: np.ndarray, ci: float) -> dict:
    values = finite_1d(values, "bootstrap values")
    alpha = (100.0 - ci) / 2.0
    quantile_levels = np.asarray([0.0, 2.5, 5.0, 25.0, 50.0, 75.0, 95.0, 97.5, 100.0])
    quantiles = np.percentile(values, quantile_levels)
    ci_lower, ci_upper = np.percentile(values, [alpha, 100.0 - alpha])
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_level": float(ci),
        "ci_method": "percentile nested bootstrap",
        "quantiles": {
            f"q{level:g}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def tau_summary(values: np.ndarray, ci: float) -> dict:
    out = percentile_summary(values, ci)
    unique, counts = np.unique(values, return_counts=True)
    out["n_unique"] = int(len(unique))
    out["distribution"] = [
        {
            "tau": float(value),
            "count": int(count),
            "probability": float(count / len(values)),
        }
        for value, count in zip(unique, counts)
    ]
    return out


def load_benchmark_methods(
    delta: Path,
    qwen_delta: Path,
    baseline_delta: Path,
    benchmark: str,
) -> dict[str, dict]:
    spec = BENCHMARKS[benchmark]
    split_path = modern_work_dir(delta, benchmark) / spec["split_file"]
    split = load_json(split_path)
    expected_unsafe_ids = [int(value) for value in split[spec["test_unsafe_key"]]]
    expected_safe_ids = [int(value) for value in split[spec["test_safe_key"]]]

    methods = {
        "S_C aligned": load_modern_method(delta, benchmark, "S_C", "S_C aligned"),
        "LEARNED fusion": load_modern_method(
            delta, benchmark, "LEARNED", "LEARNED fusion"
        ),
        "Qwen S_C aligned": load_modern_method(
            qwen_delta, benchmark, "S_C", "Qwen S_C aligned"
        ),
    }
    expected_cal_ids = methods["S_C aligned"]["cal_ids"]
    for display_name in METHOD_ORDER[3:]:
        methods[display_name] = load_baseline_method(
            baseline_delta,
            benchmark,
            display_name,
            expected_cal_ids,
            expected_unsafe_ids,
            expected_safe_ids,
        )

    reference_cal_ids = methods[METHOD_ORDER[0]]["cal_ids"]
    for display_name in METHOD_ORDER:
        method = methods[display_name]
        if method["cal_ids"] != reference_cal_ids:
            raise RuntimeError(
                f"{benchmark}: calibration order differs for {display_name}"
            )
        if len(method["cal_safe"]) != 200:
            raise RuntimeError(
                f"{benchmark}/{display_name}: expected 200 benign calibration "
                f"samples, found {len(method['cal_safe'])}"
            )
        if len(method["test_unsafe"]) != len(expected_unsafe_ids):
            raise RuntimeError(f"{benchmark}/{display_name}: unsafe test mismatch")
        if len(method["test_safe"]) != len(expected_safe_ids):
            raise RuntimeError(f"{benchmark}/{display_name}: safe test mismatch")
        for op_name, target_fpr in OPS:
            recomputed = threshold_at_fpr(method["cal_safe"], target_fpr)
            frozen = method["taus_frozen"][op_name]
            if not np.isclose(recomputed, frozen, rtol=1e-12, atol=1e-15):
                raise RuntimeError(
                    f"{benchmark}/{display_name}/{op_name}: recomputed tau "
                    f"{recomputed:.17g} != frozen tau {frozen:.17g}"
                )
    return methods


def bootstrap_benchmark(
    methods: dict[str, dict],
    benchmark: str,
    n_boot: int,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    n_cal = len(methods[METHOD_ORDER[0]]["cal_safe"])
    n_unsafe = len(methods[METHOD_ORDER[0]]["test_unsafe"])
    n_safe = len(methods[METHOD_ORDER[0]]["test_safe"])
    output = {
        method: {
            op_name: {
                "tau": np.empty(n_boot, dtype=np.float64),
                "tpr": np.empty(n_boot, dtype=np.float64),
                "fpr": np.empty(n_boot, dtype=np.float64),
                "f1": np.empty(n_boot, dtype=np.float64),
            }
            for op_name, _ in OPS
        }
        for method in METHOD_ORDER
    }

    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        size = stop - start
        # The three index matrices are deliberately shared by all methods and
        # both operating points: this is a paired nested bootstrap.
        cal_draw = rng.integers(0, n_cal, size=(size, n_cal), dtype=np.int32)
        unsafe_draw = rng.integers(
            0, n_unsafe, size=(size, n_unsafe), dtype=np.int32
        )
        safe_draw = rng.integers(0, n_safe, size=(size, n_safe), dtype=np.int32)

        for display_name in METHOD_ORDER:
            method = methods[display_name]
            sorted_cal = np.sort(method["cal_safe"][cal_draw], axis=1)
            sampled_unsafe = method["test_unsafe"][unsafe_draw]
            sampled_safe = method["test_safe"][safe_draw]
            for op_name, target_fpr in OPS:
                tau = thresholds_from_sorted(sorted_cal, target_fpr)
                tp = np.count_nonzero(sampled_unsafe >= tau[:, None], axis=1)
                fp = np.count_nonzero(sampled_safe >= tau[:, None], axis=1)
                tpr, fpr, f1 = metrics_from_counts(tp, fp, n_unsafe, n_safe)
                block = output[display_name][op_name]
                block["tau"][start:stop] = tau
                block["tpr"][start:stop] = tpr
                block["fpr"][start:stop] = fpr
                block["f1"][start:stop] = f1

        print(f"[{benchmark}] bootstrap {stop}/{n_boot}", flush=True)
    return output


def make_results(
    all_methods: dict[str, dict[str, dict]],
    raw: dict[str, dict],
    n_boot: int,
    seed: int,
    ci: float,
    paths: dict[str, Path],
) -> dict:
    results = {
        "protocol": {
            "name": "nested calibration bootstrap",
            "dataset_version": "v13 paper split (not group-disjoint)",
            "n_boot": n_boot,
            "seed": seed,
            "ci": ci,
            "ci_method": "percentile nested bootstrap",
            "calibration_resampling": (
                "resample 200 benign operating_cal examples with replacement; "
                "recompute tau in every replicate"
            ),
            "test_resampling": (
                "stratified resampling with replacement of unsafe and benign test sets"
            ),
            "pairing": (
                "the same calibration and test index draws are used for every method"
            ),
            "threshold_rule": (
                "smallest empirical threshold with safe-cal FPR <= target; "
                "unsafe iff score >= tau"
            ),
            "learned_fusion": (
                "fusion model frozen from the disjoint fusion_fit subset; only tau recalibrated"
            ),
            "metrics": ["TPR", "FPR", "F1"],
        },
        "inputs": {key: str(value) for key, value in paths.items()},
        "benchmarks": {},
    }
    for benchmark in BENCHMARKS:
        methods = all_methods[benchmark]
        bench = {
            "n_cal_safe": len(methods[METHOD_ORDER[0]]["cal_safe"]),
            "n_test_unsafe": len(methods[METHOD_ORDER[0]]["test_unsafe"]),
            "n_test_safe": len(methods[METHOD_ORDER[0]]["test_safe"]),
            "operating_points": {},
        }
        for op_name, target_fpr in OPS:
            op = {"target_fpr": target_fpr, "methods": {}}
            for display_name in METHOD_ORDER:
                method = methods[display_name]
                tau_point = threshold_at_fpr(method["cal_safe"], target_fpr)
                arrays = raw[benchmark][display_name][op_name]
                op["methods"][display_name] = {
                    "point": point_metrics(method, tau_point),
                    "tau_bootstrap": tau_summary(arrays["tau"], ci),
                    "metrics_bootstrap": {
                        metric: percentile_summary(arrays[metric], ci)
                        for metric in ("tpr", "fpr", "f1")
                    },
                    "sources": method["sources"],
                }
            bench["operating_points"][op_name] = op
        results["benchmarks"][benchmark] = bench
    return results


def write_summary_csv(results: dict, path: Path) -> None:
    fields = [
        "benchmark",
        "operating_point",
        "target_fpr",
        "method",
        "n_cal_safe",
        "n_test_unsafe",
        "n_test_safe",
        "tau_point",
        "tau_boot_mean",
        "tau_boot_sd",
        "tau_boot_median",
        "tau_ci_lower",
        "tau_ci_upper",
        "tau_n_unique",
        "tpr_point",
        "tpr_ci_lower",
        "tpr_ci_upper",
        "fpr_point",
        "fpr_ci_lower",
        "fpr_ci_upper",
        "f1_point",
        "f1_ci_lower",
        "f1_ci_upper",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark, bench in results["benchmarks"].items():
            for op_name, op in bench["operating_points"].items():
                for display_name in METHOD_ORDER:
                    method = op["methods"][display_name]
                    tau = method["tau_bootstrap"]
                    metrics = method["metrics_bootstrap"]
                    point = method["point"]
                    writer.writerow(
                        {
                            "benchmark": benchmark,
                            "operating_point": op_name,
                            "target_fpr": op["target_fpr"],
                            "method": display_name,
                            "n_cal_safe": bench["n_cal_safe"],
                            "n_test_unsafe": bench["n_test_unsafe"],
                            "n_test_safe": bench["n_test_safe"],
                            "tau_point": point["tau"],
                            "tau_boot_mean": tau["mean"],
                            "tau_boot_sd": tau["sd"],
                            "tau_boot_median": tau["median"],
                            "tau_ci_lower": tau["ci_lower"],
                            "tau_ci_upper": tau["ci_upper"],
                            "tau_n_unique": tau["n_unique"],
                            "tpr_point": point["tpr"],
                            "tpr_ci_lower": metrics["tpr"]["ci_lower"],
                            "tpr_ci_upper": metrics["tpr"]["ci_upper"],
                            "fpr_point": point["fpr"],
                            "fpr_ci_lower": metrics["fpr"]["ci_lower"],
                            "fpr_ci_upper": metrics["fpr"]["ci_upper"],
                            "f1_point": point["f1"],
                            "f1_ci_lower": metrics["f1"]["ci_lower"],
                            "f1_ci_upper": metrics["f1"]["ci_upper"],
                        }
                    )


def write_replicates_csv_gz(raw: dict, path: Path) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "replicate",
                "benchmark",
                "operating_point",
                "target_fpr",
                "method",
                "tau",
                "tpr",
                "fpr",
                "f1",
            ]
        )
        target_by_op = dict(OPS)
        for benchmark in BENCHMARKS:
            for op_name, _ in OPS:
                for display_name in METHOD_ORDER:
                    block = raw[benchmark][display_name][op_name]
                    for index in range(len(block["tau"])):
                        writer.writerow(
                            [
                                index,
                                benchmark,
                                op_name,
                                target_by_op[op_name],
                                display_name,
                                block["tau"][index],
                                block["tpr"][index],
                                block["fpr"][index],
                                block["f1"][index],
                            ]
                        )


def write_replicates_npz(raw: dict, path: Path) -> None:
    arrays = {}
    for benchmark in BENCHMARKS:
        for display_name in METHOD_ORDER:
            method_id = METHOD_IDS[display_name]
            for op_name, _ in OPS:
                for metric, values in raw[benchmark][display_name][op_name].items():
                    arrays[f"{benchmark}__{op_name}__{method_id}__{metric}"] = values
    np.savez_compressed(path, **arrays)


def fmt_tau(point: float, summary: dict) -> str:
    return f"{point:.6g} [{summary['ci_lower']:.6g}, {summary['ci_upper']:.6g}]"


def fmt_metric(point: float, summary: dict, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    digits = 1 if percent else 3
    return (
        f"{scale * point:.{digits}f} "
        f"[{scale * summary['ci_lower']:.{digits}f}, "
        f"{scale * summary['ci_upper']:.{digits}f}]"
    )


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Nested calibration bootstrap — v13 paper split",
        "",
        (
            f"{results['protocol']['n_boot']} paired nested-bootstrap replicates; "
            f"{results['protocol']['ci']:.0f}% percentile intervals. In every "
            "replicate, the 200 benign operating-calibration examples are resampled, "
            "tau is recomputed, and the unsafe/benign test sets are resampled "
            "stratified."
        ),
        "",
        (
            "The learned-fusion classifier is frozen from the disjoint `fusion_fit` "
            "subset; its operating threshold is recalibrated in every replicate."
        ),
        "",
    ]
    for benchmark, bench in results["benchmarks"].items():
        lines.extend(
            [
                f"## {benchmark}",
                "",
                (
                    f"Calibration benign n={bench['n_cal_safe']}; test unsafe "
                    f"n={bench['n_test_unsafe']}; test benign n={bench['n_test_safe']}."
                ),
                "",
            ]
        )
        for op_name, op in bench["operating_points"].items():
            lines.extend(
                [
                    f"### {op_name} (target FPR {100 * op['target_fpr']:.0f}%)",
                    "",
                    "| Method | tau [95% CI] | TPR % [95% CI] | FPR % [95% CI] | F1 [95% CI] |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for display_name in METHOD_ORDER:
                method = op["methods"][display_name]
                point = method["point"]
                tau = method["tau_bootstrap"]
                metrics = method["metrics_bootstrap"]
                lines.append(
                    f"| {display_name} | {fmt_tau(point['tau'], tau)} | "
                    f"{fmt_metric(point['tpr'], metrics['tpr'], True)} | "
                    f"{fmt_metric(point['fpr'], metrics['fpr'], True)} | "
                    f"{fmt_metric(point['f1'], metrics['f1'])} |"
                )
            lines.append("")
    lines.extend(
        [
            "The full per-replicate distribution is in "
            "`nested_calibration_bootstrap_replicates.csv.gz`; the compact NumPy "
            "version is `nested_calibration_bootstrap_replicates.npz`. The JSON file "
            "also contains the exact discrete probability mass function of tau.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def write_latex(results: dict, path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Benchmark & Target & Method & $\\tau$ & TPR (\\%) & FPR (\\%) & F1 \\\\",
        "\\midrule",
    ]
    for benchmark, bench in results["benchmarks"].items():
        first_benchmark_row = True
        for op_name, op in bench["operating_points"].items():
            for method_index, display_name in enumerate(METHOD_ORDER):
                method = op["methods"][display_name]
                point = method["point"]
                tau = method["tau_bootstrap"]
                metrics = method["metrics_bootstrap"]
                benchmark_cell = tex_escape(benchmark) if first_benchmark_row else ""
                target_cell = (
                    f"{100 * op['target_fpr']:.0f}\\%" if method_index == 0 else ""
                )
                lines.append(
                    f"{benchmark_cell} & {target_cell} & {tex_escape(display_name)} & "
                    f"{fmt_tau(point['tau'], tau)} & "
                    f"{fmt_metric(point['tpr'], metrics['tpr'], True)} & "
                    f"{fmt_metric(point['fpr'], metrics['fpr'], True)} & "
                    f"{fmt_metric(point['f1'], metrics['f1'])} \\\\"
                )
                first_benchmark_row = False
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(
        [
            "\\end{tabular}",
            "\\caption{Nested calibration bootstrap on the v13 paper split. "
            "Each entry reports the original point estimate and the 95\\% "
            "percentile interval after jointly resampling the 200 benign "
            "operating-calibration samples and the stratified test set.}",
            "\\label{tab:nested_calibration_bootstrap}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else script_dir.parent.parent
    )
    delta = (
        Path(args.delta).resolve()
        if args.delta
        else script_dir / "artifacts/probes/aligned_last_v13/delta"
    )
    qwen_delta = (
        Path(args.qwen_delta).resolve()
        if args.qwen_delta
        else script_dir / "artifacts/probes/qwen_aligned_last_v13/delta"
    )
    baseline_delta = (
        Path(args.baseline_delta).resolve()
        if args.baseline_delta
        else project_root / "probes/rebuilt_v13/delta"
    )
    out_dir = (
        Path(args.out).resolve()
        if args.out
        else script_dir
        / "artifacts/probes/aligned_last_v13/analyses/nested_calibration_bootstrap"
    )
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not 0.0 < args.ci < 100.0:
        raise ValueError("--ci must be between 0 and 100")

    paths = {
        "project_root": project_root,
        "aligned_delta": delta,
        "qwen_delta": qwen_delta,
        "baseline_delta": baseline_delta,
    }
    print("NESTED CALIBRATION BOOTSTRAP — v13 paper split")
    for key, value in paths.items():
        print(f"{key:>16}: {value}")
    print(f"{'output':>16}: {out_dir}")
    print(f"{'replicates':>16}: {args.n_boot}")
    print(f"{'seed':>16}: {args.seed}")

    all_methods = {}
    raw = {}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        print(f"\nLoading and validating {benchmark} ...", flush=True)
        methods = load_benchmark_methods(
            delta, qwen_delta, baseline_delta, benchmark
        )
        all_methods[benchmark] = methods
        print(
            f"[{benchmark}] validated: cal_safe=200, "
            f"test_unsafe={len(methods[METHOD_ORDER[0]]['test_unsafe'])}, "
            f"test_safe={len(methods[METHOD_ORDER[0]]['test_safe'])}"
        )
        raw[benchmark] = bootstrap_benchmark(
            methods,
            benchmark,
            args.n_boot,
            args.seed + 1_000_000 * benchmark_index,
            args.chunk_size,
        )

    results = make_results(
        all_methods,
        raw,
        args.n_boot,
        args.seed,
        args.ci,
        paths,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "nested_calibration_bootstrap_results.json"
    csv_path = out_dir / "nested_calibration_bootstrap_summary.csv"
    replicate_csv_path = out_dir / "nested_calibration_bootstrap_replicates.csv.gz"
    replicate_npz_path = out_dir / "nested_calibration_bootstrap_replicates.npz"
    report_path = out_dir / "nested_calibration_bootstrap_report.md"
    latex_path = out_dir / "nested_calibration_bootstrap_table.tex"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    write_summary_csv(results, csv_path)
    write_replicates_csv_gz(raw, replicate_csv_path)
    write_replicates_npz(raw, replicate_npz_path)
    write_markdown(results, report_path)
    write_latex(results, latex_path)

    print("\nCOMPLETED")
    for path in (
        report_path,
        csv_path,
        json_path,
        replicate_csv_path,
        replicate_npz_path,
        latex_path,
    ):
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested calibration bootstrap for all v13 paper methods."
    )
    parser.add_argument(
        "--project-root",
        help="SAFETY repository root (auto-detected from this script by default)",
    )
    parser.add_argument("--delta", help="aligned_last_v13/delta directory")
    parser.add_argument("--qwen-delta", help="qwen_aligned_last_v13/delta directory")
    parser.add_argument("--baseline-delta", help="probes/rebuilt_v13/delta directory")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--chunk-size", type=int, default=500)
    run(parser.parse_args())
