#!/usr/bin/env python3
"""Evaluate fixed-text image substitutions with an existing operational S_C probe.

The script deliberately does not refit the probe and does not run the backbone.
It reuses the frozen aligned-last-token cache, applies the exact nuisance
projector stored with the operational probe, and reports paired bootstrap
intervals for SIUO, TextVQA, and their safety-specific contrast.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn


REPRESENTATION = "L2(L2(hTV_last)-L2(hT_last))"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path):
    """Load trusted experiment artifacts, including NumPy-2 pickles on NumPy 1."""
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as error:
        if error.name != "numpy._core":
            raise
        # NumPy 2 serializes arrays below numpy._core.  This compatibility path
        # is used only on older analysis hosts and leaves array values unchanged.
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        with path.open("rb") as handle:
            return pickle.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def aligned_features(h_tv: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    return unit_rows(unit_rows(h_tv) - unit_rows(h_t))


def ordered_items(rows: dict, pool: str) -> list[tuple[str, dict]]:
    if pool == "siuo":
        return sorted(
            rows.items(),
            key=lambda item: (int(item[1].get("original_index", 10**12)), str(item[0])),
        )
    return sorted(rows.items(), key=lambda item: int(item[0]))


def validate_cache(cache: dict) -> None:
    required = {"representation", "rows", "swaps", "swap_maps"}
    missing = required - set(cache)
    if missing:
        raise KeyError(f"Aligned cache is missing keys: {sorted(missing)}")
    for pool in ("siuo", "textvqa"):
        if pool not in cache["rows"]:
            raise KeyError(f"Aligned cache has no {pool} rows")
    for pool in ("siuo", "textvqa_test"):
        if pool not in cache["swaps"]:
            raise KeyError(f"Aligned cache has no {pool} substitutions")


def validate_probe(probe: dict, cache: dict) -> None:
    if "clf" not in probe:
        raise KeyError("Operational probe artifact has no 'clf'")
    mode = probe.get("sc_representation")
    formula = probe.get("representation")
    if mode != "aligned" and formula != REPRESENTATION:
        raise ValueError(
            f"Probe is not the aligned S_C representation: mode={mode!r}, formula={formula!r}"
        )
    cache_layer = cache["representation"].get("layer")
    probe_layer = probe.get("language_layer")
    if cache_layer is not None and probe_layer is not None and int(cache_layer) != int(probe_layer):
        raise ValueError(f"Probe/cache layer mismatch: probe={probe_layer}, cache={cache_layer}")


def pool_arrays(cache: dict, pool: str) -> tuple[list[str], list[dict], list[dict]]:
    swap_pool = "siuo" if pool == "siuo" else "textvqa_test"
    ordered = ordered_items(cache["rows"][pool], pool)
    ids = [key for key, _ in ordered if key in cache["swaps"][swap_pool]]
    rows = [cache["rows"][pool][key] for key in ids]
    swaps = [cache["swaps"][swap_pool][key] for key in ids]
    if not ids:
        raise ValueError(f"No paired records found for {pool}")
    if len(ids) != len(cache["swaps"][swap_pool]):
        missing = sorted(set(cache["swaps"][swap_pool]) - set(ids))
        raise ValueError(f"Unmatched substitutions in {pool}: {missing[:5]}")
    for key, row, swap in zip(ids, rows, swaps):
        if str(row["id"]) != str(key) or str(swap["id"]) != str(key):
            raise ValueError(f"Paired identifier mismatch for {pool}/{key}")
        if str(swap["image_from_id"]) == str(key):
            raise ValueError(f"Self substitution in {pool}/{key}")
        if int(row["last_token_id"]) != int(swap["last_token_id"]):
            raise ValueError(f"Final textual token changed in {pool}/{key}")
        if pool == "siuo" and str(row.get("category")) == str(swap.get("image_from_category")):
            raise ValueError(f"SIUO substitution is not cross-category for {key}")
        recorded_source = cache["swap_maps"][swap_pool].get(key)
        if str(recorded_source) != str(swap["image_from_id"]):
            raise ValueError(
                f"Frozen substitution map disagrees with cached state for {pool}/{key}"
            )
    if len({str(swap["image_from_id"]) for swap in swaps}) != len(swaps):
        raise ValueError(f"Substitution map for {pool} is not one-to-one")
    return ids, rows, swaps


def score_pool(probe: dict, rows: list[dict], swaps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    h_t = np.stack([row["hT_last"] for row in rows])
    original = np.stack([row["hTV_last"] for row in rows])
    substituted = np.stack([row["hTV_last_swap"] for row in swaps])
    original_features = aligned_features(original, h_t)
    swapped_features = aligned_features(substituted, h_t)
    projector = probe.get("ortho_Q")
    if projector is not None:
        projector = np.asarray(projector, dtype=np.float64)
        if projector.ndim != 2 or projector.shape[0] != original_features.shape[1]:
            raise ValueError(
                f"Invalid nuisance projector shape {projector.shape}; feature dim={original_features.shape[1]}"
            )
        original_features = original_features - (original_features @ projector) @ projector.T
        swapped_features = swapped_features - (swapped_features @ projector) @ projector.T
    clf = probe["clf"]
    original_scores = np.asarray(clf.predict_proba(original_features)[:, 1], dtype=np.float64)
    swapped_scores = np.asarray(clf.predict_proba(swapped_features)[:, 1], dtype=np.float64)
    if not np.all(np.isfinite(original_scores)) or not np.all(np.isfinite(swapped_scores)):
        raise ValueError("Probe emitted non-finite probabilities")
    return original_scores, swapped_scores


def interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def point_summary(drop: np.ndarray) -> dict:
    ties = np.isclose(drop, 0.0, atol=1e-12)
    return {
        "n": int(len(drop)),
        "mean_score_drop": float(drop.mean()),
        "median_score_drop": float(np.median(drop)),
        "probability_original_greater": float(np.mean(drop > 0) + 0.5 * np.mean(ties)),
        "n_positive_drop": int(np.sum(drop > 0)),
        "n_negative_drop": int(np.sum(drop < 0)),
        "n_ties": int(np.sum(ties)),
    }


def bootstrap(
    unsafe_drop: np.ndarray,
    benign_drop: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    unsafe_mean = np.empty(repetitions, dtype=np.float64)
    unsafe_sign = np.empty(repetitions, dtype=np.float64)
    benign_mean = np.empty(repetitions, dtype=np.float64)
    benign_sign = np.empty(repetitions, dtype=np.float64)
    contrast = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        unsafe = unsafe_drop[rng.integers(0, len(unsafe_drop), len(unsafe_drop))]
        benign = benign_drop[rng.integers(0, len(benign_drop), len(benign_drop))]
        unsafe_ties = np.isclose(unsafe, 0.0, atol=1e-12)
        benign_ties = np.isclose(benign, 0.0, atol=1e-12)
        unsafe_mean[index] = unsafe.mean()
        benign_mean[index] = benign.mean()
        unsafe_sign[index] = np.mean(unsafe > 0) + 0.5 * np.mean(unsafe_ties)
        benign_sign[index] = np.mean(benign > 0) + 0.5 * np.mean(benign_ties)
        contrast[index] = unsafe_mean[index] - benign_mean[index]
    return {
        "unsafe_mean_score_drop": interval(unsafe_mean),
        "unsafe_probability_original_greater": interval(unsafe_sign),
        "benign_mean_score_drop": interval(benign_mean),
        "benign_probability_original_greater": interval(benign_sign),
        "safety_specific_contrast": interval(contrast),
    }


def write_predictions(
    path: Path,
    pools: list[tuple[str, list[str], list[dict], list[dict], np.ndarray, np.ndarray]],
) -> None:
    fields = [
        "pool",
        "id",
        "category",
        "image_from_id",
        "image_from_category",
        "original_score",
        "substituted_score",
        "score_drop",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pool, ids, rows, swaps, original, substituted in pools:
            for index, key in enumerate(ids):
                writer.writerow(
                    {
                        "pool": pool,
                        "id": key,
                        "category": rows[index].get("category", "unknown"),
                        "image_from_id": swaps[index]["image_from_id"],
                        "image_from_category": swaps[index].get("image_from_category", ""),
                        "original_score": float(original[index]),
                        "substituted_score": float(substituted[index]),
                        "score_drop": float(original[index] - substituted[index]),
                    }
                )


def write_report(path: Path, result: dict) -> None:
    unsafe = result["paired"]["SIUO_unsafe"]
    benign = result["paired"]["TextVQA_benign"]
    contrast = result["safety_specific_contrast"]
    ci = result["bootstrap_confidence_intervals"]
    lines = [
        f"# Paired image substitution — {result['backbone']}",
        "",
        "The operational probe is frozen: no classifier fitting, calibration, or threshold selection is performed here.",
        "Every comparison keeps the text and final textual token fixed and changes only the paired image.",
        "",
        "| Population | n | mean original−substitution | 95% CI | P(original > substitution) | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
        f"| SIUO unsafe | {unsafe['n']} | {unsafe['mean_score_drop']:+.6f} | "
        f"[{ci['unsafe_mean_score_drop']['lower']:+.6f}, {ci['unsafe_mean_score_drop']['upper']:+.6f}] | "
        f"{unsafe['probability_original_greater']:.4f} | "
        f"[{ci['unsafe_probability_original_greater']['lower']:.4f}, {ci['unsafe_probability_original_greater']['upper']:.4f}] |",
        f"| TextVQA benign | {benign['n']} | {benign['mean_score_drop']:+.6f} | "
        f"[{ci['benign_mean_score_drop']['lower']:+.6f}, {ci['benign_mean_score_drop']['upper']:+.6f}] | "
        f"{benign['probability_original_greater']:.4f} | "
        f"[{ci['benign_probability_original_greater']['lower']:.4f}, {ci['benign_probability_original_greater']['upper']:.4f}] |",
        "",
        f"Safety-specific contrast (SIUO drop − TextVQA drop): **{contrast['point']:+.6f}** "
        f"(95% CI [{contrast['ci']['lower']:+.6f}, {contrast['ci']['upper']:+.6f}]).",
        "",
        f"Verdict: **{result['verdict']}**.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_boot < 1:
        raise ValueError("--n-boot must be positive")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cache = load_pickle(args.cache)
        probe = load_pickle(args.probe)
    compatibility_warnings = [str(item.message) for item in caught]
    validate_cache(cache)
    validate_probe(probe, cache)

    siuo_ids, siuo_rows, siuo_swaps = pool_arrays(cache, "siuo")
    text_ids, text_rows, text_swaps = pool_arrays(cache, "textvqa")
    siuo_original, siuo_substituted = score_pool(probe, siuo_rows, siuo_swaps)
    text_original, text_substituted = score_pool(probe, text_rows, text_swaps)
    unsafe_drop = siuo_original - siuo_substituted
    benign_drop = text_original - text_substituted
    intervals = bootstrap(unsafe_drop, benign_drop, args.n_boot, args.seed)
    unsafe = point_summary(unsafe_drop)
    benign = point_summary(benign_drop)
    contrast_point = float(unsafe_drop.mean() - benign_drop.mean())
    criteria = {
        "unsafe_mean_ci_above_zero": intervals["unsafe_mean_score_drop"]["lower"] > 0,
        "unsafe_sign_ci_above_half": intervals["unsafe_probability_original_greater"]["lower"] > 0.5,
        "safety_specific_contrast_ci_above_zero": intervals["safety_specific_contrast"]["lower"] > 0,
        "text_and_final_token_fixed": True,
        "probe_refit": False,
    }
    supported = all(
        value for key, value in criteria.items() if key not in {"probe_refit"}
    ) and criteria["probe_refit"] is False
    result = {
        "experiment": "paired_image_substitution_with_operational_probe",
        "backbone": args.backbone,
        "representation": REPRESENTATION,
        "probe": {
            "path": str(args.probe.resolve()),
            "sha256": sha256_file(args.probe),
            "language_layer": probe.get("language_layer"),
            "nuisance_projector": probe.get("ortho_Q") is not None,
            "refit": False,
        },
        "cache": {
            "path": str(args.cache.resolve()),
            "sha256": sha256_file(args.cache),
            "metadata": cache.get("representation", {}),
        },
        "protocol": {
            "substitution_map": "frozen one-to-one derangement",
            "siuo_constraint": "cross-category image substitution",
            "benign_reference": "TextVQA test-set image derangement",
            "bootstrap": "paired within population; independent SIUO/TextVQA resampling for the contrast",
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "paired": {"SIUO_unsafe": unsafe, "TextVQA_benign": benign},
        "safety_specific_contrast": {
            "definition": "mean(SIUO original-substitution) - mean(TextVQA original-substitution)",
            "point": contrast_point,
            "ci": intervals["safety_specific_contrast"],
        },
        "bootstrap_confidence_intervals": intervals,
        "criteria": criteria,
        "verdict": "SUPPORTED" if supported else "NOT_ALL_CRITERIA_PASSED",
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "artifact_compatibility_warnings": compatibility_warnings,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    dump_json(args.out / "paired_image_substitution_results.json", result)
    write_predictions(
        args.out / "paired_image_substitution_predictions.csv",
        [
            ("SIUO_unsafe", siuo_ids, siuo_rows, siuo_swaps, siuo_original, siuo_substituted),
            ("TextVQA_benign", text_ids, text_rows, text_swaps, text_original, text_substituted),
        ],
    )
    write_report(args.out / "PAIRED_IMAGE_SUBSTITUTION_REPORT.md", result)
    print(
        f"[{args.backbone}] unsafe drop={unsafe['mean_score_drop']:+.6f}; "
        f"benign drop={benign['mean_score_drop']:+.6f}; "
        f"contrast={contrast_point:+.6f}; verdict={result['verdict']}"
    )
    print(args.out / "paired_image_substitution_results.json")


if __name__ == "__main__":
    main()
