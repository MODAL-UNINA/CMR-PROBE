#!/usr/bin/env python3
"""
Matched 2x2 representation-control ablation for aligned last-token residuals.

Conditions:
  raw_no_proj:
      r_raw = L2(h_TV - h_T)

  raw_proj:
      (I - Q_raw Q_raw^T) r_raw

  aligned_no_proj:
      r_align = L2(L2(h_TV) - L2(h_T))

  aligned_proj:
      (I - Q_align Q_align^T) r_align

All four conditions use:
- the same verified final textual token;
- the same frozen train/validation/test split;
- the same source-by-label sample weights;
- the same StandardScaler + balanced linear LogisticRegression;
- the same validation-only sigmoid calibration;
- the same test examples.

Q_raw and Q_align are fitted separately on TRAIN ONLY with the same
nuisance-fitting algorithm.

Example:
python ablate_aligned_projection.py \
  --project-root /path/to/SAFETY \
  --cache experiments/representation_comparison_v1/aligned_last_token_results/aligned_last_token_layer17.pkl \
  --dataset-dir dataset/rebuilt_v13 \
  --manifest experiments/tmm_minimal_v1/data/split_manifest_13_new.csv \
  --manifest-split-column old_split \
  --internal-cache-dir artifacts/probes/aligned_v13/combo \
  --feature-cache-dir artifacts/probes/aligned_v13/feature_cache \
  --out artifacts/matched_representation_ablation_v13 \
  --bootstrap-reps 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_external_benchmarks import load_external_protocol
from compare_htv_residual import (
    collapse_source,
    fit_nuisance_basis,
    fitted_calibrated_model,
    load_cache,
    load_manifest,
    project,
    source_weights,
)

METHODS = (
    "raw_no_proj",
    "raw_proj",
    "aligned_no_proj",
    "aligned_proj",
)

RAW_BASE = "raw"
ALIGNED_BASE = "aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Original v13 paper dataset containing probe_train/val/test.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="v13 manifest containing group_id and old_split.",
    )
    parser.add_argument(
        "--manifest-split-column",
        choices=("old_split", "split"),
        default="old_split",
        help=(
            "Manifest column used as train/validation/test assignment. "
            "Use old_split for the original non-group-disjoint v13 paper split."
        ),
    )
    parser.add_argument(
        "--internal-cache-dir",
        type=Path,
        default=None,
        help=(
            "Non-group v13 combo cache directory containing "
            "train/val/test_dh.layer17.aligned.pkl."
        ),
    )
    parser.add_argument(
        "--internal-cache-pattern",
        default="{split}_dh.layer17.aligned.pkl",
        help="Filename pattern inside --internal-cache-dir.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Directory containing train.pkl/val.pkl/test.pkl with h_visual.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    return parser.parse_args()


def unit_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def base_representations(
    h_tv: np.ndarray,
    h_t: np.ndarray,
) -> dict[str, np.ndarray]:
    h_tv = np.asarray(h_tv, dtype=np.float64)
    h_t = np.asarray(h_t, dtype=np.float64)

    raw = unit_rows(h_tv - h_t)
    aligned = unit_rows(unit_rows(h_tv) - unit_rows(h_t))

    return {
        RAW_BASE: raw,
        ALIGNED_BASE: aligned,
    }


def arrays_from_rows(
    rows: list[dict],
    htv_key: str = "hTV_last",
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([row[htv_key] for row in rows]).astype(np.float64),
        np.stack([row["hT_last"] for row in rows]).astype(np.float64),
    )


def find_dataset_dir(root: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)

    candidates.extend(
        [
            root / "dataset" / "rebuilt_v13",
            root / "data" / "legacy_v13",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if all(
            (candidate / f"probe_{split}.json").is_file()
            for split in ("train", "val", "test")
        ):
            return candidate

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate the original v13 paper dataset.\n"
        "Pass --dataset-dir explicitly.\nTried:\n" + tried
    )


def load_raw_directory(directory: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        path = directory / f"probe_{split}.json"

        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        selected = payload.get("samples", payload)

        for row in selected:
            sid = str(row["id"])

            if sid in rows:
                raise ValueError(f"Duplicate raw v13 id: {sid}")

            rows[sid] = row

    return rows


def load_v13_manifest(
    path: Path,
    split_column: str,
) -> dict[str, dict[str, str]]:
    manifest = load_manifest(path)

    for sid, row in manifest.items():
        if split_column not in row:
            raise KeyError(
                f"Manifest row {sid} has no {split_column!r} column."
            )

        split = str(row[split_column]).strip().lower()
        row["split"] = "validation" if split == "val" else split

        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unknown {split_column} value for {sid}: {split!r}"
            )

    return manifest


def find_internal_cache_dir(
    root: Path,
    explicit: Path | None,
    pattern: str,
) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)

    candidates.extend(
        [
            Path(__file__).resolve().parent
            / "artifacts" / "probes" / "aligned_v13" / "combo",
            root / "artifacts" / "probes" / "aligned_v13" / "combo",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if all(
            (candidate / pattern.format(split=split)).is_file()
            for split in ("train", "val", "test")
        ):
            return candidate

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate the non-group v13 aligned combo caches.\n"
        "Pass --internal-cache-dir explicitly.\nTried:\n" + tried
    )


def load_internal_aligned_rows(
    directory: Path,
    pattern: str,
) -> dict[str, dict]:
    cached = load_cache(directory, pattern)
    rows = {}

    for sid, row in cached.items():
        if "h_combo" not in row or "h_solo" not in row:
            raise KeyError(
                f"Aligned v13 cache row {sid} lacks h_combo/h_solo."
            )

        rows[sid] = {
            **row,
            "hTV_last": row["h_combo"],
            "hT_last": row["h_solo"],
        }

    return rows


def find_feature_cache_dir(root: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)

    candidates.extend(
        [
            Path(__file__).resolve().parent
            / "artifacts" / "probes" / "aligned_v13" / "feature_cache",
            root / "artifacts" / "probes" / "aligned_v13" / "feature_cache",
            root / "probes" / "rebuilt_v13" / "feature_cache",
            root / "probes" / "rebuilt_v12" / "feature_cache",
            root / "results" / "feature_cache",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / f"{split}.pkl").exists() for split in ("train", "val", "test")):
            return candidate

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate visual feature cache with train.pkl/val.pkl/test.pkl.\n"
        "Pass --feature-cache-dir explicitly.\nTried:\n" + tried
    )


def build_internal_data(
    root: Path,
    cache: dict,
    manifest: dict,
    feature_cache_dir: Path,
    dataset_dir: Path,
    internal_cache_dir: Path,
    internal_cache_pattern: str,
) -> dict:
    # The large cache supplies only the frozen external rows and image swaps.
    # Internal states come from the original, non-group-disjoint v13 paper run.
    stored = load_internal_aligned_rows(
        internal_cache_dir,
        internal_cache_pattern,
    )

    if set(stored) != set(manifest):
        raise ValueError(
            "Aligned internal cache IDs do not match the frozen internal manifest."
        )

    raw_rows = load_raw_directory(dataset_dir)
    visual_rows = load_cache(feature_cache_dir, "{split}.pkl")

    if set(raw_rows) != set(manifest):
        raise ValueError("Raw internal dataset IDs do not match frozen manifest.")
    if set(visual_rows) != set(manifest):
        raise ValueError(
            "Visual feature-cache IDs do not match frozen manifest. "
            "Use the cache corresponding to the same frozen split."
        )

    ids_by_split = {
        split: sorted(
            sid for sid, row in manifest.items()
            if row["split"] == split
        )
        for split in ("train", "validation", "test")
    }

    data = {}

    for split, split_ids in ids_by_split.items():
        aligned_rows = [stored[sid] for sid in split_ids]
        h_tv, h_t = arrays_from_rows(aligned_rows)
        manifest_rows = [manifest[sid] for sid in split_ids]

        data[split] = {
            "ids": split_ids,
            "rows": manifest_rows,
            "h_tv": h_tv,
            "h_t": h_t,
            "base_features": base_representations(h_tv, h_t),
            "y": np.asarray(
                [int(manifest[sid]["label_combo"]) for sid in split_ids],
                dtype=int,
            ),
            "sources": [
                collapse_source(manifest[sid]["source"])
                for sid in split_ids
            ],
            "categories": np.asarray(
                [manifest[sid]["category"] for sid in split_ids]
            ),
            "texts": [
                str(raw_rows[sid].get("text", ""))
                for sid in split_ids
            ],
            "h_visual": np.stack(
                [visual_rows[sid]["h_visual"] for sid in split_ids]
            ).astype(np.float64),
        }

    return data


def fit_matched_models(data: dict) -> tuple[dict, dict, dict]:
    train = data["train"]
    validation = data["validation"]

    weights = source_weights(train["y"], train["sources"])

    bases = {}
    for base in (RAW_BASE, ALIGNED_BASE):
        print(f"[fit] nuisance basis: {base}", flush=True)
        bases[base] = fit_nuisance_basis(
            train["base_features"][base],
            train["categories"],
            train["sources"],
            train["texts"],
            train["h_visual"],
        )

    for split in ("train", "validation", "test"):
        base = data[split]["base_features"]

        data[split]["features"] = {
            "raw_no_proj": base[RAW_BASE],
            "raw_proj": project(base[RAW_BASE], bases[RAW_BASE]),
            "aligned_no_proj": base[ALIGNED_BASE],
            "aligned_proj": project(base[ALIGNED_BASE], bases[ALIGNED_BASE]),
        }

    models = {}
    for method in METHODS:
        print(f"[fit] calibrated linear probe: {method}", flush=True)
        models[method] = fitted_calibrated_model(
            data["train"]["features"][method],
            data["train"]["y"],
            data["validation"]["features"][method],
            data["validation"]["y"],
            weights,
        )

    basis_dimensions = {
        "Q_raw": 0 if bases[RAW_BASE] is None else int(bases[RAW_BASE].shape[1]),
        "Q_aligned": (
            0 if bases[ALIGNED_BASE] is None
            else int(bases[ALIGNED_BASE].shape[1])
        ),
    }

    return models, bases, basis_dimensions


def transformed_features(
    h_tv: np.ndarray,
    h_t: np.ndarray,
    bases: dict,
) -> dict[str, np.ndarray]:
    base = base_representations(h_tv, h_t)

    return {
        "raw_no_proj": base[RAW_BASE],
        "raw_proj": project(base[RAW_BASE], bases[RAW_BASE]),
        "aligned_no_proj": base[ALIGNED_BASE],
        "aligned_proj": project(base[ALIGNED_BASE], bases[ALIGNED_BASE]),
    }


def score_arrays(
    h_tv: np.ndarray,
    h_t: np.ndarray,
    models: dict,
    bases: dict,
) -> dict[str, np.ndarray]:
    features = transformed_features(h_tv, h_t, bases)

    return {
        method: models[method].predict_proba(features[method])[:, 1]
        for method in METHODS
    }


def score_rows(
    rows: list[dict],
    models: dict,
    bases: dict,
) -> dict[str, np.ndarray]:
    h_tv, h_t = arrays_from_rows(rows)
    return score_arrays(h_tv, h_t, models, bases)


def score_swaps(
    original_rows: list[dict],
    swap_rows: list[dict],
    models: dict,
    bases: dict,
) -> dict[str, np.ndarray]:
    h_tv = np.stack(
        [row["hTV_last_swap"] for row in swap_rows]
    ).astype(np.float64)

    h_t = np.stack(
        [row["hT_last"] for row in original_rows]
    ).astype(np.float64)

    return score_arrays(h_tv, h_t, models, bases)


def metric_pair(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def confidence_intervals(values: dict[str, list[float]]) -> dict:
    return {
        key: {
            "lower": float(np.percentile(series, 2.5)),
            "upper": float(np.percentile(series, 97.5)),
        }
        for key, series in values.items()
    }


def paired_swap_bootstrap(
    original: dict[str, np.ndarray],
    swapped: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> tuple[dict, dict]:
    differences = {
        method: original[method] - swapped[method]
        for method in METHODS
    }

    point = {}
    for method, delta in differences.items():
        ties = np.isclose(delta, 0.0, atol=1e-12)

        point[method] = {
            "mean_score_change": float(delta.mean()),
            "median_score_change": float(np.median(delta)),
            "p_win": float(
                np.mean(delta > 0) + 0.5 * np.mean(ties)
            ),
            "n_positive_change": int(np.sum(delta > 0)),
            "n_negative_change": int(np.sum(delta < 0)),
            "n_ties": int(np.sum(ties)),
        }

    rng = np.random.default_rng(seed)
    n = len(next(iter(differences.values())))
    values: dict[str, list[float]] = defaultdict(list)

    for rep in range(repetitions):
        idx = rng.integers(0, n, n)

        for method, delta in differences.items():
            db = delta[idx]
            ties = np.isclose(db, 0.0, atol=1e-12)

            values[f"{method}.mean_score_change"].append(
                float(db.mean())
            )
            values[f"{method}.p_win"].append(
                float(np.mean(db > 0) + 0.5 * np.mean(ties))
            )

        if (rep + 1) % 2000 == 0:
            print(
                f"[bootstrap] paired swaps {rep + 1}/{repetitions}",
                flush=True,
            )

    return point, confidence_intervals(values)


def difference_in_differences(
    unsafe_original: dict[str, np.ndarray],
    unsafe_swapped: dict[str, np.ndarray],
    safe_original: dict[str, np.ndarray],
    safe_swapped: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> dict:
    unsafe_drop = {
        method: unsafe_original[method] - unsafe_swapped[method]
        for method in METHODS
    }
    safe_drop = {
        method: safe_original[method] - safe_swapped[method]
        for method in METHODS
    }

    point = {
        method: float(
            unsafe_drop[method].mean()
            - safe_drop[method].mean()
        )
        for method in METHODS
    }

    rng = np.random.default_rng(seed)
    nu = len(next(iter(unsafe_drop.values())))
    ns = len(next(iter(safe_drop.values())))
    values: dict[str, list[float]] = defaultdict(list)

    for rep in range(repetitions):
        ui = rng.integers(0, nu, nu)
        si = rng.integers(0, ns, ns)

        for method in METHODS:
            values[method].append(
                float(
                    unsafe_drop[method][ui].mean()
                    - safe_drop[method][si].mean()
                )
            )

        if (rep + 1) % 2000 == 0:
            print(
                f"[bootstrap] difference-in-differences "
                f"{rep + 1}/{repetitions}",
                flush=True,
            )

    return {
        "point": point,
        "confidence_intervals": confidence_intervals(values),
    }


def group_bootstrap_internal(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    members: dict[str, list[int]] = defaultdict(list)

    for i, group in enumerate(groups):
        members[str(group)].append(i)

    group_names = np.asarray(sorted(members), dtype=object)
    group_indices = {
        group: np.asarray(members[str(group)], dtype=int)
        for group in group_names
    }

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)

    accepted = 0

    while accepted < repetitions:
        chosen = rng.choice(
            group_names,
            size=len(group_names),
            replace=True,
        )

        idx = np.concatenate(
            [group_indices[group] for group in chosen]
        )

        if len(np.unique(y[idx])) < 2:
            continue

        for method in METHODS:
            metrics = metric_pair(y[idx], scores[method][idx])

            for metric, value in metrics.items():
                values[f"{method}.{metric}"].append(value)

        accepted += 1

    return confidence_intervals(values)


def internal_results(
    data: dict,
    models: dict,
    repetitions: int,
    seed: int,
) -> dict:
    test = data["test"]

    scores = {
        method: models[method].predict_proba(
            test["features"][method]
        )[:, 1]
        for method in METHODS
    }

    categories = np.asarray(
        [row["category"] for row in test["rows"]]
    )

    is_msts = np.asarray(
        [
            str(row.get("is_msts", False)).strip().lower()
            in {"1", "true", "yes", "y"}
            for row in test["rows"]
        ]
    )

    masks = {
        "C_vs_D_in_family": (
            (~is_msts) & np.isin(categories, ["C", "D"])
        ),
        "all_categories_in_family": ~is_msts,
        "MSTS_vs_in_family_C": (
            is_msts
            | ((~is_msts) & (categories == "C"))
        ),
    }

    output = {}

    for offset, (name, mask) in enumerate(masks.items()):
        y = test["y"][mask]

        selected_scores = {
            method: score[mask]
            for method, score in scores.items()
        }

        groups = np.asarray(
            [
                row["group_id"]
                for row, keep in zip(test["rows"], mask)
                if keep
            ]
        )

        output[name] = {
            "n": int(mask.sum()),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "n_groups": int(len(set(map(str, groups)))),
            "metrics": {
                method: metric_pair(y, score)
                for method, score in selected_scores.items()
            },
            "confidence_intervals": group_bootstrap_internal(
                y,
                selected_scores,
                groups,
                repetitions,
                seed + offset,
            ),
        }

    return output


def load_external_rows(
    root: Path,
    cache: dict,
) -> tuple[dict, dict]:
    splits, split_path, siuo_meta, siuo_path = load_external_protocol(root)

    keys = {
        "TextVQA_cal": list(map(str, splits["textvqa_cal"])),
        "TextVQA_test": list(map(str, splits["textvqa_test"])),
        "MM-SafetyBench_SD_test": list(
            map(str, splits["mmsafety_test"])
        ),
        "SIUO_test": sorted(
            siuo_meta,
            key=lambda key: siuo_meta[key]["original_index"],
        ),
    }

    pool_lookup = {
        "TextVQA_cal": "textvqa",
        "TextVQA_test": "textvqa",
        "MM-SafetyBench_SD_test": "mmsafety",
        "SIUO_test": "siuo",
    }

    rows = {}

    for name, pool_keys in keys.items():
        pool = pool_lookup[name]

        missing = [
            key
            for key in pool_keys
            if key not in cache["rows"][pool]
        ]

        if missing:
            raise RuntimeError(
                f"{name}: {len(missing)} rows missing from aligned cache. "
                f"Examples: {missing[:5]}"
            )

        rows[name] = [
            cache["rows"][pool][key]
            for key in pool_keys
        ]

    metadata = {
        "split_file": str(split_path),
        "siuo_frozen_ids": str(siuo_path),
        "keys": keys,
    }

    return rows, metadata


def build_swap_rows(
    cache: dict,
    external_metadata: dict,
) -> tuple[list[dict], list[dict]]:
    keys = external_metadata["keys"]

    text_rows = [
        {
            **cache["swaps"]["textvqa_test"][key],
            "hT_last": cache["rows"]["textvqa"][key]["hT_last"],
        }
        for key in keys["TextVQA_test"]
    ]

    siuo_rows = [
        {
            **cache["swaps"]["siuo"][key],
            "hT_last": cache["rows"]["siuo"][key]["hT_last"],
        }
        for key in keys["SIUO_test"]
    ]

    return text_rows, siuo_rows


def write_predictions(
    path: Path,
    external_rows: dict,
    original_scores: dict,
    text_swap_scores: dict,
    siuo_swap_scores: dict,
) -> None:
    fieldnames = [
        "pool",
        "position",
        "method",
        "original_score",
        "swapped_score",
        "original_minus_swap",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for pool_name, swap_scores in (
            ("TextVQA_test", text_swap_scores),
            ("SIUO_test", siuo_swap_scores),
        ):
            n = len(external_rows[pool_name])

            for i in range(n):
                for method in METHODS:
                    original = float(
                        original_scores[pool_name][method][i]
                    )
                    swapped = float(
                        swap_scores[method][i]
                    )

                    writer.writerow(
                        {
                            "pool": pool_name,
                            "position": i,
                            "method": method,
                            "original_score": original,
                            "swapped_score": swapped,
                            "original_minus_swap": (
                                original - swapped
                            ),
                        }
                    )


def fmt_ci(block: dict, key: str) -> str:
    ci = block["confidence_intervals"][key]
    return f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}]"


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Matched representation-control ablation",
        "",
        "This experiment isolates two factors:",
        "",
        "1. residual definition: raw vs aligned;",
        "2. nuisance projection: absent vs train-fitted projection.",
        "",
        "All four conditions use the same frozen split, source-by-label "
        "weights, StandardScaler + balanced linear logistic probe, and "
        "validation-only sigmoid calibration.",
        "",
        "## Nuisance bases",
        "",
        f"- Q_raw dimension: {result['basis_dimensions']['Q_raw']}",
        f"- Q_aligned dimension: {result['basis_dimensions']['Q_aligned']}",
        "",
        "## Internal frozen test",
        "",
    ]

    for population_name, population in result["internal"].items():
        lines.extend(
            [
                f"### {population_name}",
                "",
                "| Condition | ROC-AUC | 95% CI | PR-AUC | 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for method in METHODS:
            metrics = population["metrics"][method]

            lines.append(
                f"| {method} | "
                f"{metrics['roc_auc']:.4f} | "
                f"{fmt_ci(population, method + '.roc_auc')} | "
                f"{metrics['pr_auc']:.4f} | "
                f"{fmt_ci(population, method + '.pr_auc')} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Paired image-swap analysis",
            "",
            "Positive change means that the original pair receives a "
            "higher calibrated score than its fixed-text image substitution.",
            "",
        ]
    )

    for pool_name, block in result["paired_swaps"].items():
        lines.extend(
            [
                f"### {pool_name}",
                "",
                "| Condition | mean original-swap | 95% CI | p_win | 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for method in METHODS:
            point = block["point"][method]

            lines.append(
                f"| {method} | "
                f"{point['mean_score_change']:+.4f} | "
                f"{fmt_ci(block, method + '.mean_score_change')} | "
                f"{point['p_win']:.4f} | "
                f"{fmt_ci(block, method + '.p_win')} |"
            )

        lines.append("")

    did = result["difference_in_differences"]

    lines.extend(
        [
            "## Safety-specific difference-in-differences",
            "",
            "Defined as mean SIUO(original-swap) minus "
            "mean TextVQA(original-swap).",
            "",
            "| Condition | DiD | 95% CI |",
            "|---|---:|---:|",
        ]
    )

    for method in METHODS:
        ci = did["confidence_intervals"][method]

        lines.append(
            f"| {method} | "
            f"{did['point'][method]:+.4f} | "
            f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`raw_no_proj -> raw_proj` isolates the effect of nuisance "
            "projection for the raw residual. "
            "`aligned_no_proj -> aligned_proj` isolates the projection effect "
            "for the aligned residual. Comparing raw and aligned under the "
            "same projection status isolates the residual-definition effect.",
            "",
            "Calibrated mean changes are descriptive. p_win is reported "
            "because it is invariant to monotone score rescaling within each "
            "fitted model.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    root = args.project_root.resolve()
    cache_path = args.cache.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    dataset_dir = find_dataset_dir(
        root,
        args.dataset_dir,
    )

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else (
            root
            / "experiments" / "tmm_minimal_v1" / "data"
            / "split_manifest_13_new.csv"
        ).resolve()
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    internal_cache_dir = find_internal_cache_dir(
        root,
        args.internal_cache_dir,
        args.internal_cache_pattern,
    )

    feature_cache_dir = find_feature_cache_dir(
        root,
        args.feature_cache_dir,
    )

    print(f"[input] aligned cache: {cache_path}", flush=True)
    print(f"[input] v13 dataset:   {dataset_dir}", flush=True)
    print(
        f"[input] v13 manifest:  {manifest_path} "
        f"({args.manifest_split_column})",
        flush=True,
    )
    print(f"[input] v13 hT/hTV:    {internal_cache_dir}", flush=True)
    print(f"[input] visual cache:  {feature_cache_dir}", flush=True)

    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)

    manifest = load_v13_manifest(
        manifest_path,
        args.manifest_split_column,
    )

    data = build_internal_data(
        root=root,
        cache=cache,
        manifest=manifest,
        feature_cache_dir=feature_cache_dir,
        dataset_dir=dataset_dir,
        internal_cache_dir=internal_cache_dir,
        internal_cache_pattern=args.internal_cache_pattern,
    )

    models, bases, basis_dimensions = fit_matched_models(data)

    internal = internal_results(
        data,
        models,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )

    external_rows, external_metadata = load_external_rows(
        root,
        cache,
    )

    original_scores = {
        name: score_rows(rows, models, bases)
        for name, rows in external_rows.items()
    }

    text_swap_rows, siuo_swap_rows = build_swap_rows(
        cache,
        external_metadata,
    )

    text_swap_scores = score_swaps(
        external_rows["TextVQA_test"],
        text_swap_rows,
        models,
        bases,
    )

    siuo_swap_scores = score_swaps(
        external_rows["SIUO_test"],
        siuo_swap_rows,
        models,
        bases,
    )

    text_point, text_ci = paired_swap_bootstrap(
        original_scores["TextVQA_test"],
        text_swap_scores,
        args.bootstrap_reps,
        args.bootstrap_seed + 10,
    )

    siuo_point, siuo_ci = paired_swap_bootstrap(
        original_scores["SIUO_test"],
        siuo_swap_scores,
        args.bootstrap_reps,
        args.bootstrap_seed + 11,
    )

    paired_swaps = {
        "TextVQA_safe": {
            "point": text_point,
            "confidence_intervals": text_ci,
        },
        "SIUO_unsafe": {
            "point": siuo_point,
            "confidence_intervals": siuo_ci,
        },
    }

    did = difference_in_differences(
        original_scores["SIUO_test"],
        siuo_swap_scores,
        original_scores["TextVQA_test"],
        text_swap_scores,
        args.bootstrap_reps,
        args.bootstrap_seed + 12,
    )

    result = {
        "analysis_status": "MATCHED_REPRESENTATION_CONTROL_ABLATION",
        "representation_definition": {
            "raw": "L2(h_TV - h_T)",
            "aligned": "L2(L2(h_TV) - L2(h_T))",
            "token": (
                "same verified final textual prompt token in "
                "text-only and multimodal passes"
            ),
        },
        "conditions": {
            "raw_no_proj": (
                "raw residual without nuisance projection"
            ),
            "raw_proj": (
                "raw residual with train-fitted Q_raw projection"
            ),
            "aligned_no_proj": (
                "aligned residual without nuisance projection"
            ),
            "aligned_proj": (
                "aligned residual with train-fitted Q_aligned projection"
            ),
        },
        "matched_protocol": {
            "split": (
                "same original non-group-disjoint v13 paper "
                "train/validation/test split"
            ),
            "target": "label_combo",
            "sample_weights": (
                "same source-by-label training weights "
                "for every condition"
            ),
            "classifier": (
                "StandardScaler + LogisticRegression("
                "C=1, class_weight=balanced, random_state=42)"
            ),
            "calibration": "sigmoid on validation only",
            "nuisance_fit": (
                "Q_raw and Q_aligned fitted separately "
                "on training only with the same algorithm"
            ),
            "test_usage": (
                "no test sample used for nuisance fitting, "
                "classifier fitting, standardization, "
                "or sigmoid calibration"
            ),
        },
        "basis_dimensions": basis_dimensions,
        "split_counts": {
            split: len(data[split]["ids"])
            for split in ("train", "validation", "test")
        },
        "source_artifacts": {
            "project_root": str(root),
            "aligned_cache": str(cache_path),
            "aligned_cache_usage": (
                "external TextVQA/MM-SafetyBench/SIUO rows and swaps only"
            ),
            "v13_dataset": str(dataset_dir),
            "v13_internal_aligned_cache_dir": str(internal_cache_dir),
            "internal_manifest": str(manifest_path),
            "manifest_split_column": args.manifest_split_column,
            "feature_cache_dir": str(feature_cache_dir),
        },
        "internal": internal,
        "paired_swaps": paired_swaps,
        "difference_in_differences": did,
        "bootstrap": {
            "repetitions": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "internal_unit": (
                "group_id within the original v13 paper test split"
            ),
            "swap_unit": "paired example",
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    result_path = (
        out / "matched_representation_ablation_results.json"
    )

    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    bundle_path = (
        out / "matched_representation_ablation_frozen_bundle.pkl"
    )

    with bundle_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": 1,
                "methods": METHODS,
                "models": models,
                "bases": bases,
                "basis_dimensions": basis_dimensions,
                "representation_definition": result[
                    "representation_definition"
                ],
                "matched_protocol": result["matched_protocol"],
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    predictions_path = (
        out / "matched_representation_ablation_predictions.csv"
    )

    write_predictions(
        predictions_path,
        external_rows,
        original_scores,
        text_swap_scores,
        siuo_swap_scores,
    )

    report_path = (
        out / "MATCHED_REPRESENTATION_ABLATION_REPORT.md"
    )

    write_report(report_path, result)

    print("", flush=True)
    print("=" * 78, flush=True)
    print(
        "MATCHED REPRESENTATION-CONTROL ABLATION COMPLETE",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(f"Results : {result_path}", flush=True)
    print(f"Report  : {report_path}", flush=True)
    print(f"Bundle  : {bundle_path}", flush=True)
    print(f"Scores  : {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
