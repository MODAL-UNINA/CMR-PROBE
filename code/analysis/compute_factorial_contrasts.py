#!/usr/bin/env python3
"""Compute paired factorial contrasts for the aligned-v13 2x2 ablation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = (
    "raw_no_proj",
    "raw_proj",
    "aligned_no_proj",
    "aligned_proj",
)

CONTRASTS = {
    "alignment_without_projection": {
        "aligned_no_proj": 1.0,
        "raw_no_proj": -1.0,
    },
    "alignment_with_projection": {
        "aligned_proj": 1.0,
        "raw_proj": -1.0,
    },
    "projection_raw": {
        "raw_proj": 1.0,
        "raw_no_proj": -1.0,
    },
    "projection_aligned": {
        "aligned_proj": 1.0,
        "aligned_no_proj": -1.0,
    },
    "alignment_x_projection": {
        "aligned_proj": 1.0,
        "aligned_no_proj": -1.0,
        "raw_proj": -1.0,
        "raw_no_proj": 1.0,
    },
}


def linear_contrast(values: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(weights[key] * values[key] for key in weights))


def main() -> None:
    root = Path(__file__).resolve().parent
    prediction_path = root / "matched_representation_ablation_predictions.csv"
    output_path = root / "matched_factorial_contrasts.json"

    rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["pool"]][row["method"]].append(
                float(row["original_minus_swap"])
            )

    arrays = {
        pool: {
            method: np.asarray(values, dtype=np.float64)
            for method, values in methods.items()
        }
        for pool, methods in rows.items()
    }
    for pool in ("SIUO_test", "TextVQA_test"):
        if set(arrays[pool]) != set(METHODS):
            raise RuntimeError(f"Incomplete method set for {pool}")

    point_did = {
        method: float(
            arrays["SIUO_test"][method].mean()
            - arrays["TextVQA_test"][method].mean()
        )
        for method in METHODS
    }
    point = {
        name: linear_contrast(point_did, weights)
        for name, weights in CONTRASTS.items()
    }

    repetitions = 10_000
    seed = 20260933
    rng = np.random.default_rng(seed)
    unsafe_n = len(arrays["SIUO_test"][METHODS[0]])
    safe_n = len(arrays["TextVQA_test"][METHODS[0]])
    boot = {name: np.empty(repetitions) for name in CONTRASTS}

    for rep in range(repetitions):
        unsafe_idx = rng.integers(0, unsafe_n, unsafe_n)
        safe_idx = rng.integers(0, safe_n, safe_n)
        did = {
            method: float(
                arrays["SIUO_test"][method][unsafe_idx].mean()
                - arrays["TextVQA_test"][method][safe_idx].mean()
            )
            for method in METHODS
        }
        for name, weights in CONTRASTS.items():
            boot[name][rep] = linear_contrast(did, weights)

    output = {
        "definition": (
            "Contrasts of Gamma = mean SIUO(original-swap) minus "
            "mean TextVQA(original-swap), using common resampled indices "
            "across all four fitted ablation estimators."
        ),
        "bootstrap": {"repetitions": repetitions, "seed": seed},
        "point_gamma": point_did,
        "contrasts": {
            name: {
                "estimate": point[name],
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            }
            for name, values in boot.items()
        },
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
