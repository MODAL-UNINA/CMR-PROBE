#!/usr/bin/env python3
"""Finalize aligned-v13 multiple-mapping results from a complete checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from evaluate_v13_multiple_swap_mappings import (
    REPRESENTATION,
    hierarchical_bootstrap,
    mapping_digest,
    population_digest,
    score,
    sha256_file,
    validate_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", choices=("llava", "qwen"), required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261031)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_cache = args.base_cache.resolve()
    args.probe = args.probe.resolve()
    args.mapping_manifest = args.mapping_manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.base_cache.open("rb") as handle:
        cache = pickle.load(handle)
    with args.probe.open("rb") as handle:
        probe = pickle.load(handle)
    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    manifest = json.loads(args.mapping_manifest.read_text(encoding="utf-8"))

    cache_hash = sha256_file(args.base_cache)
    probe_hash = sha256_file(args.probe)
    manifest_hash = sha256_file(args.mapping_manifest)
    expected = {
        "schema_version": 2,
        "lineage": "aligned-v13 operational",
        "backbone": args.backbone,
        "base_cache_sha256": cache_hash,
        "probe_sha256": probe_hash,
        "mapping_manifest_sha256": manifest_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(
                f"Checkpoint lineage mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {value!r}"
            )
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Expected aligned-v13 mapping manifest schema 2")

    layer = 17 if args.backbone == "llava" else 11
    validate_probe(probe, cache, layer)
    siuo_rows = cache["rows"]["siuo"]
    siuo_keys = sorted(
        map(str, siuo_rows),
        key=lambda key: (
            int(siuo_rows[key].get("original_index", 10**12)),
            key,
        ),
    )
    text_keys = list(map(str, cache["swap_maps"]["textvqa_test"]))
    text_keys.sort(key=int)
    categories = {
        key: str(siuo_rows[key].get("category", "unknown"))
        for key in siuo_keys
    }
    if population_digest(siuo_keys, categories) != manifest["siuo_population_sha256"]:
        raise RuntimeError("SIUO population differs from the manifest")
    if population_digest(text_keys) != manifest["textvqa_population_sha256"]:
        raise RuntimeError("TextVQA population differs from the manifest")

    seeds = sorted(map(int, manifest["mappings"]))
    if len(seeds) != 20:
        raise RuntimeError(f"Expected 20 mappings, found {len(seeds)}")
    for seed in seeds:
        declared = manifest["mappings"][str(seed)]
        item = checkpoint["seeds"].get(str(seed))
        if item is None:
            raise RuntimeError(f"Checkpoint is missing seed {seed}")
        if len(item.get("siuo", {})) != len(siuo_keys):
            raise RuntimeError(f"Seed {seed} has incomplete SIUO extraction")
        if len(item.get("textvqa", {})) != len(text_keys):
            raise RuntimeError(f"Seed {seed} has incomplete TextVQA extraction")
        if mapping_digest(item["siuo_map"]) != declared["siuo_sha256"]:
            raise RuntimeError(f"Seed {seed} SIUO mapping digest mismatch")
        if mapping_digest(item["textvqa_map"]) != declared["textvqa_sha256"]:
            raise RuntimeError(f"Seed {seed} TextVQA mapping digest mismatch")

    unsafe_h_t = np.stack([siuo_rows[key]["hT_last"] for key in siuo_keys])
    unsafe_h_tv = np.stack([siuo_rows[key]["hTV_last"] for key in siuo_keys])
    benign_h_t = np.stack(
        [cache["rows"]["textvqa"][key]["hT_last"] for key in text_keys]
    )
    benign_h_tv = np.stack(
        [cache["rows"]["textvqa"][key]["hTV_last"] for key in text_keys]
    )
    unsafe_original = score(probe, unsafe_h_tv, unsafe_h_t)
    benign_original = score(probe, benign_h_tv, benign_h_t)

    unsafe_drops = []
    benign_drops = []
    per_mapping = []
    prediction_rows = []
    for seed in seeds:
        item = checkpoint["seeds"][str(seed)]
        unsafe_swapped = score(
            probe,
            np.stack([item["siuo"][key]["hTV_last_swap"] for key in siuo_keys]),
            unsafe_h_t,
        )
        benign_swapped = score(
            probe,
            np.stack(
                [item["textvqa"][key]["hTV_last_swap"] for key in text_keys]
            ),
            benign_h_t,
        )
        unsafe_drop = unsafe_original - unsafe_swapped
        benign_drop = benign_original - benign_swapped
        unsafe_drops.append(unsafe_drop)
        benign_drops.append(benign_drop)
        unsafe_ties = np.isclose(unsafe_drop, 0.0, atol=1e-12)
        per_mapping.append(
            {
                "seed": seed,
                "siuo_mean_drop": float(unsafe_drop.mean()),
                "probability_original_greater": float(
                    np.mean(unsafe_drop > 0) + 0.5 * np.mean(unsafe_ties)
                ),
                "benign_mean_drop": float(benign_drop.mean()),
                "gamma": float(unsafe_drop.mean() - benign_drop.mean()),
            }
        )
        for index, key in enumerate(siuo_keys):
            prediction_rows.append(
                {
                    "seed": seed,
                    "pool": "SIUO_unsafe",
                    "id": key,
                    "category": categories[key],
                    "image_from_id": item["siuo"][key]["image_from_id"],
                    "original_score": float(unsafe_original[index]),
                    "substituted_score": float(unsafe_swapped[index]),
                    "score_drop": float(unsafe_drop[index]),
                }
            )
        for index, key in enumerate(text_keys):
            prediction_rows.append(
                {
                    "seed": seed,
                    "pool": "TextVQA_benign",
                    "id": key,
                    "category": "safe_control",
                    "image_from_id": item["textvqa"][key]["image_from_id"],
                    "original_score": float(benign_original[index]),
                    "substituted_score": float(benign_swapped[index]),
                    "score_drop": float(benign_drop[index]),
                }
            )

    unsafe_array = np.stack(unsafe_drops)
    benign_array = np.stack(benign_drops)
    aggregate = {}
    for metric in (
        "siuo_mean_drop",
        "probability_original_greater",
        "benign_mean_drop",
        "gamma",
    ):
        values = np.asarray([row[metric] for row in per_mapping])
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std_across_mappings": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    aggregate["hierarchical_bootstrap_95ci"] = hierarchical_bootstrap(
        unsafe_array,
        benign_array,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    result = {
        "experiment": "aligned_v13_operational_multiple_swap_mappings",
        "lineage": "aligned-v13 operational",
        "backbone": args.backbone,
        "model_id": checkpoint.get("model_id"),
        "resolved_revision": checkpoint.get("resolved_revision"),
        "language_layer": checkpoint.get("language_layer"),
        "representation": REPRESENTATION,
        "base_cache": str(args.base_cache),
        "base_cache_sha256": cache_hash,
        "probe": str(args.probe),
        "probe_sha256": probe_hash,
        "mapping_manifest": str(args.mapping_manifest),
        "mapping_manifest_sha256": manifest_hash,
        "probe_refit": False,
        "n_mappings": len(seeds),
        "mapping_seeds": seeds,
        "siuo_n": len(siuo_keys),
        "textvqa_n": len(text_keys),
        "identity_checks": checkpoint.get("identity_checks", []),
        "bootstrap": {
            "method": "hierarchical resampling of mappings and shared source IDs",
            "repetitions": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "per_mapping": per_mapping,
        "aggregate": aggregate,
    }
    result_path = args.out / f"{args.backbone}_v13_multiple_swap_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions_path = (
        args.out / f"{args.backbone}_v13_multiple_swap_predictions.csv"
    )
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
