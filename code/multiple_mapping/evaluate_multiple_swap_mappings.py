#!/usr/bin/env python3
"""Evaluate multiple image-substitution mappings with a frozen v13 probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


DEFAULT_SEEDS = tuple(20261001 + index for index in range(20))
REPRESENTATION = "L2(L2(hTV_last)-L2(hT_last))"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", choices=("llava", "qwen"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261031)
    parser.add_argument("--identity-cosine-min", type=float, default=0.9999)
    parser.add_argument("--identity-relative-l2-max", type=float, default=0.02)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_digest(mapping: dict[str, str]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def population_digest(
    keys: list[str],
    categories: dict[str, str] | None = None,
) -> str:
    payload = {"keys": list(map(str, keys)), "categories": categories or {}}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_pickle(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def aligned_features(h_tv: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    return unit_rows(unit_rows(h_tv) - unit_rows(h_t))


def score(probe: dict, h_tv: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    features = aligned_features(h_tv, h_t)
    projector = probe.get("ortho_Q")
    if projector is not None:
        projector = np.asarray(projector, dtype=np.float64)
        if projector.shape[0] != features.shape[1]:
            raise ValueError(
                f"Projector dimension {projector.shape} does not match {features.shape}"
            )
        features = features - (features @ projector) @ projector.T
    values = np.asarray(probe["clf"].predict_proba(features)[:, 1], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Frozen operational probe emitted non-finite scores")
    return values


def validate_probe(probe: dict, cache: dict, layer: int) -> None:
    if "clf" not in probe:
        raise KeyError("Operational probe has no classifier")
    if (
        probe.get("sc_representation") != "aligned"
        and probe.get("representation") != REPRESENTATION
    ):
        raise ValueError("Probe is not the aligned operational representation")
    cache_layer = int(cache["representation"]["layer"])
    probe_layer = int(probe.get("language_layer", -1))
    if cache_layer != layer or probe_layer != layer:
        raise ValueError(
            f"Layer mismatch: requested={layer}, cache={cache_layer}, probe={probe_layer}"
        )


def validate_mapping(
    mapping: dict[str, str],
    keys: list[str],
    categories: dict[str, str] | None,
    expected_digest: str,
) -> dict[str, str]:
    mapping = {str(source): str(target) for source, target in mapping.items()}
    population = set(keys)
    if set(mapping) != population or set(mapping.values()) != population:
        raise RuntimeError("Mapping is not one-to-one over the frozen population")
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("Mapping contains a self-pair")
    if categories is not None and any(
        categories[source] == categories[target]
        for source, target in mapping.items()
    ):
        raise RuntimeError("SIUO mapping contains a same-category pair")
    if mapping_digest(mapping) != expected_digest:
        raise RuntimeError("Mapping digest differs from the manifest")
    return mapping


def load_backend(args: argparse.Namespace, shared):
    if args.backbone == "llava":
        model_id = args.model_id or "llava-hf/llava-1.5-7b-hf"
        layer = 17 if args.layer is None else int(args.layer)
        if layer != 17:
            raise ValueError("LLaVA aligned-v13 requires layer 17")
        model, processor = shared.load_model(
            args.project_root, args.gpu, model_id, args.revision, False
        )
        extractor = shared.extract_multimodal_last
    else:
        import extract_qwen_aligned_last_token as qwen

        model_id = args.model_id or qwen.DEFAULT_MODEL_ID
        layer = 11 if args.layer is None else int(args.layer)
        if layer != 11:
            raise ValueError("Qwen aligned-v13 requires layer 11")
        model, processor, _ = qwen.load_model(
            args.gpu, model_id, args.revision, False
        )

        def extractor(text, image, current_model, current_processor):
            return qwen.extract_multimodal_last(
                text, image, current_model, current_processor, layer
            )

    return model, processor, model_id, layer, extractor


def compare_identity(
    label: str,
    extracted: np.ndarray,
    cached: np.ndarray,
    cosine_minimum: float,
    relative_l2_maximum: float,
) -> dict[str, float | str]:
    extracted = np.asarray(extracted, dtype=np.float64)
    cached = np.asarray(cached, dtype=np.float64)
    if extracted.shape != cached.shape:
        raise RuntimeError(
            f"{label} hidden-state shape mismatch: {extracted.shape} != {cached.shape}"
        )
    denominator = max(float(np.linalg.norm(cached)), 1e-12)
    relative_l2 = float(np.linalg.norm(extracted - cached) / denominator)
    cosine = float(
        np.dot(extracted, cached)
        / max(float(np.linalg.norm(extracted)) * denominator, 1e-12)
    )
    if cosine < cosine_minimum or relative_l2 > relative_l2_maximum:
        raise RuntimeError(
            f"{label} identity check failed: cosine={cosine:.8f}, "
            f"relative_l2={relative_l2:.8f}. The loaded checkpoint or processor "
            "does not reproduce the aligned-v13 cache."
        )
    return {"label": label, "cosine": cosine, "relative_l2": relative_l2}


def hierarchical_bootstrap(
    unsafe_drops: np.ndarray,
    benign_drops: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict:
    if unsafe_drops.ndim != 2 or benign_drops.ndim != 2:
        raise ValueError("Drop arrays must be mapping by source")
    if unsafe_drops.shape[0] != benign_drops.shape[0]:
        raise ValueError("Unsafe and benign arrays use different mapping counts")
    rng = np.random.default_rng(seed)
    n_mappings, n_unsafe = unsafe_drops.shape
    n_benign = benign_drops.shape[1]
    values = {
        "siuo_mean_drop": np.empty(repetitions),
        "benign_mean_drop": np.empty(repetitions),
        "probability_original_greater": np.empty(repetitions),
        "gamma": np.empty(repetitions),
    }
    for repetition in range(repetitions):
        map_indices = rng.integers(0, n_mappings, n_mappings)
        unsafe_indices = rng.integers(0, n_unsafe, n_unsafe)
        benign_indices = rng.integers(0, n_benign, n_benign)
        unsafe = unsafe_drops[map_indices][:, unsafe_indices]
        benign = benign_drops[map_indices][:, benign_indices]
        unsafe_mean = float(unsafe.mean())
        benign_mean = float(benign.mean())
        unsafe_ties = np.isclose(unsafe, 0.0, atol=1e-12)
        values["siuo_mean_drop"][repetition] = unsafe_mean
        values["benign_mean_drop"][repetition] = benign_mean
        values["probability_original_greater"][repetition] = (
            np.mean(unsafe > 0) + 0.5 * np.mean(unsafe_ties)
        )
        values["gamma"][repetition] = unsafe_mean - benign_mean
    return {
        metric: {
            "lower": float(np.percentile(series, 2.5)),
            "upper": float(np.percentile(series, 97.5)),
        }
        for metric, series in values.items()
    }


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.base_cache = args.base_cache.resolve()
    args.probe = args.probe.resolve()
    args.mapping_manifest = args.mapping_manifest.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    tools_dir = args.project_root / "code" / "aligned_last_token"
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"Missing aligned extraction tools: {tools_dir}")
    sys.path.insert(0, str(tools_dir))
    import extract_aligned_last_token as shared

    with args.base_cache.open("rb") as handle:
        cache = pickle.load(handle)
    with args.probe.open("rb") as handle:
        probe = pickle.load(handle)
    manifest = json.loads(args.mapping_manifest.read_text(encoding="utf-8"))
    if cache.get("schema_version") != 1:
        raise ValueError("Unsupported aligned-cache schema")
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected an aligned-v13 mapping manifest (schema 2)")

    cache_hash = sha256_file(args.base_cache)
    probe_hash = sha256_file(args.probe)
    manifest_hash = sha256_file(args.mapping_manifest)
    model, processor, model_id, layer, extractor = load_backend(args, shared)
    validate_probe(probe, cache, layer)
    if cache["representation"].get("model") != model_id:
        raise RuntimeError("Requested model identifier differs from the v13 cache")
    resolved_revision = getattr(model.config, "_commit_hash", None)
    expected_revision = cache["representation"].get("revision_resolved")
    if expected_revision and resolved_revision and expected_revision != resolved_revision:
        raise RuntimeError(
            f"Model revision mismatch: {resolved_revision} != {expected_revision}"
        )

    siuo_rows_cache = cache["rows"]["siuo"]
    siuo_keys = sorted(
        map(str, siuo_rows_cache),
        key=lambda key: (
            int(siuo_rows_cache[key].get("original_index", 10**12)),
            key,
        ),
    )
    text_keys = list(map(str, cache["swap_maps"]["textvqa_test"]))
    text_keys.sort(key=int)
    categories = {
        key: str(siuo_rows_cache[key].get("category", "unknown"))
        for key in siuo_keys
    }
    if len(siuo_keys) != 167 or len(text_keys) != 1100:
        raise RuntimeError(
            f"Unexpected populations: SIUO={len(siuo_keys)}, TextVQA={len(text_keys)}"
        )
    if population_digest(siuo_keys, categories) != manifest["siuo_population_sha256"]:
        raise RuntimeError("SIUO v13 population differs from the mapping manifest")
    if population_digest(text_keys) != manifest["textvqa_population_sha256"]:
        raise RuntimeError("TextVQA v13 population differs from the mapping manifest")
    requested_seeds = list(map(int, args.seeds))
    missing = set(map(str, requested_seeds)) - set(manifest["mappings"])
    if missing:
        raise RuntimeError(f"Manifest is missing seeds: {sorted(missing)}")

    siuo_rows = shared.materialize_siuo(set(siuo_keys))
    _, textvqa = shared.load_mmsafety_textvqa()

    identity_checks = []
    first_siuo = siuo_keys[0]
    vector, metadata = extractor(
        siuo_rows_cache[first_siuo]["text"],
        siuo_rows[first_siuo]["image"],
        model,
        processor,
    )
    if int(metadata["last_token_id"]) != int(
        siuo_rows_cache[first_siuo]["last_token_id"]
    ):
        raise RuntimeError("SIUO identity check changed the final textual token")
    identity_checks.append(
        compare_identity(
            "SIUO original",
            vector,
            siuo_rows_cache[first_siuo]["hTV_last"],
            args.identity_cosine_min,
            args.identity_relative_l2_max,
        )
    )
    first_text = text_keys[0]
    vector, metadata = extractor(
        cache["rows"]["textvqa"][first_text]["text"],
        shared.get_image(dict(textvqa[int(first_text)])),
        model,
        processor,
    )
    if int(metadata["last_token_id"]) != int(
        cache["rows"]["textvqa"][first_text]["last_token_id"]
    ):
        raise RuntimeError("TextVQA identity check changed the final textual token")
    identity_checks.append(
        compare_identity(
            "TextVQA original",
            vector,
            cache["rows"]["textvqa"][first_text]["hTV_last"],
            args.identity_cosine_min,
            args.identity_relative_l2_max,
        )
    )
    print(json.dumps({"identity_checks": identity_checks}, indent=2), flush=True)

    checkpoint_path = args.out / f"{args.backbone}_v13_multiple_swap_states.pkl"
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        expected = {
            "schema_version": 2,
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
    else:
        checkpoint = {
            "schema_version": 2,
            "lineage": "aligned-v13 operational",
            "backbone": args.backbone,
            "model_id": model_id,
            "resolved_revision": resolved_revision,
            "language_layer": layer,
            "base_cache_sha256": cache_hash,
            "probe_sha256": probe_hash,
            "mapping_manifest_sha256": manifest_hash,
            "identity_checks": identity_checks,
            "seeds": {},
        }

    for seed in requested_seeds:
        declared = manifest["mappings"][str(seed)]
        siuo_mapping = validate_mapping(
            declared["siuo"],
            siuo_keys,
            categories,
            declared["siuo_sha256"],
        )
        text_mapping = validate_mapping(
            declared["textvqa"],
            text_keys,
            None,
            declared["textvqa_sha256"],
        )
        item = checkpoint["seeds"].setdefault(
            str(seed),
            {
                "siuo_map": siuo_mapping,
                "siuo_sha256": declared["siuo_sha256"],
                "textvqa_map": text_mapping,
                "textvqa_sha256": declared["textvqa_sha256"],
                "siuo": {},
                "textvqa": {},
            },
        )
        if item["siuo_map"] != siuo_mapping or item["textvqa_map"] != text_mapping:
            raise RuntimeError(f"Checkpoint mapping differs for seed {seed}")
        count = 0
        for source in siuo_keys:
            if source in item["siuo"]:
                continue
            target = siuo_mapping[source]
            vector, metadata = extractor(
                siuo_rows_cache[source]["text"],
                siuo_rows[target]["image"],
                model,
                processor,
            )
            if not np.all(np.isfinite(vector)):
                raise RuntimeError(
                    f"Non-finite Qwen/LLaVA hidden state for seed {seed}, "
                    f"SIUO source {source}, target {target}; entry not checkpointed"
                )
            if int(metadata["last_token_id"]) != int(
                siuo_rows_cache[source]["last_token_id"]
            ):
                raise RuntimeError(f"SIUO final-token mismatch for {source}")
            item["siuo"][source] = {
                "id": source,
                "image_from_id": target,
                "image_from_category": categories[target],
                "hTV_last_swap": vector,
                **metadata,
            }
            count += 1
            if count % args.checkpoint_every == 0:
                atomic_pickle(checkpoint_path, checkpoint)
        for source in text_keys:
            if source in item["textvqa"]:
                continue
            target = text_mapping[source]
            vector, metadata = extractor(
                cache["rows"]["textvqa"][source]["text"],
                shared.get_image(dict(textvqa[int(target)])),
                model,
                processor,
            )
            if not np.all(np.isfinite(vector)):
                raise RuntimeError(
                    f"Non-finite Qwen/LLaVA hidden state for seed {seed}, "
                    f"TextVQA source {source}, target {target}; entry not checkpointed"
                )
            if int(metadata["last_token_id"]) != int(
                cache["rows"]["textvqa"][source]["last_token_id"]
            ):
                raise RuntimeError(f"TextVQA final-token mismatch for {source}")
            item["textvqa"][source] = {
                "id": source,
                "image_from_id": target,
                "hTV_last_swap": vector,
                **metadata,
            }
            count += 1
            if count % args.checkpoint_every == 0:
                atomic_pickle(checkpoint_path, checkpoint)
        atomic_pickle(checkpoint_path, checkpoint)
        print(f"completed mapping seed {seed}", flush=True)

    unsafe_h_t = np.stack(
        [siuo_rows_cache[key]["hT_last"] for key in siuo_keys]
    )
    unsafe_h_tv = np.stack(
        [siuo_rows_cache[key]["hTV_last"] for key in siuo_keys]
    )
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
    for seed in requested_seeds:
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

    unsafe_drops_array = np.stack(unsafe_drops)
    benign_drops_array = np.stack(benign_drops)
    metrics = (
        "siuo_mean_drop",
        "probability_original_greater",
        "benign_mean_drop",
        "gamma",
    )
    aggregate = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in per_mapping], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std_across_mappings": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    aggregate["hierarchical_bootstrap_95ci"] = hierarchical_bootstrap(
        unsafe_drops_array,
        benign_drops_array,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    result = {
        "experiment": "aligned_v13_operational_multiple_swap_mappings",
        "lineage": "aligned-v13 operational",
        "backbone": args.backbone,
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "language_layer": layer,
        "representation": REPRESENTATION,
        "base_cache": str(args.base_cache),
        "base_cache_sha256": cache_hash,
        "probe": str(args.probe),
        "probe_sha256": probe_hash,
        "mapping_manifest": str(args.mapping_manifest),
        "mapping_manifest_sha256": manifest_hash,
        "probe_refit": False,
        "n_mappings": len(requested_seeds),
        "mapping_seeds": requested_seeds,
        "siuo_n": len(siuo_keys),
        "textvqa_n": len(text_keys),
        "identity_checks": identity_checks,
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
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
