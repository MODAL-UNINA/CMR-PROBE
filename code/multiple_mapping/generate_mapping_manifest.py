#!/usr/bin/env python3
"""Generate deterministic substitution mappings from an aligned-v13 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


DEFAULT_SEEDS = tuple(20261001 + index for index in range(20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
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


def population_digest(keys: list[str], categories: dict[str, str] | None = None) -> str:
    payload = {
        "keys": list(map(str, keys)),
        "categories": categories or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def derangement(
    keys: list[str],
    seed: int,
    categories: dict[str, str] | None = None,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    keys = list(map(str, keys))
    if categories is not None:
        grouped: dict[str, list[str]] = {}
        for key in keys:
            grouped.setdefault(categories[key], []).append(key)
        order = list(grouped)
        rng.shuffle(order)
        arranged: list[str] = []
        for category in order:
            block = np.asarray(grouped[category], dtype=object)
            arranged.extend(map(str, block[rng.permutation(len(block))]))
        maximum = max(map(len, grouped.values()))
        if maximum * 2 > len(arranged):
            raise RuntimeError("Cross-category derangement is infeasible")
        shifted = arranged[maximum:] + arranged[:maximum]
        mapping = dict(zip(arranged, shifted))
        return {key: mapping[key] for key in keys}
    for _ in range(100_000):
        shuffled = list(np.asarray(keys, dtype=object)[rng.permutation(len(keys))])
        if all(source != target for source, target in zip(keys, shuffled)):
            return dict(zip(keys, map(str, shuffled)))
    raise RuntimeError("Could not construct a derangement")


def validate(
    mapping: dict[str, str],
    keys: list[str],
    categories: dict[str, str] | None = None,
) -> None:
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


def main() -> None:
    args = parse_args()
    cache_path = args.cache.resolve()
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)

    if cache.get("schema_version") != 1:
        raise ValueError("Unsupported aligned-cache schema")
    siuo_rows = cache["rows"]["siuo"]
    siuo_keys = sorted(
        map(str, siuo_rows),
        key=lambda key: (int(siuo_rows[key].get("original_index", 10**12)), key),
    )
    text_keys = list(map(str, cache["swap_maps"]["textvqa_test"]))
    text_keys.sort(key=int)
    categories = {
        key: str(siuo_rows[key].get("category", "unknown"))
        for key in siuo_keys
    }
    if len(siuo_keys) != 167 or len(text_keys) != 1100:
        raise RuntimeError(
            f"Unexpected populations: SIUO={len(siuo_keys)}, TextVQA={len(text_keys)}"
        )

    output = {
        "schema_version": 2,
        "lineage": "aligned-v13 operational",
        "source_cache": str(cache_path),
        "source_cache_sha256": sha256_file(cache_path),
        "representation": cache["representation"],
        "siuo_n": len(siuo_keys),
        "textvqa_n": len(text_keys),
        "siuo_population_sha256": population_digest(siuo_keys, categories),
        "textvqa_population_sha256": population_digest(text_keys),
        "constraints": {
            "siuo": "one-to-one, no self-pair, different safety category",
            "textvqa": "one-to-one, no self-pair",
        },
        "mappings": {},
    }
    seen_siuo: set[str] = set()
    seen_text: set[str] = set()
    for seed in map(int, args.seeds):
        siuo_mapping = derangement(siuo_keys, seed, categories)
        text_mapping = derangement(text_keys, seed + 1)
        validate(siuo_mapping, siuo_keys, categories)
        validate(text_mapping, text_keys)
        siuo_hash = mapping_digest(siuo_mapping)
        text_hash = mapping_digest(text_mapping)
        if siuo_hash in seen_siuo or text_hash in seen_text:
            raise RuntimeError("Duplicate mapping generated for different seeds")
        seen_siuo.add(siuo_hash)
        seen_text.add(text_hash)
        output["mappings"][str(seed)] = {
            "siuo": siuo_mapping,
            "textvqa": text_mapping,
            "siuo_sha256": siuo_hash,
            "textvqa_sha256": text_hash,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "source_cache_sha256": output["source_cache_sha256"],
                "n_mappings": len(output["mappings"]),
                "siuo_n": len(siuo_keys),
                "textvqa_n": len(text_keys),
                "all_unique_and_valid": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
