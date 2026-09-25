#!/usr/bin/env python3
"""Benchmark the exact aligned CMR-PROBE cross-modal branch.

The benchmark times the two representation passes used by
``delta_last_prenorm_l2`` and the complete residual/probe/calibration step.
It intentionally does not reuse the sequence-pooled system profiler.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", choices=("llava", "qwen"), default="llava")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def synchronize(torch, gpu: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(gpu)


def summary(values) -> dict:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def driver_version(gpu: int) -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def load_backend(args):
    import extract_aligned_last_token as shared

    if args.backbone == "llava":
        model_id = args.model_id or "llava-hf/llava-1.5-7b-hf"
        layer = 17 if args.layer is None else args.layer
        if layer != 17:
            raise ValueError("The released LLaVA extractor is frozen at layer 17")
        model, processor = shared.load_model(
            args.project_root, args.gpu, model_id, args.revision, False
        )

        def text_state(text):
            return shared.extract_text_last(text, model, processor)

        def multimodal_state(text, image):
            return shared.extract_multimodal_last(text, image, model, processor)

    else:
        import extract_qwen_aligned_last_token as qwen

        model_id = args.model_id or qwen.DEFAULT_MODEL_ID
        layer = 11 if args.layer is None else args.layer
        model, processor, _ = qwen.load_model(
            args.gpu, model_id, args.revision, False
        )

        def text_state(text):
            return qwen.extract_text_last(text, model, processor, layer)

        def multimodal_state(text, image):
            return qwen.extract_multimodal_last(text, image, model, processor, layer)

    return shared, model, processor, model_id, layer, text_state, multimodal_state


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import sklearn
    import torch

    from evaluate_aligned_last_token import PRIMARY, representations

    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid CUDA device {args.gpu}; "
                f"this process exposes {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(args.gpu)

    with args.bundle.resolve().open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("primary") != PRIMARY:
        raise ValueError(f"Bundle primary feature is not {PRIMARY}")
    bundle_representation = bundle.get("representation_metadata", {})
    if args.revision is None and bundle_representation.get("revision_resolved"):
        args.revision = bundle_representation["revision_resolved"]
    shared, model, _, model_id, layer, text_state, multimodal_state = load_backend(args)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if bundle_representation:
        if bundle_representation.get("model") != model_id:
            raise RuntimeError("Frozen bundle and requested model identifier differ")
        if int(bundle_representation.get("layer", -1)) != layer:
            raise RuntimeError("Frozen bundle and requested language layer differ")
        expected_revision = bundle_representation.get("revision_resolved")
        if expected_revision and resolved_revision and expected_revision != resolved_revision:
            raise RuntimeError("Loaded model revision differs from the frozen bundle")
    revision_verification = (
        "matched recorded resolved revision"
        if resolved_revision and bundle_representation.get("revision_resolved")
        else "unverified because the original artifact did not record a resolved revision"
    )
    probe = bundle["models"][PRIMARY]

    splits, _, _, _ = shared.load_external_protocol(args.project_root)
    _, textvqa = shared.load_mmsafety_textvqa()
    keys = np.asarray(list(map(int, splits["textvqa_test"])), dtype=int)
    rng = np.random.default_rng(args.seed)
    chosen = keys[rng.permutation(len(keys))[: args.warmup + args.samples]]
    rows = [dict(textvqa[int(key)]) for key in chosen]

    def run_one(row, measured: bool) -> dict | None:
        text = shared.get_text(row)
        image = shared.get_image(row)
        if torch.cuda.is_available() and measured:
            torch.cuda.reset_peak_memory_stats(args.gpu)
        synchronize(torch, args.gpu)
        total_start = time.perf_counter()
        start = time.perf_counter()
        h_t, text_meta = text_state(text)
        synchronize(torch, args.gpu)
        text_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        h_tv, multimodal_meta = multimodal_state(text, image)
        synchronize(torch, args.gpu)
        multimodal_elapsed = time.perf_counter() - start
        if text_meta["last_token_id"] != multimodal_meta["last_token_id"]:
            raise RuntimeError("Aligned benchmark encountered a final-token mismatch")
        start = time.perf_counter()
        feature = representations(h_tv[None, :], h_t[None, :])[PRIMARY]
        score = float(probe.predict_proba(feature)[0, 1])
        synchronize(torch, args.gpu)
        head_elapsed = time.perf_counter() - start
        total_elapsed = time.perf_counter() - total_start
        if not measured:
            return None
        peak_mb = (
            float(torch.cuda.max_memory_allocated(args.gpu)) / (1024**2)
            if torch.cuda.is_available()
            else None
        )
        return {
            "text_ms": text_elapsed * 1000,
            "multimodal_ms": multimodal_elapsed * 1000,
            "residual_probe_ms": head_elapsed * 1000,
            "total_ms": total_elapsed * 1000,
            "peak_allocated_mb": peak_mb,
            "score": score,
        }

    for row in rows[: args.warmup]:
        run_one(row, measured=False)
    baseline_mb = (
        float(torch.cuda.memory_allocated(args.gpu)) / (1024**2)
        if torch.cuda.is_available()
        else None
    )
    measured = [run_one(row, measured=True) for row in rows[args.warmup :]]
    latency = {
        key: summary([row[key] for row in measured])
        for key in ("text_ms", "multimodal_ms", "residual_probe_ms", "total_ms")
    }
    peaks = [row["peak_allocated_mb"] for row in measured if row["peak_allocated_mb"] is not None]
    incremental_peaks = (
        [value - baseline_mb for value in peaks] if baseline_mb is not None else []
    )
    device_name = torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None
    device_capability = (
        list(torch.cuda.get_device_capability(args.gpu))
        if torch.cuda.is_available()
        else None
    )
    result = {
        "experiment": "exact_aligned_last_token_efficiency",
        "backbone": args.backbone,
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "revision_verification": revision_verification,
        "language_layer": layer,
        "representation": PRIMARY,
        "dataset": "TextVQA frozen test pool",
        "samples": len(measured),
        "warmup": args.warmup,
        "seed": args.seed,
        "major_forward_passes": 2,
        "latency_ms": latency,
        "throughput_samples_per_second": 1000.0 / latency["total_ms"]["mean"],
        "warmed_model_allocated_mb": baseline_mb,
        "peak_allocated_mb": summary(peaks) if peaks else None,
        "incremental_peak_allocated_mb": (
            summary(incremental_peaks) if incremental_peaks else None
        ),
        "hardware": {
            "gpu_index": args.gpu if torch.cuda.is_available() else None,
            "gpu_name": device_name,
            "gpu_compute_capability": device_capability,
            "nvidia_driver": driver_version(args.gpu) if torch.cuda.is_available() else None,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
        },
        "software": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.out / "aligned_efficiency_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (args.out / "aligned_efficiency_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
