#!/usr/bin/env python3
"""Validate that this is a portable, internally consistent aligned-v13 release."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LINEAGE_DIGEST = "3674ef403fecc41b205c4ff9c531b01c503cdb35205d4f04a0bee1b0e23aaf5e"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".txt"}
FORBIDDEN_SUFFIXES = {".bin", ".joblib", ".key", ".pkl", ".pt", ".pth", ".safetensors"}


def add(errors: list[str], message: str) -> None:
    errors.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(errors: list[str], *relative: str) -> None:
    for value in relative:
        if not (ROOT / value).is_file():
            add(errors, f"Missing required file: {value}")


def check_portability(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            add(errors, f"Forbidden binary artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            add(errors, f"Oversized tracked file: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and path != Path(__file__).resolve():
            text = path.read_text(encoding="utf-8", errors="replace")
            if "/home/" in text or "modal-workbench" in text:
                add(errors, f"Local path leaked into: {relative}")


def check_syntax(errors: list[str]) -> None:
    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            add(errors, f"Python syntax error in {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            add(errors, f"Invalid JSON in {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                next(csv.reader(handle))
        except (OSError, StopIteration, UnicodeDecodeError) as exc:
            add(errors, f"Invalid or empty CSV in {path.relative_to(ROOT)}: {exc}")


def check_split(errors: list[str]) -> None:
    manifest = ROOT / "data/manifests/aligned_v13_split_manifest.csv"
    if not manifest.is_file():
        return
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 2520, "val": 540, "test": 540}:
        add(errors, f"Unexpected aligned-v13 split counts: {dict(counts)}")
    audit = json.loads((ROOT / "data/manifests/aligned_v13_split_audit.json").read_text(encoding="utf-8"))
    observed = sha256(manifest)
    if audit.get("manifest_sha256") != observed:
        add(errors, "The split-audit digest does not match aligned_v13_split_manifest.csv")
    if observed != EXPECTED_LINEAGE_DIGEST:
        add(
            errors,
            "The supplied 2520/540/540 manifest digest does not match the rerun-lineage digest; replace it with the authoritative paper manifest.",
        )


def check_results(errors: list[str]) -> None:
    for backbone in ("llava", "qwen"):
        paired = ROOT / f"results/aligned_v13/paired_current_probe/{backbone}/paired_image_substitution_results.json"
        mapping = ROOT / f"results/aligned_v13/multiple_mapping_v13/{backbone}/{backbone}_v13_multiple_swap_results.json"
        if paired.is_file():
            result = json.loads(paired.read_text(encoding="utf-8"))
            if result.get("paired", {}).get("SIUO_unsafe", {}).get("n") != 167:
                add(errors, f"{backbone} fixed-map artifact must contain 167 SIUO rows")
        if mapping.is_file():
            result = json.loads(mapping.read_text(encoding="utf-8"))
            if result.get("n_mappings") != 20 or result.get("probe_refit") is not False:
                add(errors, f"{backbone} multiple-map artifact is not the frozen 20-map evaluation")
    mssbench = ROOT / "data/manifests/mssbench_matched_source_manifest.json"
    if mssbench.is_file():
        rows = json.loads(mssbench.read_text(encoding="utf-8")).get("rows", [])
        if len(rows) != 600:
            add(errors, "The MSSBench matched-source manifest must contain 600 query rows")


def check_checksums(errors: list[str]) -> None:
    manifest = ROOT / "FROZEN_ARTIFACTS.sha256"
    if not manifest.is_file():
        add(errors, "Missing FROZEN_ARTIFACTS.sha256")
        return
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            add(errors, f"Malformed checksum line {number}")
            continue
        path = ROOT / relative
        if not path.is_file():
            add(errors, f"Checksum target missing: {relative}")
        elif sha256(path) != expected:
            add(errors, f"Checksum mismatch: {relative}")


def main() -> int:
    errors: list[str] = []
    require(
        errors,
        "README.md",
        "data/README.md",
        "docs/REPRODUCIBILITY.md",
        "code/reporting/summarize_results.py",
        "code/reporting/generate_paper_tables.py",
        "code/aligned_v13/run_core_pipeline.sh",
        "code/aligned_v13/run_full_pipeline.sh",
        "code/aligned_v13/run_qwen_probe_training.sh",
        "code/aligned_v13/run_qwen_full_pipeline.sh",
        "code/aligned_v13/25_mssbench_matched_source.py",
        "data/manifests/aligned_v13_split_manifest.csv",
        "data/manifests/aligned_v13_split_audit.json",
        "data/manifests/primary_swap_manifest_seed20260913.json",
        "data/manifests/multiple_mapping_manifest_v13.json",
        "data/manifests/mssbench_matched_source_manifest.json",
        "results/aligned_v13/paired_current_probe/fixed_map_threshold_failures.json",
        "results/aligned_v13/matched_representation_ablation/matched_representation_ablation_results.json",
        "results/aligned_v13/mssbench_matched_source/mssbench_matched_source_results.json",
    )
    check_portability(errors)
    check_syntax(errors)
    check_split(errors)
    check_results(errors)
    check_checksums(errors)
    if errors:
        print("Release check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Release check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
