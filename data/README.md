# Data boundary

No raw internal records, source media, feature caches, model weights, or
credentials are committed. The release publishes only identifier- and
hash-based manifests plus score-only result tables.

`manifests/aligned_v13_split_manifest.csv` is the supplied 3,600-row
2,520/540/540 aligned-v13 assignment. `internal_split_manifest.csv` is the
same compatibility copy used by the existing pipeline. Read
`aligned_v13_split_audit.json` before use: its computed digest does not match
the rerun-lineage digest recorded by the current result artifacts, so it is a
release gate rather than a claim that the discrepancy is resolved.

For an authorized rerun, place `probe_train.json`, `probe_val.json`, and
`probe_test.json` in `data/legacy_v13/`; do not commit them. Each record must
provide a stable ID, labels `label_text`, `label_img`, and `label_combo`, text,
source metadata, and a locally resolvable image path. Their union must match
the final authoritative split manifest exactly.

The external manifests freeze the MM-SafetyBench/TextVQA indices, SIUO IDs,
the primary substitution map (seed 20260913), the twenty-map robustness
assignment, and MSSBench matched image-pair identifiers. Users must obtain
all external datasets and model checkpoints from their official distributors
and comply with their licenses and access requirements.
