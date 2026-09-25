# CMR-PROBE
Multimodal large language models support interactive systems in which safety-relevant meaning may arise from joint text--image interpretation. We introduce CMR-PROBE, a representation-probing framework based on an aligned cross-modal residual: the normalized difference between the hidden states of the same verified final textual token in text-only and multimodal passes. A lightweight linear probe maps this residual to a cross-modal risk score \(S_C\). We evaluate it through controlled image substitutions, frozen-threshold transfer, matched-source evaluation, and cross-backbone replication, and study integration with independently trained text and visual branches via a threshold-relative OR rule and supervised score-level fusion. Controlled substitutions show pair sensitivity on both backbones, with positive aggregate effects across 20 predeclared mappings. On MSSBench, six representation paths separate pair sensitivity from representation-specific superiority. On Qwen2-VL, the operational residual achieves strong safe/unsafe context ordering (ROC-AUC \(=.869\)) and outperforms visual-only and legacy-pair controls, although direct multimodal states and unprojected aligned residuals yield slightly higher AUCs. On LLaVA, reversed safety ordering is shared by language-conditioned estimators. Cross-dataset experiments further characterize threshold transfer and multi-branch integration under distribution shift. Thus, CMR-PROBE provides an assignment-robust pair-sensitive signal whose safety orientation and projection benefit depend on backbone and transfer domain.
Reproducibility package for CMR-PROBE aligned-v13: token-aligned cross-modal residual probes for multimodal safety.
<img width="1947" height="808" alt="new_pipeline_4" src="https://github.com/user-attachments/assets/1d85b242-3764-403d-9fe0-10ff9afc2249" />

This release contains only the current pipeline. It measures the image-conditioned change at the same verified final textual token in a text-only and multimodal pass:  

```
r = L2(L2(h_TV,last) - L2(h_T,last))
z = (I - QQᵀ) r
S_C = sigmoid(LogisticRegression(z))
```


`Q` is fitted on training data only; the operational branch does not apply a post-projection standardizer. The logistic probe is fitted on the frozen 2,520-example training partition and its sigmoid calibrator is fitted on the 540-example validation partition. All test, paired-substitution, MSSBench, and multiple-mapping analyses use that frozen estimator without refitting. The two backbones are LLaVA-1.5-7B (language layer 17) and Qwen2-VL-7B-Instruct (language layer 11). The manifest records an example-disjoint 2,520/540/540 train/validation/test assignment; it is not group-disjoint according to the post-hoc related-component audit.

## Reproduction

Use Python 3.10 or 3.11. The pinned statistical environment is NumPy 2.2.6, SciPy 1.15.3, and scikit-learn 1.7.2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-extraction.txt

# Place licensed internal files under data/legacy/, then run LLaVA.
SC_REPRESENTATION=aligned GPU_ID=0 PYTHON_BIN=python3 \
  bash code/aligned/run_core_pipeline.sh .

# Add MM-SafetyBench, TextVQA, and SIUO evaluations.
SC_REPRESENTATION=aligned GPU_ID=0 PYTHON_BIN=python3 \
  bash code/aligned/run_full_pipeline.sh .
```

The core run writes mutable caches, probes, thresholds, and local logs under `artifacts/`, which is intentionally ignored. The full run adds the external common-FPR and SIUO stages. Qwen training and the MSSBench matched-source runner require the two source scripts listed by `tools/check_release.py`; they are not included until retrieved from the designated remote machine.

## Published result paths

Committed score-only artifacts are under `results/aligned/`:

- `paired_current_probe/` — primary fixed-map SIUO/TextVQA substitution;
- `matched_representation_ablation/` — matched 2×2 residual controls;
- `mssbench_matched_source/` — matched safe/unsafe source-pair transfer;
- `multiple_mapping/` — 20 predeclared substitution mappings;
- `common_fpr/`, `nested_bootstrap/`, `efficiency/`, `multiseed/`, and
  `target_calibration/` — operating and sensitivity analyses.

Run `python3 code/reporting/summarize_results.py` for a compact report and `python3 code/reporting/generate_paper_tables.py` to regenerate paper tables from the committed JSON/CSV inputs. Before publication, run `python3 tools/check_release.py`. The checker intentionally rejects the current assembly until the remote scripts and the authoritative split manifest whose digest matches rerun lineage are supplied.

`data/README.md` specifies the data boundary; `docs/REPRODUCIBILITY.md`
documents exact protocol constraints and expected artifacts.
