# MSSBench matched-source evaluation

The Chat Task is evaluated with the same query under its official paired safe and unsafe images. No MSSBench sample is used for fitting, calibration, nuisance-projection estimation, or threshold selection.

Matched scenarios: **300**; matched queries: **600**.
Incomplete Chat Task entries excluded because no complete safe/unsafe pair was available: **0**.

| Backbone | ROC-AUC | mean unsafe-safe | P(unsafe>safe) | d_z | TPR@5% | FPR@5% | TPR@1% | FPR@1% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | 0.4691 | -0.02739 | 0.423 | -0.200 | 0.990 | 0.990 | 0.810 | 0.840 |
| Qwen2-VL-7B-Instruct | 0.8688 | +0.23733 | 0.823 | +0.716 | 0.982 | 0.325 | 0.942 | 0.287 |

Bootstrap CIs are stored in `mssbench_matched_source_results.json` and resample whole MSSBench scenarios, retaining all queries and both visual contexts inside each resampling unit.
