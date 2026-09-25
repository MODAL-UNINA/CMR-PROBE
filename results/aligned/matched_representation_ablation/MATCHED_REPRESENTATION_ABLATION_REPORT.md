# Matched representation-control ablation

This experiment isolates two factors:

1. residual definition: raw vs aligned;
2. nuisance projection: absent vs train-fitted projection.

All four conditions use the same frozen split, source-by-label weights, StandardScaler + balanced linear logistic probe, and validation-only sigmoid calibration.

## Nuisance bases

- Q_raw dimension: 39
- Q_aligned dimension: 39

## Internal frozen test

### C_vs_D_in_family

| Condition | ROC-AUC | 95% CI | PR-AUC | 95% CI |
|---|---:|---:|---:|---:|
| raw_no_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| raw_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| aligned_no_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| aligned_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |

### all_categories_in_family

| Condition | ROC-AUC | 95% CI | PR-AUC | 95% CI |
|---|---:|---:|---:|---:|
| raw_no_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| raw_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| aligned_no_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| aligned_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |

### MSTS_vs_in_family_C

| Condition | ROC-AUC | 95% CI | PR-AUC | 95% CI |
|---|---:|---:|---:|---:|
| raw_no_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| raw_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| aligned_no_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |
| aligned_proj | 1.0000 | [+1.0000, +1.0000] | 1.0000 | [+1.0000, +1.0000] |

## Paired image-swap analysis

Positive change means that the original pair receives a higher calibrated score than its fixed-text image substitution.

### TextVQA_safe

| Condition | mean original-swap | 95% CI | p_win | 95% CI |
|---|---:|---:|---:|---:|
| raw_no_proj | +0.0047 | [+0.0021, +0.0075] | 0.6345 | [+0.6064, +0.6636] |
| raw_proj | -0.0018 | [-0.0041, +0.0005] | 0.5255 | [+0.4964, +0.5555] |
| aligned_no_proj | -0.0005 | [-0.0033, +0.0024] | 0.5445 | [+0.5145, +0.5736] |
| aligned_proj | -0.0039 | [-0.0058, -0.0021] | 0.4309 | [+0.4009, +0.4609] |

### SIUO_unsafe

| Condition | mean original-swap | 95% CI | p_win | 95% CI |
|---|---:|---:|---:|---:|
| raw_no_proj | +0.0984 | [+0.0737, +0.1228] | 0.7784 | [+0.7126, +0.8383] |
| raw_proj | +0.0966 | [+0.0726, +0.1214] | 0.7485 | [+0.6826, +0.8144] |
| aligned_no_proj | +0.1051 | [+0.0799, +0.1309] | 0.7545 | [+0.6886, +0.8204] |
| aligned_proj | +0.0934 | [+0.0708, +0.1165] | 0.7365 | [+0.6647, +0.8024] |

## Safety-specific difference-in-differences

Defined as mean SIUO(original-swap) minus mean TextVQA(original-swap).

| Condition | DiD | 95% CI |
|---|---:|---:|
| raw_no_proj | +0.0937 | [+0.0691, +0.1182] |
| raw_proj | +0.0984 | [+0.0740, +0.1233] |
| aligned_no_proj | +0.1056 | [+0.0800, +0.1313] |
| aligned_proj | +0.0973 | [+0.0748, +0.1206] |

## Interpretation

`raw_no_proj -> raw_proj` isolates the effect of nuisance projection for the raw residual. `aligned_no_proj -> aligned_proj` isolates the projection effect for the aligned residual. Comparing raw and aligned under the same projection status isolates the residual-definition effect.

Calibrated mean changes are descriptive. p_win is reported because it is invariant to monotone score rescaling within each fitted model.
