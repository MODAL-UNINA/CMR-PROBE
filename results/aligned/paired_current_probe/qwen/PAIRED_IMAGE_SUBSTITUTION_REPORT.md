# Paired image substitution — qwen2-vl

The operational probe is frozen: no classifier fitting, calibration, or threshold selection is performed here.
Every comparison keeps the text and final textual token fixed and changes only the paired image.

| Population | n | mean original−substitution | 95% CI | P(original > substitution) | 95% CI |
|---|---:|---:|---:|---:|---:|
| SIUO unsafe | 167 | +0.055461 | [+0.023646, +0.087251] | 0.6287 | [0.5569, 0.7006] |
| TextVQA benign | 1100 | -0.000013 | [-0.000135, +0.000113] | 0.4409 | [0.4109, 0.4700] |

Safety-specific contrast (SIUO drop − TextVQA drop): **+0.055474** (95% CI [+0.023703, +0.087288]).

Verdict: **SUPPORTED**.
