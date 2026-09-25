# Paired image substitution — llava

The operational probe is frozen: no classifier fitting, calibration, or threshold selection is performed here.
Every comparison keeps the text and final textual token fixed and changes only the paired image.

| Population | n | mean original−substitution | 95% CI | P(original > substitution) | 95% CI |
|---|---:|---:|---:|---:|---:|
| SIUO unsafe | 167 | +0.072650 | [+0.050161, +0.095961] | 0.6766 | [0.6048, 0.7485] |
| TextVQA benign | 1100 | -0.004671 | [-0.006580, -0.002800] | 0.3100 | [0.2827, 0.3373] |

Safety-specific contrast (SIUO drop − TextVQA drop): **+0.077320** (95% CI [+0.054922, +0.100567]).

Verdict: **SUPPORTED**.
