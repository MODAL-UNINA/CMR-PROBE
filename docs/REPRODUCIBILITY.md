# Reproducibility protocol

The aligned-v13 operational score compares hidden states at the same final
textual token after verifying token identity between text-only and multimodal
prompts. LLaVA uses language layer 17; Qwen2-VL uses language layer 11. The
residual is the L2-normalized difference of individually L2-normalized states, followed by a
training-fitted nuisance projection, a balanced linear logistic probe, and a
validation-fitted sigmoid calibrator. No component is refitted for the fixed
paired map, twenty-map robustness run, or MSSBench matched-source evaluation.

The supplied split has 2,520 training, 540 validation, and 540 internal-test
records (630/135/135 per category). It is example-disjoint but not
group-disjoint: the post-hoc audit found related components that cross the
partitions. Internal performance is therefore diagnostic; it is not evidence
of independent generalization.

Fixed substitutions use seed 20260913. SIUO mappings are one-to-one,
no-self, and cross-category; TextVQA mappings are one-to-one and no-self.
The twenty additional mappings use seeds 20261001–20261020. Paired confidence
intervals use 10,000 resamples with seed 20260914; hierarchical multiple-map
intervals resample mappings and source IDs with seed 20261031.

Run the core and full commands in the repository README. Each rerun must
record model and processor revisions, layer, prompt, token-alignment checks,
package versions, CUDA and GPU details, all seeds, and SHA-256 digests of the
split, cache, probe, calibration, and mapping inputs. Never replace committed
score tables with values generated from a different frozen estimator.

The public result set excludes hidden-state caches and serialized probes. It
therefore supports score-table reproduction and audit, but exact model
extraction requires licensed models, datasets, internal records, GPU hardware,
and controlled access to the omitted frozen artifacts.
