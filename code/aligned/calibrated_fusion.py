#!/usr/bin/env python3
"""Evaluate target-calibrated score fusion."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
SEED = 42
N_REPEATS = 20

def load_cache(path: Path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if 'results_unsafe' in obj:
        return (obj['results_unsafe'], obj['results_safe'])
    if 'unsafe' in obj:
        return (obj['unsafe'], obj['safe'])
    raise KeyError(f'Unrecognized cache schema: {list(obj.keys())}')

def load_thresholds(config_path: Path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    tau_C = float(cfg['tau_C'])
    if 'internal_probes' in cfg:
        tau_T = float(cfg['internal_probes']['tau_T'])
        tau_V = float(cfg['internal_probes']['tau_V'])
    else:
        tau_T = float(cfg['tau_T'])
        tau_V = float(cfg['tau_V'])
    return {'T': tau_T, 'V': tau_V, 'C': tau_C}

def calibrate_component(unsafe_rows, safe_rows, component):
    key = f'S_{component}'
    scores_unsafe = np.array([float(r[key]) for r in unsafe_rows])
    scores_safe = np.array([float(r[key]) for r in safe_rows])
    labels = np.concatenate([np.ones(len(scores_unsafe)), np.zeros(len(scores_safe))])
    scores = np.concatenate([scores_unsafe, scores_safe])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    J = tpr - fpr
    idx = int(np.argmax(J))
    tau = float(thresholds[idx])
    return {'component': component, 'tau': tau, 'auc': auc, 'youden_J': float(J[idx]), 'tpr_cal': float(tpr[idx]), 'fpr_cal': float(fpr[idx]), 'active': bool(auc > 0.5)}

def compute_binary_metrics(pred_unsafe, pred_safe, score_unsafe=None, score_safe=None):
    pred_unsafe = np.asarray(pred_unsafe)
    pred_safe = np.asarray(pred_safe)
    tp = int(pred_unsafe.sum())
    fn = int((pred_unsafe == 0).sum())
    fp = int(pred_safe.sum())
    tn = int((pred_safe == 0).sum())
    tpr = tp / (tp + fn + 1e-09)
    fpr = fp / (fp + tn + 1e-09)
    precision = tp / (tp + fp + 1e-09)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-09)
    tnr = tn / (tn + fp + 1e-09)
    balanced_accuracy = (tpr + tnr) / 2
    auc = None
    if score_unsafe is not None and score_safe is not None:
        labels = np.concatenate([np.ones(len(score_unsafe)), np.zeros(len(score_safe))])
        scores = np.concatenate([score_unsafe, score_safe])
        auc = float(roc_auc_score(labels, scores))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': float(tpr), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(balanced_accuracy), 'auc': auc}

def evaluate_c_only(unsafe_rows, safe_rows, tau_C):
    sc_unsafe = np.array([float(r['S_C']) for r in unsafe_rows])
    sc_safe = np.array([float(r['S_C']) for r in safe_rows])
    pred_unsafe = (sc_unsafe >= tau_C).astype(int)
    pred_safe = (sc_safe >= tau_C).astype(int)
    return compute_binary_metrics(pred_unsafe, pred_safe, sc_unsafe, sc_safe)

def binary_metrics(pred_unsafe, pred_safe, score_unsafe=None, score_safe=None):
    pred_unsafe = np.asarray(pred_unsafe)
    pred_safe = np.asarray(pred_safe)
    tp = int(pred_unsafe.sum())
    fn = int((pred_unsafe == 0).sum())
    fp = int(pred_safe.sum())
    tn = int((pred_safe == 0).sum())
    tpr = tp / (tp + fn + 1e-09)
    fpr = fp / (fp + tn + 1e-09)
    precision = tp / (tp + fp + 1e-09)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-09)
    tnr = tn / (tn + fp + 1e-09)
    balanced_accuracy = (tpr + tnr) / 2
    auc = None
    if score_unsafe is not None and score_safe is not None:
        labels = np.concatenate([np.ones(len(score_unsafe)), np.zeros(len(score_safe))])
        scores = np.concatenate([score_unsafe, score_safe])
        auc = float(roc_auc_score(labels, scores))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': float(tpr), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(balanced_accuracy), 'auc': auc}

def compute_delta(rows, thresholds, active_components):
    if len(active_components) == 0:
        return np.full(len(rows), -np.inf)
    deltas = []
    for r in rows:
        margins = []
        for component in active_components:
            margins.append(float(r[f'S_{component}']) - thresholds[component])
        deltas.append(max(margins))
    return np.asarray(deltas)

def evaluate_fusion(unsafe_rows, safe_rows, thresholds, active_components):
    delta_unsafe = compute_delta(unsafe_rows, thresholds, active_components)
    delta_safe = compute_delta(safe_rows, thresholds, active_components)
    pred_unsafe = (delta_unsafe >= 0).astype(int)
    pred_safe = (delta_safe >= 0).astype(int)
    return binary_metrics(pred_unsafe, pred_safe, delta_unsafe, delta_safe)

def run_single_repeat(results_unsafe, results_safe, original_thresholds, n_cal_per_class, n_test_per_class, seed):
    rng = np.random.default_rng(seed)
    idx_u = rng.permutation(len(results_unsafe))
    unsafe_cal = [results_unsafe[int(i)] for i in idx_u[:n_cal_per_class]]
    unsafe_test = [results_unsafe[int(i)] for i in idx_u[n_cal_per_class:n_cal_per_class + n_test_per_class]]
    idx_s = rng.permutation(len(results_safe))
    safe_cal = [results_safe[int(i)] for i in idx_s[:n_cal_per_class]]
    safe_test = [results_safe[int(i)] for i in idx_s[n_cal_per_class:n_cal_per_class + n_test_per_class]]
    calibration = {}
    calibrated_thresholds = {}
    for component in ['T', 'V', 'C']:
        result = calibrate_component(unsafe_cal, safe_cal, component)
        calibration[component] = result
        calibrated_thresholds[component] = result['tau']
    active_gated = [component for component in ['T', 'V', 'C'] if calibration[component]['active']]
    fixed_full = evaluate_fusion(unsafe_test, safe_test, original_thresholds, ['T', 'V', 'C'])
    all_calibrated = evaluate_fusion(unsafe_test, safe_test, calibrated_thresholds, ['T', 'V', 'C'])
    gated_calibrated = evaluate_fusion(unsafe_test, safe_test, calibrated_thresholds, active_gated)
    c_only = evaluate_fusion(unsafe_test, safe_test, calibrated_thresholds, ['C'])
    return {'seed': seed, 'calibration': calibration, 'calibrated_thresholds': calibrated_thresholds, 'active_components': active_gated, 'fixed_full': fixed_full, 'all_calibrated': all_calibrated, 'gated_calibrated': gated_calibrated, 'c_only_calibrated': c_only}

def calibrate_tau_c(scores_unsafe: np.ndarray, scores_safe: np.ndarray):
    scores = np.concatenate([scores_unsafe, scores_safe])
    labels = np.concatenate([np.ones(len(scores_unsafe)), np.zeros(len(scores_safe))])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    J = tpr - fpr
    idx = np.argmax(J)
    tau = float(thresholds[idx])
    auc = float(roc_auc_score(labels, scores))
    return {'tau_C': tau, 'auc_cal': auc, 'tpr_cal': float(tpr[idx]), 'fpr_cal': float(fpr[idx]), 'youden_J': float(J[idx])}

def summarize_runs(runs):
    modes = ['fixed_full', 'all_calibrated', 'gated_calibrated', 'c_only_calibrated']
    metrics = ['tpr', 'fpr', 'precision', 'f1', 'balanced_accuracy', 'auc']
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for metric in metrics:
            values = np.array([r[mode][metric] for r in runs])
            summary[mode][metric] = {'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()), 'max': float(values.max())}
    summary['components'] = {}
    for component in ['T', 'V', 'C']:
        aucs = np.array([r['calibration'][component]['auc'] for r in runs])
        Js = np.array([r['calibration'][component]['youden_J'] for r in runs])
        taus = np.array([r['calibration'][component]['tau'] for r in runs])
        active = np.array([r['calibration'][component]['active'] for r in runs])
        summary['components'][component] = {'auc_mean': float(aucs.mean()), 'auc_std': float(aucs.std()), 'youden_mean': float(Js.mean()), 'youden_std': float(Js.std()), 'tau_mean': float(taus.mean()), 'tau_std': float(taus.std()), 'active_rate': float(active.mean())}
    return summary

def calibration_size_sensitivity(results_unsafe, results_safe, tau_T, tau_V, tau_C_original, calibration_sizes, n_test_per_class, n_repeats, seed):
    runs_by_size = {str(n): [] for n in calibration_sizes}
    max_cal = max(calibration_sizes)
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        idx_u = rng.permutation(len(results_unsafe))
        u_test_idx = idx_u[:n_test_per_class]
        u_cal_pool = idx_u[n_test_per_class:n_test_per_class + max_cal]
        unsafe_test = [results_unsafe[int(i)] for i in u_test_idx]
        idx_s = rng.permutation(len(results_safe))
        s_test_idx = idx_s[:n_test_per_class]
        s_cal_pool = idx_s[n_test_per_class:n_test_per_class + max_cal]
        safe_test = [results_safe[int(i)] for i in s_test_idx]
        for n_cal in calibration_sizes:
            unsafe_cal = [results_unsafe[int(i)] for i in u_cal_pool[:n_cal]]
            safe_cal = [results_safe[int(i)] for i in s_cal_pool[:n_cal]]
            sc_unsafe_cal = np.array([float(r['S_C']) for r in unsafe_cal])
            sc_safe_cal = np.array([float(r['S_C']) for r in safe_cal])
            calibration = calibrate_tau_c(sc_unsafe_cal, sc_safe_cal)
            tau_C_target = calibration['tau_C']
            metrics = evaluate_c_only(unsafe_test, safe_test, tau_C_target)
            runs_by_size[str(n_cal)].append({'tau_C': tau_C_target, 'auc_cal': calibration['auc_cal'], 'youden_J': calibration['youden_J'], 'metrics': metrics})
    results = {}
    for n_cal in calibration_sizes:
        runs = runs_by_size[str(n_cal)]
        taus = np.array([r['tau_C'] for r in runs])
        results_metrics = {}
        for metric in ['tpr', 'fpr', 'precision', 'f1', 'balanced_accuracy', 'auc']:
            values = np.array([r['metrics'][metric] for r in runs])
            results_metrics[metric] = {'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()), 'max': float(values.max())}
        results[str(n_cal)] = {'n_cal_per_class': n_cal, 'n_cal_total': 2 * n_cal, 'n_test_per_class': n_test_per_class, 'n_test_total': 2 * n_test_per_class, 'test_fixed_across_sizes': True, 'tau_C': {'mean': float(taus.mean()), 'std': float(taus.std()), 'min': float(taus.min()), 'max': float(taus.max())}, 'metrics': results_metrics}
        r = results_metrics
        print(f"\n{'=' * 72}")
        print(f'CALIBRATION SIZE = {n_cal} PER CLASS ({2 * n_cal} TOTAL)')
        print(f"{'=' * 72}")
        print(f'tau_C = {taus.mean():.6f} ± {taus.std():.6f}')
        print(f"TPR   = {r['tpr']['mean']:.4f} ± {r['tpr']['std']:.4f}")
        print(f"FPR   = {r['fpr']['mean']:.4f} ± {r['fpr']['std']:.4f}")
        print(f"F1    = {r['f1']['mean']:.4f} ± {r['f1']['std']:.4f}")
        print(f"BAcc  = {r['balanced_accuracy']['mean']:.4f} ± {r['balanced_accuracy']['std']:.4f}")
    return results

def plot_calibration_size_sensitivity(results, path):
    sizes = sorted([int(k) for k in results.keys()])
    tpr_mean = []
    tpr_std = []
    fpr_mean = []
    fpr_std = []
    f1_mean = []
    f1_std = []
    bacc_mean = []
    bacc_std = []
    for n in sizes:
        r = results[str(n)]['metrics']
        tpr_mean.append(r['tpr']['mean'])
        tpr_std.append(r['tpr']['std'])
        fpr_mean.append(r['fpr']['mean'])
        fpr_std.append(r['fpr']['std'])
        f1_mean.append(r['f1']['mean'])
        f1_std.append(r['f1']['std'])
        bacc_mean.append(r['balanced_accuracy']['mean'])
        bacc_std.append(r['balanced_accuracy']['std'])
    x = np.array(sizes)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, tpr_mean, yerr=tpr_std, marker='o', capsize=4, label='TPR')
    ax.errorbar(x, fpr_mean, yerr=fpr_std, marker='o', capsize=4, label='FPR')
    ax.errorbar(x, f1_mean, yerr=f1_std, marker='o', capsize=4, label='F1')
    ax.errorbar(x, bacc_mean, yerr=bacc_std, marker='o', capsize=4, label='Balanced Accuracy')
    ax.set_xlabel('Calibration samples for class')
    ax.set_ylabel('Score')
    ax.set_xticks(sizes)
    ax.set_ylim(0, 1.05)
    ax.set_title('Sensitivity to target calibration size')
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def print_summary(summary):
    print('\n' + '=' * 96)
    print('TARGET-CALIBRATED FUSION')
    print('=' * 96)
    print('\nComponent diagnostics')
    print('-' * 80)
    print(f"{'Comp':8s}{'AUC cal':>16s}{'Youden J':>16s}{'tau':>18s}{'Active':>14s}")
    print('-' * 80)
    for component in ['T', 'V', 'C']:
        r = summary['components'][component]
        print(f"{component:8s}{r['auc_mean']:8.4f}±{r['auc_std']:.3f} {r['youden_mean']:8.4f}±{r['youden_std']:.3f} {r['tau_mean']:10.6f}±{r['tau_std']:.6f} {100 * r['active_rate']:8.1f}%")
    print('\nPerformance')
    print('-' * 96)
    print(f"{'Mode':24s}{'TPR':>13s}{'FPR':>13s}{'F1':>13s}{'BAcc':>13s}{'AUC':>13s}")
    print('-' * 96)
    names = {'fixed_full': 'Fixed Full', 'all_calibrated': 'All-Calibrated', 'gated_calibrated': 'Gated-Calibrated', 'c_only_calibrated': 'C-only Calibrated'}
    for key, name in names.items():
        r = summary[key]
        print(f"{name:24s}{r['tpr']['mean']:8.4f}±{r['tpr']['std']:.3f} {r['fpr']['mean']:8.4f}±{r['fpr']['std']:.3f} {r['f1']['mean']:8.4f}±{r['f1']['std']:.3f} {r['balanced_accuracy']['mean']:8.4f}±{r['balanced_accuracy']['std']:.3f} {r['auc']['mean']:8.4f}±{r['auc']['std']:.3f}")
    print('=' * 96)

def plot_metrics(summary, path):
    keys = ['fixed_full', 'all_calibrated', 'gated_calibrated', 'c_only_calibrated']
    names = ['Fixed Full', 'All-Cal.', 'Gated-Cal.', 'C-only Cal.']
    tpr = [summary[k]['tpr']['mean'] for k in keys]
    fpr = [summary[k]['fpr']['mean'] for k in keys]
    f1 = [summary[k]['f1']['mean'] for k in keys]
    bacc = [summary[k]['balanced_accuracy']['mean'] for k in keys]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, tpr, width, label='TPR')
    ax.bar(x - 0.5 * width, f1, width, label='F1')
    ax.bar(x + 0.5 * width, bacc, width, label='Balanced Accuracy')
    ax.bar(x + 1.5 * width, fpr, width, label='FPR')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Target-calibrated fusion')
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_component_activity(summary, path):
    components = ['T', 'V', 'C']
    activity = [100 * summary['components'][c]['active_rate'] for c in components]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(components, activity)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Active across repeats (%)')
    ax.set_xlabel('Component')
    ax.set_title('Calibration-gated component activity')
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def run(args):
    cache_path = Path(args.cache)
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n[1] Loading cache...')
    unsafe, safe = load_cache(cache_path)
    print(f'  unsafe = {len(unsafe)}')
    print(f'  safe   = {len(safe)}')
    print('\n[2] Original thresholds...')
    original_thresholds = load_thresholds(config_path)
    for component in ['T', 'V', 'C']:
        print(f'  tau_{component} = {original_thresholds[component]:.6f}')
    print('\n[3] Target-calibrated fusion...')
    runs = []
    for repeat in range(args.n_repeats):
        result = run_single_repeat(unsafe, safe, original_thresholds, n_cal_per_class=args.n_cal_per_class, n_test_per_class=args.n_test_per_class, seed=args.seed + repeat)
        runs.append(result)
        active = '+'.join(result['active_components'])
        if not active:
            active = 'NONE'
        print(f'  repeat {repeat + 1:02d}: active={active}')
    summary = summarize_runs(runs)
    print_summary(summary)
    original_thresholds = load_thresholds(config_path)
    print('\n[4b] Calibration-size sensitivity...')
    calibration_size_results = calibration_size_sensitivity(unsafe, safe, original_thresholds['T'], original_thresholds['V'], original_thresholds['C'], calibration_sizes=[10, 25, 50], n_test_per_class=50, n_repeats=args.n_repeats, seed=args.seed)
    output = {'experiment': 'target_calibrated_fusion', 'protocol': {'cache': str(cache_path), 'n_cal_per_class': args.n_cal_per_class, 'n_test_per_class': args.n_test_per_class, 'n_cal_total': 2 * args.n_cal_per_class, 'n_test_total': 2 * args.n_test_per_class, 'n_repeats': args.n_repeats, 'seed': args.seed, 'calibration_method': 'Youden J', 'component_gate': 'AUC_cal > 0.5', 'test_used_for_calibration': False, 'probe_retrained': False, 'calibration_size_sensitivity': calibration_size_results}, 'original_thresholds': original_thresholds, 'summary': summary, 'runs': runs}
    json_path = out_dir / 'calibrated_fusion_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    plot_metrics(summary, out_dir / 'calibrated_fusion_metrics.png')
    plot_component_activity(summary, out_dir / 'calibrated_component_activity.png')
    plot_calibration_size_sensitivity(calibration_size_results, out_dir / 'target_calibration_size_sensitivity.png')
    print('\nSaved:')
    print(f'  {json_path}')
    print(f"  {out_dir / 'calibrated_fusion_metrics.png'}")
    print(f"  {out_dir / 'calibrated_component_activity.png'}")
if __name__ == '__main__':
    import sys
    CONFIG = {'cache': 'probes/rebuilt_v13/delta/eval_results_textvqa_internal.pkl', 'config': 'probes/rebuilt_v13/delta/delta_config.json', 'out': 'probes/rebuilt_v13/calibrated_fusion_mmsafety', 'n_cal_per_class': 50, 'n_test_per_class': 50, 'n_repeats': 20, 'seed': 42}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cache', default=CONFIG['cache'])
        parser.add_argument('--config', default=CONFIG['config'])
        parser.add_argument('--out', default=CONFIG['out'])
        parser.add_argument('--n_cal_per_class', type=int, default=CONFIG['n_cal_per_class'])
        parser.add_argument('--n_test_per_class', type=int, default=CONFIG['n_test_per_class'])
        parser.add_argument('--n_repeats', type=int, default=CONFIG['n_repeats'])
        parser.add_argument('--seed', type=int, default=CONFIG['seed'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
