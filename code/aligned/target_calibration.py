#!/usr/bin/env python3
"""Evaluate few-shot target-domain threshold calibration."""
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
    return (tau_T, tau_V, tau_C)

def calibrate_tau_c(scores_unsafe: np.ndarray, scores_safe: np.ndarray):
    scores = np.concatenate([scores_unsafe, scores_safe])
    labels = np.concatenate([np.ones(len(scores_unsafe)), np.zeros(len(scores_safe))])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    J = tpr - fpr
    idx = np.argmax(J)
    tau = float(thresholds[idx])
    auc = float(roc_auc_score(labels, scores))
    return {'tau_C': tau, 'auc_cal': auc, 'tpr_cal': float(tpr[idx]), 'fpr_cal': float(fpr[idx]), 'youden_J': float(J[idx])}

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

def evaluate_full(unsafe_rows, safe_rows, tau_T, tau_V, tau_C):

    def delta(rows):
        return np.array([max(float(r['S_T']) - tau_T, float(r['S_V']) - tau_V, float(r['S_C']) - tau_C) for r in rows])
    delta_unsafe = delta(unsafe_rows)
    delta_safe = delta(safe_rows)
    pred_unsafe = (delta_unsafe >= 0).astype(int)
    pred_safe = (delta_safe >= 0).astype(int)
    return compute_binary_metrics(pred_unsafe, pred_safe, delta_unsafe, delta_safe)

def run_single_repeat(results_unsafe, results_safe, tau_T, tau_V, tau_C_original, n_cal_per_class, n_test_per_class, seed):
    rng = np.random.default_rng(seed)
    idx_u = rng.permutation(len(results_unsafe))
    u_cal_idx = idx_u[:n_cal_per_class]
    u_test_idx = idx_u[n_cal_per_class:n_cal_per_class + n_test_per_class]
    idx_s = rng.permutation(len(results_safe))
    s_cal_idx = idx_s[:n_cal_per_class]
    s_test_idx = idx_s[n_cal_per_class:n_cal_per_class + n_test_per_class]
    unsafe_cal = [results_unsafe[int(i)] for i in u_cal_idx]
    unsafe_test = [results_unsafe[int(i)] for i in u_test_idx]
    safe_cal = [results_safe[int(i)] for i in s_cal_idx]
    safe_test = [results_safe[int(i)] for i in s_test_idx]
    sc_unsafe_cal = np.array([float(r['S_C']) for r in unsafe_cal])
    sc_safe_cal = np.array([float(r['S_C']) for r in safe_cal])
    calibration = calibrate_tau_c(sc_unsafe_cal, sc_safe_cal)
    tau_C_target = calibration['tau_C']
    c_fixed = evaluate_c_only(unsafe_test, safe_test, tau_C_original)
    c_calibrated = evaluate_c_only(unsafe_test, safe_test, tau_C_target)
    full_fixed = evaluate_full(unsafe_test, safe_test, tau_T, tau_V, tau_C_original)
    full_calibrated = evaluate_full(unsafe_test, safe_test, tau_T, tau_V, tau_C_target)
    return {'seed': seed, 'calibration': calibration, 'c_fixed': c_fixed, 'c_target_calibrated': c_calibrated, 'full_fixed': full_fixed, 'full_target_calibrated': full_calibrated}

def summarize_runs(runs):
    modes = ['c_fixed', 'c_target_calibrated', 'full_fixed', 'full_target_calibrated']
    metrics = ['tpr', 'fpr', 'precision', 'f1', 'balanced_accuracy', 'auc']
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for metric in metrics:
            values = np.array([r[mode][metric] for r in runs])
            summary[mode][metric] = {'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()), 'max': float(values.max())}
    taus = np.array([r['calibration']['tau_C'] for r in runs])
    summary['tau_C_target'] = {'mean': float(taus.mean()), 'std': float(taus.std()), 'median': float(np.median(taus)), 'min': float(taus.min()), 'max': float(taus.max())}
    return summary

def print_summary(summary, tau_C_original):
    print('\n' + '=' * 92)
    print('FEW-SHOT TARGET CALIBRATION')
    print('=' * 92)
    print(f'Original tau_C = {tau_C_original:.6f}')
    tau = summary['tau_C_target']
    print(f"Target tau_C   = {tau['mean']:.6f} ± {tau['std']:.6f}")
    print(f"Range          = [{tau['min']:.6f}, {tau['max']:.6f}]")
    print('-' * 92)
    print(f"{'Mode':24s}{'TPR':>13s}{'FPR':>13s}{'F1':>13s}{'BAcc':>13s}{'AUC':>13s}")
    print('-' * 92)
    names = {'c_fixed': 'C fixed', 'c_target_calibrated': 'C target-cal.', 'full_fixed': 'Full fixed', 'full_target_calibrated': 'Full target-cal.'}
    for key, name in names.items():
        r = summary[key]
        print(f"{name:24s}{r['tpr']['mean']:8.4f}±{r['tpr']['std']:.3f} {r['fpr']['mean']:8.4f}±{r['fpr']['std']:.3f} {r['f1']['mean']:8.4f}±{r['f1']['std']:.3f} {r['balanced_accuracy']['mean']:8.4f}±{r['balanced_accuracy']['std']:.3f} {r['auc']['mean']:8.4f}±{r['auc']['std']:.3f}")
    print('=' * 92)

def plot_tau_distribution(runs, tau_C_original, path):
    taus = np.array([r['calibration']['tau_C'] for r in runs])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(taus) + 1)
    ax.plot(x, taus, marker='o', label='Target-calibrated $\\tau_C$')
    ax.axhline(tau_C_original, linestyle='--', label=f'Original $\\tau_C={tau_C_original:.4f}$')
    ax.set_xlabel('Repeat')
    ax.set_ylabel('$\\tau_C$')
    ax.set_title('Target calibration stability')
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_metrics(summary, path):
    modes = ['C fixed', 'C target-cal.', 'Full fixed', 'Full target-cal.']
    keys = ['c_fixed', 'c_target_calibrated', 'full_fixed', 'full_target_calibrated']
    f1 = [summary[k]['f1']['mean'] for k in keys]
    bacc = [summary[k]['balanced_accuracy']['mean'] for k in keys]
    fpr = [summary[k]['fpr']['mean'] for k in keys]
    x = np.arange(len(modes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, f1, width, label='F1')
    ax.bar(x, bacc, width, label='Balanced Accuracy')
    ax.bar(x + width, fpr, width, label='FPR')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Fixed vs target-calibrated threshold')
    ax.legend()
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
    tau_T, tau_V, tau_C_original = load_thresholds(config_path)
    print(f'  tau_T = {tau_T:.4f}')
    print(f'  tau_V = {tau_V:.4f}')
    print(f'  tau_C = {tau_C_original:.4f}')
    print('\n[3] Target calibration...')
    runs = []
    for repeat in range(args.n_repeats):
        result = run_single_repeat(unsafe, safe, tau_T, tau_V, tau_C_original, n_cal_per_class=args.n_cal_per_class, n_test_per_class=args.n_test_per_class, seed=args.seed + repeat)
        runs.append(result)
        print(f"  repeat {repeat + 1:02d}: tau_C={result['calibration']['tau_C']:.6f}")
    summary = summarize_runs(runs)
    print_summary(summary, tau_C_original)
    output = {'experiment': 'few_shot_target_calibration', 'protocol': {'cache': str(cache_path), 'tau_T_frozen': tau_T, 'tau_V_frozen': tau_V, 'tau_C_original': tau_C_original, 'calibrated_component': 'S_C only', 'calibration_method': 'Youden J', 'n_cal_per_class': args.n_cal_per_class, 'n_test_per_class': args.n_test_per_class, 'n_cal_total': 2 * args.n_cal_per_class, 'n_test_total': 2 * args.n_test_per_class, 'n_repeats': args.n_repeats, 'seed': args.seed, 'probe_retrained': False, 'test_used_for_calibration': False}, 'summary': summary, 'runs': runs}
    json_path = out_dir / 'target_calibration_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    plot_tau_distribution(runs, tau_C_original, out_dir / 'target_calibration_tau.png')
    plot_metrics(summary, out_dir / 'target_calibration_metrics.png')
    print('\nSaved:')
    print(f'  {json_path}')
    print(f"  {out_dir / 'target_calibration_tau.png'}")
    print(f"  {out_dir / 'target_calibration_metrics.png'}")
if __name__ == '__main__':
    import sys
    CONFIG = {'cache': 'probes/rebuilt_v13/delta/eval_results_textvqa_internal.pkl', 'config': 'probes/rebuilt_v13/delta/delta_config.json', 'out': 'probes/rebuilt_v13/target_calibration_mmsafety', 'n_cal_per_class': 50, 'n_test_per_class': 50, 'n_repeats': 20, 'seed': 42}
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
