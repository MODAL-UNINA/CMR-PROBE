#!/usr/bin/env python3
"""Measure sensitivity to the combination-score threshold."""
from __future__ import annotations
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
TAU_C_GRID = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

def load_cache(path: Path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if 'results_unsafe' in obj and 'results_safe' in obj:
        return (obj['results_unsafe'], obj['results_safe'])
    if 'unsafe' in obj and 'safe' in obj:
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

def compute_metrics(pred_unsafe: np.ndarray, pred_safe: np.ndarray) -> dict:
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
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': float(tpr), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(balanced_accuracy)}

def evaluate_c_only(results_unsafe: list, results_safe: list, tau_C: float) -> dict:
    sc_unsafe = np.array([float(r['S_C']) for r in results_unsafe])
    sc_safe = np.array([float(r['S_C']) for r in results_safe])
    pred_unsafe = (sc_unsafe >= tau_C).astype(int)
    pred_safe = (sc_safe >= tau_C).astype(int)
    return compute_metrics(pred_unsafe, pred_safe)

def evaluate_full(results_unsafe: list, results_safe: list, tau_T: float, tau_V: float, tau_C: float) -> dict:

    def predict(rows):
        pred = []
        for r in rows:
            delta = max(float(r['S_T']) - tau_T, float(r['S_V']) - tau_V, float(r['S_C']) - tau_C)
            pred.append(1 if delta >= 0 else 0)
        return np.array(pred)
    pred_unsafe = predict(results_unsafe)
    pred_safe = predict(results_safe)
    return compute_metrics(pred_unsafe, pred_safe)

def run_sensitivity(results_unsafe: list, results_safe: list, tau_T: float, tau_V: float, tau_C_original: float):
    thresholds = np.unique(np.concatenate([TAU_C_GRID, [tau_C_original]]))
    thresholds.sort()
    c_only_results = []
    full_results = []
    for tau_C in thresholds:
        c_metrics = evaluate_c_only(results_unsafe, results_safe, tau_C)
        full_metrics = evaluate_full(results_unsafe, results_safe, tau_T, tau_V, tau_C)
        c_metrics['tau_C'] = float(tau_C)
        full_metrics['tau_C'] = float(tau_C)
        c_only_results.append(c_metrics)
        full_results.append(full_metrics)
    return (c_only_results, full_results)

def print_table(results: list, title: str, tau_C_original: float):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"{'tau_C':>9s} {'TPR':>8s} {'FPR':>8s} {'Prec':>8s} {'F1':>8s} {'BAcc':>8s}")
    print('-' * 80)
    for r in results:
        marker = ' <- original' if np.isclose(r['tau_C'], tau_C_original) else ''
        print(f"{r['tau_C']:9.4f} {r['tpr']:8.4f} {r['fpr']:8.4f} {r['precision']:8.4f} {r['f1']:8.4f} {r['balanced_accuracy']:8.4f}{marker}")

def plot_sensitivity(results: list, tau_C_original: float, title: str, path: Path):
    tau = np.array([r['tau_C'] for r in results])
    tpr = np.array([r['tpr'] for r in results])
    fpr = np.array([r['fpr'] for r in results])
    f1 = np.array([r['f1'] for r in results])
    balanced = np.array([r['balanced_accuracy'] for r in results])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tau, tpr, marker='o', label='TPR')
    ax.plot(tau, fpr, marker='o', label='FPR')
    ax.plot(tau, f1, marker='o', label='F1')
    ax.plot(tau, balanced, marker='o', label='Balanced Accuracy')
    ax.axvline(tau_C_original, linestyle='--', linewidth=1.5, label=f'Original $\\tau_C={tau_C_original:.4f}$')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('$\\tau_C$')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved: {path}')

def save_csv(c_only_results: list, full_results: list, path: Path):
    fields = ['mode', 'tau_C', 'tpr', 'fpr', 'precision', 'f1', 'balanced_accuracy', 'tp', 'fp', 'fn', 'tn']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for mode, results in [('C-only', c_only_results), ('T+V+C', full_results)]:
            for r in results:
                writer.writerow({'mode': mode, **r})

def run(args):
    cache_path = Path(args.cache)
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print('\n[1] Loading cache...')
    results_unsafe, results_safe = load_cache(cache_path)
    print(f'  unsafe = {len(results_unsafe)}')
    print(f'  safe   = {len(results_safe)}')
    print('\n[2] Loading thresholds...')
    tau_T, tau_V, tau_C_original = load_thresholds(config_path)
    print(f'  tau_T = {tau_T:.4f}')
    print(f'  tau_V = {tau_V:.4f}')
    print(f'  tau_C = {tau_C_original:.4f}')
    print('\n[3] S_C distribution...')
    sc_unsafe = np.array([r['S_C'] for r in results_unsafe])
    sc_safe = np.array([r['S_C'] for r in results_safe])
    print(f'  unsafe mean   = {sc_unsafe.mean():.6f}')
    print(f'  unsafe median = {np.median(sc_unsafe):.6f}')
    print(f'  unsafe max    = {sc_unsafe.max():.6f}')
    print(f'  safe mean     = {sc_safe.mean():.6f}')
    print(f'  safe median   = {np.median(sc_safe):.6f}')
    print(f'  safe max      = {sc_safe.max():.6f}')
    print('\n[4] Threshold sensitivity...')
    c_only_results, full_results = run_sensitivity(results_unsafe, results_safe, tau_T, tau_V, tau_C_original)
    print_table(c_only_results, 'C-ONLY THRESHOLD SENSITIVITY', tau_C_original)
    print_table(full_results, 'FULL PIPELINE THRESHOLD SENSITIVITY', tau_C_original)
    output = {'cache': str(cache_path), 'original_thresholds': {'tau_T': tau_T, 'tau_V': tau_V, 'tau_C': tau_C_original}, 'n_samples': {'unsafe': len(results_unsafe), 'safe': len(results_safe)}, 'note': 'Diagnostic sensitivity analysis only. No threshold was recalibrated using the test set.', 'c_only': c_only_results, 'full_pipeline': full_results}
    json_path = out_dir / 'threshold_sensitivity_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    csv_path = out_dir / 'threshold_sensitivity_results.csv'
    save_csv(c_only_results, full_results, csv_path)
    plot_sensitivity(c_only_results, tau_C_original, 'Threshold sensitivity — $S_C$ only', out_dir / 'threshold_sensitivity_c_only.png')
    plot_sensitivity(full_results, tau_C_original, 'Threshold sensitivity — Full pipeline', out_dir / 'threshold_sensitivity_full.png')
    print('\nSaved:')
    print(f'  {json_path}')
    print(f'  {csv_path}')
if __name__ == '__main__':
    import sys
    CONFIG = {'cache': 'probes/rebuilt_v13/delta/eval_results_textvqa_internal.pkl', 'config': 'probes/rebuilt_v13/delta/delta_config.json', 'out': 'probes/rebuilt_v13/threshold_sensitivity_TEXTQA'}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cache', default=CONFIG['cache'])
        parser.add_argument('--config', default=CONFIG['config'])
        parser.add_argument('--out', default=CONFIG['out'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
