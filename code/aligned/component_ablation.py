#!/usr/bin/env python3
"""Measure the contribution of detector components."""
from __future__ import annotations
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
SEED = 42
N_BOOT = 10000
COMPONENTS = {'T': ('T',), 'V': ('V',), 'C': ('C',), 'T+V': ('T', 'V'), 'T+C': ('T', 'C'), 'V+C': ('V', 'C'), 'T+V+C': ('T', 'V', 'C')}

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
    return {'T': tau_T, 'V': tau_V, 'C': tau_C}

def compute_delta(row: dict, active_components: tuple[str, ...], thresholds: dict) -> float:
    margins = []
    for component in active_components:
        score = float(row[f'S_{component}'])
        tau = thresholds[component]
        margins.append(score - tau)
    return float(max(margins))

def evaluate_configuration(results_unsafe: list, results_safe: list, active_components: tuple[str, ...], thresholds: dict) -> dict:
    delta_unsafe = np.array([compute_delta(r, active_components, thresholds) for r in results_unsafe])
    delta_safe = np.array([compute_delta(r, active_components, thresholds) for r in results_safe])
    pred_unsafe = (delta_unsafe >= 0).astype(int)
    pred_safe = (delta_safe >= 0).astype(int)
    tp = int(pred_unsafe.sum())
    fn = int((pred_unsafe == 0).sum())
    fp = int(pred_safe.sum())
    tn = int((pred_safe == 0).sum())
    tpr = tp / (tp + fn + 1e-09)
    fpr = fp / (fp + tn + 1e-09)
    precision = tp / (tp + fp + 1e-09)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-09)
    tnr = tn / (tn + fp + 1e-09)
    balanced_acc = (tpr + tnr) / 2
    labels = np.concatenate([np.ones(len(delta_unsafe)), np.zeros(len(delta_safe))])
    scores = np.concatenate([delta_unsafe, delta_safe])
    auc = roc_auc_score(labels, scores)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': float(tpr), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(balanced_acc), 'auc': float(auc), 'delta_unsafe': delta_unsafe, 'delta_safe': delta_safe}

def bootstrap_metrics(delta_unsafe: np.ndarray, delta_safe: np.ndarray, n_boot: int=N_BOOT, seed: int=SEED, ci: float=95.0):
    rng = np.random.default_rng(seed)
    n_u = len(delta_unsafe)
    n_s = len(delta_safe)
    metric_names = ['tpr', 'fpr', 'precision', 'f1', 'balanced_accuracy', 'auc']
    samples = {k: np.empty(n_boot) for k in metric_names}
    for b in range(n_boot):
        idx_u = rng.integers(0, n_u, n_u)
        idx_s = rng.integers(0, n_s, n_s)
        du = delta_unsafe[idx_u]
        ds = delta_safe[idx_s]
        pu = (du >= 0).astype(int)
        ps = (ds >= 0).astype(int)
        tp = pu.sum()
        fn = (pu == 0).sum()
        fp = ps.sum()
        tn = (ps == 0).sum()
        tpr = tp / (tp + fn + 1e-09)
        fpr = fp / (fp + tn + 1e-09)
        precision = tp / (tp + fp + 1e-09)
        f1 = 2 * precision * tpr / (precision + tpr + 1e-09)
        tnr = tn / (tn + fp + 1e-09)
        balanced_acc = (tpr + tnr) / 2
        labels = np.concatenate([np.ones(len(du)), np.zeros(len(ds))])
        scores = np.concatenate([du, ds])
        auc = roc_auc_score(labels, scores)
        samples['tpr'][b] = tpr
        samples['fpr'][b] = fpr
        samples['precision'][b] = precision
        samples['f1'][b] = f1
        samples['balanced_accuracy'][b] = balanced_acc
        samples['auc'][b] = auc
    alpha = (100 - ci) / 2
    out = {}
    for metric, values in samples.items():
        lo, hi = np.percentile(values, [alpha, 100 - alpha])
        out[metric] = {'ci_lower': float(lo), 'ci_upper': float(hi), 'std': float(values.std())}
    return out

def plot_results(results: dict, out_dir: Path):
    names = list(results.keys())
    f1 = [results[name]['f1'] for name in names]
    tpr = [results[name]['tpr'] for name in names]
    fpr = [results[name]['fpr'] for name in names]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, tpr, width, label='TPR')
    ax.bar(x, f1, width, label='F1')
    ax.bar(x + width, fpr, width, label='FPR')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_xlabel('Active components')
    ax.set_title('Component contribution analysis')
    ax.legend()
    plt.tight_layout()
    path = out_dir / 'component_ablation_plot.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved: {path}')

def save_csv(results: dict, path: Path):
    fields = ['configuration', 'components', 'tpr', 'fpr', 'precision', 'f1', 'balanced_accuracy', 'auc', 'tp', 'fp', 'fn', 'tn']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, r in results.items():
            writer.writerow({'configuration': name, 'components': '+'.join(r['components']), 'tpr': r['tpr'], 'fpr': r['fpr'], 'precision': r['precision'], 'f1': r['f1'], 'balanced_accuracy': r['balanced_accuracy'], 'auc': r['auc'], 'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn']})

def run(args):
    cache_path = Path(args.cache)
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(config_path)
    print('\n[1] Loading cache...')
    results_unsafe, results_safe = load_cache(cache_path)
    print('\n[SCORE DIAGNOSTICS]')
    for comp in ['T', 'V', 'C']:
        unsafe = np.array([r[f'S_{comp}'] for r in results_unsafe])
        safe = np.array([r[f'S_{comp}'] for r in results_safe])
        tau = thresholds[comp]
        print(f'\nS_{comp}  tau={tau:.4f}')
        print(f'  unsafe: min={unsafe.min():.4f}  mean={unsafe.mean():.4f}  median={np.median(unsafe):.4f}  max={unsafe.max():.4f}')
        print(f'  safe:   min={safe.min():.4f}  mean={safe.mean():.4f}  median={np.median(safe):.4f}  max={safe.max():.4f}')
        print(f'  unsafe >= tau: {(unsafe >= tau).mean():.2%}')
        print(f'  safe >= tau:   {(safe >= tau).mean():.2%}')
    print(f'  unsafe = {len(results_unsafe)}')
    print(f'  safe   = {len(results_safe)}')
    print('\n[2] Loading thresholds...')
    thresholds = load_thresholds(config_path)
    print(f"  tau_T = {thresholds['T']:.4f}")
    print(f"  tau_V = {thresholds['V']:.4f}")
    print(f"  tau_C = {thresholds['C']:.4f}")
    print('\n[3] Component contribution analysis...')
    all_results = {}
    for name, components in COMPONENTS.items():
        print(f'\n  [{name}] components={components}')
        r = evaluate_configuration(results_unsafe, results_safe, components, thresholds)
        ci = bootstrap_metrics(r['delta_unsafe'], r['delta_safe'], n_boot=args.n_boot, seed=args.seed)
        all_results[name] = {'components': list(components), 'tp': r['tp'], 'fp': r['fp'], 'fn': r['fn'], 'tn': r['tn'], 'tpr': round(r['tpr'], 4), 'fpr': round(r['fpr'], 4), 'precision': round(r['precision'], 4), 'f1': round(r['f1'], 4), 'balanced_accuracy': round(r['balanced_accuracy'], 4), 'auc': round(r['auc'], 4), 'bootstrap': {metric: {k: round(v, 4) for k, v in values.items()} for metric, values in ci.items()}}
        print(f"    TPR={r['tpr']:.4f}  FPR={r['fpr']:.4f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")
    output = {'cache': str(cache_path), 'config': str(config_path), 'thresholds': {'tau_T': thresholds['T'], 'tau_V': thresholds['V'], 'tau_C': thresholds['C']}, 'n_samples': {'unsafe': len(results_unsafe), 'safe': len(results_safe)}, 'n_boot': args.n_boot, 'seed': args.seed, 'results': all_results}
    json_path = out_dir / 'component_ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    csv_path = out_dir / 'component_ablation_results.csv'
    save_csv(all_results, csv_path)
    plot_results(all_results, out_dir)
    print(f"\n{'=' * 78}")
    print('COMPONENT CONTRIBUTION ANALYSIS')
    print(f"{'=' * 78}")
    print(f"{'Config':10s} {'TPR':>8s} {'FPR':>8s} {'Prec':>8s} {'F1':>8s} {'BAcc':>8s} {'AUC':>8s}")
    print('-' * 78)
    for name, r in all_results.items():
        print(f"{name:10s} {r['tpr']:8.4f} {r['fpr']:8.4f} {r['precision']:8.4f} {r['f1']:8.4f} {r['balanced_accuracy']:8.4f} {r['auc']:8.4f}")
    print(f"{'=' * 78}")
    print('\nSaved:')
    print(f'  {json_path}')
    print(f'  {csv_path}')
    print(f"  {out_dir / 'component_ablation_plot.png'}")
if __name__ == '__main__':
    import sys
    CONFIG = {'cache': 'probes/rebuilt_v13/delta/eval_results_textvqa_internal.pkl', 'config': 'probes/rebuilt_v13/delta/delta_config.json', 'out': 'probes/rebuilt_v13/component_ablation_mmsafety', 'n_boot': 10000, 'seed': 42}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cache', default=CONFIG['cache'])
        parser.add_argument('--config', default=CONFIG['config'])
        parser.add_argument('--out', default=CONFIG['out'])
        parser.add_argument('--n_boot', type=int, default=CONFIG['n_boot'])
        parser.add_argument('--seed', type=int, default=CONFIG['seed'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
