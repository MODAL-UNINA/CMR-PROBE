#!/usr/bin/env python3
"""Evaluate frozen common-FPR thresholds externally."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
METHODS = ('S_C', 'S_T+S_V', 'GEOMETRIC', 'LEARNED')

def load_test_cache(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, 'rb') as f:
        cached = pickle.load(f)
    if 'mmsafety' not in cached or 'textvqa' not in cached:
        raise KeyError(f'Unrecognized test-cache schema: {list(cached.keys())}')
    st_u, sv_u, sc_u = cached['mmsafety']
    st_s, sv_s, sc_s = cached['textvqa']
    unsafe = {'T': np.asarray(st_u, dtype=np.float64).reshape(-1), 'V': np.asarray(sv_u, dtype=np.float64).reshape(-1), 'C': np.asarray(sc_u, dtype=np.float64).reshape(-1)}
    safe = {'T': np.asarray(st_s, dtype=np.float64).reshape(-1), 'V': np.asarray(sv_s, dtype=np.float64).reshape(-1), 'C': np.asarray(sc_s, dtype=np.float64).reshape(-1)}
    rows_u = cached.get('rows_mmsafety', [])
    rows_s = cached.get('rows_textvqa', [])
    sc_representation = cached.get('sc_representation')
    if sc_representation not in ('mean', 'aligned'):
        raise ValueError('Secondary test cache is missing valid sc_representation metadata.')
    return (unsafe, safe, rows_u, rows_s, sc_representation)

def build_method_scores(scores, learned_model):
    st = scores['T']
    sv = scores['V']
    sc = scores['C']
    X = np.column_stack([st, sv, sc])
    return {'S_C': sc, 'S_T+S_V': np.maximum(st, sv), 'GEOMETRIC': np.maximum.reduce([st, sv, sc]), 'LEARNED': learned_model.predict_proba(X)[:, 1]}

def compute_metrics(scores_unsafe, scores_safe, tau):
    su = np.asarray(scores_unsafe, dtype=np.float64)
    ss = np.asarray(scores_safe, dtype=np.float64)
    pu = su >= tau
    ps = ss >= tau
    tp = int(pu.sum())
    fn = int((~pu).sum())
    fp = int(ps.sum())
    tn = int((~ps).sum())
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    bacc = 0.5 * (recall + tn / max(tn + fp, 1))
    labels = np.concatenate([np.ones(len(su), dtype=int), np.zeros(len(ss), dtype=int)])
    scores = np.concatenate([su, ss])
    auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    return {'tau': float(tau), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'recall': float(recall), 'tpr': float(recall), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(bacc), 'roc_auc': auc, 'pr_auc': pr_auc}

def run(args):
    delta_dir = Path(args.delta)
    work_dir = delta_dir / 'mmsafety_textvqa_common_fpr'
    cfg_path = Path(args.config) if args.config else work_dir / 'common_fpr_config_mmsafety_textvqa.json'
    test_path = Path(args.test_cache) if args.test_cache else work_dir / 'scores_test_mmsafety_textvqa.pkl'
    learned_path = Path(args.learned_model) if args.learned_model else work_dir / 'learned_fusion_common_fpr_mmsafety_textvqa.pkl'
    for p in (cfg_path, test_path, learned_path):
        if not p.exists():
            raise FileNotFoundError(p)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if cfg['protocol'].get('test_used', True):
        raise RuntimeError('The configuration declares test_used=True and is not a valid calibration.')
    with open(learned_path, 'rb') as f:
        learned_obj = pickle.load(f)
    learned_model = learned_obj['model']
    unsafe, safe, rows_u, rows_s, test_sc_representation = load_test_cache(test_path)
    config_sc_representation = cfg.get('protocol', {}).get('sc_representation')
    learned_sc_representation = learned_obj.get('sc_representation')
    if not (test_sc_representation == config_sc_representation == learned_sc_representation):
        raise ValueError(
            'S_C representation mismatch among secondary test cache, calibration config, and learned fusion: '
            f'{test_sc_representation}, {config_sc_representation}, {learned_sc_representation}'
        )
    n_u = len(unsafe['C'])
    n_s = len(safe['C'])
    print('\n' + '=' * 78)
    print('COMMON-FPR TEST — MM-SAFETYBENCH / TEXTVQA')
    print('=' * 78)
    print(f'unsafe test : {n_u}')
    print(f'safe test   : {n_s}')
    print(f'config      : {cfg_path}')
    print(f'test cache  : {test_path}')
    print('thresholds  : frozen during calibration')
    print('test labels : not used to select tau')
    scores_u = build_method_scores(unsafe, learned_model)
    scores_s = build_method_scores(safe, learned_model)
    output_methods = {}
    paired_cache = {}
    print('\n' + '-' * 78)
    print('COMMON-FPR RESULTS')
    print('-' * 78)
    for method in METHODS:
        method_cfg = cfg['methods'][method]
        output_methods[method] = {'operating_points': {}}
        paired_cache[method] = {'scores_unsafe': scores_u[method], 'scores_safe': scores_s[method], 'operating_points': {}}
        print(f'\n{method}')
        for op_name in ('fpr_05', 'fpr_01'):
            op_cfg = method_cfg['operating_points'][op_name]
            tau = float(op_cfg['tau'])
            m = compute_metrics(scores_u[method], scores_s[method], tau)
            output_methods[method]['operating_points'][op_name] = {'target_fpr': float(op_cfg['target_fpr']), 'tau': tau, 'calibration_fpr': float(op_cfg['fpr_cal']), 'calibration_tpr': float(op_cfg['tpr_cal']), 'test': m}
            paired_cache[method]['operating_points'][op_name] = {'tau': tau, 'pred_unsafe': (scores_u[method] >= tau).astype(np.int8), 'pred_safe': (scores_s[method] >= tau).astype(np.int8)}
            print(f"  {op_name}: tau={tau:.8f} | Recall={m['recall']:.2%} | FPR_test={m['fpr']:.2%} | F1={m['f1']:.4f} | AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f}")
    output = {'benchmark': 'MM-SafetyBench SD / TextVQA', 'protocol': {'common_fpr': True, 'sc_representation': test_sc_representation, 'threshold_selected_on_test': False, 'primary_target_fpr': 0.05, 'secondary_target_fpr': 0.01, 'quantile_matching': False, 'test_adaptation': False}, 'n_samples': {'unsafe': n_u, 'safe': n_s}, 'methods': output_methods}
    result_path = work_dir / 'common_fpr_evaluation_mmsafety_textvqa.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    idx_u = [r['idx'] for r in rows_u] if rows_u else list(range(n_u))
    idx_s = [r['idx'] for r in rows_s] if rows_s else list(range(n_s))
    paired_path = work_dir / 'common_fpr_test_scores_mmsafety_textvqa.pkl'
    with open(paired_path, 'wb') as f:
        pickle.dump({'sc_representation': test_sc_representation, 'idx_unsafe': idx_u, 'idx_safe': idx_s, 'methods': paired_cache, 'source_test_cache': str(test_path), 'source_config': str(cfg_path)}, f)
    print('\n' + '=' * 78)
    print('SAVED')
    print('=' * 78)
    print(f'metrics : {result_path}')
    print(f'paired  : {paired_path}')
if __name__ == '__main__':
    CONFIG = {'delta': 'probes/rebuilt_v13/delta', 'config': None, 'test_cache': None, 'learned_model': None}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--delta', default=CONFIG['delta'])
        parser.add_argument('--config', default=CONFIG['config'])
        parser.add_argument('--test_cache', default=CONFIG['test_cache'])
        parser.add_argument('--learned_model', default=CONFIG['learned_model'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
