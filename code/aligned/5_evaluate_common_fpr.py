#!/usr/bin/env python3
"""Evaluate frozen common-FPR thresholds on the internal test set."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
METHODS = ('S_C', 'S_T+S_V', 'GEOMETRIC', 'LEARNED')
OPERATING_POINTS = ('fpr_05', 'fpr_01')
EPS = 1e-12

def load_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pickle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_test_cache(path) -> tuple[list[dict], list[dict], str]:
    cached = load_pickle(path)
    if 'results_unsafe' not in cached or 'results_safe' not in cached:
        raise KeyError(f'Unrecognized test cache. Available keys: {list(cached.keys())}')
    unsafe = cached['results_unsafe']
    safe = cached['results_safe']
    if len(unsafe) == 0 or len(safe) == 0:
        raise ValueError(f'Test cache is empty or incomplete: unsafe={len(unsafe)}, safe={len(safe)}')
    sc_representation = cached.get('sc_representation')
    if sc_representation not in ('mean', 'aligned'):
        raise ValueError('Test cache is missing valid sc_representation metadata.')
    return (unsafe, safe, sc_representation)

def normalize_rows(rows: list[dict], group_name: str) -> list[dict]:
    out = []
    for pos, r in enumerate(rows):
        missing = [k for k in ('S_T', 'S_V', 'S_C') if k not in r]
        if missing:
            raise KeyError(f"{group_name} sample pos={pos}, idx={r.get('idx')}: missing {missing}")
        st = float(r['S_T'])
        sv = float(r['S_V'])
        sc = float(r['S_C'])
        if not np.all(np.isfinite([st, sv, sc])):
            raise ValueError(f"{group_name} sample pos={pos}, idx={r.get('idx')}: not-finite score")
        out.append({'idx': r.get('idx', pos), 'S_T': st, 'S_V': sv, 'S_C': sc})
    return out

def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = np.asarray([r['S_T'] for r in rows], dtype=np.float64)
    sv = np.asarray([r['S_V'] for r in rows], dtype=np.float64)
    sc = np.asarray([r['S_C'] for r in rows], dtype=np.float64)
    return (st, sv, sc)

def build_method_scores(rows: list[dict], learned_model) -> dict[str, np.ndarray]:
    st, sv, sc = rows_to_arrays(rows)
    X = np.column_stack([st, sv, sc])
    return {'S_C': sc, 'S_T+S_V': np.maximum(st, sv), 'GEOMETRIC': np.maximum.reduce([st, sv, sc]), 'LEARNED': learned_model.predict_proba(X)[:, 1]}

def compute_metrics(scores_unsafe: np.ndarray, scores_safe: np.ndarray, tau: float) -> dict:
    pred_u = scores_unsafe >= tau
    pred_s = scores_safe >= tau
    tp = int(pred_u.sum())
    fn = int((~pred_u).sum())
    fp = int(pred_s.sum())
    tn = int((~pred_s).sum())
    tpr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)
    precision = tp / (tp + fp + EPS)
    f1 = 2.0 * precision * tpr / (precision + tpr + EPS)
    tnr = tn / (tn + fp + EPS)
    balanced_accuracy = 0.5 * (tpr + tnr)
    labels = np.concatenate([np.ones(len(scores_unsafe), dtype=int), np.zeros(len(scores_safe), dtype=int)])
    scores = np.concatenate([scores_unsafe, scores_safe])
    auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    return {'tau': float(tau), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': float(tpr), 'recall': float(tpr), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(balanced_accuracy), 'roc_auc': auc, 'pr_auc': pr_auc}

def likely_rounded_to_4_decimals(rows: list[dict]) -> bool:
    values = np.asarray([[r['S_T'], r['S_V'], r['S_C']] for r in rows], dtype=np.float64)
    return bool(np.allclose(values, np.round(values, 4), atol=1e-12, rtol=0.0))

def build_per_sample_output(rows: list[dict], method_scores: dict[str, np.ndarray], config: dict) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        item = {'idx': r['idx'], 'S_T': r['S_T'], 'S_V': r['S_V'], 'S_C': r['S_C'], 'scores': {}, 'predictions': {}}
        for method in METHODS:
            score = float(method_scores[method][i])
            item['scores'][method] = score
            item['predictions'][method] = {}
            for op_name in OPERATING_POINTS:
                tau = float(config['methods'][method]['operating_points'][op_name]['tau'])
                item['predictions'][method][op_name] = int(score >= tau)
        out.append(item)
    return out

def run(args):
    delta_dir = Path(args.delta)
    config_path = Path(args.config) if args.config else delta_dir / 'common_fpr_config.json'
    if args.test_cache:
        test_cache_path = Path(args.test_cache)
    else:
        test_cache_path = delta_dir / 'eval_results_cache_internal.pkl'
        if not test_cache_path.exists():
            test_cache_path = delta_dir / 'eval_results_cache_v13_internal.pkl'
    learned_path = Path(args.learned_model) if args.learned_model else delta_dir / 'learned_fusion_common_fpr.pkl'
    print('\n' + '=' * 82)
    print('COMMON-FPR EVALUATION — FROZEN THRESHOLDS, TEST ONLY')
    print('=' * 82)
    print(f'config      : {config_path}')
    print(f'test cache  : {test_cache_path}')
    print(f'learned     : {learned_path}')
    print(f'unsafe set  : {args.unsafe_name}')
    print(f'safe set    : {args.safe_name}')
    print(f'suffix      : {args.suffix}')
    config = load_json(config_path)
    if config.get('protocol', {}).get('test_used', None) is not False:
        raise ValueError('common_fpr_config.json does not declare test_used=False. Confirm that it was produced by 5_calibrate_common_fpr.py.')
    for method in METHODS:
        if method not in config.get('methods', {}):
            raise KeyError(f'Method {method} missing from the config.')
        for op_name in OPERATING_POINTS:
            if op_name not in config['methods'][method].get('operating_points', {}):
                raise KeyError(f'{method}: operating point {op_name} missing.')
    learned_obj = load_pickle(learned_path)
    if not isinstance(learned_obj, dict) or 'model' not in learned_obj:
        raise ValueError('learned_fusion_common_fpr.pkl: unrecognized schema.')
    learned_model = learned_obj['model']
    print('\nLEARNED FUSION COEFFICIENTS')
    if hasattr(learned_model, 'named_steps'):
        clf = learned_model.steps[-1][1]
    else:
        clf = learned_model
    print(f'  S_T: {clf.coef_[0][0]:+.6f}')
    print(f'  S_V: {clf.coef_[0][1]:+.6f}')
    print(f'  S_C: {clf.coef_[0][2]:+.6f}')
    print(f'  intercept: {clf.intercept_[0]:+.6f}')
    raw_unsafe, raw_safe, test_sc_representation = load_test_cache(test_cache_path)
    config_sc_representation = config.get('protocol', {}).get('sc_representation')
    learned_sc_representation = learned_obj.get('sc_representation')
    if not (test_sc_representation == config_sc_representation == learned_sc_representation):
        raise ValueError(
            'S_C representation mismatch among test cache, calibration config, and learned fusion: '
            f'{test_sc_representation}, {config_sc_representation}, {learned_sc_representation}'
        )
    unsafe = normalize_rows(raw_unsafe, args.unsafe_name)
    safe = normalize_rows(raw_safe, args.safe_name)
    print(f'\ntest samples : unsafe={len(unsafe)}, safe={len(safe)}')
    rounded_cache = likely_rounded_to_4_decimals(unsafe) and likely_rounded_to_4_decimals(safe)
    if rounded_cache:
        print('[WARN] S_T/S_V/S_C appear rounded to four decimals in the test cache. Evaluation uses the available values, but a full-precision cache is preferable for very small thresholds such as S_C near 1e-3.')
    scores_u = build_method_scores(unsafe, learned_model)
    scores_s = build_method_scores(safe, learned_model)
    results = {}
    print('\n' + '-' * 82)
    print('TEST RESULTS — THRESHOLDS FROZEN DURING CALIBRATION')
    print('-' * 82)
    for method in METHODS:
        results[method] = {'score_definition': config['methods'][method].get('score_definition'), 'operating_points': {}}
        print(f'\n{method}')
        for op_name in OPERATING_POINTS:
            op_cal = config['methods'][method]['operating_points'][op_name]
            tau = float(op_cal['tau'])
            target_fpr = float(op_cal['target_fpr'])
            m = compute_metrics(scores_u[method], scores_s[method], tau)
            results[method]['operating_points'][op_name] = {'target_fpr_from_calibration': target_fpr, 'calibration_fpr': float(op_cal['fpr_cal']), 'calibration_tpr': float(op_cal['tpr_cal']), **m}
            print(f"  {op_name} | tau={tau:.8f} | Recall={m['tpr']:.2%} | FPR_test={m['fpr']:.2%} | F1={m['f1']:.4f} | AUC={m['roc_auc']:.4f}")
    per_sample_unsafe = build_per_sample_output(unsafe, scores_u, config)
    per_sample_safe = build_per_sample_output(safe, scores_s, config)
    output = {'protocol': {'name': 'common_fpr_test_evaluation', 'sc_representation': test_sc_representation, 'thresholds_frozen_from_calibration': True, 'test_used_for_threshold_selection': False, 'quantile_matching': False, 'youden_on_test': False, 'primary_operating_point': 'fpr_05', 'secondary_operating_point': 'fpr_01'}, 'source': {'common_fpr_config': str(config_path), 'test_cache': str(test_cache_path), 'learned_model': str(learned_path), 'unsafe_dataset': args.unsafe_name, 'safe_dataset': args.safe_name, 'n_unsafe': len(unsafe), 'n_safe': len(safe), 'input_scores_likely_rounded_to_4_decimals': rounded_cache}, 'methods': results}
    out_json = delta_dir / f'common_fpr_evaluation_{args.suffix}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    out_pkl = delta_dir / f'common_fpr_test_scores_{args.suffix}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'sc_representation': test_sc_representation, 'results_unsafe': per_sample_unsafe, 'results_safe': per_sample_safe, 'method_scores_unsafe': scores_u, 'method_scores_safe': scores_s, 'source_test_cache': str(test_cache_path), 'source_common_fpr_config': str(config_path), 'unsafe_name': args.unsafe_name, 'safe_name': args.safe_name}, f)
    print('\n' + '=' * 82)
    print('SAVED — NO PREVIOUS FILE WAS MODIFIED')
    print('=' * 82)
    print(f'metrics      : {out_json}')
    print(f'sample cache : {out_pkl}')
if __name__ == '__main__':
    import sys
    CONFIG = {'delta': 'probes/rebuilt_v13/delta', 'config': None, 'test_cache': None, 'learned_model': None, 'unsafe_name': 'MSTS_test', 'safe_name': 'MMStar_test', 'suffix': 'msts_mmstar'}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--delta', default=CONFIG['delta'])
        parser.add_argument('--config', default=CONFIG['config'])
        parser.add_argument('--test_cache', default=CONFIG['test_cache'])
        parser.add_argument('--learned_model', default=CONFIG['learned_model'])
        parser.add_argument('--unsafe_name', default=CONFIG['unsafe_name'])
        parser.add_argument('--safe_name', default=CONFIG['safe_name'])
        parser.add_argument('--suffix', default=CONFIG['suffix'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
