#!/usr/bin/env python3
"""Calibrate common-FPR thresholds for the external benchmark."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
PRIMARY_FPR = 0.05
SECONDARY_FPR = 0.01
METHODS = ('S_C', 'S_T+S_V', 'GEOMETRIC', 'LEARNED')

def _as_1d(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        raise ValueError(f'{name}: empty array')
    if not np.all(np.isfinite(x)):
        bad = int((~np.isfinite(x)).sum())
        raise ValueError(f'{name}: {bad} not-finite values')
    return x

def load_secondary_calibration_cache(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Cache calibration secondary not found: {path}')
    with open(path, 'rb') as f:
        cached = pickle.load(f)
    if 'mmsafety' not in cached or 'textvqa' not in cached:
        raise KeyError(f'Unrecognized cache schema. Available keys: {list(cached.keys())}')
    if len(cached['mmsafety']) != 3:
        raise ValueError("cached['mmsafety'] must be (S_T, S_V, S_C)")
    if len(cached['textvqa']) != 3:
        raise ValueError("cached['textvqa'] must be (S_T, S_V, S_C)")
    st_u, sv_u, sc_u = cached['mmsafety']
    st_s, sv_s, sc_s = cached['textvqa']
    sc_representation = cached.get('sc_representation')
    if sc_representation not in ('mean', 'aligned'):
        raise ValueError('Secondary calibration cache is missing valid sc_representation metadata.')
    out = {'sc_representation': sc_representation, 'unsafe': {'T': _as_1d(st_u, 'MM-Safety S_T'), 'V': _as_1d(sv_u, 'MM-Safety S_V'), 'C': _as_1d(sc_u, 'MM-Safety S_C')}, 'safe': {'T': _as_1d(st_s, 'TextVQA S_T'), 'V': _as_1d(sv_s, 'TextVQA S_V'), 'C': _as_1d(sc_s, 'TextVQA S_C')}, 'raw_cache': cached}
    for group in ('unsafe', 'safe'):
        lens = {k: len(v) for k, v in out[group].items()}
        if len(set(lens.values())) != 1:
            raise ValueError(f'Inconsistent lengths in {group}: {lens}')
    return out

def stratified_fit_operating_split(n_unsafe: int, n_safe: int, fit_fraction: float, seed: int):
    if not 0.1 <= fit_fraction <= 0.9:
        raise ValueError('fusion_fit_fraction must be between 0.1 and 0.9')
    rng = np.random.default_rng(seed)
    iu = rng.permutation(n_unsafe)
    is_ = rng.permutation(n_safe)
    nu_fit = max(1, min(n_unsafe - 1, int(round(n_unsafe * fit_fraction))))
    ns_fit = max(1, min(n_safe - 1, int(round(n_safe * fit_fraction))))
    return {'unsafe_fit': iu[:nu_fit], 'unsafe_operating': iu[nu_fit:], 'safe_fit': is_[:ns_fit], 'safe_operating': is_[ns_fit:]}

def stack_features(scores, idx):
    return np.column_stack([scores['T'][idx], scores['V'][idx], scores['C'][idx]])

def fit_learned_fusion(unsafe, safe, split, seed: int):
    X_u = stack_features(unsafe, split['unsafe_fit'])
    X_s = stack_features(safe, split['safe_fit'])
    X = np.vstack([X_u, X_s])
    y = np.concatenate([np.ones(len(X_u), dtype=int), np.zeros(len(X_s), dtype=int)])
    model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=seed))])
    model.fit(X, y)
    return model

def build_method_scores(scores, idx, learned_model):
    st = scores['T'][idx]
    sv = scores['V'][idx]
    sc = scores['C'][idx]
    X = np.column_stack([st, sv, sc])
    return {'S_C': sc, 'S_T+S_V': np.maximum(st, sv), 'GEOMETRIC': np.maximum.reduce([st, sv, sc]), 'LEARNED': learned_model.predict_proba(X)[:, 1]}

def threshold_at_fpr(scores_safe, target_fpr: float):
    s = _as_1d(scores_safe, 'scores_safe')
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError('target_fpr must be in [0,1)')
    candidates = np.unique(s)
    candidates = np.concatenate([candidates, [np.nextafter(float(s.max()), np.inf)]])
    best_tau = float(candidates[-1])
    best_fpr = float(np.mean(s >= best_tau))
    for tau in candidates:
        fpr = float(np.mean(s >= tau))
        if fpr <= target_fpr + 1e-15:
            best_tau = float(tau)
            best_fpr = fpr
            break
    n_fp = int(np.sum(s >= best_tau))
    return (best_tau, best_fpr, n_fp)

def operating_metrics(scores_unsafe, scores_safe, tau: float):
    pred_u = scores_unsafe >= tau
    pred_s = scores_safe >= tau
    tpr = float(pred_u.mean())
    fpr = float(pred_s.mean())
    labels = np.concatenate([np.ones(len(scores_unsafe)), np.zeros(len(scores_safe))])
    scores = np.concatenate([scores_unsafe, scores_safe])
    auc = float(roc_auc_score(labels, scores))
    return {'tpr_cal': tpr, 'fpr_cal': fpr, 'auc_cal': auc, 'tp_cal': int(pred_u.sum()), 'fn_cal': int((~pred_u).sum()), 'fp_cal': int(pred_s.sum()), 'tn_cal': int((~pred_s).sum())}

def calibrate_one_method(name, score_u, score_s, targets):
    score_definition = {'S_C': 'S_C', 'S_T+S_V': 'max(S_T, S_V)', 'GEOMETRIC': 'max(S_T, S_V, S_C)', 'LEARNED': 'LogisticRegression(S_T, S_V, S_C)'}[name]
    result = {'score_definition': score_definition, 'n_unsafe_operating_cal': int(len(score_u)), 'n_safe_operating_cal': int(len(score_s)), 'operating_points': {}}
    for target in targets:
        tau, empirical_fpr, n_fp = threshold_at_fpr(score_s, target)
        metrics = operating_metrics(score_u, score_s, tau)
        key = f'fpr_{int(round(target * 100)):02d}'
        result['operating_points'][key] = {'target_fpr': float(target), 'tau': float(tau), 'empirical_fpr_from_safe_cal': float(empirical_fpr), 'n_fp_safe_cal': int(n_fp), **metrics}
    return result

def run(args):
    delta_dir = Path(args.delta)
    work_dir = delta_dir / 'mmsafety_textvqa_common_fpr'
    work_dir.mkdir(parents=True, exist_ok=True)
    cal_cache = Path(args.cal_cache) if args.cal_cache else work_dir / 'scores_cal_mmsafety_textvqa.pkl'
    split_path = work_dir / 'eval_splits_mmsafety_textvqa.json'
    if not split_path.exists():
        raise FileNotFoundError(f'Split secondary not found: {split_path}')
    print('\n' + '=' * 78)
    print('COMMON-FPR CALIBRATION — MM-SAFETYBENCH / TEXTVQA')
    print('=' * 78)
    print(f'cache              : {cal_cache}')
    print(f'primary FPR        : {PRIMARY_FPR:.1%}')
    print(f'secondary FPR      : {SECONDARY_FPR:.1%}')
    print(f'fusion fit fraction: {args.fusion_fit_fraction:.1%}')
    print(f'split seed         : {args.split_seed}')
    data = load_secondary_calibration_cache(cal_cache)
    sc_representation = data['sc_representation']
    unsafe = data['unsafe']
    safe = data['safe']
    n_u = len(unsafe['C'])
    n_s = len(safe['C'])
    print(f'\ncalibration pool   : unsafe={n_u}, safe={n_s}')
    split = stratified_fit_operating_split(n_unsafe=n_u, n_safe=n_s, fit_fraction=args.fusion_fit_fraction, seed=args.split_seed)
    print(f"split              : fit unsafe={len(split['unsafe_fit'])}, safe={len(split['safe_fit'])} | operating unsafe={len(split['unsafe_operating'])}, safe={len(split['safe_operating'])}")
    if len(split['safe_operating']) < 100:
        print('[WARN] fewer than 100 benign samples in operating_cal: the 1% FPR point has coarse resolution.')
    learned = fit_learned_fusion(unsafe, safe, split, args.split_seed)
    model_path = work_dir / 'learned_fusion_common_fpr_mmsafety_textvqa.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': learned, 'features': ['S_T', 'S_V', 'S_C'], 'sc_representation': sc_representation, 'fit_fraction': args.fusion_fit_fraction, 'split_seed': args.split_seed, 'source_calibration_cache': str(cal_cache)}, f)
    scores_u = build_method_scores(unsafe, split['unsafe_operating'], learned)
    scores_s = build_method_scores(safe, split['safe_operating'], learned)
    targets = (PRIMARY_FPR, SECONDARY_FPR)
    methods = {}
    print('\n' + '-' * 78)
    print('OPERATING POINTS (ONLY operating_cal secondary)')
    print('-' * 78)
    for name in METHODS:
        methods[name] = calibrate_one_method(name, scores_u[name], scores_s[name], targets)
        print(f'\n{name}')
        for key, op in methods[name]['operating_points'].items():
            print(f"  {key}: tau={op['tau']:.8f} | FPR_cal={op['fpr_cal']:.2%} ({op['fp_cal']}/{len(scores_s[name])}) | TPR_cal={op['tpr_cal']:.2%} | AUC_cal={op['auc_cal']:.4f}")
    with open(split_path, 'r', encoding='utf-8') as f:
        dataset_splits = json.load(f)
    config = {'protocol': {'name': 'common_fpr_calibration_mmsafety_textvqa', 'sc_representation': sc_representation, 'test_used': False, 'primary_target_fpr': PRIMARY_FPR, 'secondary_target_fpr': SECONDARY_FPR, 'threshold_rule': 'smallest empirical threshold with safe-cal FPR <= target; unsafe iff score >= tau', 'learned_fusion_fit': 'separate fusion_fit subset of calibration only', 'threshold_calibration': 'shared operating_cal subset for all methods', 'quantile_matching': False, 'youden_for_final_operating_point': False}, 'source': {'calibration_cache': str(cal_cache), 'split_file': str(split_path), 'unsafe_dataset': 'MM-SafetyBench_SD_cal', 'safe_dataset': 'TextVQA_cal', 'n_unsafe_total': int(n_u), 'n_safe_total': int(n_s), 'source_seed': int(args.source_seed)}, 'split': {'split_seed': int(args.split_seed), 'fusion_fit_fraction': float(args.fusion_fit_fraction), 'unsafe_fit_positions': split['unsafe_fit'].tolist(), 'unsafe_operating_positions': split['unsafe_operating'].tolist(), 'safe_fit_positions': split['safe_fit'].tolist(), 'safe_operating_positions': split['safe_operating'].tolist(), 'mmsafety_cal_indices': dataset_splits['mmsafety_cal'], 'textvqa_cal_indices': dataset_splits['textvqa_cal'], 'mmsafety_test_indices': dataset_splits['mmsafety_test'], 'textvqa_test_indices': dataset_splits['textvqa_test']}, 'methods': methods, 'learned_model_path': str(model_path)}
    cfg_path = work_dir / 'common_fpr_config_mmsafety_textvqa.json'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    score_cache_path = work_dir / 'common_fpr_calibration_scores_mmsafety_textvqa.pkl'
    with open(score_cache_path, 'wb') as f:
        pickle.dump({'sc_representation': sc_representation, 'scores_unsafe_operating': scores_u, 'scores_safe_operating': scores_s, 'split': split, 'source_calibration_cache': str(cal_cache), 'split_file': str(split_path)}, f)
    print('\n' + '=' * 78)
    print('SAVED')
    print('=' * 78)
    print(f'config        : {cfg_path}')
    print(f'learned model : {model_path}')
    print(f'score cache   : {score_cache_path}')
    print('\nNo test score or test label was loaded or used.')
if __name__ == '__main__':
    CONFIG = {'delta': 'probes/rebuilt_v13/delta', 'cal_cache': None, 'source_seed': 42, 'split_seed': 2026, 'fusion_fit_fraction': 0.5}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--delta', default=CONFIG['delta'])
        parser.add_argument('--cal_cache', default=CONFIG['cal_cache'])
        parser.add_argument('--source_seed', type=int, default=CONFIG['source_seed'])
        parser.add_argument('--split_seed', type=int, default=CONFIG['split_seed'])
        parser.add_argument('--fusion_fit_fraction', type=float, default=CONFIG['fusion_fit_fraction'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
