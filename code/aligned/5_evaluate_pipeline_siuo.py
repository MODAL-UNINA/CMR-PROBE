#!/usr/bin/env python3
"""Evaluate zero-shot and calibrated transfer on SIUO."""
from __future__ import annotations
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve
from utils import load_model, extract_llama_hidden_solo, extract_sc_feature_pair, extract_clip_hidden, validate_combo_representation, SC_REPRESENTATION_CHOICES
LLAMA_LAYER = 17
DATASET_REPO = 'oneonlee/SIUO'
DATASET_CONFIG = 'siuo_gen'
DATASET_SPLIT = 'test'
EPS = 1e-09

def apply_projector(X: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    if Q is None:
        return X
    return X - X @ Q @ Q.T

def unwrap_probe(obj):
    if isinstance(obj, dict):
        clf = obj.get('clf')
        if clf is None:
            raise KeyError("Probe dictionary without key 'clf'.")
        return (clf, obj.get('ortho_Q'))
    return (obj, None)

def find_combo_probe(probes_dir: Path, sc_representation: str):
    candidates = ['1_linear_delta_h_orthogonalized', '1_linear_delta_h', '2_svm_rbf_delta_h', '3_mlp_pair']
    for name in candidates:
        path = probes_dir / 'combo' / f'probe_combo_{name}.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            validate_combo_representation(obj, sc_representation)
            clf, Q = unwrap_probe(obj)
            return (path, clf, Q)
    raise FileNotFoundError(f"No probe_combo found in {probes_dir / 'combo'}")

def to_pil(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert('RGB')
    if isinstance(image_obj, dict):
        if image_obj.get('bytes') is not None:
            from io import BytesIO
            return Image.open(BytesIO(image_obj['bytes'])).convert('RGB')
        if image_obj.get('path'):
            return Image.open(image_obj['path']).convert('RGB')
    raise TypeError(f'Unsupported image format: {type(image_obj)}')

def compute_scores(text: str, image: Image.Image, model, processor, clf_text, Q_text, clf_visual, Q_visual, clf_combo, Q_combo, clip_layer: int, sc_representation: str='aligned'):
    h_text = extract_llama_hidden_solo(text, model, processor, LLAMA_LAYER)
    ht = apply_projector(h_text.reshape(1, -1), Q_text)
    s_t = float(clf_text.predict_proba(ht)[0, 1])
    h_visual = extract_clip_hidden(image, model, processor, clip_layer)
    hv = apply_projector(h_visual.reshape(1, -1), Q_visual)
    s_v = float(clf_visual.predict_proba(hv)[0, 1])
    _, _, delta_h = extract_sc_feature_pair(
        text, image, model, processor, LLAMA_LAYER, sc_representation
    )
    dh = apply_projector(delta_h.reshape(1, -1), Q_combo)
    s_c = float(clf_combo.predict_proba(dh)[0, 1])
    return (s_t, s_v, s_c)

def verdict_from_delta(delta: float, d_cautious: float, d_restricted: float) -> str:
    if delta < 0:
        return 'SAFE'
    if delta < d_cautious:
        return 'CAUTIOUS'
    if delta < d_restricted:
        return 'RESTRICTED'
    return 'REFUSED'

def materialize_stream(dataset, max_samples=None):
    rows = []
    for i, row in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        row = dict(row)
        try:
            img = to_pil(row['image'])
            img.load()
            row['image'] = img.copy()
        except Exception as e:
            print(f'  [warn] materialize SIUO idx={i}: {e}')
            continue
        rows.append(row)
    return rows

def evaluate_siuo(rows, model, processor, clf_text, Q_text, clf_visual, Q_visual, clf_combo, Q_combo, clip_layer, tau_T, tau_V, tau_C, d_cautious, d_restricted, sc_representation='aligned'):
    results = []
    for i, row in enumerate(tqdm(rows, desc='  SIUO-original')):
        qid = row.get('question_id', i)
        text = (row.get('question') or '').strip()
        category = str(row.get('category', 'unknown'))
        if not text:
            print(f'  [warn] SIUO id={qid}: text empty')
            continue
        try:
            image = to_pil(row['image'])
            s_t, s_v, s_c = compute_scores(text, image, model, processor, clf_text, Q_text, clf_visual, Q_visual, clf_combo, Q_combo, clip_layer, sc_representation)
        except Exception as e:
            print(f'  [warn] SIUO id={qid}: {e}')
            continue
        trig_t = s_t >= tau_T
        trig_v = s_v >= tau_V
        trig_c = s_c >= tau_C
        delta = max(s_t - tau_T, s_v - tau_V, s_c - tau_C)
        results.append({'idx': int(i), 'question_id': int(qid) if isinstance(qid, (int, np.integer)) else str(qid), 'category': category, 'S_T': float(s_t), 'S_V': float(s_v), 'S_C': float(s_c), 'delta': float(delta), 'pred': int(delta >= 0), 'label': 1, 'verdict': verdict_from_delta(delta, d_cautious, d_restricted), 'trigger_T': bool(trig_t), 'trigger_V': bool(trig_v), 'trigger_C': bool(trig_c), 'trigger': ('T' if trig_t else '_') + ('V' if trig_v else '_') + ('C' if trig_c else '_')})
        if len(results) % 20 == 0:
            print(f"    valid={len(results)}  S_C={np.mean([r['trigger_C'] for r in results]):.1%}  Full={np.mean([r['pred'] for r in results]):.1%}")
    return results

def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {'n': 0}
    st = np.asarray([r['S_T'] for r in rows], dtype=float)
    sv = np.asarray([r['S_V'] for r in rows], dtype=float)
    sc = np.asarray([r['S_C'] for r in rows], dtype=float)
    pred = np.asarray([r['pred'] for r in rows], dtype=int)
    t = np.asarray([r['trigger_T'] for r in rows], dtype=bool)
    v = np.asarray([r['trigger_V'] for r in rows], dtype=bool)
    c = np.asarray([r['trigger_C'] for r in rows], dtype=bool)
    silent = ~t & ~v
    cond_c = float(c[silent].mean()) if silent.any() else float('nan')
    return {'n': len(rows), 'sc_tpr': round(float(c.mean()), 4), 'full_tpr': round(float(pred.mean()), 4), 'text_trigger_rate': round(float(t.mean()), 4), 'visual_trigger_rate': round(float(v.mean()), 4), 'c_only_rate': round(float((~t & ~v & c).mean()), 4), 'all_unimodal_safe_rate': round(float(silent.mean()), 4), 'conditional_sc_tpr_given_tv_silent': round(cond_c, 4) if np.isfinite(cond_c) else None, 'mean_S_T': round(float(st.mean()), 4), 'mean_S_V': round(float(sv.mean()), 4), 'mean_S_C': round(float(sc.mean()), 4)}

def compute_siuo_summary(results: list[dict], thresholds: dict) -> dict:
    if not results:
        raise RuntimeError('No sample SIUO valid.')
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r['category']].append(r)
    verdicts = Counter((r['verdict'] for r in results))
    return {'dataset': {'repo': DATASET_REPO, 'config': DATASET_CONFIG, 'split': DATASET_SPLIT, 'n_valid': len(results)}, 'thresholds_fixed_v13': thresholds, 'overall': summarize_group(results), 'verdict_distribution': {k: {'n': int(v), 'pct': round(100 * v / len(results), 2)} for k, v in sorted(verdicts.items())}, 'per_category': {cat: summarize_group(x) for cat, x in sorted(by_cat.items())}}

def _fixed_arrays(results: list[dict], tau_T: float, tau_V: float, tau_C: float):
    st = np.asarray([r['S_T'] for r in results], dtype=float)
    sv = np.asarray([r['S_V'] for r in results], dtype=float)
    sc = np.asarray([r['S_C'] for r in results], dtype=float)
    if not (np.all(np.isfinite(st)) and np.all(np.isfinite(sv)) and np.all(np.isfinite(sc))):
        raise ValueError('Not-finite scores in zero-shot evaluation.')
    delta = np.maximum.reduce([st - tau_T, sv - tau_V, sc - tau_C])
    pred_full = (delta >= 0).astype(int)
    pred_sc = (sc >= tau_C).astype(int)
    return (st, sv, sc, delta, pred_sc, pred_full)

def _verdict_distribution_from_delta(delta: np.ndarray, d_cautious: float, d_restricted: float) -> dict:
    verdicts = [verdict_from_delta(float(x), d_cautious, d_restricted) for x in delta]
    counts = Counter(verdicts)
    n = len(verdicts)
    order = ['SAFE', 'CAUTIOUS', 'RESTRICTED', 'REFUSED']
    return {k: {'n': int(counts.get(k, 0)), 'pct': round(100.0 * counts.get(k, 0) / n, 2) if n else 0.0} for k in order}

def compute_formal_zeroshot(siuo_results: list[dict], mmstar_test_results: list[dict], tau_T: float, tau_V: float, tau_C: float, d_cautious: float, d_restricted: float) -> dict:
    if not siuo_results:
        raise RuntimeError('No sample SIUO valid.')
    if not mmstar_test_results:
        raise RuntimeError('No sample MMStar_test valid.')
    st_p, sv_p, sc_p, d_p, scpred_p, fullpred_p = _fixed_arrays(siuo_results, tau_T, tau_V, tau_C)
    st_n, sv_n, sc_n, d_n, scpred_n, fullpred_n = _fixed_arrays(mmstar_test_results, tau_T, tau_V, tau_C)
    sc_metrics = binary_metrics(scpred_p, scpred_n)
    full_metrics = binary_metrics(fullpred_p, fullpred_n)
    labels = np.concatenate([np.ones(len(sc_p)), np.zeros(len(sc_n))])
    sc_scores = np.concatenate([sc_p, sc_n])
    full_scores = np.concatenate([d_p, d_n])
    auc_sc = float(roc_auc_score(labels, sc_scores))
    auc_full = float(roc_auc_score(labels, full_scores))
    trig_t_p = st_p >= tau_T
    trig_v_p = sv_p >= tau_V
    trig_c_p = sc_p >= tau_C
    trig_t_n = st_n >= tau_T
    trig_v_n = sv_n >= tau_V
    trig_c_n = sc_n >= tau_C
    silent_p = ~trig_t_p & ~trig_v_p
    silent_n = ~trig_t_n & ~trig_v_n

    def _safe_cond(c, silent):
        return float(c[silent].mean()) if silent.any() else None
    return {'protocol': {'name': 'Frozen zero-shot SIUO evaluation', 'positive_set': 'SIUO test', 'negative_set': 'MMStar_test', 'threshold_source': 'delta_config.json from source-domain v13 calibration', 'tau_T_frozen': float(tau_T), 'tau_V_frozen': float(tau_V), 'tau_C_frozen': float(tau_C), 'uses_target_data_for_threshold_selection': False, 'uses_quantile_matching': False, 'uses_target_calibration': False, 'note': 'SIUO and MMStar_test are evaluation-only. AUC is reported as a threshold-independent diagnostic; all operating-point metrics use the frozen source-domain thresholds.'}, 'n': {'siuo_positive': int(len(siuo_results)), 'mmstar_negative': int(len(mmstar_test_results))}, 'sc_only': {**{k: round(float(v), 4) if isinstance(v, float) else v for k, v in sc_metrics.items()}, 'auc': round(auc_sc, 4), 'mean_positive': round(float(sc_p.mean()), 4), 'mean_negative': round(float(sc_n.mean()), 4)}, 'full_pipeline': {**{k: round(float(v), 4) if isinstance(v, float) else v for k, v in full_metrics.items()}, 'auc_delta': round(auc_full, 4)}, 'diagnostics': {'siuo': {'text_trigger_rate': round(float(trig_t_p.mean()), 4), 'visual_trigger_rate': round(float(trig_v_p.mean()), 4), 'combo_trigger_rate': round(float(trig_c_p.mean()), 4), 'both_unimodal_silent_rate': round(float(silent_p.mean()), 4), 'c_only_rate': round(float((silent_p & trig_c_p).mean()), 4), 'conditional_sc_tpr_given_tv_silent': round(_safe_cond(trig_c_p, silent_p), 4) if _safe_cond(trig_c_p, silent_p) is not None else None, 'verdict_distribution': _verdict_distribution_from_delta(d_p, d_cautious, d_restricted)}, 'mmstar_test': {'text_trigger_rate': round(float(trig_t_n.mean()), 4), 'visual_trigger_rate': round(float(trig_v_n.mean()), 4), 'combo_trigger_rate': round(float(trig_c_n.mean()), 4), 'both_unimodal_silent_rate': round(float(silent_n.mean()), 4), 'c_only_rate': round(float((silent_n & trig_c_n).mean()), 4), 'conditional_sc_fpr_given_tv_silent': round(_safe_cond(trig_c_n, silent_n), 4) if _safe_cond(trig_c_n, silent_n) is not None else None, 'verdict_distribution': _verdict_distribution_from_delta(d_n, d_cautious, d_restricted)}}}

def _candidate_paths(explicit_path, roots, patterns):
    paths = []
    if explicit_path:
        paths.append(Path(explicit_path).expanduser())
    for root in roots:
        root = Path(root)
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
    out, seen = ([], set())
    for p in paths:
        p = Path(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def load_mmstar_cal_scores(delta_dir: Path, probes_dir: Path, explicit_path=None, sc_representation='aligned'):
    candidates = _candidate_paths(explicit_path, [delta_dir, probes_dir, probes_dir / 'delta'], ['scores_cal_seed42_n*_*.pkl', '**/scores_cal_seed42_n*_*.pkl'])
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if obj.get('sc_representation') != sc_representation:
                continue
            if 'mmstar' not in obj:
                continue
            mm = obj['mmstar']
            sc = np.asarray(mm[2], dtype=float)
            if len(sc) > 0 and np.all(np.isfinite(sc)):
                return (sc, path)
        except Exception:
            continue
    raise FileNotFoundError("Cache MMStar_cal not found. Specify CONFIG['calibration_cache'] with the file scores_cal_seed42_n*_*.pkl produced by 4.1_calibrate_tau_c.py.")

def load_mmstar_test_results(delta_dir: Path, probes_dir: Path, explicit_path=None, sc_representation='aligned'):
    candidates = _candidate_paths(explicit_path, [delta_dir, probes_dir, probes_dir / 'delta'], ['eval_results_cache_internal.pkl', '**/eval_results_cache_internal.pkl', 'eval_results_cache_*internal*.pkl'])
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if obj.get('sc_representation') != sc_representation:
                continue
            safe = obj.get('results_safe')
            if isinstance(safe, list) and len(safe) > 0 and ('S_C' in safe[0]):
                return (safe, path)
        except Exception:
            continue
    raise FileNotFoundError("Cache MMStar_test not found. Specify CONFIG['mmstar_eval_cache'] with eval_results_cache_internal.pkl produced by 6_evaluate_pipeline.py.")

def youden_tau(scores_pos: np.ndarray, scores_neg: np.ndarray):
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        raise ValueError('Youden calibration requires at least one positive and one negative sample.')
    if not np.all(np.isfinite(scores_pos)) or not np.all(np.isfinite(scores_neg)):
        raise ValueError('Not-finite scores in Youden calibration.')
    labels = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    scores = np.concatenate([scores_pos, scores_neg])
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    finite = np.isfinite(thr)
    if not finite.any():
        raise RuntimeError('No finite threshold produced from the ROC.')
    valid_idx = np.flatnonzero(finite)
    idx = int(valid_idx[np.argmax(j[finite])])
    return (float(thr[idx]), float(tpr[idx]), float(fpr[idx]), float(roc_auc_score(labels, scores)))

def binary_metrics(pred_pos: np.ndarray, pred_neg: np.ndarray):
    tp = int(pred_pos.sum())
    fn = int((pred_pos == 0).sum())
    fp = int(pred_neg.sum())
    tn = int((pred_neg == 0).sum())
    tpr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)
    precision = tp / (tp + fp + EPS)
    f1 = 2 * precision * tpr / (precision + tpr + EPS)
    bacc = (tpr + tn / (tn + fp + EPS)) / 2
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': float(tpr), 'fpr': float(fpr), 'precision': float(precision), 'f1': float(f1), 'balanced_accuracy': float(bacc)}

def preds_with_tau(results: list[dict], tau_T: float, tau_V: float, tau_C: float):
    return np.asarray([int(max(r['S_T'] - tau_T, r['S_V'] - tau_V, r['S_C'] - tau_C) >= 0) for r in results], dtype=int)

def sc_preds(results: list[dict], tau_C: float):
    return np.asarray([int(r['S_C'] >= tau_C) for r in results], dtype=int)

def conditional_sc_tpr(results: list[dict], tau_T: float, tau_V: float, tau_C: float):
    silent = [r for r in results if r['S_T'] < tau_T and r['S_V'] < tau_V]
    if not silent:
        return (float('nan'), 0)
    return (float(np.mean([r['S_C'] >= tau_C for r in silent])), len(silent))

def aggregate_repeats(repeats: list[dict], key_path: tuple[str, ...]):
    vals = []
    for r in repeats:
        x = r
        for key in key_path:
            x = x[key]
        if x is not None and np.isfinite(x):
            vals.append(float(x))
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {'mean': round(float(a.mean()), 4), 'std': round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0, 'min': round(float(a.min()), 4), 'max': round(float(a.max()), 4), 'n': len(a)}

def run_fewshot_calibration(siuo_results: list[dict], mmstar_cal_sc: np.ndarray, mmstar_test_results: list[dict], tau_T: float, tau_V: float, tau_C_fixed: float, n_pos: int=30, n_safe: int=30, repeats: int=20, seed: int=42):
    n_siuo = len(siuo_results)
    if n_pos >= n_siuo:
        raise ValueError(f'fewshot_n_pos={n_pos} must be < n_SIUO={n_siuo}')
    if n_safe > len(mmstar_cal_sc):
        raise ValueError(f'fewshot_n_safe={n_safe} > MMStar_cal available={len(mmstar_cal_sc)}')
    mmstar_test_sc = np.asarray([r['S_C'] for r in mmstar_test_results], dtype=float)
    if not np.all(np.isfinite(mmstar_test_sc)):
        raise ValueError('MMStar_test contains not-finite S_C values.')
    all_idx = np.arange(n_siuo)
    rep_out = []
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        cal_pos_idx = np.sort(rng.choice(all_idx, size=n_pos, replace=False))
        mask = np.ones(n_siuo, dtype=bool)
        mask[cal_pos_idx] = False
        test_pos_idx = all_idx[mask]
        cal_safe_idx = np.sort(rng.choice(len(mmstar_cal_sc), size=n_safe, replace=False))
        sc_pos_cal = np.asarray([siuo_results[i]['S_C'] for i in cal_pos_idx], dtype=float)
        sc_safe_cal = mmstar_cal_sc[cal_safe_idx]
        tau_target, cal_tpr, cal_fpr, cal_auc = youden_tau(sc_pos_cal, sc_safe_cal)
        test_pos = [siuo_results[i] for i in test_pos_idx]
        test_safe = mmstar_test_results
        sc_pred_pos_fixed = sc_preds(test_pos, tau_C_fixed)
        sc_pred_neg_fixed = sc_preds(test_safe, tau_C_fixed)
        sc_pred_pos_target = sc_preds(test_pos, tau_target)
        sc_pred_neg_target = sc_preds(test_safe, tau_target)
        sc_fixed = binary_metrics(sc_pred_pos_fixed, sc_pred_neg_fixed)
        sc_target = binary_metrics(sc_pred_pos_target, sc_pred_neg_target)
        full_pos_fixed = preds_with_tau(test_pos, tau_T, tau_V, tau_C_fixed)
        full_neg_fixed = preds_with_tau(test_safe, tau_T, tau_V, tau_C_fixed)
        full_pos_target = preds_with_tau(test_pos, tau_T, tau_V, tau_target)
        full_neg_target = preds_with_tau(test_safe, tau_T, tau_V, tau_target)
        full_fixed = binary_metrics(full_pos_fixed, full_neg_fixed)
        full_target = binary_metrics(full_pos_target, full_neg_target)
        cond_fixed, n_silent = conditional_sc_tpr(test_pos, tau_T, tau_V, tau_C_fixed)
        cond_target, _ = conditional_sc_tpr(test_pos, tau_T, tau_V, tau_target)
        y = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_safe))])
        s = np.concatenate([np.asarray([r['S_C'] for r in test_pos], dtype=float), mmstar_test_sc])
        auc_test = float(roc_auc_score(y, s))
        rep_out.append({'repeat': rep, 'seed': seed + rep, 'n_cal_pos_siuo': int(n_pos), 'n_cal_safe_mmstar': int(n_safe), 'n_test_pos_siuo': int(len(test_pos)), 'n_test_safe_mmstar': int(len(test_safe)), 'tau_C_target': float(tau_target), 'calibration': {'auc': cal_auc, 'tpr': cal_tpr, 'fpr': cal_fpr}, 'heldout_auc_sc': auc_test, 'sc_fixed': sc_fixed, 'sc_target_calibrated': sc_target, 'full_fixed': full_fixed, 'full_target_calibrated': full_target, 'conditional_sc_tpr_given_tv_silent': {'n_silent': int(n_silent), 'fixed': None if not np.isfinite(cond_fixed) else float(cond_fixed), 'target_calibrated': None if not np.isfinite(cond_target) else float(cond_target)}})
    aggregate = {'tau_C_target': aggregate_repeats(rep_out, ('tau_C_target',)), 'heldout_auc_sc': aggregate_repeats(rep_out, ('heldout_auc_sc',)), 'sc_fixed_tpr': aggregate_repeats(rep_out, ('sc_fixed', 'tpr')), 'sc_fixed_fpr': aggregate_repeats(rep_out, ('sc_fixed', 'fpr')), 'sc_target_tpr': aggregate_repeats(rep_out, ('sc_target_calibrated', 'tpr')), 'sc_target_fpr': aggregate_repeats(rep_out, ('sc_target_calibrated', 'fpr')), 'sc_target_f1': aggregate_repeats(rep_out, ('sc_target_calibrated', 'f1')), 'sc_target_bacc': aggregate_repeats(rep_out, ('sc_target_calibrated', 'balanced_accuracy')), 'full_fixed_tpr': aggregate_repeats(rep_out, ('full_fixed', 'tpr')), 'full_fixed_fpr': aggregate_repeats(rep_out, ('full_fixed', 'fpr')), 'full_target_tpr': aggregate_repeats(rep_out, ('full_target_calibrated', 'tpr')), 'full_target_fpr': aggregate_repeats(rep_out, ('full_target_calibrated', 'fpr')), 'full_target_f1': aggregate_repeats(rep_out, ('full_target_calibrated', 'f1')), 'full_target_bacc': aggregate_repeats(rep_out, ('full_target_calibrated', 'balanced_accuracy')), 'conditional_fixed': aggregate_repeats(rep_out, ('conditional_sc_tpr_given_tv_silent', 'fixed')), 'conditional_target': aggregate_repeats(rep_out, ('conditional_sc_tpr_given_tv_silent', 'target_calibrated'))}
    return {'protocol': {'method': "Repeated few-shot SIUO-assisted calibration of tau_C with Youden's J", 'positive_calibration_domain': 'SIUO', 'safe_calibration_domain': 'MMStar_cal', 'positive_test_domain': 'SIUO held-out', 'safe_test_domain': 'MMStar_test', 'n_pos_per_repeat': int(n_pos), 'n_safe_per_repeat': int(n_safe), 'n_repeats': int(repeats), 'base_seed': int(seed), 'tau_T_fixed': float(tau_T), 'tau_V_fixed': float(tau_V), 'tau_C_fixed_v13': float(tau_C_fixed), 'note': "Only tau_C is adapted. SIUO calibration positives are excluded from that repeat's SIUO test. MMStar_test is never used for calibration."}, 'aggregate': aggregate, 'repeats': rep_out}

def plot_fewshot(fewshot: dict, out_path: Path):
    reps = fewshot['repeats']
    tau = np.asarray([r['tau_C_target'] for r in reps], dtype=float)
    sc_tpr = np.asarray([r['sc_target_calibrated']['tpr'] for r in reps])
    sc_fpr = np.asarray([r['sc_target_calibrated']['fpr'] for r in reps])
    full_tpr = np.asarray([r['full_target_calibrated']['tpr'] for r in reps])
    full_fpr = np.asarray([r['full_target_calibrated']['fpr'] for r in reps])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(tau, bins=min(12, max(5, len(tau) // 2)))
    axes[0].axvline(fewshot['protocol']['tau_C_fixed_v13'], ls='--', label='fixed v13')
    axes[0].set_title('Few-shot tau_C across repeats')
    axes[0].set_xlabel('tau_C')
    axes[0].legend()
    x = np.arange(len(reps))
    axes[1].plot(x, sc_tpr, marker='o', label='S_C TPR')
    axes[1].plot(x, sc_fpr, marker='o', label='S_C FPR')
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title('S_C held-out performance')
    axes[1].set_xlabel('repeat')
    axes[1].legend()
    axes[2].plot(x, full_tpr, marker='o', label='Full TPR')
    axes[2].plot(x, full_fpr, marker='o', label='Full FPR')
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title('Full pipeline held-out performance')
    axes[2].set_xlabel('repeat')
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def run(args):
    probes_dir = Path(args.probes)
    delta_dir = Path(args.delta)
    delta_dir.mkdir(parents=True, exist_ok=True)
    print('\n[1] Loading configuration v13...')
    cfg_path = delta_dir / 'delta_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'{cfg_path} not found')
    cfg = json.load(open(cfg_path))
    config_mode = cfg.get('sc_representation_mode')
    if config_mode is not None and config_mode != args.sc_representation:
        raise ValueError(f'delta_config S_C representation mismatch: {config_mode} != {args.sc_representation}')
    internal = cfg['internal_probes']
    tau_T = float(internal['tau_T'])
    tau_V = float(internal['tau_V'])
    tau_C = float(cfg['tau_C'])
    clip_layer = int(internal.get('clip_layer', 7))
    d_cautious = internal.get('d_cautious')
    d_restricted = internal.get('d_restricted')
    if d_cautious is None or d_restricted is None:
        delta_max = 1.0 - min(tau_T, tau_V, tau_C)
        d_cautious = delta_max / 3
        d_restricted = 2 * delta_max / 3
    d_cautious = float(d_cautious)
    d_restricted = float(d_restricted)
    print(f'  tau_T={tau_T:.4f}  tau_V={tau_V:.4f}  tau_C={tau_C:.4f}')
    print(f'  LLaMA={LLAMA_LAYER}  CLIP={clip_layer}')
    print('\n[2] Loading probe...')
    with open(probes_dir / 'probe_text.pkl', 'rb') as f:
        clf_text, Q_text = unwrap_probe(pickle.load(f))
    visual_path = probes_dir / 'probe_visual_orthogonalized.pkl'
    if not visual_path.exists():
        visual_path = probes_dir / 'probe_visual.pkl'
    with open(visual_path, 'rb') as f:
        clf_visual, Q_visual = unwrap_probe(pickle.load(f))
    combo_path, clf_combo, Q_combo = find_combo_probe(probes_dir, args.sc_representation)
    print(f'  visual={visual_path.name}  combo={combo_path.name}')
    print('\n[3] Loading LLaVA...')
    model, processor = load_model(gpu=args.gpu)
    print('\n[4] SIUO from Hugging Face (streaming=True)...')
    stream = load_dataset(args.dataset_repo, args.dataset_config, split=args.dataset_split, streaming=True)
    rows = materialize_stream(stream, args.max_samples)
    print(f'  sample in RAM: {len(rows)}')
    suffix = '' if args.max_samples is None else f'_n{args.max_samples}'
    siuo_cache = delta_dir / f'eval_results_siuo_internal{suffix}.pkl'
    if siuo_cache.exists() and (not args.recompute):
        print(f'\n[5] Cache SIUO: {siuo_cache}')
        obj = pickle.load(open(siuo_cache, 'rb'))
        if obj.get('sc_representation') != args.sc_representation:
            raise ValueError(f"SIUO cache representation mismatch: {obj.get('sc_representation')} != {args.sc_representation}")
        siuo_results = obj['results_unsafe']
    else:
        print('\n[5] Evaluation original SIUO...')
        siuo_results = evaluate_siuo(rows, model, processor, clf_text, Q_text, clf_visual, Q_visual, clf_combo, Q_combo, clip_layer, tau_T, tau_V, tau_C, d_cautious, d_restricted, args.sc_representation)
        with open(siuo_cache, 'wb') as f:
            pickle.dump({'sc_representation': args.sc_representation, 'results_unsafe': siuo_results}, f)
        print(f'  [cache] {siuo_cache}')
    print('\n[6] Loading MMStar_cal / MMStar_test from the v13 caches...')
    mmstar_cal_sc, cal_path = load_mmstar_cal_scores(delta_dir, probes_dir, args.calibration_cache, args.sc_representation)
    mmstar_test_results, test_path = load_mmstar_test_results(delta_dir, probes_dir, args.mmstar_eval_cache, args.sc_representation)
    print(f'  MMStar_cal S_C : n={len(mmstar_cal_sc)}  <- {cal_path}')
    print(f'  MMStar_test    : n={len(mmstar_test_results)} <- {test_path}')
    thresholds = {'tau_T': tau_T, 'tau_V': tau_V, 'tau_C': tau_C, 'd_cautious': d_cautious, 'd_restricted': d_restricted, 'clip_layer': clip_layer, 'llama_layer': LLAMA_LAYER, 'sc_representation': args.sc_representation}
    summary = compute_siuo_summary(siuo_results, thresholds)
    summary_path = delta_dir / f'evaluation_results_siuo_internal{suffix}.json'
    json.dump(summary, open(summary_path, 'w'), indent=2)
    zeroshot = compute_formal_zeroshot(siuo_results=siuo_results, mmstar_test_results=mmstar_test_results, tau_T=tau_T, tau_V=tau_V, tau_C=tau_C, d_cautious=d_cautious, d_restricted=d_restricted)
    zeroshot['source_files'] = {'siuo_cache': str(siuo_cache), 'mmstar_test_cache': str(test_path), 'delta_config': str(cfg_path)}
    zeroshot_path = delta_dir / f'evaluation_zeroshot_siuo_vs_mmstar_internal{suffix}.json'
    json.dump(zeroshot, open(zeroshot_path, 'w'), indent=2)
    zeroshot_cache = delta_dir / f'eval_results_siuo_vs_mmstar_zeroshot_internal{suffix}.pkl'
    with open(zeroshot_cache, 'wb') as f:
        pickle.dump({'sc_representation': args.sc_representation, 'results_unsafe': siuo_results, 'results_safe': mmstar_test_results, 'thresholds_frozen': thresholds, 'protocol': zeroshot['protocol']}, f)
    print(f'  [zero-shot JSON]  {zeroshot_path}')
    print(f'  [zero-shot cache] {zeroshot_cache}')
    print('\n[7] Repeated few-shot target calibration of tau_C...')
    fewshot = run_fewshot_calibration(siuo_results=siuo_results, mmstar_cal_sc=mmstar_cal_sc, mmstar_test_results=mmstar_test_results, tau_T=tau_T, tau_V=tau_V, tau_C_fixed=tau_C, n_pos=args.fewshot_n_pos, n_safe=args.fewshot_n_safe, repeats=args.fewshot_repeats, seed=args.fewshot_seed)
    fewshot['source_files'] = {'siuo_cache': str(siuo_cache), 'mmstar_cal_cache': str(cal_path), 'mmstar_test_cache': str(test_path)}
    fewshot_path = delta_dir / f'evaluation_fewshot_siuo_internal{suffix}.json'
    json.dump(fewshot, open(fewshot_path, 'w'), indent=2)
    plot_path = delta_dir / f'evaluation_fewshot_siuo_internal{suffix}.png'
    plot_fewshot(fewshot, plot_path)
    z = zeroshot
    a = fewshot['aggregate']
    print('\n' + '=' * 78)
    print('SIUO vs MMSTAR_TEST — FORMAL ZERO-SHOT, FROZEN THRESHOLDS')
    print('=' * 78)
    print(f"n SIUO positives           : {z['n']['siuo_positive']}")
    print(f"n MMStar_test negatives    : {z['n']['mmstar_negative']}")
    print(f'tau_C frozen               : {tau_C:.6f}')
    print('-' * 78)
    print(f"S_C AUC                    : {z['sc_only']['auc']:.4f}")
    print(f"S_C TPR / FPR              : {z['sc_only']['tpr']:.2%} / {z['sc_only']['fpr']:.2%}")
    print(f"S_C Precision / F1         : {z['sc_only']['precision']:.4f} / {z['sc_only']['f1']:.4f}")
    print(f"S_C Balanced Accuracy      : {z['sc_only']['balanced_accuracy']:.4f}")
    print('-' * 78)
    print(f"Full Delta AUC             : {z['full_pipeline']['auc_delta']:.4f}")
    print(f"Full TPR / FPR             : {z['full_pipeline']['tpr']:.2%} / {z['full_pipeline']['fpr']:.2%}")
    print(f"Full Precision / F1        : {z['full_pipeline']['precision']:.4f} / {z['full_pipeline']['f1']:.4f}")
    print(f"Full Balanced Accuracy     : {z['full_pipeline']['balanced_accuracy']:.4f}")
    print('-' * 78)
    dsiuo = z['diagnostics']['siuo']
    print(f"SIUO S_T trigger           : {dsiuo['text_trigger_rate']:.2%}")
    print(f"SIUO S_V trigger           : {dsiuo['visual_trigger_rate']:.2%}")
    print(f"SIUO S_C trigger           : {dsiuo['combo_trigger_rate']:.2%}")
    print(f"SIUO T,V both silent       : {dsiuo['both_unimodal_silent_rate']:.2%}")
    if dsiuo.get('conditional_sc_tpr_given_tv_silent') is not None:
        print(f"S_C TPR | T,V silent      : {dsiuo['conditional_sc_tpr_given_tv_silent']:.2%}")
    print('\n' + '=' * 78)
    print('SIUO — REPEATED FEW-SHOT tau_C CALIBRATION')
    print('=' * 78)
    print(f'repeats                    : {args.fewshot_repeats}')
    print(f'cal/repeat                  : {args.fewshot_n_pos} SIUO + {args.fewshot_n_safe} MMStar safe')
    print(f'tau_C fixed v13             : {tau_C:.4f}')
    print(f"tau_C target                : {a['tau_C_target']['mean']:.4f} ± {a['tau_C_target']['std']:.4f}  [{a['tau_C_target']['min']:.4f}, {a['tau_C_target']['max']:.4f}]")
    print(f"Held-out AUC S_C            : {a['heldout_auc_sc']['mean']:.4f} ± {a['heldout_auc_sc']['std']:.4f}")
    print('-' * 78)
    print(f"S_C fixed TPR/FPR           : {a['sc_fixed_tpr']['mean']:.2%} / {a['sc_fixed_fpr']['mean']:.2%}")
    print(f"S_C target-cal TPR/FPR      : {a['sc_target_tpr']['mean']:.2%} ± {a['sc_target_tpr']['std']:.2%} / {a['sc_target_fpr']['mean']:.2%} ± {a['sc_target_fpr']['std']:.2%}")
    print(f"S_C target-cal F1/BAcc      : {a['sc_target_f1']['mean']:.4f} / {a['sc_target_bacc']['mean']:.4f}")
    print('-' * 78)
    print(f"Full fixed TPR/FPR          : {a['full_fixed_tpr']['mean']:.2%} / {a['full_fixed_fpr']['mean']:.2%}")
    print(f"Full target-cal TPR/FPR     : {a['full_target_tpr']['mean']:.2%} ± {a['full_target_tpr']['std']:.2%} / {a['full_target_fpr']['mean']:.2%} ± {a['full_target_fpr']['std']:.2%}")
    print(f"Full target-cal F1/BAcc     : {a['full_target_f1']['mean']:.4f} / {a['full_target_bacc']['mean']:.4f}")
    if a.get('conditional_target') is not None:
        print(f"S_C TPR | T,V silent target : {a['conditional_target']['mean']:.2%} ± {a['conditional_target']['std']:.2%}")
    print('=' * 78)
    print(f'SIUO-ONLY JSON     : {summary_path}')
    print(f'ZERO-SHOT JSON     : {zeroshot_path}')
    print(f'ZERO-SHOT CACHE    : {zeroshot_cache}')
    print(f'FEW-SHOT JSON      : {fewshot_path}')
    print(f'PLOT               : {plot_path}')
if __name__ == '__main__':
    CONFIG = {'probes': 'probes/rebuilt_v13/', 'delta': 'probes/rebuilt_v13/delta', 'gpu': 3, 'dataset_repo': DATASET_REPO, 'dataset_config': DATASET_CONFIG, 'dataset_split': DATASET_SPLIT, 'calibration_cache': 'probes/rebuilt_v13/delta/scores_cal_seed42_msts100_mmstar400.pkl', 'mmstar_eval_cache': 'probes/rebuilt_v13/delta/eval_results_cache_internal.pkl', 'fewshot_n_pos': 30, 'fewshot_n_safe': 30, 'fewshot_repeats': 20, 'fewshot_seed': 42, 'max_samples': None, 'recompute': False, 'sc_representation': 'aligned'}
    try:
        from IPython import get_ipython
        IN_INTERACTIVE = get_ipython() is not None
    except Exception:
        IN_INTERACTIVE = False
    if IN_INTERACTIVE:
        print('[interactive] Jupyter/VS Code environment detected: using CONFIG directly.')
        args = argparse.Namespace(**CONFIG)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--probes', default=CONFIG['probes'])
        parser.add_argument('--delta', default=CONFIG['delta'])
        parser.add_argument('--gpu', type=int, default=CONFIG['gpu'])
        parser.add_argument('--dataset_repo', default=CONFIG['dataset_repo'])
        parser.add_argument('--dataset_config', default=CONFIG['dataset_config'])
        parser.add_argument('--dataset_split', default=CONFIG['dataset_split'])
        parser.add_argument('--calibration_cache', default=CONFIG['calibration_cache'])
        parser.add_argument('--mmstar_eval_cache', default=CONFIG['mmstar_eval_cache'])
        parser.add_argument('--fewshot_n_pos', type=int, default=CONFIG['fewshot_n_pos'])
        parser.add_argument('--fewshot_n_safe', type=int, default=CONFIG['fewshot_n_safe'])
        parser.add_argument('--fewshot_repeats', type=int, default=CONFIG['fewshot_repeats'])
        parser.add_argument('--fewshot_seed', type=int, default=CONFIG['fewshot_seed'])
        parser.add_argument('--max_samples', type=int, default=CONFIG['max_samples'])
        parser.add_argument('--recompute', action='store_true', default=CONFIG['recompute'])
        parser.add_argument('--sc_representation', choices=SC_REPRESENTATION_CHOICES, default=CONFIG['sc_representation'])
        args = parser.parse_args()
    run(args)
