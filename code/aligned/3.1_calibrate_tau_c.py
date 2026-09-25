#!/usr/bin/env python3
"""Calibrate the combination-score threshold."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from utils import load_model, extract_llama_hidden_solo, extract_sc_feature_pair, extract_clip_hidden, sc_representation_formula, validate_combo_representation, SC_REPRESENTATION_CHOICES
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
LLAMA_LAYER = 17
MODEL_ID = 'llava-hf/llava-1.5-7b-hf'
TAU_T_FIXED = 0.5
TAU_V_FIXED = 0.5
CLIP_LAYER = 21

def compute_all_scores(text: str, image: Image.Image, model, processor, clf_combo, clf_text, clf_visual, layer: int, clip_layer: int, ortho_Q: np.ndarray | None=None, sc_representation: str='aligned') -> tuple[float, float, float]:
    h_text = extract_llama_hidden_solo(text, model, processor, layer)
    h_visual = extract_clip_hidden(image, model, processor, clip_layer)
    _, _, delta_h = extract_sc_feature_pair(text, image, model, processor, layer, sc_representation)
    dh = delta_h.reshape(1, -1)
    if ortho_Q is not None:
        dh = dh - dh @ ortho_Q @ ortho_Q.T
    s_t = float(clf_text.predict_proba(h_text.reshape(1, -1))[0, 1])
    s_v = float(clf_visual.predict_proba(h_visual.reshape(1, -1))[0, 1])
    s_c = float(clf_combo.predict_proba(dh)[0, 1])
    return (s_t, s_v, s_c)

def load_or_create_splits_v13(splits_path: Path, msts_manifest_path: Path, n_mmstar: int, n_cal_mmstar: int, seed: int=42) -> dict:
    msts_manifest_path = Path(msts_manifest_path)
    if not msts_manifest_path.exists():
        raise FileNotFoundError(f'Manifest MSTS v13 not found: {msts_manifest_path}')
    with open(msts_manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    msts_train = list(map(int, manifest['train']))
    msts_cal = list(map(int, manifest['val']))
    msts_test = list(map(int, manifest['test']))
    if (len(msts_train), len(msts_cal), len(msts_test)) != (200, 100, 100):
        raise RuntimeError(f'MSTS v13 manifest has unexpected dimensions: train={len(msts_train)}, dev={len(msts_cal)}, test={len(msts_test)}')
    s_train, s_cal, s_test = (set(msts_train), set(msts_cal), set(msts_test))
    if s_train & s_cal or s_train & s_test or s_cal & s_test:
        raise RuntimeError('Leakage in the MSTS v13 manifest: the splits are not disjoint.')
    if len(s_train | s_cal | s_test) != 400:
        raise RuntimeError('The MSTS v13 manifest does not cover exactly 400 samples.')
    if splits_path.exists():
        with open(splits_path, 'r', encoding='utf-8') as f:
            old = json.load(f)
        protocol_ok = old.get('protocol') == 'rebuilt_v13_msts_200_100_100'
        msts_ok = list(map(int, old.get('msts_cal', []))) == msts_cal and list(map(int, old.get('msts_test', []))) == msts_test
        mmstar_ok = len(old.get('mmstar_cal', [])) == n_cal_mmstar and len(old.get('mmstar_test', [])) == n_mmstar - n_cal_mmstar
        if protocol_ok and msts_ok and mmstar_ok:
            print(f'  [splits] Reusing the v13 split: {splits_path}')
            print(f'  [splits] MSTS dev={len(msts_cal)}  test={len(msts_test)}')
            print(f"  [splits] MMStar cal={len(old['mmstar_cal'])}  test={len(old['mmstar_test'])}")
            return old
        print('  [splits][warn] existing eval_splits is incompatible with v13 and will be regenerated.')
    rng = np.random.default_rng(seed=seed)
    mmstar_idx = rng.permutation(n_mmstar).tolist()
    splits = {'protocol': 'rebuilt_v13_msts_200_100_100', 'seed': seed, 'msts_manifest': str(msts_manifest_path), 'msts_train': msts_train, 'msts_cal': msts_cal, 'msts_test': msts_test, 'n_cal_mmstar': n_cal_mmstar, 'mmstar_cal': mmstar_idx[:n_cal_mmstar], 'mmstar_test': mmstar_idx[n_cal_mmstar:]}
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    print(f'  [splits] Saved protocol v13 in {splits_path}')
    print(f'  [splits] MSTS train={len(msts_train)} dev={len(msts_cal)} test={len(msts_test)}')
    print(f"  [splits] MMStar cal={len(splits['mmstar_cal'])} test={len(splits['mmstar_test'])}")
    return splits

def extract_scores_msts(dataset, indices, model, processor, clf_combo, clf_text, clf_visual, layer, clip_layer, ortho_Q=None, sc_representation='aligned'):
    s_t_list, s_v_list, s_c_list = ([], [], [])
    for i, idx in enumerate(tqdm(indices, desc='  MSTS_cal scores')):
        row = dataset[idx]
        text = row.get('prompt_text') or row.get('text', '')
        image = row['unsafe_image'].convert('RGB')
        try:
            s_t, s_v, s_c = compute_all_scores(text, image, model, processor, clf_combo, clf_text, clf_visual, layer, clip_layer, ortho_Q, sc_representation)
        except Exception as e:
            print(f'  [warn] MSTS idx={idx}: {e}')
            s_t, s_v, s_c = (0.0, 0.0, 0.0)
        s_t_list.append(s_t)
        s_v_list.append(s_v)
        s_c_list.append(s_c)
        if (i + 1) % 20 == 0:
            print(f'    [{i + 1}/{len(indices)}]  S_T={s_t:.4f}  S_V={s_v:.4f}  S_C={s_c:.4f}')
    return (np.array(s_t_list), np.array(s_v_list), np.array(s_c_list))

def extract_scores_mmstar(dataset, indices, model, processor, clf_combo, clf_text, clf_visual, layer, clip_layer, ortho_Q=None, sc_representation='aligned'):
    s_t_list, s_v_list, s_c_list = ([], [], [])
    for i, idx in enumerate(tqdm(indices, desc='  MMStar_cal scores')):
        row = dataset[idx]
        text = row.get('question', '')
        image = row['image'].convert('RGB')
        try:
            s_t, s_v, s_c = compute_all_scores(text, image, model, processor, clf_combo, clf_text, clf_visual, layer, clip_layer, ortho_Q, sc_representation)
        except Exception as e:
            print(f'  [warn] MMStar idx={idx}: {e}')
            s_t, s_v, s_c = (0.0, 0.0, 0.0)
        s_t_list.append(s_t)
        s_v_list.append(s_v)
        s_c_list.append(s_c)
        if (i + 1) % 20 == 0:
            print(f'    [{i + 1}/{len(indices)}]  S_T={s_t:.4f}  S_V={s_v:.4f}  S_C={s_c:.4f}')
    return (np.array(s_t_list), np.array(s_v_list), np.array(s_c_list))

def calibrate_threshold(name: str, scores_unsafe: np.ndarray, scores_safe: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate([scores_unsafe, scores_safe])
    labels = np.concatenate([np.ones(len(scores_unsafe)), np.zeros(len(scores_safe))])
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    J = tpr - fpr
    idx = np.argmax(J)
    tau = float(thresholds[idx])
    print(f'\n  Calibration tau_{name}:')
    print(f'    AUC = {auc:.4f}   tau_{name} = {tau:.4f}')
    print(f'    TPR = {tpr[idx]:.1%}   FPR = {fpr[idx]:.1%}')
    return (tau, float(auc))

def find_response_thresholds(tau_T: float, tau_V: float, tau_C: float) -> tuple[float, float]:
    delta_max = 1.0 - min(tau_T, tau_V, tau_C)
    d_cautious = delta_max * 1 / 3
    d_restricted = delta_max * 2 / 3
    print(f'\n  Response thresholds (geometric):')
    print(f'    delta_max    = {delta_max:.4f}')
    print(f'    SAFE         :  Delta < 0')
    print(f'    CAUTIOUS     :  0 <= Delta < {d_cautious:.4f}')
    print(f'    RESTRICTED   :  {d_cautious:.4f} <= Delta < {d_restricted:.4f}')
    print(f'    REFUSED      :  Delta >= {d_restricted:.4f}')
    return (d_cautious, d_restricted)

def plot_calibration(sc_unsafe: np.ndarray, sc_safe: np.ndarray, tau_C: float, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(sc_safe, bins=30, alpha=0.6, color='steelblue', label=f'MMStar_cal safe (n={len(sc_safe)})')
    ax.hist(sc_unsafe, bins=30, alpha=0.6, color='tomato', label=f'MSTS_cal unsafe (n={len(sc_unsafe)})')
    ax.axvline(tau_C, color='black', ls='--', lw=2, label=f'τ_C = {tau_C:.4f}')
    ax.set_xlabel('S_C')
    ax.set_ylabel('Count')
    ax.set_title('S_C distribution — calibration set')
    ax.legend()
    tpr = (sc_unsafe >= tau_C).mean()
    fpr = (sc_safe >= tau_C).mean()
    ax.text(0.05, 0.95, f'TPR={tpr:.1%}  FPR={fpr:.1%}', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2 = axes[1]
    scores = np.concatenate([sc_unsafe, sc_safe])
    labels = np.concatenate([np.ones(len(sc_unsafe)), np.zeros(len(sc_safe))])
    fpr_c, tpr_c, thr = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax2.plot(fpr_c, tpr_c, lw=2, color='steelblue', label=f'AUC={auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    idx = np.argmin(np.abs(thr - tau_C))
    ax2.scatter(fpr_c[idx], tpr_c[idx], color='tomato', s=100, zorder=5, label=f'τ_C={tau_C:.3f}  (TPR={tpr_c[idx]:.2f}, FPR={fpr_c[idx]:.2f})')
    ax2.set_xlabel('FPR (MMStar_cal)')
    ax2.set_ylabel('TPR (MSTS_cal)')
    ax2.set_title('ROC — calibration tau_C')
    ax2.legend()
    plt.suptitle(f'Calibration tau_C = {tau_C:.4f}  (Youden on MSTS_cal + MMStar_cal)', fontsize=13)
    plt.tight_layout()
    p = out_dir / 'tau_c_calibration.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {p}')

def run(args):
    from datasets import load_dataset
    probes_dir = Path(args.probes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = out_dir / 'eval_splits_v13.json'
    print('\n[1] Loading probe_combo...')
    combo_path = None
    for name in ['1_linear_delta_h_orthogonalized', '1_linear_delta_h', '2_svm_rbf_delta_h', '3_mlp_pair']:
        p = probes_dir / 'combo' / f'probe_combo_{name}.pkl'
        if p.exists():
            combo_path = p
            break
    if combo_path is None:
        raise FileNotFoundError(f'No probe_combo in {probes_dir}/combo/')
    with open(combo_path, 'rb') as f:
        combo_obj = pickle.load(f)
    validate_combo_representation(combo_obj, args.sc_representation)
    clf_combo = combo_obj['clf']
    ortho_Q = combo_obj.get('ortho_Q')
    print(f'  probe_combo: {combo_path.name}')
    print(f'  ortho_Q available: {ortho_Q is not None}' + (f'  shape={ortho_Q.shape}' if ortho_Q is not None else ''))
    print('\n[1b] Loading probe_text e probe_visual internal...')
    with open(probes_dir / 'probe_text.pkl', 'rb') as f:
        clf_text = pickle.load(f)
    with open(probes_dir / 'probe_visual.pkl', 'rb') as f:
        clf_visual = pickle.load(f)
    print('\n[2] Loading dataset...')
    print('  MSTS...')
    msts_ds = load_dataset('felfri/MSTS', split='english')
    print(f'  MSTS: {len(msts_ds)} sample total')
    print('  MMStar...')
    mmstar_ds = load_dataset('Lin-Chen/MMStar', split='val')
    print(f'  MMStar: {len(mmstar_ds)} sample total')
    print('\n[3] Managing splits...')
    splits = load_or_create_splits_v13(splits_path=splits_path, msts_manifest_path=Path(args.msts_manifest), n_mmstar=len(mmstar_ds), n_cal_mmstar=args.n_cal_mmstar, seed=42)
    # Historical downstream v13 scripts use the unversioned filename.  Keep an
    # exact compatibility copy so every stage consumes the same frozen split.
    compat_splits_path = out_dir / 'eval_splits.json'
    with open(compat_splits_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    print(f'  [splits] Compatibility alias: {compat_splits_path.name}')
    print('\n[4] Loading LLaVA...')
    model, processor = load_model(gpu=args.gpu)
    sc_formula = sc_representation_formula(args.sc_representation)
    cache_path = out_dir / f"scores_cal_v13_{args.sc_representation}_seed42_msts{len(splits['msts_cal'])}_mmstar{len(splits['mmstar_cal'])}.pkl"
    if cache_path.exists() and (not args.recompute):
        print(f'  [cache] Loading {cache_path.name}')
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('sc_representation') != args.sc_representation:
            raise ValueError(f"Calibration cache representation mismatch: {cached.get('sc_representation')} != {args.sc_representation}")
        st_msts, sv_msts, sc_msts_cal = cached['msts']
        st_mmstar, sv_mmstar, sc_mmstar_cal = cached['mmstar']
    else:
        print('  Extraction MSTS_cal...')
        st_msts, sv_msts, sc_msts_cal = extract_scores_msts(msts_ds, splits['msts_cal'], model, processor, clf_combo, clf_text, clf_visual, LLAMA_LAYER, CLIP_LAYER, ortho_Q, args.sc_representation)
        print('  Extraction MMStar_cal...')
        st_mmstar, sv_mmstar, sc_mmstar_cal = extract_scores_mmstar(mmstar_ds, splits['mmstar_cal'], model, processor, clf_combo, clf_text, clf_visual, LLAMA_LAYER, CLIP_LAYER, ortho_Q, args.sc_representation)
        with open(cache_path, 'wb') as f:
            pickle.dump({'sc_representation': args.sc_representation, 'sc_formula': sc_formula, 'msts': (st_msts, sv_msts, sc_msts_cal), 'mmstar': (st_mmstar, sv_mmstar, sc_mmstar_cal)}, f)
        print(f'  [cache] Saved {cache_path.name}')
    # Keep the historical filename expected by the downstream v13 scripts.
    compat_cache_path = out_dir / f"scores_cal_seed42_msts{len(splits['msts_cal'])}_mmstar{len(splits['mmstar_cal'])}.pkl"
    with open(compat_cache_path, 'wb') as f:
        pickle.dump({'sc_representation': args.sc_representation, 'sc_formula': sc_formula, 'msts': (st_msts, sv_msts, sc_msts_cal), 'mmstar': (st_mmstar, sv_mmstar, sc_mmstar_cal)}, f)
    print(f'  [cache] Compatibility alias: {compat_cache_path.name}')
    print("\n[6] Calibration thresholds with Youden's J...")
    tau_T_internal, auc_T_cal = calibrate_threshold('T', st_msts, st_mmstar)
    tau_V_internal, auc_V_cal = calibrate_threshold('V', sv_msts, sv_mmstar)
    tau_C, auc_cal = calibrate_threshold('C', sc_msts_cal, sc_mmstar_cal)
    print('\n[7] Computing response thresholds...')
    d_cautious, d_restricted = find_response_thresholds(TAU_T_FIXED, TAU_V_FIXED, tau_C)
    d_cautious_internal, d_restricted_internal = find_response_thresholds(tau_T_internal, tau_V_internal, tau_C)
    print('\n[8] Generating plots...')
    plot_calibration(sc_msts_cal, sc_mmstar_cal, tau_C, out_dir)
    config = {'formula': 'Delta = max(S_T - tau_T, S_V - tau_V, S_C - tau_C)', 'decision': 'UNSAFE if Delta >= 0', 'tau_T': TAU_T_FIXED, 'tau_V': TAU_V_FIXED, 'tau_C': round(tau_C, 4), 'calibration': {'method': "Youden's J", 'unsafe_set': 'MSTS_cal', 'safe_set': 'MMStar_cal', 'n_cal_msts': len(splits['msts_cal']), 'n_cal_mmstar': args.n_cal_mmstar, 'seed': 42, 'auc_cal': round(auc_cal, 4), 'tpr_at_tau': round(float((sc_msts_cal >= tau_C).mean()), 4), 'fpr_at_tau': round(float((sc_mmstar_cal >= tau_C).mean()), 4)}, 'response_thresholds': {'SAFE': 'Delta < 0', 'CAUTIOUS': f'0 <= Delta < {d_cautious:.4f}', 'RESTRICTED': f'{d_cautious:.4f} <= Delta < {d_restricted:.4f}', 'REFUSED': f'Delta >= {d_restricted:.4f}', 'd_cautious': round(d_cautious, 4), 'd_restricted': round(d_restricted, 4)}, 'sc_representation_mode': args.sc_representation, 'sc_representation': sc_formula, 'note': 'S_C is produced by the selected representation; when available, the historical orthogonal projection ortho_Q is applied before predict_proba.', 'internal_probes': {'tau_T': round(tau_T_internal, 4), 'tau_V': round(tau_V_internal, 4), 'auc_T_cal': round(auc_T_cal, 4), 'auc_V_cal': round(auc_V_cal, 4), 'clip_layer': CLIP_LAYER, 'd_cautious': round(d_cautious_internal, 4), 'd_restricted': round(d_restricted_internal, 4)}}
    cfg_path = out_dir / 'delta_config.json'
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f'  TAU_C CALIBRATION — REPORT')
    print(f"{'=' * 60}")
    print(f'  tau_T = {TAU_T_FIXED}  (fixed, KoalaAI)')
    print(f'  tau_V = {TAU_V_FIXED}  (fixed, Falconsai)')
    print(f'  tau_C = {tau_C:.4f}  (Youden on MSTS_cal + MMStar_cal)')
    print(f'  AUC calibration = {auc_cal:.4f}')
    print(f'  TPR MSTS_cal     = {(sc_msts_cal >= tau_C).mean():.1%}')
    print(f'  FPR MMStar_cal   = {(sc_mmstar_cal >= tau_C).mean():.1%}')
    print(f"{'=' * 60}")
    print(f'  CAUTIOUS   : 0 <= Delta < {d_cautious:.4f}')
    print(f'  RESTRICTED : {d_cautious:.4f} <= Delta < {d_restricted:.4f}')
    print(f'  REFUSED    : Delta >= {d_restricted:.4f}')
    print(f"{'=' * 60}")
    print(f'\n  Saved:')
    print(f'    {cfg_path}')
    print(f'    {splits_path}')
    print(f"    {out_dir / 'tau_c_calibration.png'}")
    print(f'\n  Next step:')
    print(f'    Evaluate on MSTS_test and MMStar_test (not used here):')
    print(f'    python evaluate_pipeline.py --probes {probes_dir} --delta {out_dir}')
if __name__ == '__main__':
    import sys
    CONFIG = {'probes': 'probes/rebuilt_v13/', 'out': 'probes/rebuilt_v13/delta', 'gpu': 3, 'msts_manifest': 'data/legacy_v13/msts_split_200_100_100.json', 'n_cal_mmstar': 400, 'recompute': True, 'sc_representation': 'aligned'}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--probes', default=CONFIG['probes'])
        parser.add_argument('--out', default=CONFIG['out'])
        parser.add_argument('--gpu', type=int, default=CONFIG['gpu'])
        parser.add_argument('--msts_manifest', default=CONFIG['msts_manifest'], help='Persistent MSTS v13 manifest: 200 train / 100 development / 100 test')
        parser.add_argument('--n_cal_mmstar', type=int, default=CONFIG['n_cal_mmstar'])
        parser.add_argument('--recompute', action='store_true', help='Recompute S_C even when the cache exists')
        parser.add_argument('--sc_representation', choices=SC_REPRESENTATION_CHOICES, default=CONFIG['sc_representation'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
