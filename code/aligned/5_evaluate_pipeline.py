#!/usr/bin/env python3
"""Evaluate the frozen v13 pipeline on the internal benchmark."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from tqdm import tqdm
from utils import load_model, dataset_tag, extract_llama_hidden_solo, extract_sc_feature_pair, extract_clip_hidden, validate_combo_representation, SC_REPRESENTATION_CHOICES
from scipy.spatial.distance import mahalanobis
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
LLAMA_LAYER = 17
MODEL_ID = 'llava-hf/llava-1.5-7b-hf'
CLIP_LAYER = 21

def compute_sc(text, image, model, processor, clf_combo, ortho_Q=None, sc_representation='aligned'):
    _, _, delta_h = extract_sc_feature_pair(
        text, image, model, processor, LLAMA_LAYER, sc_representation
    )
    dh = delta_h.reshape(1, -1)
    if ortho_Q is not None:
        dh = dh - dh @ ortho_Q @ ortho_Q.T
    return float(clf_combo.predict_proba(dh)[0, 1])

def load_and_validate_v13_splits(splits_path: Path, msts_manifest_path: Path) -> dict:
    if not splits_path.exists():
        raise FileNotFoundError(f'{splits_path} not found. Run first 4.1_calibrate_tau_c_v13.py')
    if not msts_manifest_path.exists():
        raise FileNotFoundError(f'Manifest MSTS v13 not found: {msts_manifest_path}')
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    with open(msts_manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if splits.get('protocol') != 'rebuilt_v13_msts_200_100_100':
        raise RuntimeError('eval_splits does not belong to the v13 protocol. Evaluation stops to avoid using the old MSTS split.')
    expected_dev = list(map(int, manifest['val']))
    expected_test = list(map(int, manifest['test']))
    got_dev = list(map(int, splits.get('msts_cal', [])))
    got_test = list(map(int, splits.get('msts_test', [])))
    if got_dev != expected_dev:
        raise RuntimeError("MSTS development in eval_splits_v13 does not match manifest['val'].")
    if got_test != expected_test:
        raise RuntimeError("MSTS final test in eval_splits_v13 does not match manifest['test'].")
    train = set(map(int, manifest['train']))
    dev = set(expected_dev)
    test = set(expected_test)
    if train & dev or train & test or dev & test:
        raise RuntimeError('Leakage in the MSTS v13 manifest: splits are not disjoint.')
    if (len(train), len(dev), len(test)) != (200, 100, 100):
        raise RuntimeError(f'Unexpected MSTS dimensions: train={len(train)}, dev={len(dev)}, test={len(test)}')
    print('  [split-check] v13 protocol verified')
    print('  [split-check] MSTS train=200 | development=100 | final test=100 | overlap=0')
    return splits

def load_combo_selected_on_validation(probes_dir: Path):
    combo_dir = probes_dir / 'combo'
    results_path = combo_dir / 'probe_combo_results.json'
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        candidates = [r for r in summary.get('results', []) if r.get('auc_val') is not None and np.isfinite(float(r['auc_val']))]
        if not candidates:
            raise RuntimeError(f'{results_path} does not contain valid results with auc_val.')
        best = max(candidates, key=lambda r: float(r['auc_val']))
        best_name = best['name']
        combo_path = combo_dir / f'probe_combo_{best_name}.pkl'
        if not combo_path.exists():
            raise FileNotFoundError(f'Probe selected on validation was not found: {combo_path}')
        with open(combo_path, 'rb') as f:
            combo_obj = pickle.load(f)
        print(f"  selected on validation: {combo_path.name} (AUC_val={float(best['auc_val']):.4f})")
        return (combo_path, combo_obj)
    fallback = combo_dir / 'probe_combo_1_linear_delta_h_orthogonalized.pkl'
    if not fallback.exists():
        raise FileNotFoundError(f'Missing {results_path} e missing the fallback {fallback}. Run before 3.1_train_probe_combo.py.')
    print(f'  [warn] {results_path.name} not found: using the explicit fallback {fallback.name}. For the final run, selection on AUC_val is preferred.')
    with open(fallback, 'rb') as f:
        combo_obj = pickle.load(f)
    return (fallback, combo_obj)

def load_external_classifiers(gpu: int):
    from transformers import pipeline as hf_pipeline
    device = gpu if torch.cuda.is_available() else -1
    print('  Loading KoalaAI/Text-Moderation...')
    text_pipe = hf_pipeline('text-classification', model='KoalaAI/Text-Moderation', device=device, top_k=None)
    print('  Loading Falconsai/nsfw_image_detection...')
    visual_pipe = hf_pipeline('image-classification', model='Falconsai/nsfw_image_detection', device=device)
    return (text_pipe, visual_pipe)

def score_text(text: str, pipe) -> float:
    results = pipe(text, truncation=True, max_length=512)
    if isinstance(results[0], list):
        results = results[0]
    for r in results:
        if r['label'].upper() == 'OK':
            return 1.0 - float(r['score'])
    return min(1.0, sum((r['score'] for r in results if r['label'].upper() != 'OK')))

def score_visual(image: Image.Image, pipe) -> float:
    results = pipe(image)
    for r in results:
        if r['label'].lower() in ('nsfw', 'unsafe', 'explicit'):
            return float(r['score'])
    for r in results:
        if r['label'].lower() in ('normal', 'safe'):
            return 1.0 - float(r['score'])
    return 0.0

def evaluate_dataset(dataset, indices: list, label: int, model, processor, clf_combo, ortho_Q, text_pipe, visual_pipe, clf_text, clf_visual, use_internal: bool, clip_layer: int, tau_T: float, tau_V: float, tau_C: float, d_cautious: float, d_restricted: float, text_field: str, image_field: str, name: str, ortho_Q_visual, sc_representation: str='aligned') -> dict:
    results = []
    for i, idx in enumerate(tqdm(indices, desc=f'  {name}')):
        row = dataset[idx]
        text = row.get(text_field, '') or ''
        image = row[image_field].convert('RGB')
        try:
            if use_internal:
                h_text = extract_llama_hidden_solo(text, model, processor, LLAMA_LAYER)
                h_visual = extract_clip_hidden(image, model, processor, clip_layer)
                s_t = float(clf_text.predict_proba(h_text.reshape(1, -1))[0, 1])
                hv = h_visual.reshape(1, -1)
                if ortho_Q_visual is not None:
                    hv = hv - hv @ ortho_Q_visual @ ortho_Q_visual.T
                s_v = float(clf_visual.predict_proba(hv)[0, 1])
                _, _, delta_h = extract_sc_feature_pair(
                    text, image, model, processor, LLAMA_LAYER, sc_representation
                )
                dh = delta_h.reshape(1, -1)
                if ortho_Q is not None:
                    dh = dh - dh @ ortho_Q @ ortho_Q.T
                s_c = float(clf_combo.predict_proba(dh)[0, 1])
            else:
                s_t = score_text(text, text_pipe)
                s_v = score_visual(image, visual_pipe)
                s_c = compute_sc(text, image, model, processor, clf_combo, ortho_Q, sc_representation)
        except Exception as e:
            print(f'  [warn] {name} idx={idx}: {e}')
            continue
        delta = max(s_t - tau_T, s_v - tau_V, s_c - tau_C)
        pred = 1 if delta >= 0 else 0
        if delta < 0:
            verdict = 'SAFE'
        elif delta < d_cautious:
            verdict = 'CAUTIOUS'
        elif delta < d_restricted:
            verdict = 'RESTRICTED'
        else:
            verdict = 'REFUSED'
        results.append({'idx': idx, 'S_T': round(s_t, 4), 'S_V': round(s_v, 4), 'S_C': round(s_c, 4), 'delta': round(delta, 4), 'pred': pred, 'label': label, 'verdict': verdict, 'trigger': ('T' if s_t >= tau_T else '_') + ('V' if s_v >= tau_V else '_') + ('C' if s_c >= tau_C else '_')})
        if (i + 1) % 20 == 0:
            correct = sum((r['pred'] == label for r in results))
            print(f'    [{i + 1}/{len(indices)}]  acc_so_far={correct / len(results):.1%}  S_C={s_c:.4f}  delta={delta:+.4f}')
    return results

def compute_metrics(results_unsafe: list, results_safe: list, tau_C: float) -> dict:
    sc_unsafe = np.array([r['S_C'] for r in results_unsafe])
    sc_safe = np.array([r['S_C'] for r in results_safe])
    tpr_c = float((sc_unsafe >= tau_C).mean())
    fpr_c = float((sc_safe >= tau_C).mean())
    pred_unsafe = np.array([r['pred'] for r in results_unsafe])
    pred_safe = np.array([r['pred'] for r in results_safe])
    tp = int(pred_unsafe.sum())
    fn = int((pred_unsafe == 0).sum())
    fp = int(pred_safe.sum())
    tn = int((pred_safe == 0).sum())
    tpr_full = tp / (tp + fn + 1e-09)
    fpr_full = fp / (fp + tn + 1e-09)
    prec = tp / (tp + fp + 1e-09)
    f1 = 2 * prec * tpr_full / (prec + tpr_full + 1e-09)
    scores = np.concatenate([sc_unsafe, sc_safe])
    labels = np.concatenate([np.ones(len(sc_unsafe)), np.zeros(len(sc_safe))])
    auc_sc = float(roc_auc_score(labels, scores))

    def trigger_stats(results):
        t = sum(('T' in r['trigger'] for r in results))
        v = sum(('V' in r['trigger'] for r in results))
        c = sum(('C' in r['trigger'] for r in results))
        n = len(results)
        return {'T': t / n, 'V': v / n, 'C': c / n}

    def verdict_dist(results):
        counts = {'SAFE': 0, 'CAUTIOUS': 0, 'RESTRICTED': 0, 'REFUSED': 0}
        for r in results:
            counts[r['verdict']] += 1
        n = len(results)
        return {k: {'n': v, 'pct': round(v / n * 100, 1)} for k, v in counts.items()}
    return {'probe_combo_only': {'tpr': round(tpr_c, 4), 'fpr': round(fpr_c, 4), 'auc': round(auc_sc, 4)}, 'full_pipeline': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'tpr': round(tpr_full, 4), 'fpr': round(fpr_full, 4), 'precision': round(prec, 4), 'f1': round(f1, 4)}, 'trigger_unsafe': trigger_stats(results_unsafe), 'trigger_safe': trigger_stats(results_safe), 'verdicts_unsafe': verdict_dist(results_unsafe), 'verdicts_safe': verdict_dist(results_safe), 'sc_stats': {'unsafe_mean': round(float(sc_unsafe.mean()), 4), 'unsafe_std': round(float(sc_unsafe.std()), 4), 'safe_mean': round(float(sc_safe.mean()), 4), 'safe_std': round(float(sc_safe.std()), 4)}}

def plot_evaluation(results_unsafe: list, results_safe: list, tau_C: float, d_cautious: float, d_restricted: float, out_dir: Path, suffix: str):
    sc_unsafe = np.array([r['S_C'] for r in results_unsafe])
    sc_safe = np.array([r['S_C'] for r in results_safe])
    sv_unsafe = np.array([r['S_V'] for r in results_unsafe])
    sv_safe = np.array([r['S_V'] for r in results_safe])
    dt_unsafe = np.array([r['delta'] for r in results_unsafe])
    dt_safe = np.array([r['delta'] for r in results_safe])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.hist(sc_safe, bins=30, alpha=0.6, color='steelblue', label=f'MMStar_test safe (n={len(sc_safe)})')
    ax.hist(sc_unsafe, bins=30, alpha=0.6, color='tomato', label=f'MSTS_test unsafe (n={len(sc_unsafe)})')
    ax.axvline(tau_C, color='black', ls='--', lw=2, label=f'τ_C={tau_C:.3f}')
    tpr = (sc_unsafe >= tau_C).mean()
    fpr = (sc_safe >= tau_C).mean()
    ax.text(0.05, 0.95, f'TPR={tpr:.1%}\nFPR={fpr:.1%}', transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('S_C')
    ax.set_ylabel('Count')
    ax.set_title('S_C distribution — held-out test')
    ax.legend(fontsize=9)
    ax2 = axes[1]
    scores = np.concatenate([sc_unsafe, sc_safe])
    labels = np.concatenate([np.ones(len(sc_unsafe)), np.zeros(len(sc_safe))])
    fpr_c, tpr_c, thr = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax2.plot(fpr_c, tpr_c, lw=2, color='steelblue', label=f'AUC={auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    idx = np.argmin(np.abs(thr - tau_C))
    ax2.scatter(fpr_c[idx], tpr_c[idx], color='tomato', s=100, zorder=5, label=f'τ_C={tau_C:.3f}')
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title('ROC — probe_combo on held-out test')
    ax2.legend()
    ax3 = axes[2]
    order = ['SAFE', 'CAUTIOUS', 'RESTRICTED', 'REFUSED']
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    x = np.arange(len(order))
    width = 0.35
    counts_unsafe = [sum((r['verdict'] == v for r in results_unsafe)) for v in order]
    counts_safe = [sum((r['verdict'] == v for r in results_safe)) for v in order]
    bars1 = ax3.bar(x - width / 2, counts_unsafe, width, color=colors, alpha=0.8, label=f'MSTS unsafe (n={len(sc_unsafe)})')
    bars2 = ax3.bar(x + width / 2, counts_safe, width, color=colors, alpha=0.4, label=f'MM-Star safe (n={len(sc_safe)})', edgecolor='black', linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(order)
    ax3.set_ylabel('Count')
    ax3.set_title('Full-pipeline decisions — held-out test')
    ax3.legend(fontsize=9)
    plt.suptitle(f'Evaluation final held-out — τ_C={tau_C:.4f}  (calibrated on MSTS_cal + MMStar_cal)', fontsize=13)
    plt.tight_layout()
    p = out_dir / f'evaluation_plot_{suffix}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {p}')

def run(args):
    from datasets import load_dataset
    probes_dir = Path(args.probes)
    delta_dir = Path(args.delta)
    out_dir = delta_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_suffix = 'v13_internal' if args.use_internal_probes else 'v13_external'
    print('\n[1] Loading configuration...')
    cfg_path = delta_dir / 'delta_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'{cfg_path} not found. Run first calibrate_tau_c.py')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    config_mode = cfg.get('sc_representation_mode')
    if config_mode is not None and config_mode != args.sc_representation:
        raise ValueError(f'delta_config S_C representation mismatch: {config_mode} != {args.sc_representation}')
    tau_C = cfg['tau_C']
    d_cautious = cfg['response_thresholds']['d_cautious']
    d_restricted = cfg['response_thresholds']['d_restricted']
    if args.use_internal_probes:
        tau_T = cfg['internal_probes']['tau_T']
        tau_V = cfg['internal_probes']['tau_V']
        d_cautious = cfg['internal_probes']['d_cautious']
        d_restricted = cfg['internal_probes']['d_restricted']
    else:
        tau_T = cfg['tau_T']
        tau_V = cfg['tau_V']
    print(f"  mode: {('INTERNAL probes' if args.use_internal_probes else 'EXTERNAL (KoalaAI/Falconsai)')}")
    print(f'  tau_T={tau_T}  tau_V={tau_V}  tau_C={tau_C}')
    print('\n[2] Loading split v13...')
    splits_path = delta_dir / 'eval_splits_v13.json'
    splits = load_and_validate_v13_splits(splits_path=splits_path, msts_manifest_path=Path(args.msts_manifest))
    print(f"  MSTS_test:   {len(splits['msts_test'])} sample (final frozen test)")
    print(f"  MMStar_test: {len(splits['mmstar_test'])} sample")
    print('\n[3] Loading probe_combo...')
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
    print(f'  {combo_path.name}')
    print(f'  ortho_Q available: {ortho_Q is not None}')
    print(f'  {combo_path.name}')
    print('\n[4] Loading models...')
    model, processor = load_model(gpu=args.gpu)
    ortho_Q_visual = None
    clf_text = clf_visual = text_pipe = visual_pipe = None
    if args.use_internal_probes:
        print('  Loading probe_text/probe_visual internal...')
        with open(probes_dir / 'probe_text.pkl', 'rb') as f:
            clf_text = pickle.load(f)
        with open(probes_dir / 'probe_visual_orthogonalized.pkl', 'rb') as f:
            obj_visual = pickle.load(f)
        clf_visual, ortho_Q_visual = (obj_visual['clf'], obj_visual.get('ortho_Q'))
    else:
        text_pipe, visual_pipe = load_external_classifiers(gpu=args.gpu)
    print('\n[5] Loading dataset...')
    msts_ds = load_dataset('felfri/MSTS', split='english')
    mmstar_ds = load_dataset('Lin-Chen/MMStar', split='val')
    cache_path = out_dir / f'eval_results_cache_{mode_suffix}.pkl'
    compat_mode_suffix = 'internal' if args.use_internal_probes else 'external'
    compat_cache_path = out_dir / f'eval_results_cache_{compat_mode_suffix}.pkl'
    expected_thresholds = {
        'tau_T': float(tau_T),
        'tau_V': float(tau_V),
        'tau_C': float(tau_C),
        'd_cautious': float(d_cautious),
        'd_restricted': float(d_restricted),
    }
    use_cache = cache_path.exists() and (not args.recompute)
    if use_cache:
        print(f'\n[6] Loading results from the cache {cache_path.name}...')
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('sc_representation') != args.sc_representation:
            raise ValueError(f"Evaluation cache representation mismatch: {cached.get('sc_representation')} != {args.sc_representation}")
        cached_thresholds = cached.get('thresholds')
        thresholds_match = isinstance(cached_thresholds, dict) and all(
            key in cached_thresholds and np.isclose(float(cached_thresholds[key]), value)
            for key, value in expected_thresholds.items()
        )
        if thresholds_match:
            results_unsafe = cached['results_unsafe']
            results_safe = cached['results_safe']
        else:
            print('  [cache] Threshold metadata is missing or stale; recomputing test decisions.')
            use_cache = False
    if not use_cache:
        print(f"\n[6a] Evaluation MSTS_test ({len(splits['msts_test'])} sample)...")
        results_unsafe = evaluate_dataset(msts_ds, splits['msts_test'], label=1, model=model, processor=processor, clf_combo=clf_combo, text_pipe=text_pipe, visual_pipe=visual_pipe, clf_text=clf_text, clf_visual=clf_visual, use_internal=args.use_internal_probes, clip_layer=CLIP_LAYER, tau_T=tau_T, tau_V=tau_V, tau_C=tau_C, d_cautious=d_cautious, d_restricted=d_restricted, text_field='prompt_text', image_field='unsafe_image', name='MSTS_test', ortho_Q=ortho_Q, ortho_Q_visual=ortho_Q_visual, sc_representation=args.sc_representation)
        print(f"\n[6b] Evaluation MMStar_test ({len(splits['mmstar_test'])} sample)...")
        results_safe = evaluate_dataset(mmstar_ds, splits['mmstar_test'], label=0, model=model, processor=processor, clf_combo=clf_combo, text_pipe=text_pipe, visual_pipe=visual_pipe, clf_text=clf_text, clf_visual=clf_visual, use_internal=args.use_internal_probes, clip_layer=CLIP_LAYER, tau_T=tau_T, tau_V=tau_V, tau_C=tau_C, d_cautious=d_cautious, d_restricted=d_restricted, text_field='question', image_field='image', name='MMStar_test', ortho_Q=ortho_Q, ortho_Q_visual=ortho_Q_visual, sc_representation=args.sc_representation)
    cache_payload = {
        'sc_representation': args.sc_representation,
        'thresholds': expected_thresholds,
        'results_unsafe': results_unsafe,
        'results_safe': results_safe,
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_payload, f)
    with open(compat_cache_path, 'wb') as f:
        pickle.dump(cache_payload, f)
    print(f'  [cache] Saved {cache_path.name}')
    print(f'  [cache] Compatibility alias: {compat_cache_path.name}')
    print('\n[7] Computing metrics...')
    metrics = compute_metrics(results_unsafe, results_safe, tau_C)
    print('\n[8] Generating plots...')
    plot_evaluation(results_unsafe, results_safe, tau_C, d_cautious, d_restricted, out_dir, mode_suffix)
    output = {'protocol': 'rebuilt_v13_msts_200_100_100', 'split_file': str(splits_path), 'msts_manifest': str(args.msts_manifest), 'probe_combo': combo_path.name, 'sc_representation': args.sc_representation, 'config': {'tau_T': tau_T, 'tau_V': tau_V, 'tau_C': tau_C, 'd_cautious': d_cautious, 'd_restricted': d_restricted}, 'metrics': metrics, 'n_samples': {'msts_test': len(results_unsafe), 'mmstar_test': len(results_safe)}}
    res_path = out_dir / f'evaluation_results_{mode_suffix}.json'
    with open(res_path, 'w') as f:
        json.dump(output, f, indent=2)
    m = metrics
    print(f"\n{'=' * 60}")
    print(f'  FINAL EVALUATION — HELD-OUT TEST v13 ({mode_suffix.upper()})')
    print(f"{'=' * 60}")
    print(f'  probe_combo only:')
    print(f"    AUC  = {m['probe_combo_only']['auc']:.4f}")
    print(f"    TPR  = {m['probe_combo_only']['tpr']:.1%}  (MSTS_test detected)")
    print(f"    FPR  = {m['probe_combo_only']['fpr']:.1%}  (MMStar_test blocked)")
    print(f'\n  pipeline complete (S_T + S_V + S_C):')
    print(f"    TP={m['full_pipeline']['tp']}  FP={m['full_pipeline']['fp']}  FN={m['full_pipeline']['fn']}  TN={m['full_pipeline']['tn']}")
    print(f"    TPR={m['full_pipeline']['tpr']:.1%}  FPR={m['full_pipeline']['fpr']:.1%}  F1={m['full_pipeline']['f1']:.4f}")
    print(f'\n  trigger on MSTS_test:')
    t = m['trigger_unsafe']
    u = m['trigger_safe']
    print(f"   trigger unsafe={t['T']:.1%}  visual={t['V']:.1%}  combo={t['C']:.1%}")
    print(f"    safe={u['T']:.1%}  visual={u['V']:.1%}  combo={u['C']:.1%}")
    print(f"{'=' * 60}")
    print(f'\n  Saved:')
    print(f'    {res_path}')
    print(f"    {out_dir / f'evaluation_plot_{mode_suffix}.png'}")
if __name__ == '__main__':
    import sys
    CONFIG = {'probes': 'probes/rebuilt_v13/', 'delta': 'probes/rebuilt_v13/delta', 'msts_manifest': 'data/legacy_v13/msts_split_200_100_100.json', 'gpu': 3, 'recompute': True, 'use_internal_probes': True, 'sc_representation': 'aligned'}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--probes', default=CONFIG['probes'])
        parser.add_argument('--delta', default=CONFIG['delta'])
        parser.add_argument('--msts_manifest', default=CONFIG['msts_manifest'])
        parser.add_argument('--gpu', type=int, default=CONFIG['gpu'])
        parser.add_argument('--recompute', action='store_true', default=CONFIG['recompute'])
        parser.add_argument('--use_internal_probes', action='store_true', default=CONFIG['use_internal_probes'])
        parser.add_argument('--sc_representation', choices=SC_REPRESENTATION_CHOICES, default=CONFIG['sc_representation'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
