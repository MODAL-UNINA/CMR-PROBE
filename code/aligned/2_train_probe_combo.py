#!/usr/bin/env python3
"""Train and select the multimodal combination probe."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utils import load_model, dataset_tag, extract_sc_feature_pair, sc_representation_formula, SC_REPRESENTATION_CHOICES, compute_sample_weights, collapse_source, resolve_local_image_path
import sys
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
LLAMA_LAYER = 17
MODEL_ID = 'llava-hf/llava-1.5-7b-hf'
SWEEP_LAYERS = [6, 8, 10, 12, 15, 17, 20, 24]
_model = None
_processor = None

def normalize_delta_h(delta_h: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(delta_h)
    if norm > 1e-08:
        return delta_h / norm
    return delta_h

def extract_delta_h(feats_unimodal: list[dict], json_path: Path, base_dir: Path, cache_path: Path, model, processor, layer: int, sc_representation: str) -> list[dict]:
    cache_key = cache_path.with_suffix(f'.layer{layer}.{sc_representation}.pkl')
    if cache_key.exists():
        print(f'  [cache] Loading {cache_key.name}')
        with open(cache_key, 'rb') as f:
            return pickle.load(f)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw_samples = {s['id']: s for s in (data['samples'] if isinstance(data, dict) else data)}
    feats_out = []
    for feat in tqdm(feats_unimodal, desc=f'  δh layer={layer} {json_path.name}'):
        sid = feat['id']
        if sid not in raw_samples:
            continue
        s = raw_samples[sid]
        img_path = resolve_local_image_path(s['local_image_path'], base_dir)
        if not img_path.exists():
            print(f'  [warn] image not found: {img_path}')
            continue
        try:
            image = Image.open(img_path).convert('RGB')
            h_solo, h_combo, delta_h = extract_sc_feature_pair(
                s['text'], image, model, processor, layer, sc_representation
            )
        except Exception as e:
            print(f'  [warn] S_C extraction ({sc_representation}) failed {sid}: {e}')
            continue
        feats_out.append({'id': sid, 'category': feat['category'], 'label_text': feat['label_text'], 'label_img': feat['label_img'], 'label_combo': feat['label_combo'], 'source': s.get('source', 'baseline'), 'h_solo': h_solo, 'h_combo': h_combo, 'delta_h': delta_h})
    with open(cache_key, 'wb') as f:
        pickle.dump(feats_out, f)
    print(f'  [cache] Saved {cache_key.name} ({len(feats_out)} samples)')
    return feats_out

def build_XY(feats: list[dict], mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == 'delta':
        X = np.stack([f['delta_h'] for f in feats])
    elif mode == 'pair':
        X = np.concatenate([np.stack([f['h_solo'] for f in feats]), np.stack([f['h_combo'] for f in feats])], axis=1)
    else:
        raise ValueError(f'unknown mode: {mode}')
    y = np.array([f['label_combo'] for f in feats])
    return (X, y)

def visual_hidden_norm_proxy(h_visual: np.ndarray) -> float:
    return float(np.linalg.norm(h_visual))

def fit_sentinel_and_get_projector(X_train: np.ndarray, sources_train: list[str], n_components: int=None) -> tuple:
    le = LabelEncoder()
    y_source = le.fit_transform(sources_train)
    n_classes = len(le.classes_)
    if n_classes < 2:
        print('  [sentinel] Only one source in the group: no direction to remove.')
        return (float('nan'), lambda X: X)
    sentinel_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='lbfgs', random_state=42)
    sentinel_clf.fit(X_train, y_source)
    if n_classes == 2:
        proba = sentinel_clf.predict_proba(X_train)[:, 1]
        auc_sentinel = roc_auc_score(y_source, proba)
    else:
        from sklearn.metrics import accuracy_score as acc_fn
        auc_sentinel = acc_fn(y_source, sentinel_clf.predict(X_train))
    print(f'  [sentinel] Sources: {list(le.classes_)}')
    print(f'  [sentinel] Source-predictability score (AUC o ACC if >2 classes) = {auc_sentinel:.4f}')
    W = sentinel_clf.coef_
    if W.ndim == 1:
        W = W.reshape(1, -1)
    Q, _ = np.linalg.qr(W.T)
    k = min(n_components, Q.shape[1]) if n_components else Q.shape[1]
    Q = Q[:, :k]

    def projector_fn(X):
        return X - X @ Q @ Q.T
    return (float(auc_sentinel), projector_fn, Q)

def fit_style_sentinel_and_get_projector(X_train: np.ndarray, aux_train: list, style_metric: str='length') -> np.ndarray:
    if style_metric == 'length':
        y_spurious = np.array([len(t.split()) for t in aux_train])
        y_spurious = (y_spurious - y_spurious.mean()) / (y_spurious.std() + 1e-08)
        model = Ridge(alpha=1.0)
    elif style_metric == 'complexity':
        y_spurious = np.array([1 if '\n' in t else 0 for t in aux_train])
        model = LogisticRegression(solver='liblinear', C=0.1)
    elif style_metric in ('image_complexity', 'visual_hidden_norm'):
        y_spurious = np.array(aux_train, dtype=np.float64)
        valid = ~np.isnan(y_spurious)
        if valid.sum() < len(y_spurious):
            print(f'  [warn] {(~valid).sum()} NaN values in {style_metric}, excluded')
        y_spurious = (y_spurious[valid] - np.nanmean(y_spurious)) / (np.nanstd(y_spurious) + 1e-08)
        X_train = X_train[valid]
        model = Ridge(alpha=1.0)
    else:
        raise ValueError('Unsupported style metric')
    print(f'[Concept Erasure] Training the bias sentinel: {style_metric}')
    model.fit(X_train, y_spurious)
    w = model.coef_.flatten()
    w_norm = np.linalg.norm(w)
    if w_norm > 1e-08:
        Q = (w / w_norm).reshape(-1, 1)
    else:
        Q = None
    return Q

def fit_visual_subspace_sentinel_and_get_projector(X_train: np.ndarray, h_visual_train: np.ndarray, n_components: int=5) -> np.ndarray | None:
    from sklearn.decomposition import PCA
    n_components = min(n_components, h_visual_train.shape[0] - 1, h_visual_train.shape[1])
    if n_components < 1:
        return None
    pca_visual = PCA(n_components=n_components, random_state=42)
    targets = pca_visual.fit_transform(h_visual_train)
    var_exp = pca_visual.explained_variance_ratio_.sum()
    print(f'  [visual subspace] PCA on h_visual: {n_components} components, explained variance={var_exp:.3f}')
    model = Ridge(alpha=1.0)
    model.fit(X_train, targets)
    W = model.coef_
    if W.ndim == 1:
        W = W.reshape(1, -1)
    Q, _ = np.linalg.qr(W.T)
    return Q[:, :n_components]

def fit_visual_style_sentinel_per_category(dh_train, img_paths_by_id, style_metric='image_complexity'):
    Q_list = []
    for cat in sorted(set((f['category'] for f in dh_train))):
        feats_cat = [f for f in dh_train if f['category'] == cat]
        X_cat = np.stack([f['delta_h'] for f in feats_cat])
        complexities_cat = [img_paths_by_id.get(f['id'], float('nan')) for f in feats_cat]
        Q = fit_style_sentinel_and_get_projector(X_cat, complexities_cat, style_metric)
        if Q is not None:
            Q_list.append(Q)
    return Q_list

def fit_sentinel_per_category(dh_train: list[dict]) -> dict:
    projectors = {}
    categories = sorted(set((f['category'] for f in dh_train)))
    for cat in categories:
        feats_cat = [f for f in dh_train if f['category'] == cat]
        sources_cat = [collapse_source(f['source']) for f in feats_cat]
        n_sources = len(set(sources_cat))
        if n_sources < 2:
            print(f'  [sentinel cat={cat}] Only one source ({sources_cat[0]}): no correction is required.')
            projectors[cat] = (float('nan'), None)
            continue
        X_cat = np.stack([f['delta_h'] for f in feats_cat])
        auc, _, Q = fit_sentinel_and_get_projector(X_cat, sources_cat)
        print(f'  [sentinel cat={cat}] sources={set(sources_cat)}  AUC/ACC pre={auc:.4f}')
        projectors[cat] = (auc, Q)
    return projectors

def fit_style_sentinel_per_category(dh_train, texts_by_id, style_metric='length'):
    Q_list = []
    for cat in sorted(set((f['category'] for f in dh_train))):
        feats_cat = [f for f in dh_train if f['category'] == cat]
        X_cat = np.stack([f['delta_h'] for f in feats_cat])
        texts_cat = [texts_by_id[f['id']] for f in feats_cat]
        Q = fit_style_sentinel_and_get_projector(X_cat, texts_cat, style_metric)
        if Q is not None:
            Q_list.append(Q)
    return Q_list

def fit_visual_subspace_sentinel_per_category(dh_train, h_visual_by_id, n_components: int=5):
    Q_list = []
    for cat in sorted(set((f['category'] for f in dh_train))):
        feats_cat = [f for f in dh_train if f['category'] == cat]
        X_cat = np.stack([f['delta_h'] for f in feats_cat])
        h_visual_cat = np.stack([h_visual_by_id[f['id']] for f in feats_cat if f['id'] in h_visual_by_id])
        if h_visual_cat.shape[0] != X_cat.shape[0]:
            print(f'  [warn] cat={cat}: mismatch h_visual available, skip')
            continue
        Q = fit_visual_subspace_sentinel_and_get_projector(X_cat, h_visual_cat, n_components)
        if Q is not None:
            Q_list.append(Q)
    return Q_list

def build_union_projector_from_list(Q_list: list) -> np.ndarray | None:
    Q_list = [Q for Q in Q_list if Q is not None]
    if not Q_list:
        return None
    Q_stack = np.concatenate(Q_list, axis=1)
    Q_union, _ = np.linalg.qr(Q_stack)
    return Q_union[:, :Q_stack.shape[1]]

def build_union_projector(projectors: dict) -> np.ndarray | None:
    Q_list = [Q for _, Q in projectors.values() if Q is not None]
    if not Q_list:
        return None
    Q_stack = np.concatenate(Q_list, axis=1)
    Q_union, _ = np.linalg.qr(Q_stack)
    return Q_union[:, :Q_stack.shape[1]]

def apply_projector(X: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    if Q is None:
        return X
    return X - X @ Q @ Q.T

def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, clf_raw, name: str, scaler=None, sample_weight=None) -> dict:
    print(f'\n  [{name}]')
    print(f'    train={len(y_tr)} (pos={y_tr.sum()}) | val={len(y_va)} (pos={y_va.sum()}) | test={len(y_te)} (pos={y_te.sum()})')
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)
    clf_raw.fit(X_tr, y_tr, sample_weight=sample_weight)
    clf_cal = CalibratedClassifierCV(clf_raw, cv='prefit', method='sigmoid')
    clf_cal.fit(X_va, y_va)
    p_va = clf_cal.predict_proba(X_va)[:, 1]
    auc_v = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float('nan')
    acc_v = accuracy_score(y_va, clf_cal.predict(X_va))
    print(f'    AUC_val={auc_v:.4f}  ACC_val={acc_v:.4f}')
    p_te = clf_cal.predict_proba(X_te)[:, 1]
    auc_t = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float('nan')
    acc_t = accuracy_score(y_te, clf_cal.predict(X_te))
    fpr, tpr, thr = roc_curve(y_te, p_te)
    tau = float(thr[np.argmax(tpr - fpr)])
    print(f'    AUC_test={auc_t:.4f}  ACC_test={acc_t:.4f}  τ_opt={tau:.4f}')
    return {'name': name, 'auc_val': round(float(auc_v), 4), 'acc_val': round(float(acc_v), 4), 'auc_test': round(float(auc_t), 4), 'acc_test': round(float(acc_t), 4), 'tau_opt': round(tau, 4), 'clf': clf_cal, 'scaler': scaler, 'p_val': p_va, 'p_test': p_te, 'y_val': y_va, 'y_test': y_te}

def plot_results(results: list[dict], out_dir: Path):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]
    for i, res in enumerate(results):
        ax_roc, ax_dist = axes[i]
        fpr, tpr, _ = roc_curve(res['y_test'], res['p_test'])
        ax_roc.plot(fpr, tpr, lw=2, color='steelblue', label=f"AUC={res['auc_test']:.4f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.axvline(0.1, color='gray', ls=':', lw=1, label='FPR=0.1')
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.set_title(f"{res['name']} — ROC curve (test)")
        ax_roc.legend()
        ax_dist.hist(res['p_test'][res['y_test'] == 0], bins=25, alpha=0.6, color='tomato', label='Safe (combo=0)')
        ax_dist.hist(res['p_test'][res['y_test'] == 1], bins=25, alpha=0.6, color='steelblue', label='Unsafe (combo=1)')
        ax_dist.axvline(res['tau_opt'], color='black', lw=1.5, ls='--', label=f"τ_opt={res['tau_opt']:.3f}")
        ax_dist.set_xlabel('Score S_C')
        ax_dist.set_ylabel('Count')
        ax_dist.set_title(f"{res['name']} — score distribution")
        ax_dist.legend()
    plt.tight_layout()
    p = out_dir / 'probe_combo_results.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {p}')

def plot_layer_sweep(sweep_results: dict, out_dir: Path):
    layers = sorted(sweep_results.keys())
    aucs = [sweep_results[l] for l in layers]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, aucs, 'o-', color='steelblue', lw=2)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='Random')
    for l, a in zip(layers, aucs):
        ax.annotate(f'{a:.3f}', (l, a), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel('LLaMA layer')
    ax.set_ylabel('AUC_val')
    ax.set_title('Layer sweep — linear probe on δh (label_combo)')
    ax.set_xticks(layers)
    ax.legend()
    plt.tight_layout()
    p = out_dir / 'layer_sweep.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f'  Layer sweep saved: {p}')

def run_pipeline(args):
    data_dir = Path(args.dataset)
    base_dir = Path(args.base_dir) if args.base_dir else data_dir
    tag = args.tag if args.tag else dataset_tag(args.dataset)
    probes_dir = Path(args.probes) / tag
    out_dir = probes_dir / 'combo'
    cache_dir = probes_dir / 'feature_cache'
    out_dir.mkdir(parents=True, exist_ok=True)
    sc_formula = sc_representation_formula(args.sc_representation)
    print(f'\n[S_C] representation={args.sc_representation}: {sc_formula}')
    print('\n[1] Loading unimodal cache...')
    feats = {}
    for split in ('train', 'val', 'test'):
        p = cache_dir / f'{split}.pkl'
        if not p.exists():
            raise FileNotFoundError(f'Cache {p} not found. Run first train_probes.py.')
        with open(p, 'rb') as f:
            feats[split] = pickle.load(f)
    print(f"  train={len(feats['train'])} | val={len(feats['val'])} | test={len(feats['test'])}")
    if not args.train_only:
        model, processor = load_model(args.gpu)
    else:
        model = processor = None
    if args.sweep_layers:
        print('\n[SWEEP] Layer sweep for the linear δh probe...')
        sweep_results = {}
        for layer in SWEEP_LAYERS:
            print(f'\n  Layer {layer}:')
            dh_tr = extract_delta_h(feats['train'], data_dir / 'probe_train.json', base_dir, out_dir / 'train_dh', model, processor, layer, args.sc_representation)
            dh_va = extract_delta_h(feats['val'], data_dir / 'probe_val.json', base_dir, out_dir / 'val_dh', model, processor, layer, args.sc_representation)
            X_tr, y_tr = build_XY(dh_tr, 'delta')
            X_va, y_va = build_XY(dh_va, 'delta')
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='lbfgs', random_state=42)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)
            clf.fit(X_tr_s, y_tr)
            p_va = clf.predict_proba(X_va_s)[:, 1]
            auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else float('nan')
            sweep_results[layer] = round(float(auc), 4)
            print(f'    AUC_val={auc:.4f}')
        best = max(sweep_results, key=sweep_results.get)
        print(f'\n  Best layer: {best}  (AUC_val={sweep_results[best]:.4f})')
        print(f'  Used LLAMA_LAYER={args.llama_layer} as default.')
        plot_layer_sweep(sweep_results, out_dir)
        sweep_path = out_dir / 'layer_sweep.json'
        with open(sweep_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
    print(f'\n[2] Extraction δh (layer={args.llama_layer})...')
    layer = args.llama_layer
    if not args.train_only:
        dh_train = extract_delta_h(feats['train'], data_dir / 'probe_train.json', base_dir, out_dir / 'train_dh', model, processor, layer, args.sc_representation)
        dh_val = extract_delta_h(feats['val'], data_dir / 'probe_val.json', base_dir, out_dir / 'val_dh', model, processor, layer, args.sc_representation)
        dh_test = extract_delta_h(feats['test'], data_dir / 'probe_test.json', base_dir, out_dir / 'test_dh', model, processor, layer, args.sc_representation)
    else:
        print('  [train_only] Loading δh from the cache...')
        dh = {}
        for split in ('train', 'val', 'test'):
            p = out_dir / f'{split}_dh.layer{layer}.{args.sc_representation}.pkl'
            if not p.exists():
                raise FileNotFoundError(f'Cache δh {p} not found.')
            with open(p, 'rb') as f:
                dh[split] = pickle.load(f)
        dh_train, dh_val, dh_test = (dh['train'], dh['val'], dh['test'])
    print(f'  Samples δh: train={len(dh_train)} | val={len(dh_val)} | test={len(dh_test)}')
    for name, dh in [('train', dh_train), ('val', dh_val), ('test', dh_test)]:
        y = np.array([f['label_combo'] for f in dh])
        print(f'  {name}: label_combo pos={y.sum()} / {len(y)} ({100 * y.mean():.1f}%)')
    print('\n[3] Training three probe_combo approaches...')
    print('\n── Sentinel probe: source + style, by category ──')
    sources_tr = [collapse_source(f['source']) for f in dh_train]
    X_tr1, y_tr = build_XY(dh_train, 'delta')
    X_va1, y_va = build_XY(dh_val, 'delta')
    X_te1, y_te = build_XY(dh_test, 'delta')
    with open(data_dir / 'probe_train.json', 'r', encoding='utf-8') as f:
        raw_train = json.load(f)['samples']
    texts_by_id = {s['id']: s['text'] for s in raw_train}
    print('\n── Computing the visual proxy from h_visual (CLIP hidden state) ──')
    visual_norm_by_id = {f['id']: visual_hidden_norm_proxy(f['h_visual']) for f in feats['train']}
    h_visual_by_id = {f['id']: f['h_visual'] for f in feats['train']}
    projectors = fit_sentinel_per_category(dh_train)
    Q_list_source = [Q for _, Q in projectors.values() if Q is not None]
    Q_list_style = fit_style_sentinel_per_category(dh_train, texts_by_id, 'length')
    Q_list_visual_style = fit_visual_style_sentinel_per_category(dh_train, visual_norm_by_id, 'visual_hidden_norm')
    Q_list_visual_subspace = fit_visual_subspace_sentinel_per_category(dh_train, h_visual_by_id, n_components=5)
    Q_union = build_union_projector_from_list(Q_list_source + Q_list_style + Q_list_visual_style + Q_list_visual_subspace)
    X_tr1_clean = apply_projector(X_tr1, Q_union)
    X_va1_clean = apply_projector(X_va1, Q_union)
    X_te1_clean = apply_projector(X_te1, Q_union)
    sample_w_tr = compute_sample_weights(y_tr, sources_tr)
    print('\n── Check post-orthogonalization ──')
    dh_train_clean = [dict(f, delta_h=X_tr1_clean[i]) for i, f in enumerate(dh_train)]
    projectors_post = fit_sentinel_per_category(dh_train_clean)
    for cat in projectors:
        auc_pre, _ = projectors[cat]
        auc_post, _ = projectors_post[cat]
        print(f'  cat={cat}:  pre={auc_pre:.4f}  →  post={auc_post:.4f}  (target ≈ chance level)')
    print('\n── Approach 1: logistic regression on δh ──')
    X_tr1, y_tr = build_XY(dh_train, 'delta')
    X_va1, y_va = build_XY(dh_val, 'delta')
    X_te1, y_te = build_XY(dh_test, 'delta')
    sc1 = StandardScaler()
    res1 = train_and_eval(X_tr1_clean, y_tr, X_va1_clean, y_va, X_te1_clean, y_te, LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='lbfgs', random_state=42), name='1_linear_delta_h_orthogonalized', scaler=None, sample_weight=sample_w_tr)
    print('\n── Approach 2: RBF SVM on δh ──')
    from sklearn.decomposition import PCA
    sc2 = StandardScaler()
    pca2 = PCA(n_components=32, random_state=42)
    X_tr2_s = sc2.fit_transform(X_tr1)
    X_tr2_p = pca2.fit_transform(X_tr2_s)
    X_va2_p = pca2.transform(sc2.transform(X_va1))
    X_te2_p = pca2.transform(sc2.transform(X_te1))
    var_exp = pca2.explained_variance_ratio_.sum()
    print(f'  PCA 128 components: explained variance = {var_exp:.3f}')
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm_clf.fit(X_tr2_p, y_tr)
    svm_cal = CalibratedClassifierCV(svm_clf, cv='prefit', method='sigmoid')
    svm_cal.fit(X_va2_p, y_va)
    p_va2 = svm_cal.predict_proba(X_va2_p)[:, 1]
    p_te2 = svm_cal.predict_proba(X_te2_p)[:, 1]
    auc_v2 = roc_auc_score(y_va, p_va2) if len(np.unique(y_va)) > 1 else float('nan')
    auc_t2 = roc_auc_score(y_te, p_te2) if len(np.unique(y_te)) > 1 else float('nan')
    acc_t2 = accuracy_score(y_te, svm_cal.predict(X_te2_p))
    fpr2, tpr2, thr2 = roc_curve(y_te, p_te2)
    tau2 = float(thr2[np.argmax(tpr2 - fpr2)])
    print(f'    AUC_val={auc_v2:.4f}')
    print(f'    AUC_test={auc_t2:.4f}  ACC_test={acc_t2:.4f}  τ_opt={tau2:.4f}')
    res2 = {'name': '2_svm_rbf_delta_h', 'auc_val': round(float(auc_v2), 4), 'acc_val': 0.0, 'auc_test': round(float(auc_t2), 4), 'acc_test': round(float(acc_t2), 4), 'tau_opt': round(tau2, 4), 'clf': svm_cal, 'scaler': sc2, 'pca': pca2, 'p_val': p_va2, 'p_test': p_te2, 'y_val': y_va, 'y_test': y_te}
    print('\n── Approach 3: MLP on concat(h_solo, h_combo) ──')
    X_tr3, _ = build_XY(dh_train, 'pair')
    X_va3, _ = build_XY(dh_val, 'pair')
    X_te3, _ = build_XY(dh_test, 'pair')
    sc3 = StandardScaler()
    pca3 = PCA(n_components=256, random_state=42)
    X_tr3_p = pca3.fit_transform(sc3.fit_transform(X_tr3))
    X_va3_p = pca3.transform(sc3.transform(X_va3))
    X_te3_p = pca3.transform(sc3.transform(X_te3))
    var_exp3 = pca3.explained_variance_ratio_.sum()
    print(f'  PCA 256 components: explained variance = {var_exp3:.3f}')
    sc4 = None
    res3 = train_and_eval(X_tr3_p, y_tr, X_va3_p, y_va, X_te3_p, y_te, MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu', solver='adam', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42), name='3_mlp_pair', scaler=sc4)
    print('\n[4] Saving models...')
    all_results = [res1, res2, res3]
    for res in all_results:
        save_obj = {
            'clf': res['clf'],
            'sc_representation': args.sc_representation,
            'representation': sc_formula,
            'language_layer': int(layer),
        }
        if 'scaler' in res and res['scaler'] is not None:
            save_obj['scaler'] = res['scaler']
        if 'pca' in res:
            save_obj['pca'] = res['pca']
        if res['name'] == '1_linear_delta_h_orthogonalized':
            save_obj['ortho_Q'] = Q_union
        p = out_dir / f"probe_combo_{res['name']}.pkl"
        with open(p, 'wb') as f:
            pickle.dump(save_obj, f)
        print(f'  Saved: {p}')
    print('\n[5] Generating plots...')
    plot_results(all_results, out_dir)
    print(f"\n{'=' * 60}")
    print(f'  LAYER={layer}   MODEL={MODEL_ID}')
    print(f"{'=' * 60}")
    print(f"  {'Approach':30s}  {'AUC_val':>8s}  {'AUC_test':>9s}  {'τ_opt':>7s}")
    print(f"  {'-' * 56}")
    for res in all_results:
        print(f"  {res['name']:30s}  {res['auc_val']:>8.4f}  {res['auc_test']:>9.4f}  {res['tau_opt']:>7.4f}")
    print(f"{'=' * 60}")
    best = max(all_results, key=lambda r: r['auc_val'])
    print(f"\n  Best approach: {best['name']}")
    print(f"  AUC_test = {best['auc_test']:.4f}")
    if best['auc_test'] < 0.6:
        print('\n  ⚠  AUC < 0.60: δh does not contain the signal combo.')
        print('     Possible causes:')
        print('     - The layer selected is incorrect → try --sweep_layers')
        print('     - Too few combination samples in the dataset')
        print('     - Combination risk is not detectable in the hidden states')
    elif best['auc_test'] < 0.75:
        print('\n  ~ Moderate AUC: the signal is weak but present.')
        print('    Try --sweep_layers to find the best layer.')
    else:
        print('\n  ✓ Good AUC: probe_combo can detect combination risk.')
    results_json = {'config': {'layer': layer, 'model_id': MODEL_ID, 'sc_representation': args.sc_representation, 'representation': sc_formula}, 'results': [{k: v for k, v in r.items() if k not in ('clf', 'scaler', 'pca', 'p_val', 'p_test', 'y_val', 'y_test')} for r in all_results]}
    res_path = out_dir / 'probe_combo_results.json'
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    print(f'\n  Results in: {res_path}')
    print(f'\nNext step:')
    print(f'  Use the best probe to compute S_C and construct Δ complete:')
    print(f'  Δ = 0.5·(S_T_norm + S_V_norm) + β·S_C_norm')
if __name__ == '__main__':
    CONFIG = {'dataset': 'data/legacy_v13/', 'base_dir': '.', 'probes': 'probes/', 'gpu': 3, 'train_only': False, 'sweep_layers': False, 'llama_layer': LLAMA_LAYER, 'tag': 'rebuilt_v13', 'sc_representation': 'aligned'}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default=CONFIG['dataset'])
        parser.add_argument('--base_dir', default=CONFIG['base_dir'])
        parser.add_argument('--probes', default=CONFIG['probes'])
        parser.add_argument('--gpu', type=int, default=CONFIG['gpu'])
        parser.add_argument('--train_only', action='store_true')
        parser.add_argument('--sweep_layers', action='store_true')
        parser.add_argument('--llama_layer', type=int, default=LLAMA_LAYER)
        parser.add_argument('--tag', default=CONFIG['tag'])
        parser.add_argument('--sc_representation', choices=SC_REPRESENTATION_CHOICES, default=CONFIG['sc_representation'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    LLAMA_LAYER = args.llama_layer
    run_pipeline(args)
