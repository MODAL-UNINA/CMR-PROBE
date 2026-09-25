#!/usr/bin/env python3
"""Train the text and visual marginal safety probes."""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE'] = '3'
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
import argparse
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import Ridge
from utils import load_model, extract_llama_hidden_solo, extract_clip_hidden, compute_sample_weights, collapse_source, resolve_local_image_path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
CLIP_LAYER = 21
LLAMA_LAYER = 17
MODEL_ID = 'llava-hf/llava-1.5-7b-hf'
SWEEP_CLIP_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
_model = None
_processor = None
_face_cascade = None

def extract_features(sample: dict, model, processor, base_dir: Path) -> dict | None:
    try:
        h_text = extract_llama_hidden_solo(sample['text'], model, processor, LLAMA_LAYER)
    except Exception as e:
        print(f"  [warn] text failed {sample['id']}: {e}")
        return None
    img_path = resolve_local_image_path(sample['local_image_path'], base_dir)
    if not img_path.exists():
        print(f'  [warn] image not found: {img_path}')
        return None
    try:
        image = Image.open(img_path).convert('RGB')
        h_visual = extract_clip_hidden(image, model, processor, CLIP_LAYER)
    except Exception as e:
        print(f"  [warn] visual failed {sample['id']}: {e}")
        return None
    return {'id': sample['id'], 'category': sample['category'], 'label_text': int(sample['label_text']), 'label_img': int(sample['label_img']), 'label_combo': int(sample['label_combo']), 'h_text': h_text, 'h_visual': h_visual, 'local_image_path': str(img_path), 'source': sample.get('source', 'baseline')}

def load_split(json_path: Path) -> list[dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['samples'] if isinstance(data, dict) else data

def backfill_source(feats: list[dict], json_path: Path) -> list[dict]:
    if feats and 'source' in feats[0]:
        return feats
    print(f"  [migrate] 'source' missing in the cache, backfill from {json_path.name}...")
    raw = load_split(json_path)
    source_by_id = {s['id']: s.get('source', 'baseline') for s in raw}
    for f in feats:
        f['source'] = source_by_id.get(f['id'], 'baseline')
    return feats

def extract_split(json_path: Path, base_dir: Path, cache_path: Path, model, processor) -> list[dict]:
    if cache_path.exists():
        print(f'  [cache] Loading {cache_path.name}')
        with open(cache_path, 'rb') as f:
            feats = pickle.load(f)
        return backfill_source(feats, json_path)
    samples = load_split(json_path)
    feats = []
    for s in tqdm(samples, desc=f'  Extraction {json_path.name}'):
        feat = extract_features(s, model, processor, base_dir)
        if feat:
            feats.append(feat)
    with open(cache_path, 'wb') as f:
        pickle.dump(feats, f)
    print(f'  [cache] Saved {cache_path.name} ({len(feats)} samples)')
    return feats

def get_XY(feats: list[dict], feature: str, label: str):
    if feature == 'h_fused':
        X = np.concatenate([np.stack([f['h_text'] for f in feats]), np.stack([f['h_visual'] for f in feats])], axis=1)
    else:
        X = np.stack([f[feature] for f in feats])
    y = np.array([f[label] for f in feats])
    return (X, y)

def _detect_faces(image):
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    arr = np.array(image.convert('L'))
    return (_face_cascade.detectMultiScale(arr, 1.1, 4), arr.shape)

def face_area_ratio(image) -> float:
    faces, shape = _detect_faces(image)
    if len(faces) == 0:
        return 0.0
    areas = [w * h for x, y, w, h in faces]
    return float(max(areas) / (shape[0] * shape[1]))

def skin_ratio(image) -> float:
    arr = np.array(image.convert('RGB'))
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return float((mask > 0).mean())

def has_person_proxy(image) -> float:
    faces, _ = _detect_faces(image)
    return float(len(faces) > 0)

def train_probe(X_tr, y_tr, X_va, y_va, name: str, sample_weight=None) -> tuple:
    print(f'\n  [{name}]  train={len(y_tr)}  val={len(y_va)}  pos_train={y_tr.sum()}  pos_val={y_va.sum()}')
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='lbfgs', random_state=42)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    clf_cal = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
    clf_cal.fit(X_va, y_va)
    proba = clf_cal.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, proba) if len(np.unique(y_va)) > 1 else float('nan')
    acc = accuracy_score(y_va, clf_cal.predict(X_va))
    print(f'    AUC_val={auc:.4f}  ACC_val={acc:.4f}')
    return (clf_cal, auc, acc)

def evaluate_probe(clf, X_te, y_te, name: str) -> dict:
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float('nan')
    acc = accuracy_score(y_te, clf.predict(X_te))
    fpr, tpr, thresholds = roc_curve(y_te, proba)
    tau_opt = float(thresholds[np.argmax(tpr - fpr)])
    print(f'  [test]  {name:12s}  AUC={auc:.4f}  ACC={acc:.4f}  τ_opt={tau_opt:.4f}')
    return {'name': name, 'auc_test': round(float(auc), 4), 'acc_test': round(float(acc), 4), 'tau_opt': round(tau_opt, 4)}

def fit_sentinel_and_get_projector(X_train: np.ndarray, sources_train: list[str], n_components: int=None) -> tuple:
    le = LabelEncoder()
    y_source = le.fit_transform(sources_train)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return (float('nan'), None)
    sentinel_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', solver='lbfgs', random_state=42, multi_class='multinomial' if n_classes > 2 else 'auto')
    sentinel_clf.fit(X_train, y_source)
    proba = sentinel_clf.predict_proba(X_train)
    auc = roc_auc_score(y_source, proba[:, 1]) if n_classes == 2 else accuracy_score(y_source, sentinel_clf.predict(X_train))
    W = sentinel_clf.coef_
    if W.ndim == 1:
        W = W.reshape(1, -1)
    Q, _ = np.linalg.qr(W.T)
    k = n_components or Q.shape[1]
    return (float(auc), Q[:, :k])

def fit_visual_sentinel_per_label(feats_unimodal: list[dict]) -> dict:
    projectors = {}
    for label_val in sorted(set((f['label_img'] for f in feats_unimodal))):
        feats_lab = [f for f in feats_unimodal if f['label_img'] == label_val]
        sources_lab = [collapse_source(f['source']) for f in feats_lab]
        n_sources = len(set(sources_lab))
        if n_sources < 2:
            print(f'  [visual sentinel label_img={label_val}] only one source ({sources_lab[0]}): no correction is required.')
            projectors[label_val] = (float('nan'), None)
            continue
        X_lab = np.stack([f['h_visual'] for f in feats_lab])
        auc, Q = fit_sentinel_and_get_projector(X_lab, sources_lab)
        print(f'  [visual sentinel label_img={label_val}] sources={set(sources_lab)}  AUC/ACC pre={auc:.4f}')
        projectors[label_val] = (auc, Q)
    return projectors

def fit_visual_style_sentinel_per_label(feats_unimodal):
    Q_list = []
    for label_val in sorted(set((f['label_img'] for f in feats_unimodal))):
        feats_lab = [f for f in feats_unimodal if f['label_img'] == label_val]
        X_lab = np.stack([f['h_visual'] for f in feats_lab])
        norms, persons, face_areas, skins = ([], [], [], [])
        for f in feats_lab:
            norms.append(np.linalg.norm(f['h_visual']))
            try:
                img = Image.open(f['local_image_path']).convert('RGB')
                persons.append(has_person_proxy(img))
                face_areas.append(face_area_ratio(img))
                skins.append(skin_ratio(img))
            except Exception:
                persons.append(np.nan)
                face_areas.append(np.nan)
                skins.append(np.nan)
        proxies = {'norm': np.array(norms, dtype=np.float64), 'person': np.array(persons, dtype=np.float64), 'face_area': np.array(face_areas, dtype=np.float64), 'skin_ratio': np.array(skins, dtype=np.float64)}
        for name, vals in proxies.items():
            valid = ~np.isnan(vals)
            if valid.sum() < 5 or np.nanstd(vals) < 1e-08:
                continue
            y = (vals[valid] - np.nanmean(vals)) / (np.nanstd(vals) + 1e-08)
            model = Ridge(alpha=1.0).fit(X_lab[valid], y)
            w = model.coef_.flatten()
            wn = np.linalg.norm(w)
            if wn > 1e-08:
                Q_list.append((w / wn).reshape(-1, 1))
    return Q_list

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

def extract_clip_all_layers(image, model, processor, layers):
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    inputs = processor(images=image, text='a', return_tensors='pt').to(device)
    captured = {}
    hooks = []
    for layer in layers:

        def make_hook(l):

            def hook_fn(module, inp, out):
                captured[l] = (out[0] if isinstance(out, tuple) else out).float().detach()
            return hook_fn
        hooks.append(model.vision_tower.vision_model.encoder.layers[layer].register_forward_hook(make_hook(layer)))
    with torch.no_grad():
        model.vision_tower(pixel_values=inputs['pixel_values'].to(dtype))
    for h in hooks:
        h.remove()
    result = {}
    for layer in layers:
        h = captured[layer]
        result[layer] = {'cls': h[:, 0, :].squeeze(0).cpu().numpy(), 'mean': h[:, 1:, :].mean(dim=1).squeeze(0).cpu().numpy()}
    return result

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-08, None)

def fisher_ratio(X, y):
    Xn = l2n(X)
    mu1, mu0 = (Xn[y == 1].mean(0), Xn[y == 0].mean(0))
    between = np.linalg.norm(mu1 - mu0) ** 2
    within = Xn[y == 1].var(0).sum() + Xn[y == 0].var(0).sum()
    return float(between / (within + 1e-08))

def run_clip_sweep(feats_train: list[dict], feats_val: list[dict], model, processor, out_dir: Path, layers=range(1, 24)) -> int:
    print(f'\n[CLIP SWEEP] Extraction multi-layer ({len(list(layers))} layer) on train...')
    cache, y = ([], [])
    for f in tqdm(feats_train, desc='  extraction'):
        img_path = Path(f.get('local_image_path', ''))
        if not img_path.exists():
            continue
        try:
            image = Image.open(img_path).convert('RGB')
            cache.append(extract_clip_all_layers(image, model, processor, layers))
            y.append(f['label_img'])
        except Exception as e:
            print(f'  [warn] {img_path}: {e}')
    y = np.array(y)
    results = {}
    for layer in layers:
        for pooling in ('mean', 'cls'):
            X = np.stack([c[layer][pooling] for c in cache])
            results[layer, pooling] = fisher_ratio(X, y)
    print(f"\n  {'Layer':>6s}  {'mean':>8s}  {'cls':>8s}")
    for layer in layers:
        print(f"  {layer:6d}  {results[layer, 'mean']:8.4f}  {results[layer, 'cls']:8.4f}")
    best_key = max(results, key=results.get)
    best_layer, best_pool = best_key
    print(f'\n  Best: layer={best_layer}  pooling={best_pool}  Fisher={results[best_key]:.4f}')
    with open(out_dir / 'layer_sweep_clip_fisher.json', 'w') as f:
        json.dump({f'{l}_{p}': v for (l, p), v in results.items()}, f, indent=2)
    return best_layer

def run_pipeline(args):
    data_dir = Path(args.dataset)
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.dataset)
    out_dir = Path(args.out)
    cache_dir = out_dir / 'feature_cache'
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    if not args.train_only:
        print('\n[1/3] Extraction hidden states...')
        model, processor = load_model(args.gpu)
        feats_train = extract_split(data_dir / 'probe_train.json', base_dir, cache_dir / 'train.pkl', model, processor)
        feats_val = extract_split(data_dir / 'probe_val.json', base_dir, cache_dir / 'val.pkl', model, processor)
        feats_test = extract_split(data_dir / 'probe_test.json', base_dir, cache_dir / 'test.pkl', model, processor)
    else:
        print('\n[1/3] Loading feature from the cache...')
        feats = {}
        for split in ('train', 'val', 'test'):
            p = cache_dir / f'{split}.pkl'
            if not p.exists():
                raise FileNotFoundError(f'Cache {p} not found. Remove --train_only.')
            with open(p, 'rb') as f:
                feats[split] = pickle.load(f)
            feats[split] = backfill_source(feats[split], data_dir / f'probe_{split}.json')
        feats_train, feats_val, feats_test = (feats['train'], feats['val'], feats['test'])
    if args.extract_only:
        print('\n[done] Only extraction completed.')
        return
    print(f'\n  Samples: train={len(feats_train)} | val={len(feats_val)} | test={len(feats_test)}')
    if args.sweep_clip_layers:
        model, processor = load_model(args.gpu)
        best_clip = run_clip_sweep(feats_train, feats_val, model, processor, out_dir)
        print(f'\n  Use --clip_layer {best_clip} for the training final.')
        if args.sweep_only:
            print('\n[done] Only sweep completed (--sweep_only).')
            return
        CLIP_LAYER = best_clip
        print(f'\n  Continuing training with CLIP_LAYER={CLIP_LAYER}...')
    print('\n[2/3] Training probe linear...')
    Xt_tr, yt_tr = get_XY(feats_train, 'h_text', 'label_text')
    Xt_va, yt_va = get_XY(feats_val, 'h_text', 'label_text')
    Xt_te, yt_te = get_XY(feats_test, 'h_text', 'label_text')
    clf_text, auc_tv, acc_tv = train_probe(Xt_tr, yt_tr, Xt_va, yt_va, 'probe_text')
    Xv_tr, yv_tr = get_XY(feats_train, 'h_visual', 'label_img')
    Xv_va, yv_va = get_XY(feats_val, 'h_visual', 'label_img')
    Xv_te, yv_te = get_XY(feats_test, 'h_visual', 'label_img')
    clf_visual, auc_vv, acc_vv = train_probe(Xv_tr, yv_tr, Xv_va, yv_va, 'probe_visual')
    print('\n── Sentinel probe_visual: source, for label_img ──')
    projectors_visual = fit_visual_sentinel_per_label(feats_train)
    Q_source_list = [Q for _, Q in projectors_visual.values() if Q is not None]
    Q_style_list = fit_visual_style_sentinel_per_label(feats_train)
    Q_all = Q_source_list + Q_style_list
    Q_visual = None
    if Q_all:
        Q_stack = np.concatenate(Q_all, axis=1)
        Q_visual, _ = np.linalg.qr(Q_stack)
        Q_visual = Q_visual[:, :Q_stack.shape[1]]
    Xv_tr_clean = apply_projector(Xv_tr, Q_visual)
    Xv_va_clean = apply_projector(Xv_va, Q_visual)
    Xv_te_clean = apply_projector(Xv_te, Q_visual)
    sample_w = compute_sample_weights(yv_tr, [f['source'] for f in feats_train])
    clf_visual_ortho, auc_vv_ortho, acc_vv_ortho = train_probe(Xv_tr_clean, yv_tr, Xv_va_clean, yv_va, 'probe_visual_orthogonalized', sample_weight=sample_w)
    visual_ortho_path = out_dir / 'probe_visual_orthogonalized.pkl'
    with open(visual_ortho_path, 'wb') as f:
        pickle.dump({'clf': clf_visual_ortho, 'ortho_Q': Q_visual}, f)
    print(f'  Saved: {visual_ortho_path}')
    print(f'\n  Comparison probe_visual (test set):')
    print('\n[3/3] Evaluation on test set...')
    res_text = evaluate_probe(clf_text, Xt_te, yt_te, 'text')
    res_visual = evaluate_probe(clf_visual, Xv_te, yv_te, 'visual')
    for clf, name in [(clf_text, 'probe_text'), (clf_visual, 'probe_visual')]:
        p = out_dir / f'{name}.pkl'
        with open(p, 'wb') as f:
            pickle.dump(clf, f)
        print(f'  Saved: {p}')
    CLIP_LAYER = args.clip_layer
    LLAMA_LAYER = args.llama_layer
    results = {'config': {'clip_layer': CLIP_LAYER, 'llama_layer': LLAMA_LAYER, 'model_id': MODEL_ID}, 'val': {'text': {'auc': round(auc_tv, 4), 'acc': round(acc_tv, 4)}, 'visual': {'auc': round(auc_vv, 4), 'acc': round(acc_vv, 4)}}, 'test': {'text': res_text, 'visual': res_visual}}
    res_path = out_dir / 'results.json'
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 54}")
    print(f'  CLIP layer={CLIP_LAYER}   LLaMA layer={LLAMA_LAYER}')
    print(f"{'=' * 54}")
    print(f"  {'Probe':12s}  {'AUC_val':>8s}  {'AUC_test':>9s}  {'τ_opt':>7s}")
    print(f"  {'-' * 44}")
    for name, auc_v, res in [('text', auc_tv, res_text), ('visual', auc_vv, res_visual)]:
        print(f"  {name:12s}  {auc_v:>8.4f}  {res['auc_test']:>9.4f}  {res['tau_opt']:>7.4f}")
    print(f"{'=' * 54}")
    print(f'\nResults in: {res_path}')
    print(f'\nNext step:')
    print(f'  python compute_delta.py --probes {out_dir}/ --dataset {data_dir}/')
if __name__ == '__main__':
    import sys
    CONFIG = {'dataset': 'data/legacy_v13/', 'base_dir': '.', 'out': 'probes/rebuilt_v13/', 'gpu': 3, 'extract_only': False, 'train_only': False, 'sweep_clip_layers': False, 'sweep_only': False, 'clip_layer': 21, 'llama_layer': 17}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default=CONFIG['dataset'])
        parser.add_argument('--base_dir', default=CONFIG['base_dir'])
        parser.add_argument('--out', default=CONFIG['out'])
        parser.add_argument('--gpu', type=int, default=CONFIG['gpu'])
        parser.add_argument('--extract_only', action='store_true')
        parser.add_argument('--train_only', action='store_true')
        parser.add_argument('--clip_layer', type=int, default=CLIP_LAYER)
        parser.add_argument('--llama_layer', type=int, default=LLAMA_LAYER)
        parser.add_argument('--sweep_clip_layers', action='store_true', help='Sweep all CLIP layers to find the best probe_visual layer')
        parser.add_argument('--sweep_only', action='store_true', help='Run only the sweep without final training')
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    CLIP_LAYER = args.clip_layer
    LLAMA_LAYER = args.llama_layer
    run_pipeline(args)
