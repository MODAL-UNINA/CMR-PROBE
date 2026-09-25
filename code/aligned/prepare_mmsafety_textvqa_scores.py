#!/usr/bin/env python3
"""Prepare frozen MM-SafetyBench and TextVQA score caches."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from utils import load_model, extract_llama_hidden_solo, extract_sc_feature_pair, extract_clip_hidden, validate_combo_representation, SC_REPRESENTATION_CHOICES
LLAMA_LAYER = 17
CLIP_LAYER = 21
SEED_DEFAULT = 42
MM_SAFETY_CONFIGS = ['EconomicHarm', 'Financial_Advice', 'Fraud', 'Gov_Decision', 'HateSpeech', 'Health_Consultation', 'Illegal_Activitiy', 'Legal_Opinion', 'Malware_Generation', 'Physical_Harm', 'Political_Lobbying', 'Privacy_Violence', 'Sex']

def load_secondary_datasets():
    print('\n[1] Loading MM-SafetyBench SD...')
    parts = []
    for scenario in MM_SAFETY_CONFIGS:
        ds = load_dataset('PKU-Alignment/MM-SafetyBench', name=scenario, split='SD')
        ds = ds.add_column('category', [scenario] * len(ds))
        parts.append(ds)
    mm_safety = concatenate_datasets(parts)
    print('[1] Loading TextVQA...')
    textvqa = load_dataset('lmms-lab/TextVQA', split='test')
    print(f'    MM-SafetyBench: {len(mm_safety)} sample | TextVQA: {len(textvqa)} sample')
    return (mm_safety, textvqa)

def create_or_load_splits(split_path: Path, mm_safety, textvqa, n_cal_unsafe: int, n_test_unsafe: int, n_cal_safe: int, n_test_safe: int, seed: int):
    if split_path.exists():
        print(f'\n[2] Existing split: {split_path}')
        with open(split_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        required = {'mmsafety_cal', 'mmsafety_test', 'textvqa_cal', 'textvqa_test'}
        missing = required - set(splits)
        if missing:
            raise KeyError(f'Incomplete split file; missing: {sorted(missing)}')
        print(f"    MM-Safety cal/test: {len(splits['mmsafety_cal'])}/{len(splits['mmsafety_test'])}")
        print(f"    TextVQA cal/test: {len(splits['textvqa_cal'])}/{len(splits['textvqa_test'])}")
        return splits
    print('\n[2] Creating a disjoint SECONDARY split...')
    total_unsafe = n_cal_unsafe + n_test_unsafe
    total_safe = n_cal_safe + n_test_safe
    if total_unsafe > len(mm_safety):
        raise ValueError(f'Required {total_unsafe} MM-SafetyBench indices, but the dataset contains {len(mm_safety)}.')
    if total_safe > len(textvqa):
        raise ValueError(f'Required {total_safe} TextVQA indices, but the dataset contains {len(textvqa)}.')
    all_u = np.arange(len(mm_safety))
    cats = np.asarray(mm_safety['category'])
    selected_u, _ = train_test_split(all_u, train_size=total_unsafe, random_state=seed, stratify=cats)
    selected_cats = cats[selected_u]
    u_cal, u_test = train_test_split(selected_u, train_size=n_cal_unsafe, test_size=n_test_unsafe, random_state=seed + 1, stratify=selected_cats)
    rng = np.random.default_rng(seed)
    perm_s = rng.permutation(len(textvqa))
    s_cal = perm_s[:n_cal_safe]
    s_test = perm_s[n_cal_safe:n_cal_safe + n_test_safe]
    assert set(u_cal).isdisjoint(set(u_test))
    assert set(s_cal).isdisjoint(set(s_test))
    splits = {'protocol': 'MM-SafetyBench SD / TextVQA common-FPR', 'seed': int(seed), 'unsafe_stratified_by': 'MM-SafetyBench category', 'mmsafety_cal': [int(x) for x in u_cal], 'mmsafety_test': [int(x) for x in u_test], 'textvqa_cal': [int(x) for x in s_cal], 'textvqa_test': [int(x) for x in s_test], 'sizes': {'mmsafety_cal': int(len(u_cal)), 'mmsafety_test': int(len(u_test)), 'textvqa_cal': int(len(s_cal)), 'textvqa_test': int(len(s_test))}}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    print(f'    Saved: {split_path}')
    print(f'    MM-Safety cal/test: {len(u_cal)}/{len(u_test)} | TextVQA cal/test: {len(s_cal)}/{len(s_test)}')
    for split_name, idxs in [('cal', u_cal), ('test', u_test)]:
        vals, counts = np.unique(cats[idxs], return_counts=True)
        print(f'\n    MM-Safety {split_name} by category:')
        for v, c in zip(vals, counts):
            print(f'      {v:24s}: {c}')
    return splits

def load_probes(probes_dir: Path, sc_representation: str):
    print('\n[3] Loading probe v13...')
    with open(probes_dir / 'probe_text.pkl', 'rb') as f:
        clf_text = pickle.load(f)
    visual_path = probes_dir / 'probe_visual_orthogonalized.pkl'
    with open(visual_path, 'rb') as f:
        visual_obj = pickle.load(f)
    if isinstance(visual_obj, dict):
        clf_visual = visual_obj['clf']
        ortho_Q_visual = visual_obj.get('ortho_Q')
    else:
        clf_visual = visual_obj
        ortho_Q_visual = None
    combo_path = None
    for name in ['1_linear_delta_h_orthogonalized', '1_linear_delta_h', '2_svm_rbf_delta_h', '3_mlp_pair']:
        p = probes_dir / 'combo' / f'probe_combo_{name}.pkl'
        if p.exists():
            combo_path = p
            break
    if combo_path is None:
        raise FileNotFoundError(f"No probe_combo found in {probes_dir / 'combo'}")
    with open(combo_path, 'rb') as f:
        combo_obj = pickle.load(f)
    validate_combo_representation(combo_obj, sc_representation)
    clf_combo = combo_obj['clf']
    ortho_Q_combo = combo_obj.get('ortho_Q')
    print(f"    text   : {probes_dir / 'probe_text.pkl'}")
    print(f'    visual : {visual_path}')
    print(f'    combo  : {combo_path.name}')
    print(f'    ortho visual: {ortho_Q_visual is not None}')
    print(f'    ortho combo : {ortho_Q_combo is not None}')
    return (clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo)

def get_text(row) -> str:
    for key in ('Question', 'question', 'prompt_text', 'text'):
        if key in row and row[key] is not None:
            return str(row[key])
    return ''

def get_image(row):
    for key in ('image', 'unsafe_image'):
        if key in row and row[key] is not None:
            return row[key].convert('RGB')
    raise KeyError('No recognized image field.')

def score_one(text, image, model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, sc_representation='aligned'):
    h_text = extract_llama_hidden_solo(text, model, processor, LLAMA_LAYER)
    h_visual = extract_clip_hidden(image, model, processor, CLIP_LAYER)
    hv = h_visual.reshape(1, -1)
    if ortho_Q_visual is not None:
        hv = hv - hv @ ortho_Q_visual @ ortho_Q_visual.T
    s_t = float(clf_text.predict_proba(h_text.reshape(1, -1))[0, 1])
    s_v = float(clf_visual.predict_proba(hv)[0, 1])
    _, _, delta_h = extract_sc_feature_pair(
        text, image, model, processor, LLAMA_LAYER, sc_representation
    )
    dh = delta_h.reshape(1, -1)
    if ortho_Q_combo is not None:
        dh = dh - dh @ ortho_Q_combo @ ortho_Q_combo.T
    s_c = float(clf_combo.predict_proba(dh)[0, 1])
    return (s_t, s_v, s_c)

def score_split(dataset, indices, dataset_name, split_name, model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, sc_representation='aligned'):
    st, sv, sc, rows = ([], [], [], [])
    desc = f'{dataset_name}_{split_name}'
    for idx in tqdm(indices, desc=desc):
        row = dataset[int(idx)]
        text = get_text(row)
        image = get_image(row)
        try:
            s_t, s_v, s_c = score_one(text, image, model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, sc_representation)
        except Exception as e:
            raise RuntimeError(f'{desc} idx={idx}: extraction failed: {e}') from e
        st.append(s_t)
        sv.append(s_v)
        sc.append(s_c)
        rows.append({'idx': int(idx), 'S_T': float(s_t), 'S_V': float(s_v), 'S_C': float(s_c)})
    return (np.asarray(st, dtype=np.float64), np.asarray(sv, dtype=np.float64), np.asarray(sc, dtype=np.float64), rows)

def save_group_cache(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)

def run(args):
    probes_dir = Path(args.probes)
    delta_dir = Path(args.delta)
    work_dir = delta_dir / 'mmsafety_textvqa_common_fpr'
    work_dir.mkdir(parents=True, exist_ok=True)
    split_path = work_dir / 'eval_splits_mmsafety_textvqa.json'
    cal_path = work_dir / 'scores_cal_mmsafety_textvqa.pkl'
    test_path = work_dir / 'scores_test_mmsafety_textvqa.pkl'
    mm_safety, textvqa = load_secondary_datasets()
    splits = create_or_load_splits(split_path=split_path, mm_safety=mm_safety, textvqa=textvqa, n_cal_unsafe=args.n_cal_unsafe, n_test_unsafe=args.n_test_unsafe, n_cal_safe=args.n_cal_safe, n_test_safe=args.n_test_safe, seed=args.seed)
    if cal_path.exists() and test_path.exists() and (not args.recompute):
        for existing in (cal_path, test_path):
            with open(existing, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('sc_representation') != args.sc_representation:
                raise ValueError(f"Cache representation mismatch in {existing}: {cached.get('sc_representation')} != {args.sc_representation}")
        print('\n[cache] Calibration and test are already present.')
        print(f'    {cal_path}')
        print(f'    {test_path}')
        return
    clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo = load_probes(probes_dir, args.sc_representation)
    print('\n[4] Loading LLaVA...')
    model, processor = load_model(gpu=args.gpu)
    if cal_path.exists() and (not args.recompute):
        print(f'\n[5] Calibration cache already present: {cal_path}')
        with open(cal_path, 'rb') as f:
            cal_cache = pickle.load(f)
        if cal_cache.get('sc_representation') != args.sc_representation:
            raise ValueError(f"Calibration cache representation mismatch: {cal_cache.get('sc_representation')} != {args.sc_representation}")
    else:
        print('\n[5] Extraction CALIBRATION...')
        st_u, sv_u, sc_u, rows_u = score_split(mm_safety, splits['mmsafety_cal'], 'MM-Safety', 'cal', model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, args.sc_representation)
        st_s, sv_s, sc_s, rows_s = score_split(textvqa, splits['textvqa_cal'], 'TextVQA', 'cal', model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, args.sc_representation)
        cal_cache = {'sc_representation': args.sc_representation, 'mmsafety': (st_u, sv_u, sc_u), 'textvqa': (st_s, sv_s, sc_s), 'rows_mmsafety': rows_u, 'rows_textvqa': rows_s, 'indices': {'mmsafety': splits['mmsafety_cal'], 'textvqa': splits['textvqa_cal']}, 'split_file': str(split_path), 'role': 'calibration'}
        save_group_cache(cal_path, cal_cache)
        print(f'    Saved: {cal_path}')
    if test_path.exists() and (not args.recompute):
        print(f'\n[6] Test cache already present: {test_path}')
        with open(test_path, 'rb') as f:
            test_cache = pickle.load(f)
        if test_cache.get('sc_representation') != args.sc_representation:
            raise ValueError(f"Test cache representation mismatch: {test_cache.get('sc_representation')} != {args.sc_representation}")
    else:
        print('\n[6] Extraction TEST...')
        st_u, sv_u, sc_u, rows_u = score_split(mm_safety, splits['mmsafety_test'], 'MM-Safety', 'test', model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, args.sc_representation)
        st_s, sv_s, sc_s, rows_s = score_split(textvqa, splits['textvqa_test'], 'TextVQA', 'test', model, processor, clf_text, clf_visual, ortho_Q_visual, clf_combo, ortho_Q_combo, args.sc_representation)
        test_cache = {'sc_representation': args.sc_representation, 'mmsafety': (st_u, sv_u, sc_u), 'textvqa': (st_s, sv_s, sc_s), 'rows_mmsafety': rows_u, 'rows_textvqa': rows_s, 'indices': {'mmsafety': splits['mmsafety_test'], 'textvqa': splits['textvqa_test']}, 'split_file': str(split_path), 'role': 'test'}
        save_group_cache(test_path, test_cache)
        print(f'    Saved: {test_path}')
    print('\n' + '=' * 78)
    print('SECONDARY COMMON-FPR DATA PREPARATION COMPLETED')
    print('=' * 78)
    print(f'split : {split_path}')
    print(f'cal   : {cal_path}')
    print(f'test  : {test_path}')
    print('\nNo threshold was estimated and no test label was used.')
if __name__ == '__main__':
    CONFIG = {'probes': 'probes/rebuilt_v13', 'delta': 'probes/rebuilt_v13/delta', 'gpu': 0, 'seed': 42, 'n_cal_unsafe': 100, 'n_test_unsafe': 200, 'n_cal_safe': 400, 'n_test_safe': 1100, 'recompute': False, 'sc_representation': 'aligned'}
    in_jupyter = 'ipykernel' in sys.argv[0] if sys.argv else False
    if not in_jupyter:
        parser = argparse.ArgumentParser()
        parser.add_argument('--probes', default=CONFIG['probes'])
        parser.add_argument('--delta', default=CONFIG['delta'])
        parser.add_argument('--gpu', type=int, default=CONFIG['gpu'])
        parser.add_argument('--seed', type=int, default=CONFIG['seed'])
        parser.add_argument('--n_cal_unsafe', type=int, default=CONFIG['n_cal_unsafe'])
        parser.add_argument('--n_test_unsafe', type=int, default=CONFIG['n_test_unsafe'])
        parser.add_argument('--n_cal_safe', type=int, default=CONFIG['n_cal_safe'])
        parser.add_argument('--n_test_safe', type=int, default=CONFIG['n_test_safe'])
        parser.add_argument('--recompute', action='store_true', default=CONFIG['recompute'])
        parser.add_argument('--sc_representation', choices=SC_REPRESENTATION_CHOICES, default=CONFIG['sc_representation'])
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**CONFIG)
    run(args)
