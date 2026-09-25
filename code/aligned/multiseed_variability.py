#!/usr/bin/env python3
"""Run the frozen five-seed resplit and recalibration analysis."""
from __future__ import annotations
import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
SCRIPT_CANDIDATES = {'train': ['2_train_probes.py'], 'combo': ['3.1_train_probe_combo.py'], 'tau_c': ['4.1_calibrate_tau_c.py'], 'tau_tv': ['4.2_calibrate_tau_tv.py', '4.2_calibrate_tau_tv(1).py'], 'eval_primary': ['6_evaluate_pipeline.py'], 'eval_secondary': ['6_evaluate_pipeline_2.py']}

def _load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _samples_from_doc(doc):
    return doc['samples'] if isinstance(doc, dict) else doc

def _write_samples_like(template_doc, samples, path: Path, seed: int, split_name: str):
    if isinstance(template_doc, dict):
        out = dict(template_doc)
        out['samples'] = samples
        meta = dict(out.get('metadata', {}))
        meta.update({'multiseed_seed': int(seed), 'multiseed_split': split_name, 'multiseed_protocol': 'MSTS train/development resplit only (200/100); MSTS final test and all not-MSTS assignments frozen'})
        out['metadata'] = meta
    else:
        out = samples
    _dump_json(out, path)

def _is_msts_sample(sample: dict) -> bool:
    sid = str(sample.get('id', '')).lower()
    src = str(sample.get('source', '')).lower()
    cat = str(sample.get('category', '')).lower()
    path = str(sample.get('local_image_path', '')).lower()
    return sid.startswith('msts_') or sid.startswith('d_hf_msts_') or 'felfri/msts' in src or cat.startswith('msts_') or ('/msts_images/' in path) or path.startswith('msts_images/') or ('msts_row' in sample)

def _msts_row(sample: dict) -> int:
    if 'msts_row' in sample:
        return int(sample['msts_row'])
    sid = str(sample.get('id', ''))
    m = re.search('msts[_-](?:default[_-])?(\\d+)$', sid, flags=re.I)
    if m:
        return int(m.group(1))
    raise ValueError(f'Could not derive msts_row from sample {sid!r}')

def make_internal_train_val_split(reference_dataset: Path, out_dataset: Path, seed: int):
    p_tr = reference_dataset / 'probe_train.json'
    p_va = reference_dataset / 'probe_val.json'
    p_te = reference_dataset / 'probe_test.json'
    for p in (p_tr, p_va, p_te):
        if not p.exists():
            raise FileNotFoundError(f'File split internal not found: {p}')
    doc_tr, doc_va, doc_te = (_load_json(p_tr), _load_json(p_va), _load_json(p_te))
    tr, va, te = (_samples_from_doc(doc_tr), _samples_from_doc(doc_va), _samples_from_doc(doc_te))
    non_tr = [x for x in tr if not _is_msts_sample(x)]
    non_va = [x for x in va if not _is_msts_sample(x)]
    non_te = [x for x in te if not _is_msts_sample(x)]
    m_tr = [x for x in tr if _is_msts_sample(x)]
    m_va = [x for x in va if _is_msts_sample(x)]
    m_te = [x for x in te if _is_msts_sample(x)]
    if (len(m_tr), len(m_va), len(m_te)) != (200, 100, 100):
        raise ValueError(f'MSTS v13 expects 200/100/100 samples in probe_train/val/test, found {len(m_tr)}/{len(m_va)}/{len(m_te)}')
    pool = m_tr + m_va
    rows = [_msts_row(x) for x in pool]
    if len(set(rows)) != 300:
        raise ValueError('MSTS train+dev contains msts_row duplicates.')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    new_m_tr = [pool[int(i)] for i in perm[:200]]
    new_m_va = [pool[int(i)] for i in perm[200:300]]
    new_tr = non_tr + new_m_tr
    new_va = non_va + new_m_va
    new_te = non_te + m_te
    out_dataset.mkdir(parents=True, exist_ok=True)
    _write_samples_like(doc_tr, new_tr, out_dataset / 'probe_train.json', seed, 'train')
    _write_samples_like(doc_va, new_va, out_dataset / 'probe_val.json', seed, 'development')
    _write_samples_like(doc_te, new_te, out_dataset / 'probe_test.json', seed, 'final_test_fixed')
    manifest = {'dataset': 'felfri/MSTS', 'hf_split': 'english', 'seed': int(seed), 'protocol': '200_train_100_development_100_final_test; final test frozen from v13', 'train': [_msts_row(x) for x in new_m_tr], 'val': [_msts_row(x) for x in new_m_va], 'test': [_msts_row(x) for x in m_te]}
    manifest_path = out_dataset / 'msts_split_200_100_100.json'
    _dump_json(manifest, manifest_path)
    if set(manifest['train']) & set(manifest['val']):
        raise RuntimeError('MSTS train/development leakage in the multi-seed manifest')
    if set(manifest['train']) & set(manifest['test']):
        raise RuntimeError('MSTS train/test leakage in the multi-seed manifest')
    if set(manifest['val']) & set(manifest['test']):
        raise RuntimeError('MSTS development/test leakage in the multi-seed manifest')
    info = {'seed': int(seed), 'n_train': len(new_tr), 'n_val': len(new_va), 'n_test': len(new_te), 'msts_train': 200, 'msts_development': 100, 'msts_test_fixed': 100, 'non_msts_assignments': 'frozen_from_v13', 'msts_manifest': str(manifest_path)}
    _dump_json(info, out_dataset / 'multiseed_split_info.json')
    return info

def _shuffle_partition(pool: Iterable[int], n_first: int, n_second: int, rng) -> tuple[list[int], list[int]]:
    pool = list(dict.fromkeys((int(x) for x in pool)))
    if len(pool) < n_first + n_second:
        raise ValueError(f'Pool too much piccolo: {len(pool)} < {n_first}+{n_second}. The reference split does not contain enough samples.')
    perm = rng.permutation(pool).tolist()
    return (perm[:n_first], perm[n_first:n_first + n_second])

def make_primary_eval_split(reference_split: Path, out_path: Path, seed: int, msts_manifest_path: Path | None=None) -> dict:
    ref = _load_json(reference_split)
    rng = np.random.default_rng(seed)
    if msts_manifest_path is not None:
        man = _load_json(msts_manifest_path)
        m_cal = [int(x) for x in man['val']]
        m_test = [int(x) for x in man['test']]
    else:
        m_cal_n, m_test_n = (len(ref['msts_cal']), len(ref['msts_test']))
        m_cal, m_test = _shuffle_partition(ref['msts_cal'] + ref['msts_test'], m_cal_n, m_test_n, rng)
    s_cal_n, s_test_n = (len(ref['mmstar_cal']), len(ref['mmstar_test']))
    s_cal, s_test = _shuffle_partition(ref['mmstar_cal'] + ref['mmstar_test'], s_cal_n, s_test_n, rng)
    out = dict(ref)
    out.update({'seed': int(seed), 'msts_cal': m_cal, 'msts_test': m_test, 'mmstar_cal': s_cal, 'mmstar_test': s_test, 'multiseed_protocol': 'MSTS seed-specific development + frozen v13 final test; MMStar reshuffled within reference cal+test pool'})
    _dump_json(out, out_path)
    return out

def sync_eval_splits_v13(delta_dir: Path, manifest_path: Path, seed: int) -> Path:
    manifest = _load_json(manifest_path)
    eval_path = delta_dir / 'eval_splits.json'
    if not eval_path.exists():
        raise FileNotFoundError(f'Seed-specific primary split not found: {eval_path}')
    base = _load_json(eval_path)
    v13_path = delta_dir / 'eval_splits_v13.json'
    if v13_path.exists():
        try:
            obj = _load_json(v13_path)
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
    else:
        obj = {}
    m_train = [int(x) for x in manifest['train']]
    m_dev = [int(x) for x in manifest['val']]
    m_test = [int(x) for x in manifest['test']]
    mm_cal = [int(x) for x in base.get('mmstar_cal', [])]
    mm_test = [int(x) for x in base.get('mmstar_test', [])]
    obj.update({'seed': int(seed), 'msts_train': m_train, 'msts_cal': m_dev, 'msts_val': m_dev, 'msts_dev': m_dev, 'msts_development': m_dev, 'development': m_dev, 'val': m_dev, 'msts_test': m_test, 'msts_final_test': m_test, 'mmstar_cal': mm_cal, 'mmstar_test': mm_test, 'n_cal_msts': len(m_dev), 'n_cal_mmstar': len(mm_cal), 'multiseed_protocol': 'MSTS: seed-specific 200 train + 100 development, frozen 100 final-test; MMStar: seed-specific reshuffle within reference cal+test pool'})
    msts_block = obj.get('msts') if isinstance(obj.get('msts'), dict) else {}
    msts_block.update({'train': m_train, 'val': m_dev, 'dev': m_dev, 'development': m_dev, 'cal': m_dev, 'test': m_test, 'final_test': m_test})
    obj['msts'] = msts_block
    mm_block = obj.get('mmstar') if isinstance(obj.get('mmstar'), dict) else {}
    mm_block.update({'cal': mm_cal, 'test': mm_test})
    obj['mmstar'] = mm_block
    if 'MSTS' in obj or 'MMStar' in obj:
        mb = obj.get('MSTS') if isinstance(obj.get('MSTS'), dict) else {}
        mb.update({'train': m_train, 'val': m_dev, 'dev': m_dev, 'development': m_dev, 'cal': m_dev, 'test': m_test, 'final_test': m_test})
        obj['MSTS'] = mb
        sb = obj.get('MMStar') if isinstance(obj.get('MMStar'), dict) else {}
        sb.update({'cal': mm_cal, 'test': mm_test})
        obj['MMStar'] = sb
    _dump_json(obj, v13_path)
    chk = _load_json(v13_path)
    if chk['msts_development'] != m_dev:
        raise RuntimeError("eval_splits_v13 synchronization failed: development != manifest['val']")
    if chk['msts_test'] != m_test:
        raise RuntimeError("eval_splits_v13 synchronization failed: test != manifest['test']")
    if chk['mmstar_cal'] != mm_cal or chk['mmstar_test'] != mm_test:
        raise RuntimeError('eval_splits_v13 synchronization failed: MMStar does not match eval_splits.json')
    print(f'[splits v13] synchronized {v13_path.name}: MSTS dev/test={len(m_dev)}/{len(m_test)}, MMStar cal/test={len(mm_cal)}/{len(mm_test)}')
    print(f'[splits v13] manifest used: {manifest_path}')
    print(f'[splits v13] MSTS dev first5={m_dev[:5]} | test first5={m_test[:5]}')
    return v13_path

def make_binary_reserved_split(reference_split: Path, out_path: Path, seed: int) -> dict:
    ref = _load_json(reference_split)
    rng = np.random.default_rng(seed)
    out = dict(ref)
    for cls in ('unsafe', 'safe'):
        k_cal = f'{cls}_cal'
        k_test = f'{cls}_test'
        if k_cal not in ref or k_test not in ref:
            raise KeyError(f'{reference_split} does not contain {k_cal}/{k_test}')
        n_cal, n_test = (len(ref[k_cal]), len(ref[k_test]))
        cal, test = _shuffle_partition(ref[k_cal] + ref[k_test], n_cal, n_test, rng)
        out[k_cal] = cal
        out[k_test] = test
    out['seed'] = seed
    out['multiseed_protocol'] = 'reshuffle within reference reserved cal+test pools only'
    _dump_json(out, out_path)
    return out

def resolve_scripts(script_dir: Path) -> dict[str, Path]:
    resolved = {}
    for key, candidates in SCRIPT_CANDIDATES.items():
        for name in candidates:
            p = script_dir / name
            if p.exists():
                resolved[key] = p
                break
        if key not in resolved:
            raise FileNotFoundError(f"Script '{key}' not found in {script_dir}. Searched: {candidates}")
    return resolved

def inject_tau_c_score_cache(text: str) -> str:
    if 'MULTISEED_TAU_C_SCORE_CACHE' in text:
        return text
    if not re.search('(?m)^\\s*import\\s+pickle\\b', text):
        text = 'import pickle\n' + text
    pat = re.compile('(?m)^(?P<indent>\\s*)(?P<lhs1>[A-Za-z_]\\w*)\\s*,\\s*(?P<lhs2>[A-Za-z_]\\w*)\\s*=\\s*calibrate_threshold\\(\\s*["\\\']C["\\\']\\s*,\\s*(?P<unsafe>[A-Za-z_]\\w*)\\s*,\\s*(?P<safe>[A-Za-z_]\\w*)\\s*\\)\\s*$')
    m = pat.search(text)
    if not m:
        raise RuntimeError("Could not patch 4.1_calibrate_tau_c.py: calibrate_threshold('C', unsafe_scores, safe_scores) was not found.")
    indent = m.group('indent')
    unsafe = m.group('unsafe')
    safe = m.group('safe')
    lines = [m.group(0), f'{indent}# MULTISEED_TAU_C_SCORE_CACHE', f'{indent}_ms_n_msts = int(len({unsafe}))', f'{indent}_ms_n_mmstar = int(len({safe}))', f'{indent}_ms_cache_obj = {{', f'{indent}    "sc_representation": args.sc_representation,', f'{indent}    "msts": (None, None, {unsafe}),', f'{indent}    "mmstar": (None, None, {safe}),', f'{indent}}}', f'{indent}for _ms_name in (', f'{indent}    f"scores_cal_seed42_msts{{_ms_n_msts}}_mmstar{{_ms_n_mmstar}}.pkl",', f'{indent}    f"scores_cal_seed42_n{{_ms_n_msts}}_{{_ms_n_mmstar}}.pkl",', f'{indent}):', f'{indent}    _ms_path = out_dir / _ms_name', f'{indent}    with open(_ms_path, "wb") as _ms_f:', f'{indent}        pickle.dump(_ms_cache_obj, _ms_f)', f'{indent}    print(f"  [multiseed cache] Saved {{_ms_path.name}}")']
    injected = '\n'.join(lines)
    return text[:m.start()] + injected + text[m.end():]

def _find_tau_c_score_cache(delta_dir: Path, sc_representation: str | None=None):
    candidates = sorted(delta_dir.glob('*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            if not isinstance(obj, dict) or 'msts' not in obj or 'mmstar' not in obj:
                continue
            if sc_representation is not None and obj.get('sc_representation') != sc_representation:
                continue
            msts = obj['msts']
            mmstar = obj['mmstar']
            if not isinstance(msts, (tuple, list)) or not isinstance(mmstar, (tuple, list)):
                continue
            if len(msts) < 3 or len(mmstar) < 3:
                continue
            if msts[2] is None or mmstar[2] is None:
                continue
            return p
        except Exception:
            continue
    return None

def patch_script_text(text: str, *, seed: int, clip_layer: int, llama_layer: int, n_cal_msts: int, n_cal_mmstar: int, n_cal_tv: int, secondary: bool=False, script_key: str='') -> str:
    text = re.sub('(?m)^\\s*CLIP_LAYER\\s*=\\s*\\d+[^\\n]*$', f'CLIP_LAYER = {clip_layer}', text)
    text = re.sub('(?m)^\\s*LLAMA_LAYER\\s*=\\s*\\d+[^\\n]*$', f'LLAMA_LAYER = {llama_layer}', text)
    text = re.sub('random_state\\s*=\\s*42', f'random_state={seed}', text)
    text = re.sub('np\\.random\\.default_rng\\(seed\\s*=\\s*42\\)', f'np.random.default_rng(seed={seed})', text)
    text = re.sub('np\\.random\\.default_rng\\(42\\)', f'np.random.default_rng({seed})', text)
    text = re.sub('(?m)([\\"\']seed[\\"\']\\s*:\\s*)42', f'\\g<1>{seed}', text)
    text = re.sub('(?m)(\\bseed\\s*=\\s*)42(?=\\s*[,\\)])', f'\\g<1>{seed}', text)
    if script_key == 'tau_c':
        text = inject_tau_c_score_cache(text)
    if secondary:
        text = text.replace('scores_cal_seed42_n200_400.pkl', f'scores_cal_seed42_n{n_cal_msts}_{n_cal_mmstar}.pkl')
        text = text.replace('st_cal_seed42_n400.pkl', f'st_cal_seed42_n{n_cal_tv}.pkl')
        text = text.replace('sv_cal_seed42_n400.pkl', f'sv_cal_seed42_n{n_cal_tv}.pkl')
    return text

def stage_scripts(scripts: dict[str, Path], stage_dir: Path, **patch_kwargs) -> dict[str, Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, src in scripts.items():
        text = src.read_text(encoding='utf-8')
        text = patch_script_text(text, secondary=key == 'eval_secondary', script_key=key, **patch_kwargs)
        dst = stage_dir / src.name
        compile(text, str(dst), 'exec')
        dst.write_text(text, encoding='utf-8')
        out[key] = dst
    return out

def script_supports_option(script_path: Path, option: str) -> bool:
    text = script_path.read_text(encoding='utf-8')
    pat = f"""add_argument\\s*\\(\\s*[\\"']{re.escape(option)}[\\"']"""
    return re.search(pat, text) is not None

def ensure_secondary_cache_compat(delta_dir: Path, *, seed: int, n_cal_msts: int, n_cal_mmstar: int, n_cal_tv: int, sc_representation: str) -> None:
    import shutil

    def newest(paths):
        paths = [x for x in paths if x.exists()]
        return max(paths, key=lambda x: x.stat().st_mtime) if paths else None
    sc_patterns = [f'scores_cal_v13_seed{seed}_msts{n_cal_msts}_mmstar{n_cal_mmstar}.pkl', f'scores_cal_v13_seed42_msts{n_cal_msts}_mmstar{n_cal_mmstar}.pkl', f'scores_cal_seed{seed}_msts{n_cal_msts}_mmstar{n_cal_mmstar}.pkl', f'scores_cal_seed{seed}_n{n_cal_msts}_{n_cal_mmstar}.pkl', f'scores_cal_seed42_msts{n_cal_msts}_mmstar{n_cal_mmstar}.pkl', f'scores_cal_seed42_n{n_cal_msts}_{n_cal_mmstar}.pkl']
    candidate_paths = [delta_dir / name for name in sc_patterns]
    candidate_paths += list(delta_dir.glob('scores_cal_v13*.pkl'))
    candidate_paths += list(delta_dir.glob('scores_cal_seed*.pkl'))
    valid_paths = []
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if obj.get('sc_representation') == sc_representation:
                valid_paths.append(path)
        except Exception:
            continue
    sc_src = newest(valid_paths)
    if sc_src is None:
        sc_src = _find_tau_c_score_cache(delta_dir, sc_representation)
    if sc_src is None:
        raise FileNotFoundError(f'No tau_C score cache was found in {delta_dir}. Rerun the runner with --resume: stage 4.1 will run again and explicitly save the reference S_C values for the secondary benchmark.')
    sc_aliases = [delta_dir / f'scores_cal_seed42_msts{n_cal_msts}_mmstar{n_cal_mmstar}.pkl', delta_dir / f'scores_cal_seed42_n{n_cal_msts}_{n_cal_mmstar}.pkl']
    for dst in sc_aliases:
        if dst.resolve() == sc_src.resolve() if dst.exists() else False:
            continue
        if not dst.exists() or dst.stat().st_mtime < sc_src.stat().st_mtime:
            shutil.copy2(sc_src, dst)
            print(f'[compat secondary] {dst.name} <- {sc_src.name}')
    for prefix in ('st_cal', 'sv_cal'):
        candidates = [delta_dir / f'{prefix}_seed{seed}_n{n_cal_tv}.pkl', delta_dir / f'{prefix}_seed42_n{n_cal_tv}.pkl']
        src = newest(candidates)
        if src is None:
            src = newest(list(delta_dir.glob(f'{prefix}_seed*_n*.pkl')))
        if src is None:
            continue
        dst = delta_dir / f'{prefix}_seed42_n{n_cal_tv}.pkl'
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
            print(f'[compat secondary] {dst.name} <- {src.name}')

def run_command(cmd: list[str], *, cwd: Path, env: dict, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 100)
    print('RUN:', ' '.join(cmd))
    print('LOG:', log_path)
    print('=' * 100)
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            log.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

def _combo_model_exists(probes_dir: Path, sc_representation: str) -> bool:
    combo = probes_dir / 'combo'
    for path in combo.glob('probe_combo_*.pkl'):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if obj.get('sc_representation') == sc_representation:
                return True
        except Exception:
            continue
    return False

def _patch_config_seed(delta_dir: Path, seed: int):
    cfg_path = delta_dir / 'delta_config.json'
    if not cfg_path.exists():
        return
    cfg = _load_json(cfg_path)
    if isinstance(cfg.get('calibration'), dict):
        cfg['calibration']['seed'] = seed
    cfg['multiseed_seed'] = seed
    cfg['multiseed_protocol'] = 'train/val resplit within original train+val; internal test fixed; external cal/test reshuffled within v13 held-out pools'
    _dump_json(cfg, cfg_path)

def _result_file_primary(delta_dir: Path, require_exists: bool=True) -> Path:
    preferred = [delta_dir / 'evaluation_results_v13_internal.json', delta_dir / 'evaluation_results_internal.json']
    for p in preferred:
        if p.exists():
            return p
    matches = sorted(delta_dir.glob('evaluation_results*_internal.json'), key=lambda p: p.stat().st_mtime)
    matches = [p for p in matches if '_2_internal' not in p.name]
    if matches:
        return matches[-1]
    if require_exists:
        raise FileNotFoundError(f'No evaluation_results*_internal.json primary in {delta_dir}')
    return preferred[0]

def _result_file_secondary(delta_dir: Path, require_exists: bool=True) -> Path:
    preferred = [delta_dir / 'evaluation_results_v13_internal_2.json', delta_dir / 'evaluation_results_2_internal_trial.json', delta_dir / 'evaluation_results_2_internal.json']
    for p in preferred:
        if p.exists():
            return p
    matches = sorted(list(delta_dir.glob('evaluation_results*2*internal*.json')) + list(delta_dir.glob('evaluation_results*v13*internal*2*.json')), key=lambda p: p.stat().st_mtime)
    seen, unique = (set(), [])
    for p in matches:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if unique:
        return unique[-1]
    if require_exists:
        raise FileNotFoundError(f'No secondary-evaluation JSON was found in {delta_dir}')
    return preferred[0]

def _result_matches_representation(path: Path, sc_representation: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = _load_json(path)
        return obj.get('sc_representation') == sc_representation or obj.get('protocol', {}).get('sc_representation') == sc_representation
    except Exception:
        return False

def extract_result_row(seed: int, benchmark: str, result_path: Path) -> dict:
    obj = _load_json(result_path)
    m = obj['metrics']
    sc = m['probe_combo_only']
    full = m['full_pipeline']
    tpr, fpr = (float(full['tpr']), float(full['fpr']))
    row = {'seed': seed, 'benchmark': benchmark, 'tau_T': obj.get('config', {}).get('tau_T', np.nan), 'tau_V': obj.get('config', {}).get('tau_V', np.nan), 'tau_C': obj.get('config', {}).get('tau_C', np.nan), 'sc_auc': sc.get('auc', np.nan), 'sc_tpr': sc.get('tpr', np.nan), 'sc_fpr': sc.get('fpr', np.nan), 'full_tpr': tpr, 'full_fpr': fpr, 'full_precision': full.get('precision', np.nan), 'full_f1': full.get('f1', np.nan), 'full_balanced_accuracy': 0.5 * (tpr + (1.0 - fpr)), 'result_file': str(result_path)}
    ua = obj.get('unsupervised_adapted', {}).get('metrics')
    if ua:
        uf = ua['full_pipeline']
        utpr, ufpr = (float(uf['tpr']), float(uf['fpr']))
        row.update({'adapted_full_tpr': utpr, 'adapted_full_fpr': ufpr, 'adapted_full_f1': uf.get('f1', np.nan), 'adapted_full_balanced_accuracy': 0.5 * (utpr + (1.0 - ufpr)), 'adapted_sc_auc': ua['probe_combo_only'].get('auc', np.nan)})
    return row

def summarize_rows(rows: list[dict]) -> dict:
    out = {}
    for bench in sorted(set((r['benchmark'] for r in rows))):
        rr = [r for r in rows if r['benchmark'] == bench]
        numeric_keys = []
        for k in rr[0].keys():
            if k in {'seed', 'benchmark', 'result_file'}:
                continue
            vals = []
            ok = True
            for r in rr:
                try:
                    v = float(r.get(k, np.nan))
                    vals.append(v)
                except Exception:
                    ok = False
                    break
            if ok:
                numeric_keys.append(k)
        stats = {'n_seeds': len(rr), 'seeds': [int(r['seed']) for r in rr], 'metrics': {}}
        for k in numeric_keys:
            arr = np.array([float(r.get(k, np.nan)) for r in rr], dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            stats['metrics'][k] = {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 'min': float(arr.min()), 'max': float(arr.max())}
        out[bench] = stats
    return out

def save_aggregate(rows: list[dict], output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = ['seed', 'benchmark', 'tau_T', 'tau_V', 'tau_C', 'sc_auc', 'sc_tpr', 'sc_fpr', 'full_tpr', 'full_fpr', 'full_precision', 'full_f1', 'full_balanced_accuracy', 'adapted_sc_auc', 'adapted_full_tpr', 'adapted_full_fpr', 'adapted_full_f1', 'adapted_full_balanced_accuracy', 'result_file']
    all_keys = set().union(*(r.keys() for r in rows))
    fields = [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))
    csv_path = output_root / 'multiseed_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path = output_root / 'multiseed_results.json'
    _dump_json({'runs': rows, 'summary': summarize_rows(rows)}, json_path)
    print('\n' + '=' * 100)
    print('MULTI-SEED SUMMARY')
    print('=' * 100)
    summary = summarize_rows(rows)
    for bench, s in summary.items():
        print(f"\n[{bench}] n={s['n_seeds']} seeds={s['seeds']}")
        for key in ['sc_auc', 'sc_tpr', 'sc_fpr', 'full_tpr', 'full_fpr', 'full_f1', 'full_balanced_accuracy', 'adapted_sc_auc', 'adapted_full_tpr', 'adapted_full_fpr', 'adapted_full_f1', 'adapted_full_balanced_accuracy']:
            if key in s['metrics']:
                z = s['metrics'][key]
                print(f"  {key:32s}: {z['mean']:.4f} +- {z['std']:.4f}")
    print(f'\nCSV : {csv_path}')
    print(f'JSON: {json_path}')

def main(args):
    dataset = Path(args.dataset).resolve()
    base_dir = Path(args.base_dir).resolve()
    script_dir = Path(args.script_dir).resolve()
    ref_delta = Path(args.reference_delta).resolve()
    output_root = Path(args.output_root).resolve()
    ref_eval = Path(args.reference_eval_splits).resolve() if args.reference_eval_splits else ref_delta / 'eval_splits.json'
    ref_toxic = Path(args.reference_toxic_splits).resolve() if args.reference_toxic_splits else ref_delta / 'eval_splits_toxicchat.json'
    ref_harm = Path(args.reference_harmful_splits).resolve() if args.reference_harmful_splits else ref_delta / 'harmful_contents_reserved.json'
    if not ref_eval.exists():
        raise FileNotFoundError(f'Reference eval split not found: {ref_eval}')
    if not ref_toxic.exists():
        raise FileNotFoundError(f'Reference toxic-chat split not found: {ref_toxic}\nPass it with --reference-toxic-splits.')
    if not ref_harm.exists():
        raise FileNotFoundError(f'Reference harmful-contents split not found: {ref_harm}\nPass it with --reference-harmful-splits.')
    scripts = resolve_scripts(script_dir)
    ref_eval_obj = _load_json(ref_eval)
    n_cal_msts = len(ref_eval_obj['msts_cal'])
    n_cal_mmstar = len(ref_eval_obj['mmstar_cal'])
    n_cal_tv = 400
    if ref_toxic.exists():
        ref_tox_obj = _load_json(ref_toxic)
        n_cal_tv = len(ref_tox_obj['unsafe_cal'])
        if len(ref_tox_obj['safe_cal']) != n_cal_tv:
            raise ValueError('Reference toxic-chat: unsafe_cal and safe_cal have different sizes.')
    rows = []
    for seed in args.seeds:
        print('\n' + '#' * 110)
        print(f'# SEED {seed}')
        print('#' * 110)
        seed_root = output_root / f'seed_{seed}'
        dataset_seed = seed_root / 'dataset_split'
        probes_dir = seed_root / 'probes'
        delta_dir = probes_dir / 'delta'
        stage_dir = seed_root / '_staged_scripts'
        logs_dir = seed_root / 'logs'
        delta_dir.mkdir(parents=True, exist_ok=True)
        split_info_path = dataset_seed / 'multiseed_split_info.json'
        msts_manifest_seed = dataset_seed / 'msts_split_200_100_100.json'
        make_internal_train_val_split(dataset, dataset_seed, seed)
        if not msts_manifest_seed.exists():
            raise FileNotFoundError(f'Manifest MSTS seed-specific not found: {msts_manifest_seed}')
        make_primary_eval_split(ref_eval, delta_dir / 'eval_splits.json', seed, msts_manifest_path=msts_manifest_seed)
        sync_eval_splits_v13(delta_dir, msts_manifest_seed, seed)
        if args.benchmark in ('secondary', 'both'):
            if not (args.resume and (delta_dir / 'eval_splits_toxicchat.json').exists()):
                make_binary_reserved_split(ref_toxic, delta_dir / 'eval_splits_toxicchat.json', seed)
            if not (args.resume and (delta_dir / 'harmful_contents_reserved.json').exists()):
                make_binary_reserved_split(ref_harm, delta_dir / 'harmful_contents_reserved.json', seed)
        staged = stage_scripts(scripts, stage_dir, seed=seed, clip_layer=args.clip_layer, llama_layer=args.llama_layer, n_cal_msts=n_cal_msts, n_cal_mmstar=n_cal_mmstar, n_cal_tv=n_cal_tv)
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)
        env['MULTISEED_SEED'] = str(seed)
        env['PYTHONPATH'] = str(script_dir) + os.pathsep + env.get('PYTHONPATH', '')
        py = sys.executable
        if not (args.resume and (probes_dir / 'probe_text.pkl').exists() and (probes_dir / 'probe_visual_orthogonalized.pkl').exists()):
            run_command([py, str(staged['train']), '--dataset', str(dataset_seed), '--base_dir', str(base_dir), '--out', str(probes_dir), '--gpu', str(args.gpu), '--clip_layer', str(args.clip_layer), '--llama_layer', str(args.llama_layer)], cwd=script_dir, env=env, log_path=logs_dir / '01_train_probes.log')
        else:
            print('[resume] train_probes already completed.')
        if not (args.resume and _combo_model_exists(probes_dir, args.sc_representation)):
            run_command([py, str(staged['combo']), '--dataset', str(dataset_seed), '--base_dir', str(base_dir), '--probes', str(seed_root), '--tag', probes_dir.name, '--gpu', str(args.gpu), '--llama_layer', str(args.llama_layer), '--sc_representation', args.sc_representation], cwd=script_dir, env=env, log_path=logs_dir / '02_train_probe_combo.log')
        else:
            print('[resume] probe_combo already completed.')
        _need_tau_c_score_cache = args.benchmark in ('secondary', 'both') and _find_tau_c_score_cache(delta_dir, args.sc_representation) is None
        _tau_cfg_path = delta_dir / 'delta_config.json'
        _tau_cfg_matches = False
        if _tau_cfg_path.exists():
            try:
                _tau_cfg_matches = _load_json(_tau_cfg_path).get('sc_representation_mode') == args.sc_representation
            except Exception:
                _tau_cfg_matches = False
        _tau_c_complete = _tau_cfg_matches and (not _need_tau_c_score_cache)
        _tau_c_executed = False
        if not (args.resume and _tau_c_complete):
            if args.resume and (delta_dir / 'delta_config.json').exists() and _need_tau_c_score_cache:
                print('[resume] tau_C is already calibrated, but the secondary S_C cache is missing: rerun only 4.1.')
            tau_cmd = [py, str(staged['tau_c']), '--probes', str(probes_dir), '--out', str(delta_dir), '--gpu', str(args.gpu)]
            tau_cmd += ['--sc_representation', args.sc_representation]
            if script_supports_option(staged['tau_c'], '--msts_manifest'):
                tau_cmd += ['--msts_manifest', str(msts_manifest_seed)]
            elif script_supports_option(staged['tau_c'], '--n_cal_msts'):
                tau_cmd += ['--n_cal_msts', str(n_cal_msts)]
            else:
                raise RuntimeError('4.1_calibrate_tau_c.py exposes neither --msts_manifest nor --n_cal_msts: the MSTS protocol cannot be determined.')
            if script_supports_option(staged['tau_c'], '--n_cal_mmstar'):
                tau_cmd += ['--n_cal_mmstar', str(n_cal_mmstar)]
            if script_supports_option(staged['tau_c'], '--recompute'):
                tau_cmd += ['--recompute']
            run_command(tau_cmd, cwd=script_dir, env=env, log_path=logs_dir / '03_calibrate_tau_c.log')
            _tau_c_executed = True
        else:
            print('[resume] tau_C already calibrated.')
        sync_eval_splits_v13(delta_dir, msts_manifest_seed, seed)
        can_run_tv = ref_toxic.exists() and ref_harm.exists()
        if can_run_tv:
            if not (delta_dir / 'eval_splits_toxicchat.json').exists():
                make_binary_reserved_split(ref_toxic, delta_dir / 'eval_splits_toxicchat.json', seed)
            if not (delta_dir / 'harmful_contents_reserved.json').exists():
                make_binary_reserved_split(ref_harm, delta_dir / 'harmful_contents_reserved.json', seed)
            st_cache = delta_dir / f'st_cal_seed42_n{n_cal_tv}.pkl'
            sv_cache = delta_dir / f'sv_cal_seed42_n{n_cal_tv}.pkl'
            if not (args.resume and st_cache.exists() and sv_cache.exists() and (not _tau_c_executed)):
                if args.resume and _tau_c_executed and st_cache.exists() and sv_cache.exists():
                    print('[resume] rerunning 4.2 to align delta_config after the new 4.1 output.')
                run_command([py, str(staged['tau_tv']), '--probes', str(probes_dir), '--out', str(delta_dir), '--gpu', str(args.gpu), '--n_cal', str(n_cal_tv), '--harmful_splits', str(delta_dir / 'harmful_contents_reserved.json'), '--recompute'], cwd=script_dir, env=env, log_path=logs_dir / '04_calibrate_tau_tv.log')
            else:
                print('[resume] tau_T/tau_V already calibrated.')
        else:
            print('[WARN] reference toxic/harmful not available: skipping 4.2_calibrate_tau_tv.py.')
        _patch_config_seed(delta_dir, seed)
        if args.benchmark in ('primary', 'both'):
            sync_eval_splits_v13(delta_dir, msts_manifest_seed, seed)
            result_primary = _result_file_primary(delta_dir, require_exists=False)
            if not (args.resume and _result_matches_representation(result_primary, args.sc_representation)):
                eval_cmd = [py, str(staged['eval_primary']), '--probes', str(probes_dir), '--delta', str(delta_dir), '--gpu', str(args.gpu), '--use_internal_probes', '--recompute']
                eval_cmd += ['--sc_representation', args.sc_representation]
                if script_supports_option(staged['eval_primary'], '--msts_manifest'):
                    eval_cmd += ['--msts_manifest', str(msts_manifest_seed)]
                if script_supports_option(staged['eval_primary'], '--splits_path'):
                    eval_cmd += ['--splits_path', str(delta_dir / 'eval_splits_v13.json')]
                elif script_supports_option(staged['eval_primary'], '--eval_splits'):
                    eval_cmd += ['--eval_splits', str(delta_dir / 'eval_splits_v13.json')]
                run_command(eval_cmd, cwd=script_dir, env=env, log_path=logs_dir / '05_eval_primary.log')
            else:
                print('[resume] evaluation primary already present.')
            result_primary = _result_file_primary(delta_dir, require_exists=True)
            rows.append(extract_result_row(seed, 'MSTS_MMStar', result_primary))
            save_aggregate(rows, output_root)
        if args.benchmark in ('secondary', 'both'):
            ensure_secondary_cache_compat(delta_dir, seed=seed, n_cal_msts=n_cal_msts, n_cal_mmstar=n_cal_mmstar, n_cal_tv=n_cal_tv, sc_representation=args.sc_representation)
            expected_secondary = _result_file_secondary(delta_dir, require_exists=False)
            if not (args.resume and _result_matches_representation(expected_secondary, args.sc_representation)):
                run_command([py, str(staged['eval_secondary']), '--probes', str(probes_dir), '--delta', str(delta_dir), '--gpu', str(args.gpu), '--use_internal_probes', '--recompute', '--sc_representation', args.sc_representation], cwd=script_dir, env=env, log_path=logs_dir / '06_eval_secondary.log')
            else:
                print('[resume] evaluation secondary already present.')
            result_secondary = _result_file_secondary(delta_dir)
            rows.append(extract_result_row(seed, 'MMSafety_TextVQA', result_secondary))
            save_aggregate(rows, output_root)
    save_aggregate(rows, output_root)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Multi-seed and multi-split runner for the SAFETY pipeline')
    parser.add_argument('--dataset', required=True, help='Internal v13 dataset with probe_train/val/test.json')
    parser.add_argument('--reference-delta', required=True, help='Delta directory from the official v13 run, used only as a leakage-safe reference')
    parser.add_argument('--output-root', required=True, help='New directory for all multi-seed runs')
    parser.add_argument('--base-dir', default='.', help='Root used to resolve local_image_path')
    parser.add_argument('--script-dir', default=str(Path(__file__).resolve().parent), help='Directory containing the original scripts and utils.py')
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--benchmark', choices=['primary', 'secondary', 'both'], default='both')
    parser.add_argument('--clip-layer', type=int, default=21, help='CLIP layer used consistently in training, calibration, and evaluation')
    parser.add_argument('--llama-layer', type=int, default=17, help='LLaMA layer used consistently in training, calibration, and evaluation')
    parser.add_argument('--sc-representation', choices=['mean', 'aligned'], default='aligned', help='S_C feature used consistently in training, calibration, and evaluation')
    parser.add_argument('--resume', action='store_true', help='Skip steps already completed for each seed')
    parser.add_argument('--reference-eval-splits', default=None, help='Override for the reference eval_splits.json')
    parser.add_argument('--reference-toxic-splits', default=None, help='Override for the reference eval_splits_toxicchat.json')
    parser.add_argument('--reference-harmful-splits', default=None, help='Override for the reference harmful_contents_reserved.json')
    main(parser.parse_args())
