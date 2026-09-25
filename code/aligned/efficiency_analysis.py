#!/usr/bin/env python3
"""Benchmark v13 inference time, throughput, and memory."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import pickle
import statistics
import time
from pathlib import Path
import numpy as np
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from datasets import load_dataset
from PIL import Image
from utils import load_model, extract_llama_hidden_solo, extract_sc_feature_pair, extract_clip_hidden, validate_combo_representation, SC_REPRESENTATION_CHOICES
LLAMA_LAYER = 17
CLIP_LAYER = 21

def cuda_sync(device):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

def reset_peak_memory(device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

def get_cuda_memory_mb(device):
    if not torch.cuda.is_available():
        return {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'peak_allocated_mb': 0.0, 'peak_reserved_mb': 0.0}
    return {'allocated_mb': torch.cuda.memory_allocated(device) / 1024 ** 2, 'reserved_mb': torch.cuda.memory_reserved(device) / 1024 ** 2, 'peak_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024 ** 2, 'peak_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024 ** 2}

def timed_call(fn, device, *args, **kwargs):
    cuda_sync(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    cuda_sync(device)
    elapsed = time.perf_counter() - t0
    return (out, elapsed)

def apply_projector(X: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    if Q is None:
        return X
    return X - X @ Q @ Q.T

def count_parameters(model):
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (int(total), int(trainable))

def percentile(values, q):
    if not values:
        return float('nan')
    return float(np.percentile(np.asarray(values, dtype=float), q))

def summarize_values(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {'n': 0, 'mean': None, 'std': None, 'median': None, 'p25': None, 'p75': None, 'p90': None, 'p95': None, 'min': None, 'max': None}
    return {'n': int(len(vals)), 'mean': round(float(vals.mean()), 6), 'std': round(float(vals.std()), 6), 'median': round(float(np.median(vals)), 6), 'p25': round(float(np.percentile(vals, 25)), 6), 'p75': round(float(np.percentile(vals, 75)), 6), 'p90': round(float(np.percentile(vals, 90)), 6), 'p95': round(float(np.percentile(vals, 95)), 6), 'min': round(float(vals.min()), 6), 'max': round(float(vals.max()), 6)}

def load_frozen_pipeline(probes_dir: Path, delta_dir: Path, sc_representation: str):
    with open(probes_dir / 'probe_text.pkl', 'rb') as f:
        obj_text = pickle.load(f)
    if isinstance(obj_text, dict):
        clf_text = obj_text['clf']
        ortho_Q_text = obj_text.get('ortho_Q')
    else:
        clf_text = obj_text
        ortho_Q_text = None
    visual_path = probes_dir / 'probe_visual_orthogonalized.pkl'
    if not visual_path.exists():
        visual_path = probes_dir / 'probe_visual.pkl'
    with open(visual_path, 'rb') as f:
        obj_visual = pickle.load(f)
    if isinstance(obj_visual, dict):
        clf_visual = obj_visual['clf']
        ortho_Q_visual = obj_visual.get('ortho_Q')
    else:
        clf_visual = obj_visual
        ortho_Q_visual = None
    combo_path = None
    for name in ['1_linear_delta_h_orthogonalized', '1_linear_delta_h', '2_svm_rbf_delta_h', '3_mlp_pair']:
        p = probes_dir / 'combo' / f'probe_combo_{name}.pkl'
        if p.exists():
            combo_path = p
            break
    if combo_path is None:
        raise FileNotFoundError(f"No combo probe found under {probes_dir / 'combo'}")
    with open(combo_path, 'rb') as f:
        combo_obj = pickle.load(f)
    validate_combo_representation(combo_obj, sc_representation)
    clf_combo = combo_obj['clf']
    ortho_Q_combo = combo_obj.get('ortho_Q')
    cfg_path = delta_dir / 'delta_config.json'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    config_mode = cfg.get('sc_representation_mode')
    if config_mode is not None and config_mode != sc_representation:
        raise ValueError(f'delta_config S_C representation mismatch: {config_mode} != {sc_representation}')
    if 'internal_probes' not in cfg:
        raise KeyError('delta_config.json has no internal_probes block.')
    tau_T = float(cfg['internal_probes']['tau_T'])
    tau_V = float(cfg['internal_probes']['tau_V'])
    tau_C = float(cfg['tau_C'])
    return {'clf_text': clf_text, 'ortho_Q_text': ortho_Q_text, 'clf_visual': clf_visual, 'ortho_Q_visual': ortho_Q_visual, 'clf_combo': clf_combo, 'ortho_Q_combo': ortho_Q_combo, 'tau_T': tau_T, 'tau_V': tau_V, 'tau_C': tau_C, 'combo_path': str(combo_path), 'visual_path': str(visual_path)}

def load_benchmark_dataset(name: str):
    if name == 'mmstar':
        ds = load_dataset('Lin-Chen/MMStar', split='val')
        return (ds, 'question', 'image')
    if name == 'msts':
        ds = load_dataset('felfri/MSTS', split='english')
        return (ds, 'prompt_text', 'unsafe_image')
    if name == 'textvqa':
        ds = load_dataset('lmms-lab/TextVQA', split='test')
        return (ds, 'question', 'image')
    raise ValueError(f'Unsupported dataset: {name}')

def choose_indices(n_total: int, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    n = min(n_samples, n_total)
    return rng.choice(n_total, size=n, replace=False).tolist()

def benchmark_sample(text, image, model, processor, pipeline, device, sc_representation='aligned'):

    def run_text():
        h_text = extract_llama_hidden_solo(text, model, processor, LLAMA_LAYER)
        ht = h_text.reshape(1, -1)
        ht = apply_projector(ht, pipeline['ortho_Q_text'])
        s_t = float(pipeline['clf_text'].predict_proba(ht)[0, 1])
        return (h_text, s_t)
    (h_text, s_t), t_text = timed_call(run_text, device=device)

    def run_visual():
        h_visual = extract_clip_hidden(image, model, processor, CLIP_LAYER)
        hv = h_visual.reshape(1, -1)
        hv = apply_projector(hv, pipeline['ortho_Q_visual'])
        s_v = float(pipeline['clf_visual'].predict_proba(hv)[0, 1])
        return (h_visual, s_v)
    (h_visual, s_v), t_visual = timed_call(run_visual, device=device)

    def run_combo():
        _, h_combo, delta_h = extract_sc_feature_pair(
            text, image, model, processor, LLAMA_LAYER, sc_representation
        )
        dh = delta_h.reshape(1, -1)
        dh = apply_projector(dh, pipeline['ortho_Q_combo'])
        s_c = float(pipeline['clf_combo'].predict_proba(dh)[0, 1])
        return (h_combo, s_c)
    (h_combo, s_c), t_combo = timed_call(run_combo, device=device)
    cuda_sync(device)
    t0 = time.perf_counter()
    delta = max(s_t - pipeline['tau_T'], s_v - pipeline['tau_V'], s_c - pipeline['tau_C'])
    pred = int(delta >= 0)
    cuda_sync(device)
    t_fusion = time.perf_counter() - t0
    t_total = t_text + t_visual + t_combo + t_fusion
    return {'S_T': s_t, 'S_V': s_v, 'S_C': s_c, 'delta': delta, 'pred': pred, 'latency_text_s': t_text, 'latency_visual_s': t_visual, 'latency_combo_s': t_combo, 'latency_fusion_s': t_fusion, 'latency_total_s': t_total}

def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def run(args):
    probes_dir = Path(args.probes)
    delta_dir = Path(args.delta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print('\n[1] Loading frozen v13 pipeline...')
    pipeline = load_frozen_pipeline(probes_dir, delta_dir, args.sc_representation)
    print(f"  combo probe : {pipeline['combo_path']}")
    print(f"  visual probe: {pipeline['visual_path']}")
    print(f"  thresholds  : T={pipeline['tau_T']:.4f}, V={pipeline['tau_V']:.4f}, C={pipeline['tau_C']:.4f}")
    print('\n[2] Loading MLLM...')
    model, processor = load_model(args.gpu)
    model.eval()
    total_params, trainable_params = count_parameters(model)
    print(f'  total parameters    : {total_params:,}')
    print(f'  trainable parameters: {trainable_params:,}')
    print('\n[3] Loading dataset...')
    ds, text_field, image_field = load_benchmark_dataset(args.dataset)
    print(f'  dataset={args.dataset}, n={len(ds)}')
    all_indices = choose_indices(len(ds), args.n_samples + args.warmup, args.seed)
    warmup_indices = all_indices[:args.warmup]
    test_indices = all_indices[args.warmup:]
    print(f'\n[4] Warm-up ({len(warmup_indices)} samples)...')
    with torch.inference_mode():
        for idx in warmup_indices:
            row = ds[idx]
            text = row.get(text_field, '') or ''
            image = row[image_field].convert('RGB')
            try:
                benchmark_sample(text, image, model, processor, pipeline, device=device, sc_representation=args.sc_representation)
            except Exception as e:
                print(f'  [warmup warn] idx={idx}: {e}')
    cuda_sync(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_sync(device)
    baseline_mem = get_cuda_memory_mb(device)
    reset_peak_memory(device)
    print(f'\n[5] Benchmarking {len(test_indices)} samples...')
    results = []
    global_start = time.perf_counter()
    with torch.inference_mode():
        for i, idx in enumerate(test_indices):
            row = ds[idx]
            text = row.get(text_field, '') or ''
            try:
                image = row[image_field].convert('RGB')
                reset_peak_memory(device)
                sample = benchmark_sample(text, image, model, processor, pipeline, device=device, sc_representation=args.sc_representation)
                mem = get_cuda_memory_mb(device)
                sample.update({'idx': idx, 'dataset': args.dataset, 'peak_allocated_mb': mem['peak_allocated_mb'], 'peak_reserved_mb': mem['peak_reserved_mb'], 'incremental_peak_allocated_mb': max(0.0, mem['peak_allocated_mb'] - baseline_mem['allocated_mb']), 'incremental_peak_reserved_mb': max(0.0, mem['peak_reserved_mb'] - baseline_mem['reserved_mb'])})
                results.append(sample)
            except Exception as e:
                print(f'  [warn] idx={idx}: {e}')
            if (i + 1) % 10 == 0:
                print(f'  [{i + 1}/{len(test_indices)}] processed')
    cuda_sync(device)
    wall_time = time.perf_counter() - global_start
    if not results:
        raise RuntimeError('No samples were successfully benchmarked.')
    latency_text = [r['latency_text_s'] for r in results]
    latency_visual = [r['latency_visual_s'] for r in results]
    latency_combo = [r['latency_combo_s'] for r in results]
    latency_fusion = [r['latency_fusion_s'] for r in results]
    latency_total = [r['latency_total_s'] for r in results]
    peak_alloc = [r['peak_allocated_mb'] for r in results]
    peak_res = [r['peak_reserved_mb'] for r in results]
    inc_alloc = [r['incremental_peak_allocated_mb'] for r in results]
    inc_res = [r['incremental_peak_reserved_mb'] for r in results]
    mean_latency = float(np.mean(latency_total))
    throughput_from_mean = 1.0 / mean_latency
    throughput_wall = len(results) / wall_time
    summary = {'experiment': {'pipeline': 'v13 frozen internal probes', 'sc_representation': args.sc_representation, 'dataset': args.dataset, 'requested_samples': args.n_samples, 'successful_samples': len(results), 'warmup_samples': args.warmup, 'seed': args.seed, 'gpu_id': args.gpu, 'llama_layer': LLAMA_LAYER, 'clip_layer': CLIP_LAYER}, 'model': {'total_parameters': total_params, 'trainable_parameters_at_inference': trainable_params}, 'thresholds': {'tau_T': pipeline['tau_T'], 'tau_V': pipeline['tau_V'], 'tau_C': pipeline['tau_C']}, 'latency_seconds': {'text': summarize_values(latency_text), 'visual': summarize_values(latency_visual), 'combo': summarize_values(latency_combo), 'fusion': summarize_values(latency_fusion), 'total': summarize_values(latency_total)}, 'latency_ms': {'text_mean': round(float(np.mean(latency_text)) * 1000, 3), 'visual_mean': round(float(np.mean(latency_visual)) * 1000, 3), 'combo_mean': round(float(np.mean(latency_combo)) * 1000, 3), 'fusion_mean': round(float(np.mean(latency_fusion)) * 1000, 3), 'total_mean': round(mean_latency * 1000, 3), 'total_median': round(float(np.median(latency_total)) * 1000, 3), 'total_p95': round(float(np.percentile(latency_total, 95)) * 1000, 3)}, 'throughput': {'samples_per_second_from_mean_latency': round(throughput_from_mean, 6), 'samples_per_second_wall_clock': round(throughput_wall, 6)}, 'memory_mb': {'warmed_model_allocated': round(baseline_mem['allocated_mb'], 3), 'warmed_model_reserved': round(baseline_mem['reserved_mb'], 3), 'peak_allocated': summarize_values(peak_alloc), 'peak_reserved': summarize_values(peak_res), 'incremental_peak_allocated': summarize_values(inc_alloc), 'incremental_peak_reserved': summarize_values(inc_res)}, 'forward_passes_per_sample': {'text_only_llama': 1, 'vision_tower': 1, 'multimodal_llama': 1, 'total_major_forward_passes': 3}, 'notes': ['Dataset loading time is excluded.', 'Warm-up samples are excluded from latency statistics.', 'CUDA synchronization is performed around every timed block.', 'Memory is measured after warm-up to avoid counting one-time CUDA initialization.', 'Incremental peak memory is measured relative to the warmed model footprint.', 'Probe inference and cross-modal residual computation are included.']}
    summary_rows = [{'method': 'Ours-v13', 'dataset': args.dataset, 'n': len(results), 'latency_mean_ms': summary['latency_ms']['total_mean'], 'latency_median_ms': summary['latency_ms']['total_median'], 'latency_p95_ms': summary['latency_ms']['total_p95'], 'throughput_samples_s': summary['throughput']['samples_per_second_wall_clock'], 'gpu_peak_allocated_mb_mean': summary['memory_mb']['peak_allocated']['mean'], 'gpu_incremental_peak_mb_mean': summary['memory_mb']['incremental_peak_allocated']['mean'], 'major_forward_passes': 3}]
    with open(out_dir / 'efficiency_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    write_csv(out_dir / 'efficiency_per_sample.csv', results)
    write_csv(out_dir / 'efficiency_summary.csv', summary_rows)
    print('\n' + '=' * 72)
    print('EFFICIENCY ANALYSIS — Ours v13')
    print('=' * 72)
    print(f'Samples              : {len(results)}')
    print(f"Mean latency          : {summary['latency_ms']['total_mean']:.2f} ms")
    print(f"Median latency        : {summary['latency_ms']['total_median']:.2f} ms")
    print(f"P95 latency           : {summary['latency_ms']['total_p95']:.2f} ms")
    print(f"Throughput            : {summary['throughput']['samples_per_second_wall_clock']:.3f} samples/s")
    print('\nComponent latency:')
    print(f"  S_T                 : {summary['latency_ms']['text_mean']:.2f} ms")
    print(f"  S_V                 : {summary['latency_ms']['visual_mean']:.2f} ms")
    print(f"  S_C                 : {summary['latency_ms']['combo_mean']:.2f} ms")
    print(f"  Fusion              : {summary['latency_ms']['fusion_mean']:.4f} ms")
    print('\nCUDA memory:')
    print(f"  Warm model allocated: {summary['memory_mb']['warmed_model_allocated']:.1f} MB")
    print(f"  Mean peak allocated : {summary['memory_mb']['peak_allocated']['mean']:.1f} MB")
    print(f"  Incremental peak    : {summary['memory_mb']['incremental_peak_allocated']['mean']:.1f} MB")
    print('\nSaved:')
    print(f"  {out_dir / 'efficiency_results.json'}")
    print(f"  {out_dir / 'efficiency_per_sample.csv'}")
    print(f"  {out_dir / 'efficiency_summary.csv'}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes', required=True, help='Frozen v13 probes directory.')
    parser.add_argument('--delta', required=True, help='Directory containing delta_config.json.')
    parser.add_argument('--dataset', choices=['mmstar', 'msts', 'textvqa'], default='mmstar')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sc_representation', choices=SC_REPRESENTATION_CHOICES, default='aligned')
    args = parser.parse_args()
    run(args)
