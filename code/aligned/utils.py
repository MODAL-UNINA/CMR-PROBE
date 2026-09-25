#!/usr/bin/env python3
"""Shared LLaVA loading and hidden-state extraction utilities."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from pathlib import Path
_model = None
_processor = None

SC_REPRESENTATION_CHOICES = ('mean', 'aligned')
SC_REPRESENTATION_FORMULAS = {
    'mean': 'L2(mean(hTV)-mean(hT))',
    'aligned': 'L2(L2(hTV_last)-L2(hT_last))',
}

def validate_sc_representation(value: str) -> str:
    value = str(value).strip().lower()
    if value not in SC_REPRESENTATION_CHOICES:
        choices = ', '.join(SC_REPRESENTATION_CHOICES)
        raise ValueError(f'Unknown S_C representation {value!r}; choose one of: {choices}.')
    return value

def sc_representation_formula(value: str) -> str:
    return SC_REPRESENTATION_FORMULAS[validate_sc_representation(value)]

def load_model(gpu: int=0, MODEL_ID: str='llava-hf/llava-1.5-7b-hf') -> tuple:
    global _model, _processor
    if _model is not None:
        return (_model, _processor)
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f'[load] Loading {MODEL_ID} on {device}...')
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map={'': device})
    _model.eval()
    print('[load] Model pronto.\n')
    return (_model, _processor)

def get_language_model(model):
    if hasattr(model, 'language_model'):
        return model.language_model
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model
    raise AttributeError(f'Language model not found in {type(model)}')

def get_vision_tower(model):
    if hasattr(model, 'vision_tower'):
        return model.vision_tower
    if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        return model.model.vision_tower
    raise AttributeError(f'Vision tower not found in {type(model)}')

def collapse_source(source: str) -> str:
    if source.startswith('onullusoy/harmful-contents'):
        return 'onullusoy/harmful-contents'
    if source.startswith('huggan/wikiart'):
        return 'huggan/wikiart'
    return source

def dataset_tag(dataset_path: str) -> str:
    p = Path(dataset_path).resolve()
    return p.name

def resolve_local_image_path(raw_path: str | Path, base_dir: str | Path) -> Path:
    """Resolve dataset image paths, including paths viewed through an SFTP mount.

    The historical v13 JSON files contain absolute paths rooted at the original
    ``.../Valentina/SAFETY`` checkout.  When that checkout is accessed through
    GVFS/SFTP, those absolute paths do not exist on the client, while the same
    suffix below ``SAFETY`` is available below ``base_dir``.  Relative paths
    keep their original v13 behaviour.
    """
    path = Path(raw_path)
    root = Path(base_dir)
    candidates = [path]

    if path.is_absolute():
        raw = path.as_posix()
        for marker in ('/SAFETY/', '/dataset/'):
            if marker not in raw:
                continue
            suffix = raw.split(marker, 1)[1]
            candidate = root / suffix if marker == '/SAFETY/' else root / 'dataset' / suffix
            candidates.append(candidate)
    else:
        candidates.append(root / path)
        candidates.append(Path('dataset') / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]

def extract_llama_hidden_solo(text: str, model, processor, LLAMA_LAYER) -> np.ndarray:
    language_model = get_language_model(model)
    device = next(language_model.parameters()).device
    prompt = f'USER: {text}\nASSISTANT:'
    inputs = processor(text=prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = language_model(input_ids=inputs['input_ids'], output_hidden_states=True, return_dict=True)
    h = out.hidden_states[LLAMA_LAYER + 1]
    return h.mean(dim=1).squeeze(0).float().cpu().numpy()

def extract_llama_hidden_multimodal(text: str, image: Image.Image, model, processor, layer: int=17) -> np.ndarray:
    dtype = next(model.model.parameters()).dtype
    device = next(model.model.parameters()).device
    prompt = f'USER: <image>\n{text}\nASSISTANT:'
    inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], pixel_values=inputs['pixel_values'].to(dtype), output_hidden_states=True)
    h = out.hidden_states[layer + 1]
    return h.mean(dim=1).squeeze(0).float().cpu().numpy()

def _l2_vector(x: np.ndarray, eps: float=1e-08) -> np.ndarray:
    """Return a float32 unit vector, preserving a near-zero vector."""
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    return x / norm if norm > eps else x

def extract_aligned_last_token_pair(
    text: str,
    image: Image.Image,
    model,
    processor,
    layer: int=17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the aligned last textual token and its normalized residual.

    The first two outputs are ``hT_last`` and ``hTV_last``.  The third is the
    only representation used by the aligned S_C probe:

        L2(L2(hTV_last) - L2(hT_last))

    The text-only and multimodal prompts intentionally end with the exact same
    textual suffix.  We additionally verify that their final input token ids
    match before comparing the two hidden states.
    """
    language_model = get_language_model(model)
    lm_device = next(language_model.parameters()).device
    text_prompt = f'USER: {text}\nASSISTANT:'
    text_inputs = processor(text=text_prompt, return_tensors='pt').to(lm_device)
    with torch.no_grad():
        text_out = language_model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs.get('attention_mask'),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    h_text = text_out.hidden_states[layer + 1][0, -1].float().cpu().numpy()

    core = model.model if hasattr(model, 'model') else model
    mm_device = next(core.parameters()).device
    mm_dtype = next(core.parameters()).dtype
    mm_prompt = f'USER: <image>\n{text}\nASSISTANT:'
    mm_inputs = processor(text=mm_prompt, images=image, return_tensors='pt').to(mm_device)
    if int(text_inputs['input_ids'][0, -1]) != int(mm_inputs['input_ids'][0, -1]):
        raise RuntimeError('The T and TV prompts do not end in the same token id.')
    with torch.no_grad():
        mm_out = core(
            input_ids=mm_inputs['input_ids'],
            attention_mask=mm_inputs.get('attention_mask'),
            pixel_values=mm_inputs['pixel_values'].to(mm_dtype),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    h_combo = mm_out.hidden_states[layer + 1][0, -1].float().cpu().numpy()
    delta = _l2_vector(_l2_vector(h_combo) - _l2_vector(h_text))
    return (
        np.asarray(h_text, dtype=np.float32),
        np.asarray(h_combo, dtype=np.float32),
        np.asarray(delta, dtype=np.float32),
    )

def extract_sc_feature_pair(
    text: str,
    image: Image.Image,
    model,
    processor,
    layer: int=17,
    sc_representation: str='aligned',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two states and the feature used by the S_C probe.

    ``mean`` exactly reproduces the historical v13 representation:
    ``L2(mean(hTV) - mean(hT))``. ``aligned`` uses the matched final textual
    token and normalizes both states before subtraction:
    ``L2(L2(hTV_last) - L2(hT_last))``.
    """
    mode = validate_sc_representation(sc_representation)
    if mode == 'aligned':
        return extract_aligned_last_token_pair(text, image, model, processor, layer)
    h_text = extract_llama_hidden_solo(text, model, processor, layer)
    h_combo = extract_llama_hidden_multimodal(text, image, model, processor, layer)
    delta = _l2_vector(h_combo - h_text)
    return (
        np.asarray(h_text, dtype=np.float32),
        np.asarray(h_combo, dtype=np.float32),
        np.asarray(delta, dtype=np.float32),
    )

def validate_combo_representation(combo_obj: dict, expected: str) -> str:
    """Fail fast when a probe from one S_C representation is scored as another."""
    expected = validate_sc_representation(expected)
    actual = combo_obj.get('sc_representation')
    if actual is None:
        formula = combo_obj.get('representation')
        if formula == SC_REPRESENTATION_FORMULAS['aligned']:
            actual = 'aligned'
        elif formula == SC_REPRESENTATION_FORMULAS['mean']:
            actual = 'mean'
    if actual is None:
        raise ValueError(
            'The combo probe has no S_C representation metadata. Retrain it with '
            '3.1_train_probe_combo.py before scoring.'
        )
    actual = validate_sc_representation(actual)
    if actual != expected:
        raise ValueError(
            f'Combo probe representation mismatch: model={actual}, requested={expected}.'
        )
    return actual

def extract_clip_hidden(image, model, processor, CLIP_LAYER, pooling='mean'):
    vision_tower = get_vision_tower(model)
    device = next(vision_tower.parameters()).device
    dtype = next(vision_tower.parameters()).dtype
    inputs = processor(images=image, text='a', return_tensors='pt').to(device)
    pixel_values = inputs['pixel_values'].to(device=device, dtype=dtype)
    with torch.no_grad():
        out = vision_tower(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    if out.hidden_states is None:
        raise RuntimeError('The vision tower returned no hidden_states.')
    layer_idx = CLIP_LAYER + 1
    if layer_idx >= len(out.hidden_states):
        raise IndexError(f'CLIP_LAYER={CLIP_LAYER} not valid. Hidden states available: {len(out.hidden_states)}')
    h = out.hidden_states[layer_idx].float()
    if pooling == 'cls':
        pooled = h[:, 0, :]
    elif pooling == 'mean':
        pooled = h[:, 1:, :].mean(dim=1)
    else:
        raise ValueError(f'Unrecognized pooling method: {pooling}')
    return pooled.squeeze(0).cpu().numpy()

def compute_sample_weights(y_combo: np.ndarray, sources: list[str]) -> np.ndarray:
    sources = np.array(sources)
    weights = np.ones(len(y_combo), dtype=float)
    cells = {}
    for src, lab in zip(sources, y_combo):
        cells[src, lab] = cells.get((src, lab), 0) + 1
    n_cells = len(cells)
    total = len(y_combo)
    target_per_cell = total / n_cells
    for i, (src, lab) in enumerate(zip(sources, y_combo)):
        cell_size = cells[src, lab]
        weights[i] = target_per_cell / cell_size
    print(f'  [weights] {n_cells} cells (source × label_combo)')
    print(f'  [weights] weight min={weights.min():.3f}  max={weights.max():.3f}  mean={weights.mean():.3f}')
    return weights
