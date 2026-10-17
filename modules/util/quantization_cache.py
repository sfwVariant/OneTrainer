#v1.0.0.0.000
"""
Quantized weight cache.

Saves and restores the *quantized state* of a model component (transformer, text
encoder, VAE, ...) so that an identical run does not have to re-quantize the same
weights.

Only the quantized tensors (weight + scale) of the cacheable linear layers are
stored - never the full base component. Caches are independent per component.

Supported formats: INT_W8A8, FLOAT_W8A8, and (optionally) plain FLOAT_8.
Excluded: NF4, compressed W8A8, SVDQuant, GGUF.

Cache files live in ``config.quantization.cache_dir`` (which the trainer already
points at ``<cache_dir>/quantization``) and are named like
``krea_2_transformer_int_w8a8_a93f1742.safetensors``.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import modules.util.multi_gpu_util as multi

from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.enum.DataType import DataType
from safetensors import safe_open
from safetensors.torch import save_file

#bump this whenever the on-disk layout changes; it is part of the cache identity, so a
#bumped version produces a brand new cache file instead of reusing a stale one
CACHE_VERSION = 1


@dataclass
class QuantizationCacheInfo:
    cache_dir: str
    cache_version: int
    model_type: str
    component: str
    source_identity: str
    weight_dtype: str
    layer_filter: str
    layer_filter_preset: str
    layer_filter_regex: bool
    readable_name: str
    cache_path: str
    identity: dict


def _log(message: str):
    if multi.is_master():
        print(message)


def _identity_json(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _compute_hash(identity: dict) -> str:
    canonical = _identity_json(identity)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _build_source_identity(config: TrainConfig, config_part: TrainModelPartConfig) -> str | None:
    # HF revisions are not used anywhere in the codebase, so identity is derived from the
    # config's model names: the base model plus an explicit per-component override, if any.
    base_model = config.base_model_name
    if not base_model:
        return None
    override = config_part.model_name
    if override:
        return f"{base_model}|{override}"
    return base_model


def build_cache_info(
        config: TrainConfig,
        cache_component: str,
        config_part: TrainModelPartConfig,
) -> QuantizationCacheInfo | None:
    """Build the cache identity / file location for a component, or None if the identity
    cannot be established (e.g. no base model name)."""
    cache_dir = config.quantization.cache_dir
    if cache_dir is None:
        return None

    model_type = str(config.model_type)
    component = str(cache_component)
    source_identity = _build_source_identity(config, config_part)
    if source_identity is None:
        return None

    weight_dtype = str(config_part.weight_dtype)
    layer_filter = config.quantization.layer_filter
    layer_filter_preset = config.quantization.layer_filter_preset
    layer_filter_regex = config.quantization.layer_filter_regex

    identity = {
        "cache_version": CACHE_VERSION,
        "model_type": model_type,
        "component": component,
        "source_identity": source_identity,
        "weight_dtype": weight_dtype,
        "layer_filter": layer_filter,
        "layer_filter_preset": layer_filter_preset,
        "layer_filter_regex": layer_filter_regex,
    }

    cache_hash = _compute_hash(identity)
    readable_name = f"{model_type.lower()}_{component.lower()}_{weight_dtype.lower()}"
    cache_path = os.path.join(cache_dir, f"{readable_name}_{cache_hash}.safetensors")

    return QuantizationCacheInfo(
        cache_dir=cache_dir,
        cache_version=CACHE_VERSION,
        model_type=model_type,
        component=component,
        source_identity=source_identity,
        weight_dtype=weight_dtype,
        layer_filter=layer_filter,
        layer_filter_preset=layer_filter_preset,
        layer_filter_regex=layer_filter_regex,
        readable_name=readable_name,
        cache_path=cache_path,
        identity=identity,
    )


def _is_cacheable_instance(sub_module, LinearW8A8, LinearFp8) -> bool:
    #the module must be one of the concrete cacheable linear classes, and - for the
    #compressed-capable W8A8 - it must not actually be compressed
    if isinstance(sub_module, LinearW8A8):
        return not sub_module._compressed
    if isinstance(sub_module, LinearFp8):
        return True
    return False


def _linear_classes():
    #imported lazily to avoid a circular import: LinearW8A8 <- quantization_util <-
    #quantization_cache
    from modules.module.quantized.LinearFp8 import LinearFp8
    from modules.module.quantized.LinearW8A8 import LinearW8A8
    return LinearW8A8, LinearFp8


def has_cacheable_modules(module) -> bool:
    LinearW8A8, LinearFp8 = _linear_classes()
    for _, sub_module in module.named_modules():
        if _is_cacheable_instance(sub_module, LinearW8A8, LinearFp8):
            return True
    return False


def _collect_state(module) -> dict:
    """Collect the quantized tensors (weight + scale) of every cacheable linear layer,
    keyed by their path inside the component (e.g. ``blocks.0.attn.qkv.weight``)."""
    LinearW8A8, LinearFp8 = _linear_classes()
    state = {}
    for name, sub_module in module.named_modules():
        if name == "":
            continue
        if _is_cacheable_instance(sub_module, LinearW8A8, LinearFp8):
            state[f"{name}.weight"] = sub_module.weight.detach()
            state[f"{name}.scale"] = sub_module.scale.detach()
    return state


def _expected_keys(module) -> set:
    LinearW8A8, LinearFp8 = _linear_classes()
    keys = set()
    for name, sub_module in module.named_modules():
        if name == "":
            continue
        if _is_cacheable_instance(sub_module, LinearW8A8, LinearFp8):
            keys.add(f"{name}.weight")
            keys.add(f"{name}.scale")
    return keys


def save_quantization_cache(module, cache_info: QuantizationCacheInfo) -> None:
    """Persist the quantized state to disk. Writes to a temp file first, then atomically
    renames it into place so a crash can never leave a half-written cache behind."""
    state = _collect_state(module)
    if not state:
        return

    #safetensors requires CPU tensors
    cpu_state = {k: v.cpu() for k, v in state.items()}

    metadata = {
        "cache_version": str(cache_info.cache_version),
        "cache_identity": _identity_json(cache_info.identity),
        "model_type": cache_info.model_type,
        "component": cache_info.component,
        "source_identity": cache_info.source_identity,
        "weight_dtype": cache_info.weight_dtype,
    }

    os.makedirs(cache_info.cache_dir, exist_ok=True)
    tmp_path = f"{cache_info.cache_path}.tmp"
    save_file(cpu_state, tmp_path, metadata=metadata)
    os.replace(tmp_path, cache_info.cache_path)


def _load_state_into_modules(module, state: dict, train_dtype: DataType) -> None:
    LinearW8A8, LinearFp8 = _linear_classes()
    compute_dtype = train_dtype.torch_dtype()
    for name, sub_module in module.named_modules():
        if name == "":
            continue
        if not _is_cacheable_instance(sub_module, LinearW8A8, LinearFp8):
            continue

        device = sub_module.weight.device
        #the weight Parameter must be non-grad before it can hold the non-float quantized data
        sub_module.requires_grad_(False)
        #replace the (still unquantized, original-dtype) parameter data with the cached
        #quantized tensors; assigning .data also changes the parameter's dtype
        sub_module.weight.data = state[f"{name}.weight"].to(device=device)
        sub_module.scale.data = state[f"{name}.scale"].to(device=device)
        #the normal quantization flow sets compute_dtype before quantizing; mirror that
        sub_module.compute_dtype = compute_dtype
        sub_module.finalize_cached_quantization()


def try_load_quantization_cache(module, cache_info: QuantizationCacheInfo, train_dtype: DataType) -> bool:
    """Attempt to restore the component's quantized state from the cache. Returns True if
    the cache existed, validated, and was restored; False otherwise (caller falls back to
    quantizing normally)."""
    if not os.path.isfile(cache_info.cache_path):
        return False

    try:
        with safe_open(cache_info.cache_path, framework="pt") as f:
            metadata = f.metadata() or {}
            keys = list(f.keys())
            state = {k: f.get_tensor(k) for k in keys}
    except Exception as e:
        _log(f"Failed to read quantization cache {cache_info.cache_path}: {e}")
        return False

    #the full identity must match, otherwise the cache belongs to a different component /
    #source / format / filter / version and has to be ignored
    if metadata.get("cache_version") != str(cache_info.cache_version) \
            or metadata.get("cache_identity") != _identity_json(cache_info.identity):
        _log("Quantization cache identity mismatch; ignoring cache")
        return False

    if set(state.keys()) != _expected_keys(module):
        _log("Quantization cache key mismatch; ignoring cache")
        return False

    try:
        _load_state_into_modules(module, state, train_dtype)
    except Exception as e:
        _log(f"Failed to restore quantized weights from cache: {e}")
        return False

    return True
