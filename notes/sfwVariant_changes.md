# Changes from official repo

- Added saving & loading quantized models so they don't get re-quantized every time
- Added option checkbox in General tab to unload text encoder fully after caching
  - Massive RAM saver if you don't need the text encoder for anything after caching is done
  - Automatically disabled if: `Cache Only` is enabled, or `Latent Caching` is disabled, or training a text encoder lora
  - Unloads after first cache, i.e. everything prior to epoch 1
      - And unloads when `Clear cache before training` is disabled
  - Does NOT intelligently check other settings like Concept text variations
     - PROBLEM: OneTrainer makes extra variations on the fly, so it'll reload the text encoder every time it needs to make new variations
     - SOLUTION: if you're doing multiple variations, do a `Cache Only` run first to pre-cache all variations and then turn off `Clear cache before training` for the real run


Specs below (there are two).

###########
##########
########

# Quantized Weight Cache — v1

## Task

Create a new feature for automatically saving & loading quantized model states when executing training runs. The goal is to reduce instances of re-quantizing models on every training run, which is the current behaviour.

## Scope

Cache straightforward quantization formats only:

- `INT_W8A8`
- `FLOAT_W8A8`
- optionally plain FP8 if implementation proves equally simple

Unsupported formats continue using existing quantization unchanged. Exclude NF4, compressed W8A8 and SVDQuant from v1.

## Files

### `modules/util/quantization_cache.py` — new

Handle cache identity, save/load and validation.

Cache files live under the existing:

```text
<cache_dir>/quantization/
```

Generate a readable filename plus hash:

```text
krea_2_transformer_int_w8a8_a93f1742.safetensors
```

Hash inputs should include:

```python
{
    "cache_version": 1,
    "model_type": ...,
    "component": ...,          # e.g. "transformer"
    "source_identity": ...,    # path/repo + size/mtime or HF revision
    "weight_dtype": ...,
    "layer_filter": ...,
    "layer_filter_preset": ...,
    "layer_filter_regex": ...,
}
```

Save only state belonging to quantized modules rather than duplicating the entire base component.

```python
def try_load_quantization_cache(module, cache_info) -> bool:
    path = get_cache_path(cache_info)
    if not path.exists():
        return False

    state = load_file(path)
    load_quantized_state(module, state)
    finalize_cached_quantization(module)
    return True
```

Use atomic save (`temporary file -> rename`) so interrupted training cannot leave a valid-looking partial cache.

---

### `modules/module/quantized/LinearW8A8.py` — edit

Add an explicit restoration hook for state not represented in `state_dict()`:

```python
def finalize_cached_quantization(self):
    self.requires_grad_(False)
    self.__is_quantized = True
```

This allows cached `weight` and `scale` tensors to be loaded without calling `quantize()` again.

If plain FP8 is included, add the equivalent hook to `LinearFp8`.

---

### `modules/util/quantization_util.py` — edit

Extend `quantize_layers()` with optional cache information.

Current conceptual flow:

```python
quantize_layers(...)
```

becomes:

```python
if cache_supported and try_load_quantization_cache(module, cache_info):
    print(f"Loaded quantized weights from cache: {cache_path}")
    return

quantize_layers_normally(...)

if cache_supported:
    save_quantization_cache(module, cache_info)
```

The existing quantization loop remains the fallback and source of truth.

Cache support should be selected by quantized module type rather than assuming every `QuantizedModuleMixin` implementation is serializable.

---

### `modules/modelSetup/BaseModelSetup.py` — edit

Pass component identity into `quantize_layers()` so independent model parts receive independent caches:

```python
quantize_layers(
    module,
    self.train_device,
    train_dtype,
    config,
    compress=config_part.weight_dtype.is_compressed(),
    cache_component=attr,
)
```

For example:

```text
transformer -> its own cache
text_encoder -> its own cache
vae -> its own cache
```

Direct/legacy `quantize_layers()` callers can retain the existing signature via a default `cache_component=None`; caching can simply be disabled when insufficient identity information is available.

## Expected behaviour

First Krea2 Raw run:

```text
Quantizing model weights: 100% ...
Saving quantized weight cache...
```

Subsequent identical run:

```text
Loading cached quantized weights: ...transformer_int_w8a8_....safetensors
```

No quantization pass occurs.

Changing the base model, transformer override, quantization dtype, layer filters, or cache format produces a different key and therefore automatically falls back to normal quantization and creates a new cache.



###########
##########
########

# Text encoder unloading

# Task Summary

Add an option which allows the text encoder to be unloaded from memory during training.

* Add a new checkbox option in the `General` tab (and associated json config setting): `Unload text encoder after caching`
    * Positioned after the `Temp Device` option
* When set to true, this causes the text encoder to be fully unloaded from VRAM/RAM once the initial caching step is completed (`Latent Caching` enabled) in order to free memory
    * Similarly, the text encoder should also be unloaded if the cache is being re-used due to `Clear cache before training` being disabled
* The option is implicitly ignored (i.e. taken as false) when training text encoder loras, and when `Latent Caching` is disabled

# Scope

Support only `Krea 2` and `Anima` models.

# Spec

## 1. Configuration and UI

**`modules/util/config/TrainConfig.py`**
- Add `unload_text_encoder_after_caching: bool`, default `False`.
- Existing configurations should retain the default.

**`modules/ui/BaseTrainUIView.py`**
- Add a checkbox labelled `Unload text encoder after caching` immediately below `Temp Device` (row 16).

## 2. Unloading logic

**`modules/trainer/GenericTrainer.py`**
- Add an effective-enabled check requiring:
  - Option enabled.
  - Latent caching enabled.
  - No text encoder or embedding training.
- Trigger unloading after the initial cache is complete, including validation caching where applicable.
- When reusing an existing complete cache, unload immediately after confirming its contents are sufficient.
- Check for cache misses before subsequent epochs; reload the encoder if additional encoding becomes necessary.

```python
def should_unload_text_encoder(config):
    return (
        config.unload_text_encoder_after_caching
        and config.latent_caching
        and not any(
            getattr(config, part).train
            for part in config.model_type.text_encoder_parts()
        )
        and not config.train_any_embedding()
        and not config.train_any_output_embedding()
    )
```

**`modules/model/BaseModel.py`**
- Add a generic `unload_text_encoders()` method using `model_type.text_encoder_parts()`.
- Remove the actual encoder modules and associated references, rather than moving them to `temp_device`.
- Release allocated memory using garbage collection and CUDA cache cleanup.
- Ensure later `materialize()` / `evict()` operations safely ignore unloaded components.

**`modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`** and relevant model setups
- Release encoder references retained by caching pipeline nodes and callbacks. Otherwise, removing `model.text_encoder` alone will not free its memory.
- Guard setup routines that call `.eval()`, `.train()`, or otherwise access unloaded encoders.

## 3. Cache reuse optimisation

- Ensure encoder unload occurs after cache verification when cache is being reused.

**`modules/modelLoader/`** and **`modules/trainer/GenericTrainer.py`**

## Other

**Important:** OneTrainer currently creates its data loader in `GenericTrainer.start()`, but populates caches through `start_next_epoch()`. Unloading must therefore happen after the required cache operations finish, not immediately after constructing the data loader.