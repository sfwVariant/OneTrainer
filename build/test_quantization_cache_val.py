import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.module.quantized.LinearW8A8 import LinearW8A8
from modules.util.enum.DataType import DataType
from modules.util.quantization_cache import (
    build_cache_info,
    save_quantization_cache,
    try_load_quantization_cache,
    _identity_json,
    CACHE_VERSION,
)
from safetensors.torch import save_file


class FakeQuantConfig:
    def __init__(self):
        self.cache_dir = "build/test_quant_cache_val"
        self.layer_filter = ""
        self.layer_filter_preset = "full"
        self.layer_filter_regex = False
        self.svd_dtype = DataType.NONE


class FakeConfig:
    def __init__(self, base_model_name="krea-2"):
        self.model_type = type("MT", (), {"__str__": lambda self: "KREA_2"})()
        self.base_model_name = base_model_name
        self.quantization = FakeQuantConfig()


class FakePart:
    def __init__(self, weight_dtype, model_name=""):
        self.weight_dtype = weight_dtype
        self.model_name = model_name


def make_comp():
    class Comp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = LinearW8A8(torch.int8, in_features=16, out_features=16, bias=True)
    c = Comp()
    with torch.no_grad():
        c.qkv.weight.data = torch.randn(16, 16, dtype=torch.bfloat16) * 0.5
        c.qkv.bias.data = torch.randn(16, dtype=torch.bfloat16) * 0.5
    return c


def main():
    shutil.rmtree("build/test_quant_cache_val", ignore_errors=True)
    os.makedirs("build/test_quant_cache_val", exist_ok=True)
    train_dtype = DataType.BFLOAT_16
    ok = True

    # 1. identity fields each change the hash / path
    base_info = build_cache_info(FakeConfig("krea-2"), "transformer", FakePart(DataType.INT_W8A8))
    variations = {
        "base_model": build_cache_info(FakeConfig("other"), "transformer", FakePart(DataType.INT_W8A8)),
        "component": build_cache_info(FakeConfig("krea-2"), "vae", FakePart(DataType.INT_W8A8)),
        "weight_dtype": build_cache_info(FakeConfig("krea-2"), "transformer", FakePart(DataType.FLOAT_W8A8)),
        "override": build_cache_info(FakeConfig("krea-2"), "transformer", FakePart(DataType.INT_W8A8, model_name="custom-model")),
    }
    for name, info in variations.items():
        changed = info.cache_path != base_info.cache_path
        print(f"  {'PASS' if changed else 'FAIL'} path changes with {name}")
        ok &= changed

    # 2. correct cache file loads
    comp = make_comp()
    with torch.no_grad():
        comp.qkv.compute_dtype = train_dtype.torch_dtype()
        comp.qkv.quantize(device=torch.device("cpu"))
    save_quantization_cache(comp, base_info)
    comp2 = make_comp()
    loaded = try_load_quantization_cache(comp2, base_info, train_dtype)
    print(f"  {'PASS' if loaded else 'FAIL'} correct cache file loads")
    ok &= loaded
    print(f"  {'PASS' if comp2.qkv._LinearW8A8__is_quantized else 'FAIL'} restored layer marked quantized")
    ok &= comp2.qkv._LinearW8A8__is_quantized

    # 3. defensive: a file at the correct path but with a mismatched identity is ignored
    bad_metadata = {
        "cache_version": str(CACHE_VERSION),
        #tamper the identity so it no longer matches
        "cache_identity": _identity_json({**base_info.identity, "source_identity": "tampered"}),
    }
    from safetensors import safe_open
    import time
    with safe_open(base_info.cache_path, framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    time.sleep(0.5)  #let Windows release the memory-mapped section
    #write the tampered file to a temp path, then rename into place (avoids in-place lock)
    tmp_path = base_info.cache_path + ".tampered"
    save_file(tensors, tmp_path, metadata=bad_metadata)
    os.replace(tmp_path, base_info.cache_path)
    comp3 = make_comp()
    ignored = try_load_quantization_cache(comp3, base_info, train_dtype)
    print(f"  {'PASS' if not ignored else 'FAIL'} tampered identity is ignored (returns False)")
    ok &= not ignored
    print(f"  {'PASS' if not comp3.qkv._LinearW8A8__is_quantized else 'FAIL'} ignored layer NOT marked quantized")
    ok &= not comp3.qkv._LinearW8A8__is_quantized

    # 4. wrong key set is ignored
    save_file({k: v for k, v in tensors.items() if "scale" in k}, base_info.cache_path + ".badkeys",
              metadata={"cache_version": str(CACHE_VERSION), "cache_identity": _identity_json(base_info.identity)})
    time.sleep(0.2)
    os.replace(base_info.cache_path + ".badkeys", base_info.cache_path)
    comp4 = make_comp()
    ignored_keys = try_load_quantization_cache(comp4, base_info, train_dtype)
    print(f"  {'PASS' if not ignored_keys else 'FAIL'} wrong key set is ignored (returns False)")
    ok &= not ignored_keys

    print()
    print("ALL OK" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
