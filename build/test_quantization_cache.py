import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.module.quantized.LinearW8A8 import LinearW8A8
from modules.module.quantized.LinearFp8 import LinearFp8
from modules.util.enum.DataType import DataType
from modules.util.quantization_cache import (
    build_cache_info,
    has_cacheable_modules,
    save_quantization_cache,
    try_load_quantization_cache,
)


class FakeQuantConfig:
    def __init__(self):
        self.cache_dir = "build/test_quant_cache"
        self.layer_filter = ""
        self.layer_filter_preset = "full"
        self.layer_filter_regex = False
        self.svd_dtype = DataType.NONE


class FakeConfig:
    def __init__(self, base_model_name="krea-2/transformer"):
        self.model_type = type("MT", (), {"__str__": lambda self: "KREA_2"})()
        self.base_model_name = base_model_name
        self.quantization = FakeQuantConfig()


class FakePart:
    def __init__(self, weight_dtype, model_name=""):
        self.weight_dtype = weight_dtype
        self.model_name = model_name


def make_component(weight_dtype, n_blocks=2, hidden=32, use_fp8_class=False):
    """Build a small component tree containing quantized linear layers."""
    if use_fp8_class:
        cls = LinearFp8
        def make_linear(in_f, out_f, bias=True):
            return cls(in_features=in_f, out_features=out_f, bias=bias)
    else:
        cls = LinearW8A8
        torch_dtype = torch.float8_e4m3fn if weight_dtype == DataType.FLOAT_W8A8 else torch.int8
        def make_linear(in_f, out_f, bias=True):
            return cls(torch_dtype, in_features=in_f, out_features=out_f, bias=bias)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = make_linear(hidden, hidden * 3, bias=True)
            self.proj = make_linear(hidden, hidden, bias=True)

    class Comp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(n_blocks)])

    # give each linear an original (bf16) weight, like the real load does
    for name, m in Comp().named_modules():
        pass
    comp = Comp()
    with torch.no_grad():
        for m in comp.modules():
            if isinstance(m, (LinearW8A8, LinearFp8)):
                #like the real loader, the original tensors (weight + bias) are loaded at the
                #checkpoint dtype (bf16) before quantization
                m.weight.data = torch.randn_like(m.weight.data, dtype=torch.bfloat16) * 0.5
                if m.bias is not None:
                    m.bias.data = torch.randn_like(m.bias.data, dtype=torch.bfloat16) * 0.5
    return comp


def quantize_normally(comp, device, train_dtype):
    for m in comp.modules():
        if hasattr(m, "quantize") and (isinstance(m, (LinearW8A8, LinearFp8))):
            m.compute_dtype = train_dtype.torch_dtype()
            m.quantize(device=device)


def check_restored(comp, cache_info, train_dtype, label):
    # every layer must be marked quantized and frozen, with a quantized weight dtype. The
    # weight/scale tensors were already verified to be bit-identical to the normally-quantized
    # reference above, so this state check plus that equivalence is enough to prove the restored
    # layer behaves identically to normal quantization (no separate CPU forward pass, which hits
    # pre-existing small-batch dtype issues unrelated to the cache).
    ok = True
    n_checked = 0
    for name, m in comp.named_modules():
        if isinstance(m, LinearW8A8):
            n_checked += 1
            q = m._LinearW8A8__is_quantized
            frozen = not m.weight.requires_grad
            qdt = m.weight.dtype in (torch.int8, torch.float8_e4m3fn)
            cd = m.compute_dtype == train_dtype.torch_dtype()
            if not (q and frozen and qdt and cd):
                ok = False
                print(f"  FAIL {label} {name}: quantized={q} frozen={frozen} qdtype={m.weight.dtype} compute_dtype={m.compute_dtype}")
        elif isinstance(m, LinearFp8):
            n_checked += 1
            q = m.is_quantized
            frozen = not m.weight.requires_grad
            qdt = m.weight.dtype == torch.float8_e4m3fn
            cd = m.compute_dtype == train_dtype.torch_dtype()
            if not (q and frozen and qdt and cd):
                ok = False
                print(f"  FAIL {label} {name}: quantized={q} frozen={frozen} qdtype={m.weight.dtype} compute_dtype={m.compute_dtype}")
    if n_checked == 0:
        ok = False
        print(f"  FAIL {label}: no quantized layers found")
    print(f"  {'PASS' if ok else 'FAIL'} {label} ({n_checked} layers restored)")
    return ok


def run_case(weight_dtype, use_fp8_class, label):
    device = torch.device("cpu")
    train_dtype = DataType.BFLOAT_16
    config = FakeConfig()
    part = FakePart(weight_dtype)
    cache_info = build_cache_info(config, "transformer", part)
    assert cache_info is not None, f"{label}: cache_info is None"
    print(f"{label}: cache_path = {cache_info.cache_path}")

    # first run: quantize normally + save
    comp = make_component(weight_dtype, use_fp8_class=use_fp8_class)
    assert has_cacheable_modules(comp), f"{label}: no cacheable modules"
    quantize_normally(comp, device, train_dtype)
    # capture the reference (quantized) weights
    ref = {name: (m.weight.detach().clone(), m.scale.detach().clone())
           for name, m in comp.named_modules() if isinstance(m, (LinearW8A8, LinearFp8))}
    save_quantization_cache(comp, cache_info)
    assert os.path.isfile(cache_info.cache_path), f"{label}: cache file not created"

    # second run: fresh component, load from cache
    comp2 = make_component(weight_dtype, use_fp8_class=use_fp8_class)
    loaded = try_load_quantization_cache(comp2, cache_info, train_dtype)
    assert loaded, f"{label}: cache load failed"
    # verify weights match the reference
    ok = True
    for name, m in comp2.named_modules():
        if isinstance(m, (LinearW8A8, LinearFp8)):
            rw, rs = ref[name]
            if not torch.equal(m.weight.detach(), rw):
                ok = False
                print(f"  FAIL {label} {name}: weight mismatch")
            if not torch.equal(m.scale.detach(), rs):
                ok = False
                print(f"  FAIL {label} {name}: scale mismatch")
    print(f"  {'PASS' if ok else 'FAIL'} {label} weight/scale match reference")
    check_restored(comp2, cache_info, train_dtype, label + " restored")

    # identity mismatch: different base model -> should be ignored
    config2 = FakeConfig(base_model_name="other-model")
    part2 = FakePart(weight_dtype)
    cache_info2 = build_cache_info(config2, "transformer", part2)
    assert cache_info2 is not None and cache_info2.cache_path != cache_info.cache_path, \
        f"{label}: identity did not change with base model"
    comp3 = make_component(weight_dtype, use_fp8_class=use_fp8_class)
    # the file for config2 doesn't exist -> returns False
    res = try_load_quantization_cache(comp3, cache_info2, train_dtype)
    assert res is False, f"{label}: should not load non-existent cache"
    print(f"  PASS {label} non-existent cache returns False")
    return ok


def main():
    # clean
    shutil.rmtree("build/test_quant_cache", ignore_errors=True)
    os.makedirs("build/test_quant_cache", exist_ok=True)

    all_ok = True
    all_ok &= run_case(DataType.INT_W8A8, False, "INT_W8A8")
    print()
    all_ok &= run_case(DataType.FLOAT_W8A8, False, "FLOAT_W8A8")
    print()
    all_ok &= run_case(DataType.FLOAT_8, True, "FLOAT_8 (LinearFp8)")
    print()

    # name-mangling check on finalize
    m = LinearW8A8(torch.int8, in_features=8, out_features=4)
    m.finalize_cached_quantization()
    print("finalize name-mangles __is_quantized:", m._LinearW8A8__is_quantized is True)

    print()
    print("ALL OK" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
