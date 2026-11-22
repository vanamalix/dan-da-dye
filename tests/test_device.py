import importlib
import sys

import pytest

from dandadye import device
from dandadye.device import choose, gpu_flavour, pick_device, pick_dtype


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (True, False, "cuda"),
                                               (False, True, "mps"), (False, False, "cpu")])
def test_auto_prefers_cuda_then_mps_then_cpu(cuda, mps, expected):
    assert pick_device("auto", cuda=cuda, mps=mps) == expected
    assert pick_device(None, cuda=cuda, mps=mps) == expected


def test_explicit_devices():
    assert pick_device("cpu", cuda=True, mps=True) == "cpu"
    assert pick_device("cuda:1", cuda=True) == "cuda:1"
    assert pick_device("ROCm", cuda=True) == "cuda"          # ROCm torch uses the cuda device type
    assert pick_device("rocm:1", cuda=True) == "cuda:1"
    assert pick_device("mps", mps=True) == "mps"
    with pytest.raises(ValueError, match="no CUDA/ROCm"):
        pick_device("cuda")
    with pytest.raises(ValueError, match="MPS"):
        pick_device("mps")
    with pytest.raises(ValueError, match="unknown device"):
        pick_device("tpu")


def test_dtype_rules():
    assert pick_dtype("cuda", bf16=True) == "bfloat16"
    assert pick_dtype("cuda:1", bf16=False) == "float16"
    assert pick_dtype("mps", bf16=True) == "bfloat16"
    assert pick_dtype("mps", bf16=False) == "float32"
    assert pick_dtype("cpu", bf16=True) == "float32"
    assert pick_dtype("cpu", "bf16") == "bfloat16"            # explicit wins
    assert pick_dtype("cuda", "fp32", bf16=True) == "float32"
    with pytest.raises(ValueError, match="unknown dtype"):
        pick_dtype("cpu", "int4")


def test_gpu_flavour():
    assert gpu_flavour("6.4.1", None) == "rocm"
    assert gpu_flavour(None, "12.8") == "cuda"
    assert gpu_flavour(None, None) is None


def test_choose_reports_rocm_and_cuda():
    rocm = dict(torch="2.15", cuda=True, bf16_cuda=True, hip="6.4.1", cuda_version=None, name="Radeon")
    c = choose(flags=rocm)
    assert (c.device, c.dtype, c.flavour, c.is_gpu) == ("cuda", "bfloat16", "rocm", True)
    assert "ROCm 6.4.1" in c.detail and "Radeon" in c.detail
    nv = dict(torch="2.15", cuda=True, bf16_cuda=False, hip=None, cuda_version="12.8")
    c = choose(flags=nv)
    assert (c.dtype, c.flavour) == ("float16", "cuda") and "CUDA 12.8" in c.detail
    c = choose(flags=dict(torch=None))
    assert (c.device, c.dtype, c.is_gpu) == ("cpu", "float32", False) and "torch not installed" in c.detail


def test_device_module_needs_no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    mod = importlib.reload(device)
    try:
        assert mod.probe()["torch"] is None
        assert mod.choose().device == "cpu"
    finally:
        monkeypatch.undo()
        importlib.reload(device)


def test_magi_worker_defines_each_function_once():
    """A second pick_device later in the file silently replaced the cuda-aware one (and needed an unimported torch)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src/dandadye/detect/worker_magi_v2.py"
    names = [n.name for n in ast.parse(src.read_text()).body if isinstance(n, ast.FunctionDef)]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})
