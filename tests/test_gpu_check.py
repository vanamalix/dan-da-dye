"""scripts/gpu-check: a light import-and-run test on CPU, with no diffusion step and no network."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gpu-check"


def _load():
    loader = importlib.machinery.SourceFileLoader("gpu_check_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_is_executable():
    assert SCRIPT.stat().st_mode & 0o111


def test_choose_device_order():
    m = _load()
    assert m.choose_device("auto", True, True) == "cuda"
    assert m.choose_device("auto", False, True) == "mps"
    assert m.choose_device("auto", False, False) == "cpu"
    assert m.choose_device("cpu", True, True) == "cpu"


def test_runs_on_cpu(capsys):
    m = _load()
    assert m.main(["--device", "cpu", "--skip-diffusion", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["ok"] and info["device"] == "cpu" and info["backend"] == "CPU"
    assert info["matmul"]["fp32"]["ok"] and "diffusion" not in info


def test_expect_gpu_fails_on_cpu(capsys):
    m = _load()
    assert m.main(["--device", "cpu", "--skip-diffusion", "--expect-gpu"]) == 1
    assert "FAILED" in capsys.readouterr().out
