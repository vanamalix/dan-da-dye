"""scripts/setup.sh --dry-run on Linux: fast, no network, writes nothing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SETUP = REPO / "scripts" / "setup.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux path of setup.sh")


def run(root, *args):
    env = {**os.environ, "DANDADYE_PYTHON": sys.executable}
    env.pop("TORCH_INDEX_URL", None)
    return subprocess.run(["bash", str(SETUP), str(root), *args], capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("flavour, index", [("cpu", "whl/cpu"), ("rocm", "whl/rocm"), ("cuda", "whl/cu")])
def test_dry_run_uses_right_index_and_extras(tmp_path, flavour, index):
    root = tmp_path / "data"
    r = run(root, "--dry-run", "--torch", flavour)
    assert r.returncode == 0, r.stderr
    assert not root.exists() and list(tmp_path.iterdir()) == []
    pip_torch = [ln for ln in r.stdout.splitlines() if "pip install -q torch torchvision" in ln]
    assert len(pip_torch) == 2                                   # main venv and Magi venv
    assert all(f"https://download.pytorch.org/{index}" in ln for ln in pip_torch)
    assert "diffusers" in r.stdout and "mlx" not in r.stdout     # main venv extras on Linux
    assert r.stdout.index("pip install -q torch") < r.stdout.index("requirements-magi.txt")
    assert "hf download" in r.stdout


def test_dry_run_auto_without_gpu_is_cpu_and_writes_nothing(tmp_path):
    if os.path.exists("/dev/kfd"):
        pytest.skip("an AMD GPU is visible here, so auto would pick rocm")
    r = run(tmp_path / "data", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.iterdir()) == []


def test_refuses_data_root_inside_repo():
    r = run(REPO / "scratch-data", "--dry-run", "--torch", "cpu")
    assert r.returncode == 2
    assert "outside the repo" in r.stderr
    assert not (REPO / "scratch-data").exists()


def test_bad_torch_value_is_a_usage_error(tmp_path):
    assert run(tmp_path / "d", "--dry-run", "--torch", "tpu").returncode == 2
