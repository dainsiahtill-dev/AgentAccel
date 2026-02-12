from __future__ import annotations

import pytest

import accel.gpu_runtime as gpu_runtime


def test_resolve_gpu_runtime_off_policy_disables_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_torch_cuda",
        lambda: {
            "installed": True,
            "version": "2.x",
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 2,
            "devices": ["GPU0", "GPU1"],
            "reason": "",
        },
    )
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_nvidia_smi",
        lambda: {"available": True, "reason": "", "raw": ["GPU0", "GPU1"]},
    )
    result = gpu_runtime.resolve_gpu_runtime(
        {"enabled": True, "policy": "off", "device": "cuda:0"}
    )
    assert bool(result.get("use_gpu", True)) is False
    assert str(result.get("effective_device", "")) == "cpu"
    assert str(result.get("reason", "")) == "policy_off"


def test_resolve_gpu_runtime_auto_selects_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_torch_cuda",
        lambda: {
            "installed": True,
            "version": "2.x",
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 2,
            "devices": ["GPU0", "GPU1"],
            "reason": "",
        },
    )
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_nvidia_smi",
        lambda: {"available": True, "reason": "", "raw": ["GPU0", "GPU1"]},
    )
    result = gpu_runtime.resolve_gpu_runtime(
        {"enabled": True, "policy": "auto", "device": "auto"}
    )
    assert bool(result.get("use_gpu", False)) is True
    assert str(result.get("effective_device", "")) == "cuda:0"
    assert str(result.get("reason", "")) == "gpu_enabled"


def test_resolve_gpu_runtime_force_raises_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_torch_cuda",
        lambda: {
            "installed": True,
            "version": "2.x",
            "cuda_available": False,
            "cuda_version": "",
            "device_count": 0,
            "devices": [],
            "reason": "",
        },
    )
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_nvidia_smi",
        lambda: {"available": True, "reason": "", "raw": ["GPU0"]},
    )

    result = gpu_runtime.resolve_gpu_runtime(
        {"enabled": True, "policy": "force", "device": "auto"},
        raise_on_force_unavailable=False,
    )
    assert bool(result.get("force_error", False)) is True
    assert bool(result.get("use_gpu", True)) is False
    assert str(result.get("reason", "")) == "cuda_not_available"

    with pytest.raises(RuntimeError):
        gpu_runtime.resolve_gpu_runtime(
            {"enabled": True, "policy": "force", "device": "auto"},
            raise_on_force_unavailable=True,
        )


def test_resolve_gpu_runtime_invalid_device_with_auto_policy_falls_back_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_torch_cuda",
        lambda: {
            "installed": True,
            "version": "2.x",
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 1,
            "devices": ["GPU0"],
            "reason": "",
        },
    )
    monkeypatch.setattr(
        gpu_runtime,
        "_detect_nvidia_smi",
        lambda: {"available": True, "reason": "", "raw": ["GPU0"]},
    )

    result = gpu_runtime.resolve_gpu_runtime(
        {"enabled": True, "policy": "auto", "device": "cuda:7"}
    )
    assert bool(result.get("use_gpu", True)) is False
    assert str(result.get("effective_device", "")) == "cpu"
    assert str(result.get("reason", "")) == "invalid_cuda_device"
