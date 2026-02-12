from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_GPU_POLICIES = {"off", "auto", "force"}
_CUDA_DEVICE_RE = re.compile(r"^cuda(?::(\d+))?$", re.IGNORECASE)


def _normalize_bool(value: Any, default_value: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default_value
    token = str(value).strip().lower()
    return token in {"1", "true", "yes", "on"}


def normalize_gpu_policy(value: Any, default_value: str = "off") -> str:
    token = str(value or default_value).strip().lower()
    if token in _GPU_POLICIES:
        return token
    fallback = str(default_value or "off").strip().lower()
    return fallback if fallback in _GPU_POLICIES else "off"


def normalize_gpu_device(value: Any, default_value: str = "auto") -> str:
    token = str(value or default_value).strip().lower()
    if token in {"auto", "cpu"}:
        return token
    if _CUDA_DEVICE_RE.match(token):
        if token == "cuda":
            return "cuda:0"
        return token
    fallback = str(default_value or "auto").strip().lower()
    if fallback in {"auto", "cpu"}:
        return fallback
    if _CUDA_DEVICE_RE.match(fallback):
        return "cuda:0" if fallback == "cuda" else fallback
    return "auto"


def normalize_gpu_model_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(Path(os.path.abspath(text)))


def normalize_gpu_config(gpu_cfg: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(gpu_cfg or {})
    return {
        "enabled": _normalize_bool(payload.get("enabled", False), False),
        "policy": normalize_gpu_policy(payload.get("policy", "off"), "off"),
        "device": normalize_gpu_device(payload.get("device", "auto"), "auto"),
        "embedding_model_path": normalize_gpu_model_path(
            payload.get("embedding_model_path", "")
        ),
        "reranker_model_path": normalize_gpu_model_path(
            payload.get("reranker_model_path", "")
        ),
    }


def _detect_nvidia_smi() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "reason": "nvidia-smi not found", "raw": []}
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        return {"available": False, "reason": str(exc), "raw": []}
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": proc.stderr.strip() or f"exit={proc.returncode}",
            "raw": [line.strip() for line in proc.stdout.splitlines() if line.strip()],
        }
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {"available": bool(rows), "reason": "", "raw": rows}


def _detect_torch_cuda() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-untyped]
    except Exception as exc:
        return {
            "installed": False,
            "version": "",
            "cuda_available": False,
            "cuda_version": "",
            "device_count": 0,
            "devices": [],
            "reason": f"torch_unavailable:{exc.__class__.__name__}",
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices: list[str] = []
    if cuda_available:
        for idx in range(device_count):
            try:
                devices.append(str(torch.cuda.get_device_name(idx)))
            except Exception:
                devices.append(f"cuda:{idx}")
    return {
        "installed": True,
        "version": str(getattr(torch, "__version__", "")),
        "cuda_available": cuda_available,
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
        "device_count": device_count,
        "devices": devices,
        "reason": "",
    }


def _resolve_requested_cuda_index(device: str) -> int:
    match = _CUDA_DEVICE_RE.match(str(device).strip().lower())
    if not match:
        return -1
    index = match.group(1)
    if index is None:
        return 0
    try:
        return int(index)
    except ValueError:
        return -1


def resolve_gpu_runtime(
    gpu_cfg: dict[str, Any] | None,
    *,
    raise_on_force_unavailable: bool = False,
) -> dict[str, Any]:
    config = normalize_gpu_config(gpu_cfg)
    torch_cuda = _detect_torch_cuda()
    nvidia_smi = _detect_nvidia_smi()

    embedding_model_path = str(config.get("embedding_model_path", ""))
    reranker_model_path = str(config.get("reranker_model_path", ""))
    embedding_exists = bool(embedding_model_path and Path(embedding_model_path).exists())
    reranker_exists = bool(reranker_model_path and Path(reranker_model_path).exists())

    enabled = bool(config.get("enabled", False))
    policy = str(config.get("policy", "off"))
    device = str(config.get("device", "auto"))

    use_gpu = False
    effective_device = "cpu"
    reason = "disabled_by_config"
    force_error = False

    cuda_available = bool(torch_cuda.get("cuda_available", False))
    cuda_count = int(torch_cuda.get("device_count", 0) or 0)

    if enabled and policy != "off":
        if device == "cpu":
            reason = "cpu_requested"
        elif not cuda_available:
            reason = "cuda_not_available"
            if policy == "force":
                force_error = True
        else:
            if device == "auto":
                requested_index = 0
            else:
                requested_index = _resolve_requested_cuda_index(device)
            if requested_index < 0 or requested_index >= cuda_count:
                reason = "invalid_cuda_device"
                if policy == "force":
                    force_error = True
            else:
                use_gpu = True
                effective_device = f"cuda:{requested_index}"
                reason = "gpu_enabled"
    elif enabled and policy == "off":
        reason = "policy_off"

    result = {
        "enabled": enabled,
        "policy": policy,
        "device": device,
        "use_gpu": bool(use_gpu),
        "effective_device": str(effective_device),
        "reason": str(reason),
        "force_error": bool(force_error),
        "embedding_model_path": embedding_model_path,
        "reranker_model_path": reranker_model_path,
        "embedding_model_exists": bool(embedding_exists),
        "reranker_model_exists": bool(reranker_exists),
        "torch": torch_cuda,
        "nvidia_smi": nvidia_smi,
    }

    if force_error and raise_on_force_unavailable:
        raise RuntimeError(
            "GPU policy is 'force' but CUDA runtime is unavailable or requested device is invalid"
        )
    return result
