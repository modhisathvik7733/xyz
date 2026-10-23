"""Small utilities shared by training and benchmarking."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def apply_overrides(cfg: dict, overrides: Optional[Iterable[str]]) -> dict:
    """Apply ``--set a.b=value`` overrides; values are parsed as YAML."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set expects a non-empty key, got {item!r}")

        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw_value)
    return cfg


def save_json(path: str | os.PathLike, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    os.replace(tmp, path)


def load_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def format_int(value: int) -> str:
    return f"{value:,}"


def get_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * coeff


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
