import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(default: str = "cuda"):
    return torch.device(default if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def write_json(data: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

