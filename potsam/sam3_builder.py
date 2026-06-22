# -*- coding: utf-8 -*-
"""SAM3 build/runtime helpers for POTSAM."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model_builder import build_sam3_video_model


AMP_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}


def get_amp_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key not in AMP_DTYPES:
        raise ValueError(f"Unsupported amp dtype: {name}")
    return AMP_DTYPES[key]


def resolve_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_bpe_path(cfg: Dict) -> str:
    bpe_path = str(cfg.get("bpe_path", "")).strip()
    if bpe_path:
        return bpe_path
    sam3_root = os.path.abspath(os.path.join(os.path.dirname(sam3.__file__), ".."))
    return f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"


def prepare_output_dir(output_root: str) -> str:
    beijing_tz = timezone(timedelta(hours=8))
    run_dir = os.path.join(output_root, datetime.now(beijing_tz).strftime("%m%d%H%M"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_frozen_image_model(cfg: Dict, device: torch.device):
    model = build_sam3_image_model(
        bpe_path=resolve_bpe_path(cfg),
        checkpoint_path=cfg["sam3_checkpoint"],
        compile=bool(cfg.get("compile", False)),
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model, "num_interactive_steps_val"):
        model.num_interactive_steps_val = 0
    return model


def build_video_model_runtime(cfg: Dict, device: torch.device):
    model = build_sam3_video_model(
        checkpoint_path=cfg["sam3_checkpoint"],
        bpe_path=resolve_bpe_path(cfg),
        compile=bool(cfg.get("compile", False)),
        device=str(device),
    )
    model.eval()
    return model


def maybe_set_attr(obj, attr_name: str, value):
    """Set runtime attr only when value is not None and attr exists."""
    if value is None:
        return
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, float(value))
