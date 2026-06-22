# -*- coding: utf-8 -*-
"""Data utilities for POTSAM."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str, with_views: bool = False):
    """Load RGB image. with_views=True returns (proc, orig)."""
    orig = Image.open(path).convert("RGB")
    proc = orig.copy()
    if with_views:
        return proc, orig
    return proc


def load_label_map(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"label map must be 2D, got shape={arr.shape}")
    return arr.astype(np.int64)


def _resolve_path(path: str, config_dir: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((Path(config_dir) / p).resolve())


def normalize_config_paths(cfg: Dict, config_path: str) -> Dict:
    """Resolve relative paths in cfg against config file directory."""
    cfg = dict(cfg)
    cfg_path = str(Path(config_path).expanduser().resolve())
    cfg_dir = str(Path(cfg_path).parent)
    cfg["_config_path"] = cfg_path
    cfg["_config_dir"] = cfg_dir

    single_path_keys = [
        "sam3_checkpoint",
        "bpe_path",
        "output_dir",
        "train_image_dir",
        "train_label_dir",
        "val_image_dir",
        "val_label_dir",
        "test_image_dir",
        "test_label_dir",
        "track_token_path",
        "track_token_run_dir",
        "track_preprocessed_frames_dir",
    ]
    list_path_keys = [
        "train_image_paths",
        "train_label_paths",
        "val_image_paths",
        "val_label_paths",
        "test_image_paths",
        "test_label_paths",
    ]

    for k in single_path_keys:
        v = cfg.get(k, None)
        if isinstance(v, str) and v.strip():
            cfg[k] = _resolve_path(v.strip(), cfg_dir)

    for k in list_path_keys:
        v = cfg.get(k, None)
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            cfg[k] = [_resolve_path(str(x), cfg_dir) for x in v]

    return cfg


def _numeric_stem(path: str) -> Tuple[int, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"\d+", stem)
    if m is None:
        return (10**18, stem)
    return (int(m.group(0)), stem)


def list_images_sorted(image_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(image_dir):
        p = os.path.join(image_dir, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in _IMAGE_EXTS:
            paths.append(p)
    paths.sort(key=_numeric_stem)
    return paths


def read_split_pairs(cfg: Dict, split: str) -> Tuple[List[str], List[str]]:
    """
    Read image/label pairs from either *_dir or *_paths.

    split: "train" | "val" | "test"
    """
    image_dir = cfg.get(f"{split}_image_dir", "")
    label_dir = cfg.get(f"{split}_label_dir", "")
    image_paths_key = f"{split}_image_paths"
    label_paths_key = f"{split}_label_paths"

    if image_dir and label_dir:
        image_paths = list_images_sorted(image_dir)
        if not image_paths:
            raise RuntimeError(f"No images found in {image_dir}")
        label_paths: List[str] = []
        for p in image_paths:
            name = os.path.basename(p)
            lp = os.path.join(label_dir, name)
            if not os.path.isfile(lp):
                raise RuntimeError(f"Missing label for image: {name}")
            label_paths.append(lp)
        return image_paths, label_paths

    image_paths = list(cfg.get(image_paths_key, []))
    label_paths = list(cfg.get(label_paths_key, []))
    if len(image_paths) != len(label_paths):
        raise RuntimeError(f"{image_paths_key} and {label_paths_key} must have same length")
    if not image_paths:
        raise RuntimeError(f"No data provided for split `{split}`")
    return image_paths, label_paths


def sample_n_shot(
    image_paths: List[str],
    label_paths: List[str],
    n_shot: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if n_shot <= 0:
        return image_paths, label_paths
    total = len(image_paths)
    if n_shot == 1:
        return [image_paths[0]], [label_paths[0]]
    if n_shot >= total:
        return image_paths, label_paths
    rng = np.random.RandomState(int(seed))
    idx = np.sort(rng.choice(total, size=int(n_shot), replace=False))
    out_images = [image_paths[int(i)] for i in idx.tolist()]
    out_labels = [label_paths[int(i)] for i in idx.tolist()]
    return out_images, out_labels


def load_split(cfg: Dict, split: str) -> Tuple[List[Image.Image], List[Image.Image], List[np.ndarray], List[str]]:
    """Load one split; return (proc_images, viz_images, label_maps, image_paths)."""
    image_paths, label_paths = read_split_pairs(cfg, split)
    proc_images: List[Image.Image] = []
    viz_images: List[Image.Image] = []
    labels: List[np.ndarray] = []
    for img_path, lbl_path in zip(image_paths, label_paths):
        proc, orig = load_image(img_path, with_views=True)
        proc_images.append(proc)
        viz_images.append(orig)
        labels.append(load_label_map(lbl_path))
    return proc_images, viz_images, labels, image_paths


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float((inter + 1e-5) / (union + 1e-5))


def binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + 1e-5) / (denom + 1e-5))
