# -*- coding: utf-8 -*-
"""Query gray/std prior fitting and filtering."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


def _odd_kernel(ksize: int) -> int:
    k = max(1, int(ksize))
    if (k % 2) == 0:
        k += 1
    return k


def gray_std_from_image(image_pil: Image.Image, std_ksize: int) -> Tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(image_pil.convert("L"), dtype=np.float32) / 255.0
    k = _odd_kernel(std_ksize)
    if k <= 1:
        return gray.astype(np.float32), np.zeros_like(gray, dtype=np.float32)

    x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    mean2 = F.avg_pool2d(x * x, kernel_size=k, stride=1, padding=k // 2)
    std = torch.sqrt((mean2 - mean * mean).clamp_min(0.0))[0, 0].numpy()
    return gray.astype(np.float32), std.astype(np.float32)


def fit_instance_prior_models(
    train_images: List[Image.Image],
    train_labels: List[np.ndarray],
    class_values: List[int],
    std_ksize: int,
    device: torch.device,
    eps: float = 1e-6,
    min_pixels: int = 4,
) -> List[Dict[str, torch.Tensor]]:
    """Fit per-class Gaussian on GT instance-level (gray_mean, std_mean)."""
    models: List[Dict[str, torch.Tensor]] = []
    min_pixels = max(1, int(min_pixels))

    for class_val in class_values:
        inst_feats: List[np.ndarray] = []
        for img, lbl_np in zip(train_images, train_labels):
            gray, std = gray_std_from_image(img, std_ksize=std_ksize)
            binary = (lbl_np == int(class_val)).astype(np.uint8)
            if int(binary.sum()) <= 0:
                continue
            comp_map, n_comp = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
            for comp_id in range(1, int(n_comp) + 1):
                m = comp_map == comp_id
                pix_n = int(m.sum())
                if pix_n < min_pixels:
                    continue
                inst_feats.append(
                    np.array([float(gray[m].mean()), float(std[m].mean())], dtype=np.float32)
                )

        if not inst_feats:
            raise RuntimeError(f"No valid GT instances for class value {class_val}")

        feat_arr = np.stack(inst_feats, axis=0).astype(np.float32)
        mu = feat_arr.mean(axis=0)
        sigma = np.maximum(feat_arr.std(axis=0), float(eps))
        models.append(
            {
                "mu_inst": torch.tensor(mu, device=device, dtype=torch.float32),
                "sigma_inst": torch.tensor(sigma, device=device, dtype=torch.float32),
                "n_instances": torch.tensor([feat_arr.shape[0]], device=device, dtype=torch.long),
            }
        )

    return models


def query_region_features_from_binary_masks(
    query_masks: torch.Tensor,
    gray_map: torch.Tensor,
    std_map: torch.Tensor,
    min_pixels: int = 4,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    query_masks: (Q,H,W) bool/float in {0,1}
    gray_map/std_map: (H,W)
    Return: feats (Q,2), valid (Q,)
    """
    if query_masks.dim() != 3:
        raise ValueError(f"query_masks must be (Q,H,W), got {tuple(query_masks.shape)}")

    q_num = int(query_masks.shape[0])
    if q_num <= 0:
        dev = gray_map.device
        return (
            torch.zeros((0, 2), device=dev, dtype=torch.float32),
            torch.zeros((0,), device=dev, dtype=torch.bool),
        )

    qf = query_masks.float().reshape(q_num, -1)
    gray_flat = gray_map.float().reshape(-1)
    std_flat = std_map.float().reshape(-1)
    mass = qf.sum(dim=1)
    denom = mass.clamp_min(float(eps))

    gray_mean = (qf @ gray_flat) / denom
    std_mean = (qf @ std_flat) / denom
    feats = torch.stack([gray_mean, std_mean], dim=1)
    valid = mass >= float(max(1, int(min_pixels)))
    return feats, valid
