# -*- coding: utf-8 -*-
"""Visualization utilities for train/infer/track."""

from __future__ import annotations

import math
import os
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_train_val_loss_curve(
    save_path: str,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    title: str,
    val_epochs: Sequence[int] | None = None,
):
    fig = plt.figure(figsize=(9, 4.5), dpi=160)
    ax = plt.gca()
    xs_tr = list(range(1, len(train_losses) + 1))

    ax.plot(xs_tr, list(train_losses), label="train")
    if len(val_losses) > 0:
        xs_va = list(range(1, len(val_losses) + 1)) if val_epochs is None else list(val_epochs)
        ax.plot(xs_va, list(val_losses), label="val")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _instance_map_per_class(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = label_map.shape
    inst_map = np.zeros((h, w), dtype=np.int32)
    inst_id = 1
    for c in range(1, num_classes + 1):
        binary = label_map == c
        if not binary.any():
            continue
        visited = np.zeros((h, w), dtype=np.uint8)
        ys, xs = np.where(binary)
        for y0, x0 in zip(ys, xs):
            if visited[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = 1
            inst_map[y0, x0] = inst_id
            while stack:
                y, x = stack.pop()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or (not binary[ny, nx]):
                        continue
                    visited[ny, nx] = 1
                    inst_map[ny, nx] = inst_id
                    stack.append((ny, nx))
            inst_id += 1
    return inst_map


def _boundary_mask(label_map: np.ndarray, inst_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if label_map[y, x] == 0:
                continue
            cur_lbl = label_map[y, x]
            cur_inst = inst_map[y, x]
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    boundary[y, x] = 1
                    break
                if label_map[ny, nx] != cur_lbl or inst_map[ny, nx] != cur_inst:
                    boundary[y, x] = 1
                    break
    return boundary


def _label_to_rgba_with_boundaries(label_map: np.ndarray, num_classes: int, alpha: float) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    h, w = label_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    inst_map = _instance_map_per_class(label_map, num_classes)
    boundary = _boundary_mask(label_map, inst_map).astype(bool)

    for c in range(1, num_classes + 1):
        color = cmap((c - 1) % 20)
        mask = label_map == c
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = alpha

    overlay[boundary, 0] = 0.0
    overlay[boundary, 1] = 0.0
    overlay[boundary, 2] = 0.0
    overlay[boundary, 3] = 1.0
    return overlay


def save_multiclass_figure(
    save_path: str,
    image_np: np.ndarray,
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    metric_value: float,
    alpha: float,
    num_classes: int,
    title_prefix: str,
):
    pred_np = pred_labels.detach().cpu().numpy().astype(np.int64)
    gt_np = gt_labels.detach().cpu().numpy().astype(np.int64)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=160)
    axes[0].imshow(image_np)
    axes[0].axis("off")
    axes[0].set_title(f"{title_prefix} image")

    axes[1].imshow(image_np)
    axes[1].imshow(_label_to_rgba_with_boundaries(pred_np, num_classes, alpha=alpha))
    axes[1].axis("off")
    axes[1].set_title(f"{title_prefix} pred IoU={metric_value:.4f}")

    axes[2].imshow(image_np)
    axes[2].imshow(_label_to_rgba_with_boundaries(gt_np, num_classes, alpha=alpha))
    axes[2].axis("off")
    axes[2].set_title(f"{title_prefix} gt")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_query_mask_grid(
    save_path: str,
    image_np: np.ndarray,
    query_masks: List[np.ndarray],
    title: str,
    query_scores: List[float] | None = None,
    query_ids: List[int] | None = None,
    max_queries: int = 64,
    cols: int = 8,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_total = len(query_masks)
    n_show = n_total if max_queries <= 0 else min(n_total, int(max_queries))
    cols = max(1, int(cols))
    rows = max(1, int(math.ceil(float(n_show + 1) / float(cols))))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows), dpi=160)
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(image_np)
    axes[0].axis("off")
    axes[0].set_title("image")

    for i in range(n_show):
        ax = axes[i + 1]
        m = query_masks[i].astype(np.uint8) * 255
        ax.imshow(m, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if query_ids is not None and i < len(query_ids):
            q_name = f"q{int(query_ids[i])}"
        else:
            q_name = f"q{i}"
        if query_scores is not None and i < len(query_scores):
            ax.set_title(f"{q_name} s={float(query_scores[i]):.2f}")
        else:
            ax.set_title(q_name)

    for i in range(n_show + 1, len(axes)):
        axes[i].axis("off")

    shown_text = f"show={n_show}/{n_total}" if n_total > n_show else f"show={n_total}"
    fig.suptitle(f"{title} ({shown_text})")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_class_figure(
    save_path: str,
    image_np: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    axes[0].imshow(image_np)
    axes[0].axis("off")
    axes[0].set_title("image")

    axes[1].imshow(pred_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    axes[1].set_title("pred")

    axes[2].imshow(gt_mask.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
    axes[2].axis("off")
    axes[2].set_title("gt")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
