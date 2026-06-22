# -*- coding: utf-8 -*-
"""Token initialization helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox


def connected_components_boxes(binary: np.ndarray, connectivity: int = 8):
    assert binary.ndim == 2
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    h, w = binary.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps = []
    ys, xs = np.where(binary)
    if ys.size == 0:
        return comps

    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue

        stack = [(int(y0), int(x0))]
        visited[y0, x0] = 1

        area = 0
        x_min = x_max = int(x0)
        y_min = y_max = int(y0)

        while stack:
            y, x = stack.pop()
            area += 1
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or (not binary[ny, nx]):
                    continue
                visited[ny, nx] = 1
                stack.append((ny, nx))

        comps.append((area, x_min, y_min, x_max, y_max))

    return comps


def boxes_from_label_map(label_map: np.ndarray, target_value: int) -> List[List[float]]:
    binary = (label_map == int(target_value))
    comps = connected_components_boxes(binary, connectivity=8)
    if not comps:
        return []
    comps.sort(key=lambda t: t[0], reverse=True)
    boxes = []
    for _, x_min, y_min, x_max, y_max in comps:
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        boxes.append([float(x_min), float(y_min), float(bw), float(bh)])
    return boxes


def make_geometric_prompt_from_boxes(model, image_size_wh, boxes_xywh, device):
    width, height = image_size_wh
    geo_prompt = model._get_dummy_prompt()
    if len(boxes_xywh) == 0:
        raise RuntimeError("No boxes to encode")

    box_xywh = torch.tensor(boxes_xywh, dtype=torch.float32, device=device).view(-1, 4)
    box_cxcywh = box_xywh_to_cxcywh(box_xywh).view(-1, 4)
    norm_boxes = normalize_bbox(box_cxcywh, width, height).tolist()

    boxes = torch.tensor(norm_boxes, device=device, dtype=torch.float32).view(-1, 1, 4)
    labels = torch.ones((len(boxes_xywh), 1), device=device, dtype=torch.bool)
    geo_prompt.append_boxes(boxes, labels)
    return geo_prompt


def extract_box_tokens_and_cls(prompt: torch.Tensor, n_boxes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_boxes <= 0:
        raise RuntimeError("n_boxes<=0: cannot extract box tokens")
    seq_len, batch, _ = prompt.shape
    if batch != 1:
        raise RuntimeError(f"Expect B=1, got B={batch}")
    if seq_len < (n_boxes + 1):
        raise RuntimeError(f"prompt.S={seq_len} < n_boxes+1={n_boxes+1}")
    box_tokens = prompt[:n_boxes, 0, :].float()
    cls_token = prompt[n_boxes, 0, :].float()
    return box_tokens, cls_token
