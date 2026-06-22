# -*- coding: utf-8 -*-
"""Loss helpers for POTSAM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_cosine_penalty(token_matrix: torch.Tensor) -> torch.Tensor:
    if token_matrix.dim() != 2 or token_matrix.shape[0] < 2:
        return torch.zeros((), device=token_matrix.device)
    normed = token_matrix / (token_matrix.norm(dim=1, keepdim=True) + 1e-12)
    sim = normed @ normed.t()
    off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    return off_diag.clamp_min(0.0).mean()


def dice_loss_from_probs(pred_prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred_prob.ndim == 2:
        pred_prob = pred_prob.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    pred = pred_prob.reshape(pred_prob.shape[0], -1)
    gt = target.reshape(target.shape[0], -1).float()
    inter = (pred * gt).sum(dim=1)
    denom = pred.sum(dim=1) + gt.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    targets_f = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets_f + (1.0 - prob) * (1.0 - targets_f)
    mod = (1.0 - p_t).pow(float(gamma))
    if 0.0 <= float(alpha) <= 1.0:
        alpha_t = float(alpha) * targets_f + (1.0 - float(alpha)) * (1.0 - targets_f)
        loss = alpha_t * mod * bce
    else:
        loss = mod * bce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
