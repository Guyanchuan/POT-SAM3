# -*- coding: utf-8 -*-
"""Static-image inference/evaluation for POTSAM."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from potsam.data import (
        binary_dice,
        binary_iou,
        load_image,
        load_label_map,
        normalize_config_paths,
        read_split_pairs,
    )
    from potsam.prior import query_region_features_from_binary_masks
    from potsam.sam3_builder import build_frozen_image_model, get_amp_dtype, resolve_device
    from potsam.sam3_runner import Sam3Runner
    from potsam.viz import save_class_figure
else:
    from .data import (
        binary_dice,
        binary_iou,
        load_image,
        load_label_map,
        normalize_config_paths,
        read_split_pairs,
    )
    from .prior import query_region_features_from_binary_masks
    from .sam3_builder import build_frozen_image_model, get_amp_dtype, resolve_device
    from .sam3_runner import Sam3Runner
    from .viz import save_class_figure


def latest_run_dir(output_root: str) -> str:
    dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    if not dirs:
        raise RuntimeError(f"No run directories in {output_root}")
    numeric = [d for d in dirs if d.isdigit()]
    if numeric:
        numeric.sort(key=lambda x: int(x))
        return os.path.join(output_root, numeric[-1])
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(output_root, d)))
    return os.path.join(output_root, dirs[-1])


def load_checkpoint(run_dir: str) -> Tuple[List[torch.Tensor], Dict]:
    pt_files = glob.glob(os.path.join(run_dir, "trained_tokens.pt"))
    if not pt_files:
        raise RuntimeError(f"No trained token checkpoint in {run_dir}")
    data = torch.load(pt_files[0], map_location="cpu")
    token_keys = sorted(
        [k for k in data.keys() if str(k).startswith("T_class")],
        key=lambda k: int(str(k).replace("T_class", "")),
    )
    tokens = [data[k] for k in token_keys]
    return tokens, data


def remove_speckles(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= int(min_area)
    keep[0] = False
    return keep[labels]


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    areas[0] = -1
    best = int(np.argmax(areas))
    return labels == best


def run_infer(cfg: Dict):
    run_dir = latest_run_dir(cfg["output_dir"])
    infer_dir = os.path.join(run_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)

    tokens_cpu, ckpt_data = load_checkpoint(run_dir)
    prior_enabled = bool(cfg.get("query_prior_enabled", False))
    ckpt_prior = ckpt_data.get("query_prior", None)

    device = resolve_device(cfg["device"])
    amp_dtype = get_amp_dtype(cfg["amp_dtype"])

    model = build_frozen_image_model(cfg, device)
    runner = Sam3Runner(
        model,
        device,
        cfg["use_autocast"],
        amp_dtype,
        prior_std_ksize=int(cfg.get("query_prior_std_ksize", 9)),
    )

    image_paths, label_paths = read_split_pairs(cfg, "test")
    test_images = []
    test_images_viz = []
    test_labels = []
    for img_path, lbl_path in zip(image_paths, label_paths):
        proc, orig = load_image(img_path, with_views=True)
        test_images.append(proc)
        test_images_viz.append(orig)
        test_labels.append(load_label_map(lbl_path))

    num_classes = len(cfg["class_label_values"])
    token_list = [t.detach().float().to(device) for t in tokens_cpu]
    if len(token_list) != num_classes:
        raise RuntimeError(f"Token/class mismatch: {len(token_list)} vs {num_classes}")

    prior_models: List[Dict[str, torch.Tensor]] = []
    if prior_enabled and isinstance(ckpt_prior, dict):
        cls_priors = ckpt_prior.get("classes", [])
        if len(cls_priors) == num_classes:
            for cp in cls_priors:
                prior_models.append(
                    {
                        "mu_inst": cp["mu_inst"].detach().float().to(device),
                        "sigma_inst": cp["sigma_inst"].detach().float().to(device),
                    }
                )
        else:
            prior_enabled = False
    else:
        prior_enabled = False

    query_score_threshold = float(cfg.get("query_score_threshold", 0.5))
    infer_logit_threshold = float(cfg.get("infer_logit_threshold", 0.5))
    infer_query_topk = int(cfg.get("infer_query_topk", 0))
    infer_query_lcc = bool(cfg.get("infer_query_lcc", True))
    infer_det_nms_thresh = float(cfg.get("infer_det_nms_thresh", 0.0))
    speckle_min_area = int(cfg.get("speckle_min_area", 0))
    use_detector_path = bool(cfg.get("infer_use_detector_path", True))

    prior_n_sigma = float(cfg.get("query_prior_n_sigma", 3.0))
    prior_min_pixels = int(cfg.get("query_prior_min_pixels", 4))

    per_class_acc = {c: {"iou": [], "dice": []} for c in range(1, num_classes + 1)}
    per_image = []

    for idx, (img, lbl_np) in tqdm(list(enumerate(zip(test_images, test_labels))), total=len(test_images), desc="infer"):
        with torch.no_grad():
            state = runner.prepare_image(img, need_prior_maps=prior_enabled)
            pred_masks = []

            for class_i, token in enumerate(token_list):
                if use_detector_path:
                    out = runner.forward_detector_raw(state, token)
                else:
                    out = runner.forward_find_raw(state, token)

                pred_masks_lowres = out.get("pred_masks", None)
                q_up, q_scores = runner.query_logits_and_scores(out, state["H"], state["W"])
                if q_up.numel() <= 0:
                    pred_masks.append(np.zeros((state["H"], state["W"]), dtype=np.bool_))
                    continue

                keep_idx = runner.select_queries(
                    q_scores=q_scores,
                    q_logits_lowres=pred_masks_lowres[0] if torch.is_tensor(pred_masks_lowres) else None,
                    score_threshold=query_score_threshold,
                    nms_threshold=infer_det_nms_thresh,
                )

                if keep_idx.numel() <= 0:
                    pred_masks.append(np.zeros((state["H"], state["W"]), dtype=np.bool_))
                    continue

                q_up_keep = q_up[keep_idx]
                q_scores_keep = q_scores[keep_idx]
                q_bin = q_up_keep >= infer_logit_threshold

                if prior_enabled and q_bin.numel() > 0:
                    q_feats, q_valid = query_region_features_from_binary_masks(
                        query_masks=q_bin,
                        gray_map=state["gray_map"],
                        std_map=state["std_map"],
                        min_pixels=prior_min_pixels,
                    )
                    mu = prior_models[class_i]["mu_inst"].view(1, 2)
                    sigma = prior_models[class_i]["sigma_inst"].clamp_min(1e-6).view(1, 2)
                    z = torch.abs((q_feats - mu) / sigma)
                    keep_prior = q_valid & (z[:, 0] <= prior_n_sigma) & (z[:, 1] <= prior_n_sigma)
                    if keep_prior.numel() > 0:
                        q_up_keep = q_up_keep[keep_prior]
                        q_scores_keep = q_scores_keep[keep_prior]
                        q_bin = q_bin[keep_prior]

                if q_up_keep.numel() <= 0:
                    pred_masks.append(np.zeros((state["H"], state["W"]), dtype=np.bool_))
                    continue

                if infer_query_topk > 0 and q_up_keep.shape[0] > infer_query_topk:
                    _, top_order = torch.topk(q_scores_keep, k=infer_query_topk, largest=True, sorted=True)
                    q_up_keep = q_up_keep[top_order]

                q_bin = q_up_keep >= infer_logit_threshold
                if infer_query_lcc and q_bin.numel() > 0:
                    q_np = q_bin.detach().cpu().numpy().astype(np.bool_)
                    q_np = np.stack([largest_component(m) for m in q_np], axis=0)
                    q_bin = torch.from_numpy(q_np).to(device=q_up_keep.device, dtype=torch.bool)

                pred_mask = torch.any(q_bin, dim=0).detach().cpu().numpy().astype(np.bool_)
                pred_masks.append(pred_mask)

        gt = np.zeros_like(lbl_np, dtype=np.int64)
        for c, v in enumerate(cfg["class_label_values"], start=1):
            gt[lbl_np == int(v)] = c

        if speckle_min_area > 0:
            pred_masks = [remove_speckles(m, speckle_min_area) for m in pred_masks]

        per_item = {"image_index": idx, "classes": {}}
        for c in range(1, num_classes + 1):
            gt_mask = gt == c
            pred_mask = pred_masks[c - 1]
            iou = binary_iou(pred_mask, gt_mask)
            dice = binary_dice(pred_mask, gt_mask)
            per_class_acc[c]["iou"].append(iou)
            per_class_acc[c]["dice"].append(dice)
            per_item["classes"][str(c)] = {"iou": iou, "dice": dice}

            vis_path = os.path.join(infer_dir, "class_viz", f"class_{c:02d}", f"test_{idx:03d}.png")
            save_class_figure(
                save_path=vis_path,
                image_np=np.array(test_images_viz[idx]),
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                title=f"test {idx} class {c} IoU={iou:.4f}",
            )

        per_image.append(per_item)

    summary = {"classes": {}}
    all_iou: List[float] = []
    all_dice: List[float] = []
    for c in range(1, num_classes + 1):
        iou_mean = float(np.mean(per_class_acc[c]["iou"])) if per_class_acc[c]["iou"] else 0.0
        dice_mean = float(np.mean(per_class_acc[c]["dice"])) if per_class_acc[c]["dice"] else 0.0
        summary["classes"][str(c)] = {"iou": iou_mean, "dice": dice_mean}
        all_iou.extend(per_class_acc[c]["iou"])
        all_dice.extend(per_class_acc[c]["dice"])

    summary["overall"] = {
        "iou": float(np.mean(all_iou)) if all_iou else 0.0,
        "dice": float(np.mean(all_dice)) if all_dice else 0.0,
    }

    metrics_path = os.path.join(infer_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== Summary ===\n")
        for cls, vals in summary["classes"].items():
            f.write(f"class {cls}: IoU={vals['iou']:.4f} Dice={vals['dice']:.4f}\n")
        f.write(f"overall: IoU={summary['overall']['iou']:.4f} Dice={summary['overall']['dice']:.4f}\n")
        f.write("\n=== Per-image ===\n")
        for item in per_image:
            f.write(f"image {item['image_index']}\n")
            for cls, vals in item["classes"].items():
                f.write(f"  class {cls}: IoU={vals['iou']:.4f} Dice={vals['dice']:.4f}\n")
            f.write("\n")

    print("=== Summary ===")
    for cls, vals in summary["classes"].items():
        print(f"class {cls}: IoU={vals['iou']:.4f} Dice={vals['dice']:.4f}")
    print(f"overall: IoU={summary['overall']['iou']:.4f} Dice={summary['overall']['dice']:.4f}")
    print(f"[infer] results saved to: {infer_dir}")


def main():
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.yaml",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_cfg)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = normalize_config_paths(cfg, args.config)
    run_infer(cfg)


if __name__ == "__main__":
    main()
