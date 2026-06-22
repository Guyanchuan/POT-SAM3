# -*- coding: utf-8 -*-
"""Detector+tracker video segmentation with trained token prompt."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from potsam.data import list_images_sorted, load_label_map, normalize_config_paths, read_split_pairs
    from potsam.sam3_builder import build_video_model_runtime, maybe_set_attr, resolve_device
    from potsam.track_prompt import (
        build_prompt_records_from_forward,
        load_prompt_records,
        run_reverse_prompt_tracking,
        save_prompt_records,
    )
else:
    from .data import list_images_sorted, load_label_map, normalize_config_paths, read_split_pairs
    from .sam3_builder import build_video_model_runtime, maybe_set_attr, resolve_device
    from .track_prompt import (
        build_prompt_records_from_forward,
        load_prompt_records,
        run_reverse_prompt_tracking,
        save_prompt_records,
    )


def _latest_run_dir(output_root: str) -> str:
    dirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    if not dirs:
        raise RuntimeError(f"No run directories in {output_root}")
    numeric = [d for d in dirs if d.isdigit()]
    if numeric:
        numeric.sort(key=lambda x: int(x))
        return os.path.join(output_root, numeric[-1])
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(output_root, d)))
    return os.path.join(output_root, dirs[-1])


def _load_latest_run_config(base_config_path: str) -> Dict:
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg = normalize_config_paths(base_cfg, base_config_path)

    output_root = str(base_cfg["output_dir"])
    run_dir = _latest_run_dir(output_root)
    run_cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(run_cfg_path):
        raise RuntimeError(f"Missing run config: {run_cfg_path}")

    with open(run_cfg_path, "r", encoding="utf-8") as f:
        run_cfg_raw = yaml.safe_load(f)

    # Prefer run-local resolution first.
    run_cfg = normalize_config_paths(run_cfg_raw, run_cfg_path)
    # Fallback: if copied run config still stores repo-relative paths (for example
    # "../data/..."), resolving against outputs/<run_id>/config.yaml will be wrong.
    # In that case, resolve against the base config path used for tracking.
    test_image_dir = str(run_cfg.get("test_image_dir", "")).strip()
    if test_image_dir and (not os.path.isdir(test_image_dir)):
        run_cfg = normalize_config_paths(run_cfg_raw, base_cfg["_config_path"])

    run_cfg["_track_latest_run_dir"] = run_dir
    run_cfg["_track_latest_run_config"] = run_cfg_path
    return run_cfg


def _overlay_mask(img_rgb: np.ndarray, mask_bool: np.ndarray, color=(255, 255, 255), alpha: float = 0.45) -> np.ndarray:
    out = img_rgb.astype(np.float32).copy()
    if mask_bool is not None and mask_bool.any():
        c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        sel = mask_bool.astype(bool)
        out[sel] = (1.0 - alpha) * out[sel] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def _dilate_bool(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        up = np.zeros_like(out)
        up[1:] = out[:-1]
        down = np.zeros_like(out)
        down[:-1] = out[1:]
        left = np.zeros_like(out)
        left[:, 1:] = out[:, :-1]
        right = np.zeros_like(out)
        right[:, :-1] = out[:, 1:]
        out = out | up | down | left | right
    return out


def _overlay_mask_boundary(img_rgb: np.ndarray, mask_bool: np.ndarray, color=(0, 255, 0), width: int = 2) -> np.ndarray:
    m = mask_bool.astype(bool)
    if not m.any():
        return img_rgb
    up = np.zeros_like(m)
    up[1:] = m[:-1]
    down = np.zeros_like(m)
    down[:-1] = m[1:]
    left = np.zeros_like(m)
    left[:, 1:] = m[:, :-1]
    right = np.zeros_like(m)
    right[:, :-1] = m[:, 1:]
    interior = m & up & down & left & right
    boundary = m & (~interior)
    if width > 1:
        boundary = _dilate_bool(boundary, iterations=width - 1)
    out = img_rgb.copy()
    out[boundary] = np.array(color, dtype=np.uint8)
    return out


def _binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float((inter + 1e-5) / (union + 1e-5))


def _prepare_tracker_jpg_sequence(image_paths: List[str], out_dir: str) -> str:
    frame_dir = os.path.join(out_dir, "tracker_frames_jpg")
    os.makedirs(frame_dir, exist_ok=True)
    for i, p in enumerate(image_paths):
        dst = os.path.join(frame_dir, f"{i:05d}.jpg")
        ext = os.path.splitext(p)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            shutil.copyfile(p, dst)
        else:
            Image.open(p).convert("RGB").save(dst, format="JPEG", quality=95, subsampling=0)
    return frame_dir


def _extract_obj_mask(
    obj_ids: List[int],
    video_res_masks,
    target_obj_id: int,
    h: int,
    w: int,
) -> np.ndarray:
    if video_res_masks is None or not obj_ids:
        return np.zeros((h, w), dtype=np.bool_)
    obj_ids = [int(x) for x in obj_ids]
    if int(target_obj_id) not in obj_ids:
        return np.zeros((h, w), dtype=np.bool_)

    idx = obj_ids.index(int(target_obj_id))
    m = video_res_masks[idx]
    if torch.is_tensor(m):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 3:
        m = m[0]

    mask = m > 0
    if mask.shape != (h, w):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    return mask.astype(np.bool_)


def _color_for_obj_id(obj_id: int) -> Tuple[int, int, int]:
    palette = [
        (255, 0, 0),
        (255, 165, 0),
        (255, 255, 0),
        (0, 180, 0),
        (0, 255, 255),
        (0, 128, 255),
        (180, 0, 255),
        (255, 0, 180),
        (210, 105, 30),
        (255, 105, 97),
    ]
    return palette[int(obj_id) % len(palette)]


def _draw_obj_scores(
    img_rgb: np.ndarray,
    frame_obj_masks: Dict[int, np.ndarray],
    frame_obj_tracker_scores: Dict[int, float],
    frame_obj_det_scores: Dict[int, float],
    obj_ids: List[int],
) -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    row = 0
    for obj_id in obj_ids:
        trk_score = frame_obj_tracker_scores.get(int(obj_id), None)
        det_score = frame_obj_det_scores.get(int(obj_id), None)
        trk_text = "NA" if trk_score is None else f"{trk_score:.2f}"
        det_text = "NA" if det_score is None else f"{det_score:.2f}"
        text = f"id{int(obj_id)} t:{trk_text} d:{det_text}"
        mask = frame_obj_masks.get(int(obj_id), None)
        if mask is not None and bool(mask.any()):
            ys, xs = np.where(mask)
            cx = int(xs.mean())
            cy = int(ys.mean())
            draw.text((cx, cy), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        else:
            draw.text((8, 8 + 18 * row), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            row += 1
    return np.array(pil)


def _mask_to_xywh_norm(mask_bool: np.ndarray, w: int, h: int) -> Optional[List[float]]:
    if mask_bool is None or (not mask_bool.any()):
        return None
    ys, xs = np.where(mask_bool)
    if ys.size <= 0 or xs.size <= 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    bw = x1 - x0 + 1.0
    bh = y1 - y0 + 1.0
    return [
        x0 / float(w),
        y0 / float(h),
        bw / float(w),
        bh / float(h),
    ]


def _resolve_track_token_path(cfg: Dict) -> Optional[str]:
    token_path = str(cfg.get("track_token_path", "")).strip()
    if token_path:
        return token_path

    run_dir = str(cfg.get("track_token_run_dir", "")).strip()
    if run_dir:
        return os.path.join(run_dir, "trained_tokens.pt")

    output_dir = str(cfg.get("output_dir", "")).strip()
    if not output_dir:
        return None
    try:
        latest_run = _latest_run_dir(output_dir)
    except Exception:
        return None
    return os.path.join(latest_run, "trained_tokens.pt")


def _load_track_token_prompt(cfg: Dict, class_idx: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    token_path = _resolve_track_token_path(cfg)
    if not token_path or not os.path.isfile(token_path):
        raise RuntimeError("Token file not found. Set track_token_path or track_token_run_dir")

    data = torch.load(token_path, map_location="cpu")
    token_key = str(cfg.get("track_token_key", "")).strip() or f"T_class{int(class_idx)}"
    if token_key not in data:
        keys = sorted([k for k in data.keys() if str(k).startswith("T_class")], key=str)
        if not keys:
            raise RuntimeError("No T_class* key in token file")
        token_key = keys[0]

    token = data[token_key]
    if not isinstance(token, torch.Tensor):
        raise RuntimeError(f"Token `{token_key}` is not tensor")

    token = token.detach().float().to(device)
    if token.dim() == 1:
        token = token.view(1, -1)
    if token.dim() != 2:
        raise RuntimeError(f"Token `{token_key}` must be [S,D] or [D], got {tuple(token.shape)}")

    prompt_embed = token.unsqueeze(1).contiguous()
    prompt_mask = torch.zeros((1, prompt_embed.shape[0]), device=prompt_embed.device, dtype=torch.bool)
    meta = f"path={token_path}, key={token_key}, shape={tuple(prompt_embed.shape)}"
    return prompt_embed, prompt_mask, meta


@contextmanager
def _inject_token_as_text_prompt(video_model, token_prompt_embed: torch.Tensor, token_prompt_mask: torch.Tensor):
    old_encode_prompt = video_model.detector._encode_prompt

    def _wrapped_encode_prompt(_self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["encode_text"] = False
        kwargs["visual_prompt_embed"] = token_prompt_embed
        kwargs["visual_prompt_mask"] = token_prompt_mask
        return old_encode_prompt(*args, **kwargs)

    video_model.detector._encode_prompt = types.MethodType(_wrapped_encode_prompt, video_model.detector)
    try:
        yield
    finally:
        video_model.detector._encode_prompt = old_encode_prompt


@contextmanager
def _force_tracking_only(video_model):
    old_det_track_one_frame = video_model._det_track_one_frame

    def _wrapped_det_track_one_frame(_self, *args, **kwargs):
        kwargs["allow_new_detections"] = False
        return old_det_track_one_frame(*args, **kwargs)

    video_model._det_track_one_frame = types.MethodType(_wrapped_det_track_one_frame, video_model)
    try:
        yield
    finally:
        video_model._det_track_one_frame = old_det_track_one_frame


@contextmanager
def _safe_video_postprocess_output(video_model):
    old_postprocess_output = video_model._postprocess_output

    def _wrapped_postprocess_output(
        _self,
        inference_state,
        out,
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        try:
            return old_postprocess_output(
                inference_state=inference_state,
                out=out,
                removed_obj_ids=removed_obj_ids,
                suppressed_obj_ids=suppressed_obj_ids,
                unconfirmed_obj_ids=unconfirmed_obj_ids,
            )
        except KeyError:
            obj_id_to_mask = {int(k): v for k, v in (out.get("obj_id_to_mask", {}) or {}).items()}
            obj_id_to_score = {int(k): float(v) for k, v in (out.get("obj_id_to_score", {}) or {}).items()}
            obj_id_to_tracker_score = {
                int(k): float(v)
                for k, v in (out.get("obj_id_to_tracker_score", {}) or {}).items()
            }
            for obj_id in obj_id_to_mask.keys():
                if obj_id not in obj_id_to_score:
                    obj_id_to_score[obj_id] = obj_id_to_tracker_score.get(obj_id, 0.0)
                if obj_id not in obj_id_to_tracker_score:
                    obj_id_to_tracker_score[obj_id] = obj_id_to_score.get(obj_id, 0.0)

            out_fixed = dict(out)
            out_fixed["obj_id_to_mask"] = obj_id_to_mask
            out_fixed["obj_id_to_score"] = obj_id_to_score
            out_fixed["obj_id_to_tracker_score"] = obj_id_to_tracker_score

            return old_postprocess_output(
                inference_state=inference_state,
                out=out_fixed,
                removed_obj_ids=removed_obj_ids,
                suppressed_obj_ids=suppressed_obj_ids,
                unconfirmed_obj_ids=unconfirmed_obj_ids,
            )

    video_model._postprocess_output = types.MethodType(_wrapped_postprocess_output, video_model)
    try:
        yield
    finally:
        video_model._postprocess_output = old_postprocess_output


@contextmanager
def _cap_detector_nms_candidates(max_candidates: int):
    if int(max_candidates) <= 0:
        yield
        return

    import sam3.model.sam3_image as sam3_image_mod

    old_nms_masks = sam3_image_mod.nms_masks
    max_candidates = int(max_candidates)

    def _wrapped_nms_masks(pred_probs: torch.Tensor, pred_masks: torch.Tensor, prob_threshold: float, iou_threshold: float):
        is_valid = pred_probs > prob_threshold
        valid_idx = torch.where(is_valid)[0]
        if valid_idx.numel() <= max_candidates:
            return old_nms_masks(pred_probs, pred_masks, prob_threshold, iou_threshold)

        topk_pos = torch.topk(pred_probs[valid_idx], k=max_candidates, sorted=False).indices
        selected_idx = valid_idx[topk_pos]

        keep = torch.zeros_like(is_valid, dtype=torch.bool)
        sub_keep = old_nms_masks(
            pred_probs[selected_idx],
            pred_masks[selected_idx],
            prob_threshold=-1e9,
            iou_threshold=iou_threshold,
        )
        keep[selected_idx] = sub_keep
        return keep

    sam3_image_mod.nms_masks = _wrapped_nms_masks
    try:
        yield
    finally:
        sam3_image_mod.nms_masks = old_nms_masks


def run_track(cfg: Dict):
    image_paths, label_paths = read_split_pairs(cfg, "test")
    label_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}
    if not image_paths:
        raise RuntimeError("No test images")

    class_values = [int(v) for v in cfg.get("class_label_values", [255])]
    class_val = int(class_values[0])
    class_idx = 1

    anchor_img = Image.open(image_paths[0]).convert("RGB")
    width, height = anchor_img.size

    latest_run_dir = str(cfg.get("_track_latest_run_dir", "")).strip()
    if not latest_run_dir:
        latest_run_dir = _latest_run_dir(str(cfg["output_dir"]))
    beijing_tz = timezone(timedelta(hours=8))
    out_dir = os.path.join(latest_run_dir, f"track_{datetime.now(beijing_tz).strftime('%m%d%H%M%S')}")
    forward_dir = os.path.join(out_dir, "forward")
    bidir_dir = os.path.join(out_dir, "bidir")
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(bidir_dir, exist_ok=True)

    preprocessed_dir = str(cfg.get("track_preprocessed_frames_dir", "")).strip()
    if preprocessed_dir:
        tracker_frames = list_images_sorted(preprocessed_dir)
        if not tracker_frames:
            raise RuntimeError(f"No images found in track_preprocessed_frames_dir: {preprocessed_dir}")
        if len(tracker_frames) != len(image_paths):
            raise RuntimeError(
                "track_preprocessed_frames_dir frame count mismatch: "
                f"{len(tracker_frames)} vs test frames {len(image_paths)}"
            )
        if all(os.path.splitext(p)[1].lower() in {".jpg", ".jpeg"} for p in tracker_frames):
            tracker_video_dir = preprocessed_dir
            print(f"[track] reuse preprocessed JPEG frames: {tracker_video_dir}", flush=True)
        else:
            tracker_video_dir = _prepare_tracker_jpg_sequence(tracker_frames, out_dir)
            print(f"[track] converted preprocessed frames to JPEG: {tracker_video_dir}", flush=True)
    else:
        tracker_video_dir = _prepare_tracker_jpg_sequence(image_paths, out_dir)
        print(f"[track] prepared JPEG frames: {tracker_video_dir}", flush=True)

    device = resolve_device(cfg["device"])
    model = build_video_model_runtime(cfg, device)

    maybe_set_attr(model, "score_threshold_detection", cfg.get("track_score_threshold_detection", None))
    maybe_set_attr(model, "det_nms_thresh", cfg.get("track_det_nms_thresh", None))
    maybe_set_attr(model, "new_det_thresh", cfg.get("track_new_det_thresh", None))
    maybe_set_attr(model, "assoc_iou_thresh", cfg.get("track_assoc_iou_thresh", None))
    maybe_set_attr(model, "trk_assoc_iou_thresh", cfg.get("track_trk_assoc_iou_thresh", None))
    maybe_set_attr(
        model,
        "suppress_overlapping_based_on_recent_occlusion_threshold",
        cfg.get("track_suppress_overlapping_threshold", None),
    )

    track_remove_obj_score_threshold = float(cfg.get("track_remove_obj_score_threshold", -1.0))
    track_nms_max_candidates = int(cfg.get("track_nms_max_candidates", 128))
    track_gt_boundary_width = int(cfg.get("track_gt_boundary_width", 2))
    track_save_binary_masks = bool(cfg.get("track_save_binary_masks", False))
    track_binary_mask_dirname = str(cfg.get("track_binary_mask_dirname", "binary_masks")).strip() or "binary_masks"
    forward_binary_dir = os.path.join(forward_dir, track_binary_mask_dirname)
    if track_save_binary_masks:
        os.makedirs(forward_binary_dir, exist_ok=True)

    token_text_placeholder = str(cfg.get("track_token_text_placeholder", "token")).strip() or "token"
    token_prompt_embed, token_prompt_mask, token_prompt_meta = _load_track_token_prompt(
        cfg=cfg,
        class_idx=class_idx,
        device=str(device),
    )

    forward_obj_masks: Dict[int, Dict[int, np.ndarray]] = {}
    forward_obj_tracker_scores: Dict[int, Dict[int, float]] = {}
    forward_obj_det_scores: Dict[int, Dict[int, float]] = {}
    tracked_obj_ids_set = set()
    removed_obj_ids = set()
    obj_first_frame: Dict[int, int] = {}
    obj_seed_box_norm: Dict[int, List[float]] = {}
    obj_seed_mask: Dict[int, np.ndarray] = {}
    ious_forward: List[float] = []
    lines_forward: List[str] = []

    def _write_metrics(dir_path: str, ious: List[float], lines: List[str]):
        with open(os.path.join(dir_path, "metrics.txt"), "w", encoding="utf-8") as f:
            mean_iou = float(np.mean(ious)) if ious else 0.0
            f.write(f"mean_iou={mean_iou:.6f}\n")
            f.write(f"num_frames={len(ious)}\n")
            f.write("\n".join(lines) + "\n")

    # Pass-1: forward with auto-detection enabled.
    inference_state = model.init_state(resource_path=tracker_video_dir, video_loader_type="cv2")
    inference_state["visual_prompt_embed"] = token_prompt_embed
    inference_state["visual_prompt_mask"] = token_prompt_mask
    with (
        _cap_detector_nms_candidates(track_nms_max_candidates),
        _inject_token_as_text_prompt(model, token_prompt_embed, token_prompt_mask),
        _safe_video_postprocess_output(model),
    ):
        _, init_out = model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=token_text_placeholder,
            boxes_xywh=None,
            box_labels=None,
        )
        if init_out is not None:
            init_ids = np.asarray(init_out.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
            print(f"[track][forward] add_prompt frame=0000 objs={int(init_ids.size)}", flush=True)

        for frame_idx, out in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=len(image_paths),
            reverse=False,
        ):
            if out is None:
                continue
            frame_idx = int(frame_idx)
            if frame_idx < 0 or frame_idx >= len(image_paths):
                continue

            raw_obj_ids = [int(x) for x in np.asarray(out.get("out_obj_ids", []), dtype=np.int64).tolist()]
            out_masks = out.get("out_binary_masks", None)
            out_probs = np.asarray(out.get("out_probs", []), dtype=np.float32).reshape(-1)
            tracker_score_frame = (
                inference_state.get("tracker_metadata", {})
                .get("obj_id_to_tracker_score_frame_wise", {})
                .get(frame_idx, {})
            )

            frame_masks: Dict[int, np.ndarray] = {}
            frame_scores: Dict[int, float] = {}
            frame_tracker_scores: Dict[int, float] = {}
            frame_det_scores: Dict[int, float] = {}

            def _remove_obj_runtime(obj_id: int):
                frame_masks.pop(obj_id, None)
                frame_scores.pop(obj_id, None)
                frame_tracker_scores.pop(obj_id, None)
                frame_det_scores.pop(obj_id, None)
                removed_obj_ids.add(obj_id)
                try:
                    model.remove_object(inference_state=inference_state, obj_id=int(obj_id), is_user_action=False)
                except Exception:
                    pass

            for idx, obj_id in enumerate(raw_obj_ids):
                if int(obj_id) in removed_obj_ids:
                    continue
                mask = _extract_obj_mask(raw_obj_ids, out_masks, int(obj_id), height, width)
                frame_masks[int(obj_id)] = mask

                trk_score = None
                det_score = None
                if isinstance(tracker_score_frame, dict) and (int(obj_id) in tracker_score_frame):
                    trk_score = float(tracker_score_frame[int(obj_id)])
                if idx < len(out_probs):
                    det_score = float(out_probs[idx])

                if trk_score is not None:
                    frame_tracker_scores[int(obj_id)] = float(np.clip(trk_score, 0.0, 1.0))
                if det_score is not None:
                    frame_det_scores[int(obj_id)] = float(np.clip(det_score, 0.0, 1.0))
                if trk_score is not None:
                    frame_scores[int(obj_id)] = float(np.clip(trk_score, 0.0, 1.0))
                elif det_score is not None:
                    frame_scores[int(obj_id)] = float(np.clip(det_score, 0.0, 1.0))

            if track_remove_obj_score_threshold >= 0.0 and frame_scores:
                to_remove = [
                    int(obj_id)
                    for obj_id, s in frame_scores.items()
                    if float(s) < track_remove_obj_score_threshold
                ]
                for obj_id in to_remove:
                    _remove_obj_runtime(obj_id)

            for obj_id, mask in frame_masks.items():
                tracked_obj_ids_set.add(int(obj_id))
                if (obj_id not in obj_first_frame) and mask.any():
                    obj_first_frame[int(obj_id)] = int(frame_idx)
                    box_norm = _mask_to_xywh_norm(mask, width, height)
                    if box_norm is not None:
                        obj_seed_box_norm[int(obj_id)] = box_norm
                        obj_seed_mask[int(obj_id)] = mask.copy()

            forward_obj_masks[frame_idx] = frame_masks
            forward_obj_tracker_scores[frame_idx] = frame_tracker_scores
            forward_obj_det_scores[frame_idx] = frame_det_scores

            union_mask = np.zeros((height, width), dtype=np.bool_)
            for m in frame_masks.values():
                union_mask |= m
            if track_save_binary_masks:
                mask_u8 = (union_mask.astype(np.uint8) * 255)
                Image.fromarray(mask_u8).save(
                    os.path.join(forward_binary_dir, f"mask_{frame_idx:04d}.png")
                )

            img_np = np.array(Image.open(image_paths[frame_idx]).convert("RGB"))
            stem = os.path.splitext(os.path.basename(image_paths[frame_idx]))[0]
            lbl_path = label_lookup.get(stem, None)
            if lbl_path is not None:
                gt_mask = load_label_map(lbl_path) == int(class_val)
                iou = _binary_iou(union_mask, gt_mask)
                ious_forward.append(iou)
                lines_forward.append(f"frame={frame_idx:04d} iou={iou:.6f}")
                vis = _overlay_mask_boundary(
                    img_np, gt_mask, color=(0, 255, 0), width=track_gt_boundary_width
                )
                iou_text = f"{iou:.6f}"
            else:
                vis = img_np
                iou_text = "NA"

            for obj_id in sorted(frame_masks.keys()):
                obj_mask = frame_masks.get(int(obj_id), None)
                if obj_mask is None or not obj_mask.any():
                    continue
                vis = _overlay_mask(vis, obj_mask, color=_color_for_obj_id(int(obj_id)), alpha=0.35)

            frame_obj_ids = sorted(frame_masks.keys())
            vis = _draw_obj_scores(vis, frame_masks, frame_tracker_scores, frame_det_scores, frame_obj_ids)
            vis_path = os.path.join(forward_dir, f"track_{frame_idx:04d}.png")
            Image.fromarray(vis).save(vis_path)
            print(
                f"[track][forward] frame={frame_idx:04d} objs={len(frame_obj_ids)} "
                f"removed={len(removed_obj_ids)} iou={iou_text}",
                flush=True,
            )

    # Pass-2: reverse tracking-only via prompt record + SAM2-style prompt tracker helper.
    prompt_records = build_prompt_records_from_forward(
        obj_first_frame=obj_first_frame,
        obj_seed_box_norm=obj_seed_box_norm,
    )
    reverse_prompt_record_path = os.path.join(out_dir, "reverse_prompts.json")
    save_prompt_records(
        path=reverse_prompt_record_path,
        tracker_video_dir=tracker_video_dir,
        prompt_records=prompt_records,
        meta={
            "source": "track.py_forward_pass",
            "num_frames": int(len(image_paths)),
            "class_value": int(class_val),
        },
    )
    prompt_payload = load_prompt_records(reverse_prompt_record_path)
    prompt_records_loaded = list(prompt_payload.get("prompt_records", []))
    prompt_tracker_video_dir = str(prompt_payload.get("tracker_video_dir", "")).strip() or tracker_video_dir

    reverse_obj_masks, reverse_obj_tracker_scores, reverse_obj_det_scores, reverse_stats = run_reverse_prompt_tracking(
        model=model,
        image_paths=image_paths,
        tracker_video_dir=prompt_tracker_video_dir,
        prompt_records=prompt_records_loaded,
        min_start_frame=1,
        score_stop_threshold=track_remove_obj_score_threshold,
    )
    reverse_candidate_count = int(reverse_stats.get("reverse_candidate_count", 0))
    reverse_run_count = int(reverse_stats.get("reverse_run_count", 0))
    reverse_start_frame = int(reverse_stats.get("reverse_start_frame", -1))
    reverse_stop_frames = dict(reverse_stats.get("reverse_stop_frames", {}))
    reverse_prompt_total_count = int(reverse_stats.get("prompt_total_count", 0))
    reverse_prompt_valid_count = int(reverse_stats.get("prompt_valid_count", 0))
    reverse_prompt_bound_count = int(reverse_stats.get("prompt_bound_count", 0))

    # Forward + reverse fusion (forward has priority).
    fused_obj_masks: Dict[int, Dict[int, np.ndarray]] = {}
    fused_obj_tracker_scores: Dict[int, Dict[int, float]] = {}
    fused_obj_det_scores: Dict[int, Dict[int, float]] = {}
    ious_bidir: List[float] = []
    lines_bidir: List[str] = []
    for frame_idx in range(len(image_paths)):
        frame_masks = dict(forward_obj_masks.get(frame_idx, {}))
        frame_tracker_scores = dict(forward_obj_tracker_scores.get(frame_idx, {}))
        frame_det_scores = dict(forward_obj_det_scores.get(frame_idx, {}))

        for obj_id, rev_mask in reverse_obj_masks.get(frame_idx, {}).items():
            obj_id = int(obj_id)
            if obj_id in frame_masks and frame_masks[obj_id] is not None:
                frame_masks[obj_id] = np.logical_or(frame_masks[obj_id], rev_mask)
            else:
                frame_masks[obj_id] = rev_mask

            rev_trk = reverse_obj_tracker_scores.get(frame_idx, {}).get(obj_id, None)
            if rev_trk is not None:
                if obj_id in frame_tracker_scores:
                    frame_tracker_scores[obj_id] = max(float(frame_tracker_scores[obj_id]), float(rev_trk))
                else:
                    frame_tracker_scores[obj_id] = float(rev_trk)

            rev_det = reverse_obj_det_scores.get(frame_idx, {}).get(obj_id, None)
            if rev_det is not None:
                if obj_id in frame_det_scores:
                    frame_det_scores[obj_id] = max(float(frame_det_scores[obj_id]), float(rev_det))
                else:
                    frame_det_scores[obj_id] = float(rev_det)

        fused_obj_masks[frame_idx] = frame_masks
        fused_obj_tracker_scores[frame_idx] = frame_tracker_scores
        fused_obj_det_scores[frame_idx] = frame_det_scores

        union_mask = np.zeros((height, width), dtype=np.bool_)
        for m in frame_masks.values():
            union_mask |= m

        img_np = np.array(Image.open(image_paths[frame_idx]).convert("RGB"))
        stem = os.path.splitext(os.path.basename(image_paths[frame_idx]))[0]
        lbl_path = label_lookup.get(stem, None)
        if lbl_path is not None:
            gt_mask = load_label_map(lbl_path) == int(class_val)
            iou = _binary_iou(union_mask, gt_mask)
            ious_bidir.append(iou)
            lines_bidir.append(f"frame={frame_idx:04d} iou={iou:.6f}")
            vis = _overlay_mask_boundary(img_np, gt_mask, color=(0, 255, 0), width=track_gt_boundary_width)
        else:
            vis = img_np

        for obj_id in sorted(frame_masks.keys()):
            obj_mask = frame_masks.get(int(obj_id), None)
            if obj_mask is None or not obj_mask.any():
                continue
            vis = _overlay_mask(vis, obj_mask, color=_color_for_obj_id(int(obj_id)), alpha=0.35)
        frame_obj_ids = sorted(frame_masks.keys())
        vis = _draw_obj_scores(vis, frame_masks, frame_tracker_scores, frame_det_scores, frame_obj_ids)
        Image.fromarray(vis).save(os.path.join(bidir_dir, f"track_{frame_idx:04d}.png"))

    _write_metrics(forward_dir, ious_forward, lines_forward)
    _write_metrics(bidir_dir, ious_bidir, lines_bidir)

    # Consolidated IoU report for all visualization groups.
    iou_report_path = os.path.join(out_dir, "iou_groups.txt")
    with open(iou_report_path, "w", encoding="utf-8") as f:
        def _dump_group(name: str, ious: List[float], lines: List[str]):
            mean_iou = float(np.mean(ious)) if ious else 0.0
            f.write(f"[{name}]\n")
            f.write(f"mean_iou={mean_iou:.6f}\n")
            f.write(f"num_frames={len(ious)}\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

        _dump_group("forward", ious_forward, lines_forward)
        _dump_group("bidir", ious_bidir, lines_bidir)

    tracked_obj_ids = sorted({int(k) for fm in fused_obj_masks.values() for k in fm.keys()})
    mean_forward = float(np.mean(ious_forward)) if ious_forward else 0.0
    mean_bidir = float(np.mean(ious_bidir)) if ious_bidir else 0.0

    info_lines = [
        f"track_latest_run_dir={latest_run_dir}",
        f"track_latest_run_config={cfg.get('_track_latest_run_config', '')}",
        f"test_image_dir={cfg.get('test_image_dir', '')}",
        f"test_label_dir={cfg.get('test_label_dir', '')}",
        f"tracker_video_dir={tracker_video_dir}",
        f"track_preprocessed_frames_dir={preprocessed_dir}",
        f"reverse_prompt_record_path={reverse_prompt_record_path}",
        f"reverse_prompt_count={len(prompt_records_loaded)}",
        f"class_value={class_val}",
        f"class_idx={class_idx}",
        "forward_pass_auto_detection=true",
        "reverse_pass_auto_detection=false",
        "reverse_pass_api=tracker.add_new_points_or_box(box_prompt)",
        f"track_remove_obj_score_threshold={track_remove_obj_score_threshold}",
        f"track_nms_max_candidates={track_nms_max_candidates}",
        f"track_save_binary_masks={track_save_binary_masks}",
        f"track_binary_mask_dir={forward_binary_dir if track_save_binary_masks else ''}",
        f"track_token_prompt={token_prompt_meta}",
        f"tracked_obj_ids={tracked_obj_ids}",
        f"removed_obj_ids={sorted(int(x) for x in removed_obj_ids)}",
        f"reverse_candidate_count={reverse_candidate_count}",
        f"reverse_run_count={reverse_run_count}",
        f"reverse_start_frame={reverse_start_frame}",
        f"reverse_stop_frames={reverse_stop_frames}",
        f"reverse_prompt_total_count={reverse_prompt_total_count}",
        f"reverse_prompt_valid_count={reverse_prompt_valid_count}",
        f"reverse_prompt_bound_count={reverse_prompt_bound_count}",
        f"total_frames={len(image_paths)}",
        f"num_eval_frames_forward={len(ious_forward)}",
        f"num_eval_frames_bidir={len(ious_bidir)}",
        f"mean_iou_forward={mean_forward:.6f}",
        f"mean_iou_bidir={mean_bidir:.6f}",
        f"forward_dir={forward_dir}",
        f"bidir_dir={bidir_dir}",
        f"compile={bool(cfg.get('compile', False))}",
        f"device={str(device)}",
    ]

    if hasattr(model, "score_threshold_detection"):
        info_lines.append(f"score_threshold_detection={float(model.score_threshold_detection):.6f}")
    if hasattr(model, "det_nms_thresh"):
        info_lines.append(f"det_nms_thresh={float(model.det_nms_thresh):.6f}")
    if hasattr(model, "new_det_thresh"):
        info_lines.append(f"new_det_thresh={float(model.new_det_thresh):.6f}")
    if hasattr(model, "assoc_iou_thresh"):
        info_lines.append(f"assoc_iou_thresh={float(model.assoc_iou_thresh):.6f}")
    if hasattr(model, "trk_assoc_iou_thresh"):
        info_lines.append(f"trk_assoc_iou_thresh={float(model.trk_assoc_iou_thresh):.6f}")
    if hasattr(model, "suppress_overlapping_based_on_recent_occlusion_threshold"):
        info_lines.append(
            "suppress_overlapping_based_on_recent_occlusion_threshold="
            f"{float(model.suppress_overlapping_based_on_recent_occlusion_threshold):.6f}"
        )

    with open(os.path.join(out_dir, "track_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")

    print(f"[track] saved to: {out_dir}")
    print(f"[track] tracked objects={len(tracked_obj_ids)}")
    print(f"[track] forward mean IoU: {mean_forward:.6f} ({len(ious_forward)} frames)")
    print(f"[track] bidir mean IoU: {mean_bidir:.6f} ({len(ious_bidir)} frames)")
    return out_dir


def main():
    default_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "config.yaml",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_cfg)
    args = parser.parse_args()

    cfg = _load_latest_run_config(args.config)
    print(f"[track] using latest run config: {cfg.get('_track_latest_run_config', '')}", flush=True)
    run_track(cfg)


if __name__ == "__main__":
    main()
