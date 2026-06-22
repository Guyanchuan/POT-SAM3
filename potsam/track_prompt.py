# -*- coding: utf-8 -*-
"""SAM2-style prompt-based reverse tracking helpers for POTSAM."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from potsam.data import load_label_map, normalize_config_paths, read_split_pairs
    from potsam.sam3_builder import build_video_model_runtime, maybe_set_attr, resolve_device
else:
    from .data import load_label_map, normalize_config_paths, read_split_pairs
    from .sam3_builder import build_video_model_runtime, maybe_set_attr, resolve_device


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
        run_cfg = yaml.safe_load(f)
    run_cfg = normalize_config_paths(run_cfg, run_cfg_path)
    run_cfg["_track_latest_run_dir"] = run_dir
    run_cfg["_track_latest_run_config"] = run_cfg_path
    return run_cfg


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


def build_prompt_records_from_forward(
    obj_first_frame: Dict[int, int],
    obj_seed_box_norm: Dict[int, List[float]],
) -> List[Dict]:
    records: List[Dict] = []
    for obj_id in sorted(int(x) for x in obj_first_frame.keys()):
        frame_idx = int(obj_first_frame.get(int(obj_id), -1))
        box = obj_seed_box_norm.get(int(obj_id), None)
        if frame_idx < 0 or box is None:
            continue
        records.append(
            {
                "obj_id": int(obj_id),
                "frame_idx": int(frame_idx),
                "box_xywh_norm": [float(v) for v in box],
            }
        )
    return records


def save_prompt_records(path: str, tracker_video_dir: str, prompt_records: List[Dict], meta: Optional[Dict] = None):
    payload = {
        "tracker_video_dir": str(tracker_video_dir),
        "prompt_records": prompt_records,
        "meta": {} if meta is None else meta,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_prompt_records(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid prompt record format: {path}")
    return payload


def run_reverse_prompt_tracking(
    model,
    image_paths: List[str],
    tracker_video_dir: str,
    prompt_records: List[Dict],
    min_start_frame: int = 1,
    score_stop_threshold: float = -1.0,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], Dict]:
    if not image_paths:
        raise RuntimeError("No image paths for reverse prompt tracking")
    first_img = Image.open(image_paths[0]).convert("RGB")
    width, height = first_img.size
    width, height = int(width), int(height)

    reverse_obj_masks: Dict[int, Dict[int, np.ndarray]] = {}
    reverse_obj_tracker_scores: Dict[int, Dict[int, float]] = {}
    reverse_obj_det_scores: Dict[int, Dict[int, float]] = {}
    reverse_run_count = 0
    reverse_candidate_count = 0
    prompt_total_count = int(len(prompt_records))
    prompt_skipped_by_min_frame = 0
    prompt_skipped_invalid_box = 0
    prompt_skipped_missing_obj_id = 0
    obj_prompt_frame: Dict[int, int] = {}
    obj_runtime_id: Dict[int, int] = {}
    obj_stop_frame: Dict[int, int] = {}

    # Follow the official SAM2-style example:
    # predictor = sam3_model.tracker; predictor.backbone = sam3_model.detector.backbone.
    predictor = model.tracker
    predictor.backbone = model.detector.backbone

    valid_prompts: List[Dict] = []
    for rec in prompt_records:
        obj_id = int(rec.get("obj_id", -1))
        first_frame = int(rec.get("frame_idx", -1))
        box_xywh_norm = rec.get("box_xywh_norm", None)
        if obj_id < 0 or first_frame < int(min_start_frame):
            if first_frame < int(min_start_frame):
                prompt_skipped_by_min_frame += 1
            continue
        if not isinstance(box_xywh_norm, (list, tuple)) or len(box_xywh_norm) != 4:
            prompt_skipped_invalid_box += 1
            continue
        valid_prompts.append(
            {
                "obj_id": int(obj_id),
                "frame_idx": int(first_frame),
                "box_xywh_norm": [float(v) for v in box_xywh_norm],
            }
        )

    if not valid_prompts:
        stats = {
            "reverse_candidate_count": 0,
            "reverse_run_count": 0,
            "reverse_start_frame": -1,
            "reverse_stop_frames": {},
            "score_stop_threshold": float(score_stop_threshold),
            "prompt_total_count": int(prompt_total_count),
            "prompt_valid_count": 0,
            "prompt_bound_count": 0,
            "prompt_skipped_by_min_frame": int(prompt_skipped_by_min_frame),
            "prompt_skipped_invalid_box": int(prompt_skipped_invalid_box),
            "prompt_skipped_missing_obj_id": int(prompt_skipped_missing_obj_id),
            "obj_runtime_id_map": {},
        }
        return reverse_obj_masks, reverse_obj_tracker_scores, reverse_obj_det_scores, stats

    reverse_candidate_count = len(valid_prompts)
    reverse_start_frame = max(int(r["frame_idx"]) for r in valid_prompts)
    for rec in sorted(valid_prompts, key=lambda x: int(x["frame_idx"]), reverse=True):
        obj_id = int(rec["obj_id"])
        first_frame = int(rec["frame_idx"])
        x, y, bw, bh = [float(v) for v in rec["box_xywh_norm"]]
        box_xyxy_norm = np.asarray([[x, y, x + bw, y + bh]], dtype=np.float32)

        # One state per object: easy lifecycle control and hard stop.
        rev_state = predictor.init_state(video_path=tracker_video_dir)
        _, init_obj_ids, _, _ = predictor.add_new_points_or_box(
            inference_state=rev_state,
            frame_idx=first_frame,
            obj_id=obj_id,
            box=box_xyxy_norm,
        )
        init_obj_ids = [int(x) for x in init_obj_ids]
        if not init_obj_ids or int(obj_id) not in init_obj_ids:
            prompt_skipped_missing_obj_id += 1
            continue
        obj_prompt_frame[int(obj_id)] = int(first_frame)
        obj_runtime_id[int(obj_id)] = int(obj_id)
        reverse_run_count += 1

        for rev_frame_idx, rev_obj_ids, _, rev_video_masks, rev_obj_scores in predictor.propagate_in_video(
            inference_state=rev_state,
            start_frame_idx=first_frame,
            max_frame_num_to_track=first_frame + 1,
            reverse=True,
            tqdm_disable=True,
            propagate_preflight=True,
        ):
            rev_frame_idx = int(rev_frame_idx)
            if rev_frame_idx < 0 or rev_frame_idx >= len(image_paths):
                continue
            rev_obj_ids = [int(x) for x in rev_obj_ids]
            if int(obj_id) not in rev_obj_ids:
                continue

            sel_idx = rev_obj_ids.index(int(obj_id))
            if torch.is_tensor(rev_obj_scores):
                score_arr = rev_obj_scores.detach().float().view(-1).cpu().numpy()
            else:
                score_arr = np.asarray(rev_obj_scores, dtype=np.float32).reshape(-1)
            if sel_idx >= len(score_arr):
                continue

            score_prob = float(1.0 / (1.0 + np.exp(-float(score_arr[sel_idx]))))
            score_prob = float(np.clip(score_prob, 0.0, 1.0))
            if score_stop_threshold >= 0.0 and score_prob < float(score_stop_threshold):
                obj_stop_frame[int(obj_id)] = int(rev_frame_idx)
                break

            m = _extract_obj_mask(rev_obj_ids, rev_video_masks, int(obj_id), height, width)
            if m is None or (not m.any()):
                continue
            reverse_obj_masks.setdefault(rev_frame_idx, {})[int(obj_id)] = m
            reverse_obj_tracker_scores.setdefault(rev_frame_idx, {})[int(obj_id)] = score_prob
            reverse_obj_det_scores.setdefault(rev_frame_idx, {})[int(obj_id)] = score_prob

    stats = {
        "reverse_candidate_count": int(reverse_candidate_count),
        "reverse_run_count": int(reverse_run_count),
        "reverse_start_frame": int(reverse_start_frame),
        "reverse_stop_frames": {str(k): int(v) for k, v in sorted(obj_stop_frame.items())},
        "score_stop_threshold": float(score_stop_threshold),
        "prompt_total_count": int(prompt_total_count),
        "prompt_valid_count": int(len(valid_prompts)),
        "prompt_bound_count": int(len(obj_runtime_id)),
        "prompt_skipped_by_min_frame": int(prompt_skipped_by_min_frame),
        "prompt_skipped_invalid_box": int(prompt_skipped_invalid_box),
        "prompt_skipped_missing_obj_id": int(prompt_skipped_missing_obj_id),
        "obj_runtime_id_map": {str(k): int(v) for k, v in sorted(obj_runtime_id.items())},
    }
    return reverse_obj_masks, reverse_obj_tracker_scores, reverse_obj_det_scores, stats


def _overlay_mask(img_rgb: np.ndarray, mask_bool: np.ndarray, color=(255, 255, 255), alpha: float = 0.45) -> np.ndarray:
    out = img_rgb.astype(np.float32).copy()
    if mask_bool is not None and mask_bool.any():
        c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        sel = mask_bool.astype(bool)
        out[sel] = (1.0 - alpha) * out[sel] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def _binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float((inter + 1e-5) / (union + 1e-5))


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


def main():
    default_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_cfg)
    parser.add_argument("--prompt-record", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    cfg = _load_latest_run_config(args.config)
    prompt_payload = load_prompt_records(args.prompt_record)

    image_paths, label_paths = read_split_pairs(cfg, "test")
    label_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}
    if not image_paths:
        raise RuntimeError("No test images")

    tracker_video_dir = str(prompt_payload.get("tracker_video_dir", "")).strip()
    if not tracker_video_dir or not os.path.isdir(tracker_video_dir):
        run_dir = str(cfg.get("_track_latest_run_dir", _latest_run_dir(str(cfg["output_dir"]))))
        tracker_video_dir = _prepare_tracker_jpg_sequence(image_paths, run_dir)

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

    prompt_records = list(prompt_payload.get("prompt_records", []))

    reverse_obj_masks, _, _, stats = run_reverse_prompt_tracking(
        model=model,
        image_paths=image_paths,
        tracker_video_dir=tracker_video_dir,
        prompt_records=prompt_records,
        score_stop_threshold=float(cfg.get("track_remove_obj_score_threshold", -1.0)),
    )

    if args.out_dir:
        out_dir = args.out_dir
    else:
        beijing_tz = timezone(timedelta(hours=8))
        run_dir = str(cfg.get("_track_latest_run_dir", _latest_run_dir(str(cfg["output_dir"]))))
        out_dir = os.path.join(run_dir, f"track_prompt_{datetime.now(beijing_tz).strftime('%m%d%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    class_val = int(list(cfg.get("class_label_values", [255]))[0])
    ious: List[float] = []
    lines: List[str] = []
    for frame_idx, img_path in enumerate(image_paths):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        frame_masks = reverse_obj_masks.get(frame_idx, {})
        union = np.zeros(img_np.shape[:2], dtype=np.bool_)
        vis = img_np.copy()
        for obj_id in sorted(frame_masks.keys()):
            m = frame_masks[int(obj_id)]
            union |= m
            vis = _overlay_mask(vis, m, color=_color_for_obj_id(int(obj_id)), alpha=0.35)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = label_lookup.get(stem, None)
        if lbl_path is not None:
            gt_mask = load_label_map(lbl_path) == class_val
            iou = _binary_iou(union, gt_mask)
            ious.append(iou)
            lines.append(f"frame={frame_idx:04d} iou={iou:.6f}")
        Image.fromarray(vis).save(os.path.join(out_dir, f"reverse_{frame_idx:04d}.png"))

    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"mean_iou={float(np.mean(ious)) if ious else 0.0:.6f}\n")
        f.write(f"num_frames={len(ious)}\n")
        f.write(f"reverse_candidate_count={stats['reverse_candidate_count']}\n")
        f.write(f"reverse_run_count={stats['reverse_run_count']}\n")
        f.write(f"reverse_start_frame={stats.get('reverse_start_frame', -1)}\n")
        f.write(f"reverse_stop_frames={stats.get('reverse_stop_frames', {})}\n")
        f.write(f"score_stop_threshold={stats.get('score_stop_threshold', -1.0)}\n")
        f.write(f"prompt_total_count={stats.get('prompt_total_count', 0)}\n")
        f.write(f"prompt_valid_count={stats.get('prompt_valid_count', 0)}\n")
        f.write(f"prompt_bound_count={stats.get('prompt_bound_count', 0)}\n")
        f.write(f"prompt_skipped_by_min_frame={stats.get('prompt_skipped_by_min_frame', 0)}\n")
        f.write(f"prompt_skipped_invalid_box={stats.get('prompt_skipped_invalid_box', 0)}\n")
        f.write(f"prompt_skipped_missing_obj_id={stats.get('prompt_skipped_missing_obj_id', 0)}\n")
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(out_dir, "track_prompt_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"prompt_record={args.prompt_record}\n")
        f.write(f"tracker_video_dir={tracker_video_dir}\n")
        f.write(f"prompt_count={len(prompt_records)}\n")
        f.write(f"reverse_candidate_count={stats['reverse_candidate_count']}\n")
        f.write(f"reverse_run_count={stats['reverse_run_count']}\n")
        f.write(f"reverse_start_frame={stats.get('reverse_start_frame', -1)}\n")
        f.write(f"reverse_stop_frames={stats.get('reverse_stop_frames', {})}\n")
        f.write(f"score_stop_threshold={stats.get('score_stop_threshold', -1.0)}\n")
        f.write(f"prompt_total_count={stats.get('prompt_total_count', 0)}\n")
        f.write(f"prompt_valid_count={stats.get('prompt_valid_count', 0)}\n")
        f.write(f"prompt_bound_count={stats.get('prompt_bound_count', 0)}\n")
        f.write(f"prompt_skipped_by_min_frame={stats.get('prompt_skipped_by_min_frame', 0)}\n")
        f.write(f"prompt_skipped_invalid_box={stats.get('prompt_skipped_invalid_box', 0)}\n")
        f.write(f"prompt_skipped_missing_obj_id={stats.get('prompt_skipped_missing_obj_id', 0)}\n")
        f.write(f"obj_runtime_id_map={stats.get('obj_runtime_id_map', {})}\n")
    print(f"[track_prompt] saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
