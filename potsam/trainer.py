# -*- coding: utf-8 -*-
"""POTSAM training loop (freeze SAM3, train prompt tokens only)."""

from __future__ import annotations

import math
import os
import shutil
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from sam3.train.loss.loss_fns import CORE_LOSS_KEY, SemanticSegCriterion

from .data import load_image, load_label_map, read_split_pairs, sample_n_shot
from .losses import dice_loss_from_probs, focal_loss_with_logits, token_cosine_penalty
from .prior import fit_instance_prior_models
from .sam3_builder import build_frozen_image_model, get_amp_dtype, prepare_output_dir, resolve_device
from .sam3_runner import Sam3Runner
from .tokens import boxes_from_label_map
from .viz import save_multiclass_figure, save_query_mask_grid, save_train_val_loss_curve


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = resolve_device(cfg["device"])
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.amp_dtype = get_amp_dtype(cfg["amp_dtype"])

        self.model = build_frozen_image_model(cfg, self.device)
        self.runner = Sam3Runner(
            self.model,
            self.device,
            cfg["use_autocast"],
            self.amp_dtype,
            prior_std_ksize=int(cfg.get("query_prior_std_ksize", 9)),
        )

        self.aug_cache: List[Dict[str, Any]] = []

    def _random_augment(
        self,
        img: Image.Image,
        lbl: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[Image.Image, np.ndarray]:
        aug_applied = False
        img_out = img.copy()
        lbl_out = lbl.copy()

        if rng.rand() < float(self.cfg.get("aug_prob_rot90", 0.0)):
            k = int(rng.choice([1, 2, 3]))
            if k == 1:
                img_out = img_out.transpose(Image.ROTATE_90)
                lbl_out = np.rot90(lbl_out, 1)
            elif k == 2:
                img_out = img_out.transpose(Image.ROTATE_180)
                lbl_out = np.rot90(lbl_out, 2)
            else:
                img_out = img_out.transpose(Image.ROTATE_270)
                lbl_out = np.rot90(lbl_out, 3)
            aug_applied = True

        if rng.rand() < float(self.cfg.get("aug_prob_hflip", 0.0)):
            img_out = img_out.transpose(Image.FLIP_LEFT_RIGHT)
            lbl_out = np.fliplr(lbl_out)
            aug_applied = True

        if rng.rand() < float(self.cfg.get("aug_prob_vflip", 0.0)):
            img_out = img_out.transpose(Image.FLIP_TOP_BOTTOM)
            lbl_out = np.flipud(lbl_out)
            aug_applied = True

        blur_prob = float(self.cfg.get("aug_prob_blur", 0.0))
        if blur_prob > 0.0 and rng.rand() < blur_prob:
            img_out = img_out.filter(ImageFilter.GaussianBlur(radius=float(self.cfg.get("aug_blur_sigma", 0.5))))
            aug_applied = True

        contrast_prob = float(self.cfg.get("aug_prob_contrast", 0.0))
        if contrast_prob > 0.0 and rng.rand() < contrast_prob:
            cmin = float(self.cfg.get("aug_contrast_min", 0.9))
            cmax = float(self.cfg.get("aug_contrast_max", 1.1))
            factor = float(rng.uniform(cmin, cmax))
            img_out = ImageEnhance.Contrast(img_out).enhance(factor)
            aug_applied = True

        if not aug_applied:
            img_out = img_out.transpose(Image.FLIP_LEFT_RIGHT)
            lbl_out = np.fliplr(lbl_out)

        return img_out, lbl_out.astype(np.int64)

    @staticmethod
    def _apply_geometric_transform(
        img: Image.Image,
        lbl: np.ndarray,
        rot_k: int,
        do_h: bool,
        do_v: bool,
    ) -> Tuple[Image.Image, np.ndarray]:
        out_img = img.copy()
        out_lbl = lbl.copy()
        if rot_k == 1:
            out_img = out_img.transpose(Image.ROTATE_90)
            out_lbl = np.rot90(out_lbl, 1)
        elif rot_k == 2:
            out_img = out_img.transpose(Image.ROTATE_180)
            out_lbl = np.rot90(out_lbl, 2)
        elif rot_k == 3:
            out_img = out_img.transpose(Image.ROTATE_270)
            out_lbl = np.rot90(out_lbl, 3)
        if do_h:
            out_img = out_img.transpose(Image.FLIP_LEFT_RIGHT)
            out_lbl = np.fliplr(out_lbl)
        if do_v:
            out_img = out_img.transpose(Image.FLIP_TOP_BOTTOM)
            out_lbl = np.flipud(out_lbl)
        return out_img, out_lbl.astype(np.int64)

    def _random_geometric_augment(
        self,
        img: Image.Image,
        lbl: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[Image.Image, np.ndarray, Tuple[int, bool, bool]]:
        rot_k = int(rng.randint(0, 4))
        do_h = bool(rng.randint(0, 2))
        do_v = bool(rng.randint(0, 2))
        img_aug, lbl_aug = self._apply_geometric_transform(img, lbl, rot_k, do_h, do_v)
        return img_aug, lbl_aug, (rot_k, do_h, do_v)

    @staticmethod
    def _invert_geometric_tensor(x: torch.Tensor, geom: Tuple[int, bool, bool]) -> torch.Tensor:
        rot_k, do_h, do_v = int(geom[0]), bool(geom[1]), bool(geom[2])
        y = x
        if do_v:
            y = torch.flip(y, dims=[-2])
        if do_h:
            y = torch.flip(y, dims=[-1])
        if rot_k != 0:
            y = torch.rot90(y, k=(4 - rot_k) % 4, dims=[-2, -1])
        return y

    def _build_aug_cache(self, train_images: List[Image.Image], train_labels: List[np.ndarray]) -> List[Dict[str, Any]]:
        cache: List[Dict[str, Any]] = []
        for base_idx, (base_img, base_lbl) in enumerate(zip(train_images, train_labels)):
            for rot_k in [0, 1, 2, 3]:
                for do_h in [False, True]:
                    for do_v in [False, True]:
                        img, lbl = self._apply_geometric_transform(base_img, base_lbl, rot_k, do_h, do_v)
                        cache.append(
                            {
                                "image": img,
                                "label": lbl.astype(np.int64),
                                "geom": (int(rot_k), bool(do_h), bool(do_v)),
                                "base_idx": int(base_idx),
                            }
                        )
        return cache

    def _init_tokens(self, train_images: List[Image.Image], train_labels: List[np.ndarray]) -> List[torch.Tensor]:
        init_first_only = bool(self.cfg.get("init_from_first_image", True))
        modes_cfg = self.cfg.get("token_init_modes", ["cls", "prompt_mean", "prompt_max"])
        if isinstance(modes_cfg, str):
            modes = [s.strip().lower() for s in modes_cfg.split(",") if s.strip()]
        else:
            modes = [str(s).strip().lower() for s in modes_cfg if str(s).strip()]
        modes = [m for m in modes if m in {"cls", "prompt_mean", "prompt_max", "random"}]
        if not modes:
            raise RuntimeError("token_init_modes is empty")

        use_images = [train_images[0]] if init_first_only else train_images
        use_labels = [train_labels[0]] if init_first_only else train_labels

        token_inits: List[torch.Tensor] = []
        for class_idx, class_val in enumerate(self.cfg["class_label_values"], start=1):
            cls_tokens_all = []
            prompt_tokens_all = []
            for img, lbl_np in zip(use_images, use_labels):
                boxes = boxes_from_label_map(lbl_np, int(class_val))
                if not boxes:
                    continue
                tokens_cpu = self.runner.encode_boxes_to_box_tokens_cpu(img, boxes)
                cls_tokens_all.append(tokens_cpu[-1].view(1, -1))
                prompt_tokens_all.append(tokens_cpu[:-1])

            if not cls_tokens_all or not prompt_tokens_all:
                raise RuntimeError(f"No GT boxes to init for class value {class_val}")

            cls_tokens_cpu = torch.cat(cls_tokens_all, dim=0)
            prompt_tokens_cpu = torch.cat(prompt_tokens_all, dim=0)
            cls_token = cls_tokens_cpu.mean(dim=0)
            prompt_mean = prompt_tokens_cpu.mean(dim=0)
            prompt_max = prompt_tokens_cpu.max(dim=0).values

            token_list = []
            if "cls" in modes:
                token_list.append(cls_token)
            if "prompt_mean" in modes:
                token_list.append(prompt_mean)
            if "prompt_max" in modes:
                token_list.append(prompt_max)
            if "random" in modes:
                rnd = torch.randn_like(cls_token)
                scale = cls_token.norm().clamp_min(1e-6)
                rnd = rnd / rnd.norm().clamp_min(1e-6) * scale
                token_list.append(rnd)

            t_init = torch.stack(token_list, dim=0)
            multiplier = int(self.cfg.get("token_init_multiplier", 1))
            if multiplier > 1:
                t_init = t_init.repeat(multiplier, 1)

            print(
                f"[init][class{class_idx}] value={class_val} "
                f"n_prompt_tokens={prompt_tokens_cpu.shape[0]} D={prompt_tokens_cpu.shape[1]} "
                f"init={'+'.join(modes)}"
            )
            token_inits.append(t_init.to(self.device))

        return token_inits

    def _build_instance_targets(self, label_np: np.ndarray, class_val: int) -> Dict[str, torch.Tensor]:
        semantic_mask = torch.from_numpy((label_np == int(class_val)).copy()).to(device=self.device, dtype=torch.bool)
        if semantic_mask.any():
            cc_map, n = ndimage.label(
                semantic_mask.detach().cpu().numpy().astype(np.uint8),
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            inst_list: List[torch.Tensor] = []
            for inst_id in range(1, int(n) + 1):
                m = torch.from_numpy(cc_map == inst_id).to(device=self.device, dtype=torch.bool)
                if m.any():
                    inst_list.append(m)
            if inst_list:
                inst_masks = torch.stack(inst_list, dim=0)
            else:
                inst_masks = torch.zeros((0, semantic_mask.shape[0], semantic_mask.shape[1]), device=self.device, dtype=torch.bool)
        else:
            inst_masks = torch.zeros((0, semantic_mask.shape[0], semantic_mask.shape[1]), device=self.device, dtype=torch.bool)

        return {
            "semantic_masks": semantic_mask.unsqueeze(0),
            "instance_masks": inst_masks,
            "num_instances": torch.tensor([int(inst_masks.shape[0])], device=self.device, dtype=torch.long),
        }

    @staticmethod
    def _largest_cc(mask_np: np.ndarray) -> np.ndarray:
        if (not mask_np.any()) or mask_np.ndim != 2:
            return mask_np
        h, w = mask_np.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        best = None
        best_area = 0
        ys, xs = np.where(mask_np)
        for y0, x0 in zip(ys, xs):
            if visited[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = 1
            pix_y: List[int] = []
            pix_x: List[int] = []
            while stack:
                y, x = stack.pop()
                pix_y.append(y)
                pix_x.append(x)
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or (not mask_np[ny, nx]):
                        continue
                    visited[ny, nx] = 1
                    stack.append((ny, nx))
            if len(pix_y) > best_area:
                best_area = len(pix_y)
                best = (pix_y, pix_x)
        out = np.zeros_like(mask_np, dtype=np.bool_)
        if best is not None:
            out[np.asarray(best[0], dtype=np.int64), np.asarray(best[1], dtype=np.int64)] = True
        return out

    def run(self):
        cfg = self.cfg
        out_dir = prepare_output_dir(cfg["output_dir"])
        cfg_path = cfg.get("_config_path", "")
        if cfg_path and os.path.isfile(cfg_path):
            shutil.copy2(cfg_path, os.path.join(out_dir, "config.yaml"))
        print(f"[output] root={cfg['output_dir']} run_dir={out_dir}")

        train_image_paths, train_label_paths = read_split_pairs(cfg, "train")
        train_n_shot = int(cfg.get("train_n_shot", 0))
        if train_n_shot > 0:
            train_image_paths, train_label_paths = sample_n_shot(
                train_image_paths,
                train_label_paths,
                train_n_shot,
                int(cfg.get("train_n_shot_seed", 2026)),
            )
            print(f"[n-shot] sampled {len(train_image_paths)} images")

        train_images: List[Image.Image] = []
        train_images_viz: List[Image.Image] = []
        train_labels: List[np.ndarray] = []
        for img_path, lbl_path in zip(train_image_paths, train_label_paths):
            proc, orig = load_image(img_path, with_views=True)
            train_images.append(proc)
            train_images_viz.append(orig)
            train_labels.append(load_label_map(lbl_path))

        if bool(cfg.get("overfit_single", False)):
            train_images = [train_images[0]]
            train_labels = [train_labels[0]]
            train_images_viz = [train_images_viz[0]]

        tokens_init = self._init_tokens(train_images, train_labels)

        prior_enabled = bool(cfg.get("query_prior_enabled", False))
        prior_models: List[Dict[str, torch.Tensor]] = []
        if prior_enabled:
            prior_models = fit_instance_prior_models(
                train_images=train_images,
                train_labels=train_labels,
                class_values=[int(v) for v in cfg["class_label_values"]],
                std_ksize=int(cfg.get("query_prior_std_ksize", 9)),
                device=self.device,
                eps=float(cfg.get("eps", 1e-6)),
                min_pixels=int(cfg.get("query_prior_min_pixels", 4)),
            )
            print("[prior] fitted GT-instance gray/std distribution")

        one_shot = (len(train_images) == 1) and (len(train_labels) == 1)
        use_cache = one_shot or bool(cfg.get("overfit_single", False))
        if use_cache:
            self.aug_cache = self._build_aug_cache(train_images, train_labels)
            if not self.aug_cache:
                raise RuntimeError("Aug cache is empty")

        tokens = [torch.nn.Parameter(t.detach().float().clone(), requires_grad=True) for t in tokens_init]

        query_main_weight = float(cfg.get("query_main_weight", 1.0))
        aux_semantic_weight = float(cfg.get("aux_semantic_weight", 0.0))
        query_score_threshold = float(cfg.get("query_score_threshold", 0.5))
        train_match_iou_threshold = float(cfg.get("train_match_iou_threshold", 0.8))
        train_unmatch_iou_threshold = float(cfg.get("train_unmatch_iou_threshold", 0.5))
        if train_unmatch_iou_threshold >= train_match_iou_threshold:
            raise RuntimeError("train_unmatch_iou_threshold must be < train_match_iou_threshold")

        instance_score_weight = float(cfg.get("instance_score_weight", 1.0))
        instance_mask_focal_weight = float(cfg.get("instance_mask_focal_weight", 1.0))
        instance_mask_dice_weight = float(cfg.get("instance_mask_dice_weight", 0.5))
        query_focal_alpha = float(cfg.get("query_focal_alpha", 0.8))
        query_focal_gamma = float(cfg.get("query_focal_gamma", 2.0))

        semantic_loss_weight = float(cfg.get("semantic_loss_weight", 1.0))
        semantic_dice_weight = float(cfg.get("semantic_dice_weight", 0.5))
        inv_consistency_weight = float(cfg.get("inv_consistency_weight", 0.0))
        token_reg_weight = float(cfg.get("token_reg_weight", 0.0))

        use_detector_path = bool(cfg.get("train_use_detector_path", True))
        query_main_enabled = query_main_weight != 0.0
        sem_enabled = (aux_semantic_weight != 0.0) and (
            (semantic_loss_weight != 0.0) or (semantic_dice_weight != 0.0)
        )

        sem_criterion = None
        if sem_enabled:
            sem_criterion = SemanticSegCriterion(
                weight_dict={
                    "loss_semantic_seg": semantic_loss_weight,
                    "loss_semantic_dice": semantic_dice_weight,
                },
                focal=bool(cfg.get("semantic_use_focal", True)),
                focal_alpha=float(cfg.get("semantic_focal_alpha", 0.8)),
                focal_gamma=float(cfg.get("semantic_focal_gamma", 2.0)),
                downsample=bool(cfg.get("semantic_downsample", True)),
                presence_head=False,
                presence_loss=False,
            )

        batch_size = int(cfg.get("batch_size", 1))
        grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
        train_epochs = int(cfg["train_epochs"])
        print_every = int(cfg.get("print_every", 10))

        lr_base = float(cfg["lr"])
        lr_schedule = str(cfg.get("lr_schedule", "cosine")).strip().lower()
        lr_warmup_epochs = max(0, int(cfg.get("lr_warmup_epochs", 20)))
        lr_warmup_start_ratio = float(cfg.get("lr_warmup_start_ratio", 0.1))
        lr_min_ratio = float(cfg.get("lr_min_ratio", 0.001))
        lr_plateau_shorten = bool(cfg.get("lr_plateau_shorten", True))

        patience = int(cfg.get("per_class_patience", 100))
        min_delta = float(cfg.get("per_class_min_delta", 1e-3))
        min_epochs = int(cfg.get("per_class_min_epochs", 100))

        token_ema = bool(cfg.get("token_ema", False))
        token_ema_decay = min(max(float(cfg.get("token_ema_decay", 0.99)), 0.0), 0.999999)
        token_ema_warmup_steps = max(0, int(cfg.get("token_ema_warmup_steps", 0)))

        vis_logit_threshold = float(cfg.get("infer_logit_threshold", 0.5))
        infer_query_topk = int(cfg.get("infer_query_topk", 0))
        infer_query_lcc = bool(cfg.get("infer_query_lcc", True))
        query_grid_max = int(cfg.get("query_grid_max_queries", 10))

        rng = np.random.RandomState(int(cfg.get("aug_seed", 0)))

        def _lr_factor(epoch_1based: int) -> float:
            if lr_schedule in {"none", "constant"}:
                return 1.0
            if lr_schedule != "cosine":
                raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")
            if lr_warmup_epochs > 0 and epoch_1based <= lr_warmup_epochs:
                progress = float(epoch_1based) / float(lr_warmup_epochs)
                return lr_warmup_start_ratio + (1.0 - lr_warmup_start_ratio) * progress
            denom = max(1, train_epochs - lr_warmup_epochs)
            progress = float(epoch_1based - lr_warmup_epochs) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

        def _set_lr(optimizer: torch.optim.Optimizer, epoch_1based: int, plateau_wait_local: int, schedule_progress: float):
            if lr_schedule in {"none", "constant"}:
                lr_now = lr_base
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                return lr_now, schedule_progress

            if lr_warmup_epochs > 0 and epoch_1based <= lr_warmup_epochs:
                lr_now = lr_base * _lr_factor(epoch_1based)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                return lr_now, schedule_progress

            denom = max(1, train_epochs - lr_warmup_epochs)
            base_progress = float(epoch_1based - lr_warmup_epochs) / float(denom)
            base_progress = min(max(base_progress, 0.0), 1.0)
            progress = base_progress
            if lr_plateau_shorten and patience > 0 and plateau_wait_local > 0:
                plateau_progress = min(1.0, float(plateau_wait_local) / float(patience))
                progress = max(progress, plateau_progress)
            schedule_progress = max(schedule_progress, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * schedule_progress))
            factor = lr_min_ratio + (1.0 - lr_min_ratio) * cosine
            lr_now = lr_base * factor
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            return lr_now, schedule_progress

        for class_idx, class_val in enumerate(cfg["class_label_values"], start=1):
            token_param = tokens[class_idx - 1]
            optimizer = torch.optim.AdamW([token_param], lr=lr_base, weight_decay=0.0)

            train_losses: List[float] = []
            val_losses: List[float] = []
            val_epochs: List[int] = []

            ema_token = token_param.detach().clone() if token_ema else None
            ema_steps = 0

            best_main_loss = float("inf")
            best_token = (ema_token if ema_token is not None else token_param).detach().clone()
            plateau_wait = 0
            schedule_progress = 0.0

            class_aug_cache = None
            class_aug_cache_by_base = None
            if use_cache:
                class_aug_cache = []
                class_aug_cache_by_base = {}
                for cache_item in self.aug_cache:
                    targets = self._build_instance_targets(cache_item["label"], int(class_val))
                    base_idx = int(cache_item.get("base_idx", 0))
                    item = {
                        "image": cache_item["image"],
                        "targets": targets,
                        "geom": cache_item["geom"],
                        "base_idx": base_idx,
                    }
                    class_aug_cache.append(item)
                    class_aug_cache_by_base.setdefault(base_idx, []).append(item)

            train_eval_state = self.runner.prepare_image(train_images[0])

            def _forward(state_local: Dict[str, Any], token_override: torch.Tensor | None = None):
                t = token_param if token_override is None else token_override
                if use_detector_path:
                    return self.runner.forward_detector_raw(state_local, t)
                return self.runner.forward_find_raw(state_local, t)

            def _eval_token() -> torch.Tensor:
                if token_ema and ema_token is not None:
                    return ema_token
                return token_param

            def _update_ema():
                nonlocal ema_steps, ema_token
                if (not token_ema) or (ema_token is None):
                    return
                ema_steps += 1
                cur_decay = token_ema_decay
                if token_ema_warmup_steps > 0 and ema_steps < token_ema_warmup_steps:
                    cur_decay = token_ema_decay * (float(ema_steps) / float(token_ema_warmup_steps))
                with torch.no_grad():
                    ema_token.mul_(cur_decay).add_(token_param.detach(), alpha=(1.0 - cur_decay))

            def _detector_keep_indices(out_local: Dict[str, Any]):
                pred_masks = out_local.get("pred_masks", None)
                pred_logits = out_local.get("pred_logits", None)
                if (not torch.is_tensor(pred_masks)) or pred_masks.numel() <= 0:
                    z = torch.zeros((0,), device=self.device, dtype=torch.long)
                    return z, torch.zeros((0,), device=self.device, dtype=torch.float32)

                q_num = int(pred_masks.shape[1])
                if (
                    torch.is_tensor(pred_logits)
                    and pred_logits.dim() >= 3
                    and pred_logits.shape[1] >= q_num
                ):
                    q_scores = torch.sigmoid(pred_logits[0, :q_num, 0].float())
                else:
                    q_scores = torch.ones((q_num,), device=pred_masks.device, dtype=torch.float32)
                keep = q_scores > query_score_threshold
                keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)
                return keep_idx, q_scores

            def _instance_query_losses(out_local: Dict[str, Any], state_local: Dict[str, Any], targets_local: Dict[str, torch.Tensor]):
                if not query_main_enabled:
                    return torch.zeros((), device=self.device)

                pred_masks = out_local.get("pred_masks", None)
                if (not torch.is_tensor(pred_masks)) or pred_masks.numel() <= 0:
                    return torch.zeros((), device=self.device)

                # Use low-resolution query masks directly for matching and mask supervision.
                q_logits = pred_masks[0].float()  # (Q, h, w)
                q_num = int(q_logits.shape[0])
                if q_num <= 0:
                    return torch.zeros((), device=self.device)

                q_prob = torch.sigmoid(q_logits).clamp(1e-6, 1.0 - 1e-6)
                inst_masks = targets_local.get("instance_masks", None)
                if not torch.is_tensor(inst_masks):
                    inst_masks = torch.zeros((0, q_logits.shape[-2], q_logits.shape[-1]), device=q_logits.device, dtype=torch.bool)
                if inst_masks.numel() > 0 and (
                    int(inst_masks.shape[-2]) != int(q_logits.shape[-2])
                    or int(inst_masks.shape[-1]) != int(q_logits.shape[-1])
                ):
                    inst_masks = (
                        F.interpolate(inst_masks.float().unsqueeze(1), size=q_logits.shape[-2:], mode="nearest")[:, 0] > 0.5
                    )

                pos_mask = torch.zeros((q_num,), device=q_logits.device, dtype=torch.bool)
                neg_mask = torch.zeros((q_num,), device=q_logits.device, dtype=torch.bool)
                assigned_gt_idx = torch.full((q_num,), -1, device=q_logits.device, dtype=torch.long)

                if inst_masks.shape[0] > 0:
                    q_flat = q_prob.reshape(q_num, -1)
                    g_flat = inst_masks.float().reshape(inst_masks.shape[0], -1)
                    inter = torch.matmul(q_flat, g_flat.t())
                    q_sum = q_flat.sum(dim=1, keepdim=True)
                    g_sum = g_flat.sum(dim=1).unsqueeze(0)
                    soft_iou = inter / (q_sum + g_sum - inter + 1e-6)
                    best_iou, best_gt_idx = torch.max(soft_iou, dim=1)

                    cost = (1.0 - soft_iou).detach().cpu().numpy()
                    if cost.size > 0:
                        row_ind, col_ind = linear_sum_assignment(cost)
                        if len(row_ind) > 0:
                            row_t = torch.from_numpy(row_ind).to(device=q_logits.device, dtype=torch.long)
                            col_t = torch.from_numpy(col_ind).to(device=q_logits.device, dtype=torch.long)
                            pos_mask[row_t] = True
                            assigned_gt_idx[row_t] = col_t

                    remain = ~pos_mask
                    extra_pos = remain & (best_iou >= train_match_iou_threshold)
                    pos_mask = pos_mask | extra_pos
                    assigned_gt_idx[extra_pos] = best_gt_idx[extra_pos]
                    neg_mask = remain & (best_iou < train_unmatch_iou_threshold)
                else:
                    neg_mask = torch.ones((q_num,), device=q_up.device, dtype=torch.bool)

                score_supervised = pos_mask | neg_mask
                pred_logits = out_local.get("pred_logits", None)
                if torch.is_tensor(pred_logits):
                    q_score_logits = pred_logits[0, :, 0].float()
                    if q_score_logits.numel() != q_num:
                        q_score_logits = torch.zeros((q_num,), device=q_logits.device, dtype=q_logits.dtype)
                else:
                    q_score_logits = torch.zeros((q_num,), device=q_logits.device, dtype=q_logits.dtype)

                loss_score = torch.zeros((), device=q_logits.device)
                if score_supervised.any():
                    logits = q_score_logits[score_supervised]
                    targets = pos_mask[score_supervised].float()
                    loss_score = focal_loss_with_logits(
                        logits,
                        targets,
                        alpha=query_focal_alpha,
                        gamma=query_focal_gamma,
                        reduction="mean",
                    )

                loss_mask = torch.zeros((), device=q_logits.device)
                if pos_mask.any() and inst_masks.shape[0] > 0:
                    pos_logits = q_logits[pos_mask]
                    pos_probs = q_prob[pos_mask]
                    gt_masks = inst_masks[assigned_gt_idx[pos_mask]].float()
                    mask_focal = focal_loss_with_logits(
                        pos_logits,
                        gt_masks,
                        alpha=query_focal_alpha,
                        gamma=query_focal_gamma,
                        reduction="mean",
                    )
                    mask_dice = dice_loss_from_probs(pos_probs, gt_masks)
                    loss_mask = instance_mask_focal_weight * mask_focal + instance_mask_dice_weight * mask_dice

                return instance_score_weight * loss_score + loss_mask

            def _semantic_prob(out_local: Dict[str, Any]) -> torch.Tensor:
                sem = out_local.get("semantic_seg", None)
                if not torch.is_tensor(sem) or sem.numel() <= 0:
                    return torch.zeros((1, 1), device=self.device, dtype=torch.float32)
                return torch.sigmoid(sem[0, 0].float()).clamp(1e-6, 1.0 - 1e-6)

            def _collect_query_masks_for_vis(out_local: Dict[str, Any], state_local: Dict[str, Any]):
                pred_masks = out_local.get("pred_masks", None)
                if not torch.is_tensor(pred_masks):
                    return [], [], []

                keep_idx, q_scores_all = _detector_keep_indices(out_local)
                if keep_idx.numel() <= 0:
                    return [], [], []

                keep_scores = q_scores_all[keep_idx]
                if infer_query_topk > 0 and keep_idx.numel() > infer_query_topk:
                    top_vals, top_order = torch.topk(keep_scores, k=infer_query_topk, largest=True, sorted=True)
                    keep_idx = keep_idx[top_order]
                    keep_scores = top_vals

                q_masks: List[np.ndarray] = []
                q_scores: List[float] = []
                q_ids: List[int] = []
                for rank, q_idx in enumerate(keep_idx.tolist()):
                    q_logit = pred_masks[0, int(q_idx)].float().unsqueeze(0).unsqueeze(0)
                    q_up = F.interpolate(q_logit, size=(state_local["H"], state_local["W"]), mode="bilinear", align_corners=False)[0, 0]
                    q_bin = (q_up >= vis_logit_threshold).detach().cpu().numpy().astype(np.bool_)
                    if infer_query_lcc:
                        q_bin = self._largest_cc(q_bin)
                    q_masks.append(q_bin)
                    q_scores.append(float(keep_scores[rank]))
                    q_ids.append(int(q_idx))
                return q_masks, q_scores, q_ids

            def _pred_from_queries(out_local: Dict[str, Any], state_local: Dict[str, Any]) -> torch.Tensor:
                q_up, _ = self.runner.query_logits_and_scores(out_local, state_local["H"], state_local["W"])
                if q_up.numel() <= 0:
                    return torch.zeros((state_local["H"], state_local["W"]), device=self.device, dtype=torch.bool)

                keep_idx, _ = _detector_keep_indices(out_local)
                if keep_idx.numel() <= 0:
                    return torch.zeros((state_local["H"], state_local["W"]), device=self.device, dtype=torch.bool)

                q_bin = q_up[keep_idx] >= vis_logit_threshold
                if infer_query_lcc and q_bin.numel() > 0:
                    q_list = []
                    for qb in q_bin:
                        q_np = self._largest_cc(qb.detach().cpu().numpy().astype(np.bool_))
                        q_list.append(torch.from_numpy(q_np).to(device=q_up.device, dtype=torch.bool))
                    q_bin = torch.stack(q_list, dim=0) if q_list else q_bin

                return q_bin.any(dim=0)

            def _eval_snapshot(tag: str):
                tr_state = train_eval_state
                eval_t = _eval_token()
                tr_out = _forward(tr_state, token_override=eval_t)

                tr_pred = _pred_from_queries(tr_out, tr_state)
                tr_q_masks, tr_q_scores, tr_q_ids = _collect_query_masks_for_vis(tr_out, tr_state)

                tr_gt = torch.from_numpy((train_labels[0] == int(class_val)).copy()).to(device=tr_pred.device)

                def _iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
                    inter = torch.logical_and(pred, gt).sum().item()
                    union = torch.logical_or(pred, gt).sum().item()
                    if union == 0:
                        return 1.0
                    return float((inter + 1e-6) / (union + 1e-6))

                tr_iou = _iou(tr_pred, tr_gt)

                save_multiclass_figure(
                    save_path=os.path.join(out_dir, f"class{class_idx:02d}_train_{tag}.png"),
                    image_np=np.array(train_images_viz[0]),
                    pred_labels=tr_pred.to(dtype=torch.int64),
                    gt_labels=tr_gt.to(dtype=torch.int64),
                    metric_value=tr_iou,
                    alpha=float(cfg.get("overlay_alpha", 0.45)),
                    num_classes=1,
                    title_prefix=f"train class{class_idx}",
                )
                save_query_mask_grid(
                    save_path=os.path.join(out_dir, f"class{class_idx:02d}_train_queries_{tag}.png"),
                    image_np=np.array(train_images_viz[0]),
                    query_masks=tr_q_masks,
                    title=f"train class{class_idx} queries",
                    query_scores=tr_q_scores,
                    query_ids=tr_q_ids,
                    max_queries=query_grid_max,
                    cols=8,
                )

            _eval_snapshot("before")

            for epoch in range(1, train_epochs + 1):
                lr_now, schedule_progress = _set_lr(optimizer, epoch, plateau_wait, schedule_progress)

                batch = []
                if use_cache:
                    if (
                        inv_consistency_weight > 0.0
                        and batch_size >= 2
                        and class_aug_cache_by_base is not None
                        and len(class_aug_cache_by_base) > 0
                    ):
                        base_keys = list(class_aug_cache_by_base.keys())
                        pair_num = batch_size // 2
                        for _ in range(pair_num):
                            base_k = base_keys[int(rng.randint(0, len(base_keys)))]
                            views = class_aug_cache_by_base[base_k]
                            if len(views) >= 2:
                                i = int(rng.randint(0, len(views)))
                                j = int(rng.randint(0, len(views) - 1))
                                if j >= i:
                                    j += 1
                                pair_items = [views[i], views[j]]
                            else:
                                pair_items = [views[0], views[0]]
                            for item in pair_items:
                                with torch.no_grad():
                                    state = self.runner.prepare_image(item["image"])
                                batch.append((state, item["targets"], item["geom"]))
                        if (batch_size % 2) == 1:
                            item = class_aug_cache[int(rng.randint(0, len(class_aug_cache)))]
                            with torch.no_grad():
                                state = self.runner.prepare_image(item["image"])
                            batch.append((state, item["targets"], item["geom"]))
                    else:
                        idxs = [int(rng.randint(0, len(class_aug_cache))) for _ in range(batch_size)]
                        for idx in idxs:
                            item = class_aug_cache[idx]
                            with torch.no_grad():
                                state = self.runner.prepare_image(item["image"])
                            batch.append((state, item["targets"], item["geom"]))
                else:
                    if inv_consistency_weight > 0.0 and batch_size >= 2:
                        pair_num = batch_size // 2
                        for _ in range(pair_num):
                            idx = int(rng.randint(0, len(train_images)))
                            img = train_images[idx]
                            lbl = train_labels[idx]
                            img_a, lbl_a, geom_a = self._random_geometric_augment(img, lbl, rng)
                            img_b, lbl_b, geom_b = self._random_geometric_augment(img, lbl, rng)
                            with torch.no_grad():
                                state_a = self.runner.prepare_image(img_a)
                                state_b = self.runner.prepare_image(img_b)
                            batch.append((state_a, self._build_instance_targets(lbl_a, int(class_val)), geom_a))
                            batch.append((state_b, self._build_instance_targets(lbl_b, int(class_val)), geom_b))
                        if (batch_size % 2) == 1:
                            idx = int(rng.randint(0, len(train_images)))
                            img, lbl = self._random_augment(train_images[idx], train_labels[idx], rng)
                            with torch.no_grad():
                                state = self.runner.prepare_image(img)
                            batch.append((state, self._build_instance_targets(lbl, int(class_val)), (0, False, False)))
                    else:
                        idxs = [int(rng.randint(0, len(train_images))) for _ in range(batch_size)]
                        for idx in idxs:
                            img, lbl = self._random_augment(train_images[idx], train_labels[idx], rng)
                            with torch.no_grad():
                                state = self.runner.prepare_image(img)
                            batch.append((state, self._build_instance_targets(lbl, int(class_val)), (0, False, False)))

                loss_acc = 0.0
                query_loss_acc = 0.0
                valid = 0
                grad_updates = 0
                det_keep_acc = 0.0
                gt_inst_acc = 0.0
                fwd_time = 0.0
                bwd_time = 0.0

                optimizer.zero_grad(set_to_none=True)
                sample_records = []
                for state, targets, geom in batch:
                    t0 = time.perf_counter()
                    out = _forward(state)
                    loss_query = _instance_query_losses(out, state, targets) if query_main_enabled else torch.zeros((), device=self.device)
                    loss_aux = torch.zeros((), device=self.device)
                    if sem_enabled:
                        loss_aux = sem_criterion(out_dict=out, targets=targets, is_aux=False)[CORE_LOSS_KEY]
                    t1 = time.perf_counter()
                    fwd_time += t1 - t0

                    keep_idx, _ = _detector_keep_indices(out)
                    sample_records.append(
                        {
                            "state": state,
                            "geom": geom,
                            "out": out,
                            "loss_query": loss_query,
                            "loss_aux": loss_aux,
                            "det_keep": int(keep_idx.numel()),
                            "gt_inst": int(targets["num_instances"][0].item()),
                        }
                    )

                if inv_consistency_weight > 0.0 and sem_enabled and len(sample_records) >= 2:
                    t0 = time.perf_counter()
                    pair_num = len(sample_records) // 2
                    for pair_i in range(pair_num):
                        ia = 2 * pair_i
                        ib = ia + 1
                        rec_a = sample_records[ia]
                        rec_b = sample_records[ib]
                        sem_a = _semantic_prob(rec_a["out"])
                        sem_b = _semantic_prob(rec_b["out"])
                        can_a = self._invert_geometric_tensor(sem_a, rec_a["geom"])
                        can_b = self._invert_geometric_tensor(sem_b, rec_b["geom"])
                        if can_a.shape != can_b.shape:
                            can_b = F.interpolate(can_b.unsqueeze(0).unsqueeze(0), size=can_a.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
                        loss_cons_a = F.l1_loss(can_a, can_b.detach())
                        loss_cons_b = F.l1_loss(can_b, can_a.detach())
                        rec_a["loss_aux"] = rec_a["loss_aux"] + 0.5 * inv_consistency_weight * loss_cons_a
                        rec_b["loss_aux"] = rec_b["loss_aux"] + 0.5 * inv_consistency_weight * loss_cons_b
                    fwd_time += time.perf_counter() - t0

                for rec in sample_records:
                    loss = query_main_weight * rec["loss_query"] + aux_semantic_weight * rec["loss_aux"]
                    if token_reg_weight > 0.0 and token_param.shape[0] > 1:
                        loss = loss * (1.0 + token_reg_weight * token_cosine_penalty(token_param))

                    loss_acc += float(loss.detach().item())
                    query_loss_acc += float(rec["loss_query"].detach().item())
                    valid += 1
                    det_keep_acc += float(rec["det_keep"])
                    gt_inst_acc += float(rec["gt_inst"])

                    loss_scaled = loss / float(grad_accum_steps)
                    if not loss_scaled.requires_grad:
                        continue

                    t0 = time.perf_counter()
                    loss_scaled.backward()
                    t1 = time.perf_counter()
                    bwd_time += t1 - t0
                    grad_updates += 1

                    if (grad_updates % grad_accum_steps) == 0:
                        grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
                        if grad_clip_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_([token_param], max_norm=grad_clip_norm)
                        optimizer.step()
                        _update_ema()
                        optimizer.zero_grad(set_to_none=True)
                        bwd_time += time.perf_counter() - t1

                if valid == 0:
                    continue

                if (grad_updates % grad_accum_steps) != 0:
                    grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_([token_param], max_norm=grad_clip_norm)
                    optimizer.step()
                    _update_ema()
                    optimizer.zero_grad(set_to_none=True)

                train_loss = loss_acc / float(valid)
                query_main_loss = query_loss_acc / float(valid)
                train_losses.append(train_loss)

                cur_main_loss = float(query_main_loss)
                if cur_main_loss < (best_main_loss - min_delta):
                    best_main_loss = cur_main_loss
                    best_token = _eval_token().detach().clone()
                    plateau_wait = 0
                else:
                    plateau_wait += 1

                should_print = (epoch % print_every) == 0 or epoch == 1 or epoch == train_epochs
                if should_print:
                    det_keep_mean = det_keep_acc / float(valid)
                    gt_inst_mean = gt_inst_acc / float(valid)

                    print(
                        f"[train][class{class_idx}] epoch {epoch:04d}/{train_epochs} "
                        f"loss={train_loss:.6f} "
                        f"det_keep={det_keep_mean:.2f}/{gt_inst_mean:.2f} "
                        f"lr={lr_now:.6f} time(fwd)={fwd_time:.3f}s time(bwd)={bwd_time:.3f}s"
                    )

                if epoch >= min_epochs and plateau_wait >= patience:
                    print(
                        f"[train][class{class_idx}] early-stop epoch={epoch} "
                        f"best_main_loss={best_main_loss:.6f} cur_main_loss={cur_main_loss:.6f}"
                    )
                    break

            save_train_val_loss_curve(
                save_path=os.path.join(out_dir, f"loss_curve_class{class_idx:02d}.png"),
                train_losses=train_losses,
                val_losses=val_losses,
                title=f"loss curves class {class_idx}",
                val_epochs=val_epochs,
            )

            with torch.no_grad():
                token_param.copy_(best_token)
                if token_ema and ema_token is not None:
                    ema_token.copy_(best_token)

            _eval_snapshot("after")

        ckpt_path = os.path.join(out_dir, "trained_tokens.pt")
        to_save = {f"T_class{i + 1}": t.detach().float().cpu() for i, t in enumerate(tokens)}
        if prior_enabled:
            to_save["query_prior"] = {
                "std_ksize": int(cfg.get("query_prior_std_ksize", 9)),
                "classes": [
                    {
                        "mu_inst": prior_models[i]["mu_inst"].detach().float().cpu(),
                        "sigma_inst": prior_models[i]["sigma_inst"].detach().float().cpu(),
                        "n_instances": prior_models[i]["n_instances"].detach().cpu(),
                    }
                    for i in range(len(prior_models))
                ],
            }

        torch.save(to_save, ckpt_path)
        print(f"[done] saved tokens: {ckpt_path}")
        print(f"[done] outputs in: {out_dir}")
        return out_dir
