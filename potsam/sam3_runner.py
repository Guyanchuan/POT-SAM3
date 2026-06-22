# -*- coding: utf-8 -*-
"""SAM3 frozen runner for token prompting."""

from __future__ import annotations

import types
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.perflib.nms import nms_masks

from .tokens import make_geometric_prompt_from_boxes, extract_box_tokens_and_cls


def tree_shallow_copy(x):
    if isinstance(x, dict):
        return {k: tree_shallow_copy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [tree_shallow_copy(v) for v in x]
    if isinstance(x, tuple):
        return tuple(tree_shallow_copy(v) for v in x)
    return x


def make_prompt_from_token_param(token_param: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if token_param.dim() == 1:
        prompt = token_param.view(1, 1, -1)
    elif token_param.dim() == 2:
        prompt = token_param.unsqueeze(1)
    else:
        raise ValueError(f"token dim must be 1 or 2, got {token_param.dim()}")
    mask = torch.zeros((1, int(prompt.shape[0])), device=token_param.device, dtype=torch.bool)
    return prompt, mask


class Sam3Runner:
    def __init__(
        self,
        model,
        device: torch.device,
        use_autocast: bool,
        amp_dtype: torch.dtype,
        prior_std_ksize: int = 9,
    ):
        self.model = model
        self.device = device
        self.use_autocast = bool(use_autocast)
        self.amp_dtype = amp_dtype
        self.prior_std_ksize = max(1, int(prior_std_ksize))
        if (self.prior_std_ksize % 2) == 0:
            self.prior_std_ksize += 1
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        self.default_text_prompt = "visual"

    def _autocast_ctx(self):
        if self.use_autocast and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _ensure_language_features(self, backbone_out: Dict[str, Any], batch_size: int = 1):
        required = ("language_features", "language_mask", "language_embeds")
        if all(k in backbone_out for k in required):
            return
        captions = [self.default_text_prompt for _ in range(max(1, int(batch_size)))]
        text_outputs = self.model.backbone.forward_text(captions, device=self.device)
        backbone_out.update(text_outputs)

    def prepare_image(self, image_pil: Image.Image, need_prior_maps: bool = False) -> Dict[str, Any]:
        st = self.processor.set_image(image_pil)
        self._ensure_language_features(st["backbone_out"], batch_size=1)
        state = {
            "backbone_out_base": st["backbone_out"],
            "find_input": self.processor.find_stage,
            "W": image_pil.size[0],
            "H": image_pil.size[1],
        }
        if need_prior_maps:
            gray_np = np.asarray(image_pil.convert("L"), dtype=np.float32) / 255.0
            gray = torch.from_numpy(gray_np).to(device=self.device, dtype=torch.float32)
            if self.prior_std_ksize > 1:
                x = gray.unsqueeze(0).unsqueeze(0)
                mean = F.avg_pool2d(
                    x,
                    kernel_size=self.prior_std_ksize,
                    stride=1,
                    padding=self.prior_std_ksize // 2,
                )
                mean2 = F.avg_pool2d(
                    x * x,
                    kernel_size=self.prior_std_ksize,
                    stride=1,
                    padding=self.prior_std_ksize // 2,
                )
                std = torch.sqrt((mean2 - mean * mean).clamp_min(0.0))[0, 0]
            else:
                std = torch.zeros_like(gray)
            state["gray_map"] = gray
            state["std_map"] = std
        return state

    @torch.no_grad()
    def encode_boxes_to_box_tokens_cpu(self, image_pil: Image.Image, boxes_xywh: List[List[float]]) -> torch.Tensor:
        st = self.processor.set_image(image_pil)
        find_input = self.processor.find_stage
        backbone_out = tree_shallow_copy(st["backbone_out"])
        self._ensure_language_features(backbone_out, batch_size=1)

        geo_prompt = make_geometric_prompt_from_boxes(
            model=self.model,
            image_size_wh=image_pil.size,
            boxes_xywh=boxes_xywh,
            device=self.device,
        )
        n_boxes = len(boxes_xywh)

        with self._autocast_ctx():
            prompt, _, _ = self.model._encode_prompt(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=geo_prompt,
                encode_text=False,
            )

        box_tokens, cls_token = extract_box_tokens_and_cls(prompt, n_boxes=n_boxes)
        tokens = torch.cat([box_tokens, cls_token.view(1, -1)], dim=0)
        return tokens.detach().float().cpu().contiguous()

    @contextmanager
    def _inject_visual_token_prompt(self, prompt_embed: torch.Tensor, prompt_mask: torch.Tensor):
        old_encode_prompt = self.model._encode_prompt

        def _wrapped_encode_prompt(_self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["encode_text"] = False
            kwargs["visual_prompt_embed"] = prompt_embed
            kwargs["visual_prompt_mask"] = prompt_mask
            return old_encode_prompt(*args, **kwargs)

        self.model._encode_prompt = types.MethodType(_wrapped_encode_prompt, self.model)
        try:
            yield
        finally:
            self.model._encode_prompt = old_encode_prompt

    def forward_find_raw(self, image_state: Dict[str, Any], token_param: torch.Tensor) -> Dict[str, torch.Tensor]:
        prompt, mask = make_prompt_from_token_param(token_param)
        backbone_in = tree_shallow_copy(image_state["backbone_out_base"])
        self._ensure_language_features(backbone_in, batch_size=1)

        with self._autocast_ctx():
            backbone_out, encoder_out, _ = self.model._run_encoder(
                backbone_out=backbone_in,
                find_input=image_state["find_input"],
                prompt=prompt,
                prompt_mask=mask,
            )

            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {"encoder_out": encoder_out, "backbone_out": backbone_out},
            }
            out, hs = self.model._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=mask,
                encoder_out=encoder_out,
            )
            self.model._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=image_state["find_input"].img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=mask,
                hs=hs,
            )
        return out

    def forward_detector_raw(self, image_state: Dict[str, Any], token_param: torch.Tensor) -> Dict[str, torch.Tensor]:
        prompt_embed, prompt_mask = make_prompt_from_token_param(token_param)
        backbone_in = tree_shallow_copy(image_state["backbone_out_base"])
        self._ensure_language_features(backbone_in, batch_size=1)
        geometric_prompt = self.model._get_dummy_prompt(num_prompts=1)

        with self._autocast_ctx(), self._inject_visual_token_prompt(prompt_embed, prompt_mask):
            out = self.model.forward_grounding(
                backbone_out=backbone_in,
                find_input=image_state["find_input"],
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
        return out

    @staticmethod
    def query_logits_and_scores(
        out: Dict[str, torch.Tensor],
        out_h: int,
        out_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q_up_logits, q_scores) with shapes (Q,H,W), (Q,)."""
        pred_masks = out.get("pred_masks", None)
        pred_logits = out.get("pred_logits", None)
        if (not torch.is_tensor(pred_masks)) or pred_masks.numel() <= 0:
            dev = pred_logits.device if torch.is_tensor(pred_logits) else torch.device("cpu")
            return (
                torch.zeros((0, out_h, out_w), device=dev, dtype=torch.float32),
                torch.zeros((0,), device=dev, dtype=torch.float32),
            )
        q_logits = pred_masks[0].float()
        q_up = F.interpolate(
            q_logits.unsqueeze(1), size=(out_h, out_w), mode="bilinear", align_corners=False
        )[:, 0]
        q_num = int(q_up.shape[0])

        if (
            torch.is_tensor(pred_logits)
            and pred_logits.dim() >= 3
            and pred_logits.shape[0] > 0
            and pred_logits.shape[1] >= q_num
        ):
            q_scores = torch.sigmoid(pred_logits[0, :q_num, 0].float())
        else:
            q_scores = torch.ones((q_num,), device=q_up.device, dtype=q_up.dtype)
        return q_up, q_scores

    @staticmethod
    def select_queries(
        q_scores: torch.Tensor,
        q_logits_lowres: torch.Tensor | None,
        score_threshold: float,
        nms_threshold: float,
    ) -> torch.Tensor:
        """Return kept query indices after score threshold (+optional mask-NMS)."""
        if q_scores.numel() <= 0:
            return torch.zeros((0,), device=q_scores.device, dtype=torch.long)

        keep = q_scores > float(score_threshold)
        if nms_threshold > 0.0 and torch.is_tensor(q_logits_lowres) and q_logits_lowres.numel() > 0:
            try:
                keep_nms = nms_masks(
                    pred_probs=q_scores,
                    pred_masks=q_logits_lowres.float(),
                    prob_threshold=float(score_threshold),
                    iou_threshold=float(nms_threshold),
                )
                keep = keep & keep_nms
            except Exception:
                pass

        return torch.nonzero(keep, as_tuple=False).view(-1)
